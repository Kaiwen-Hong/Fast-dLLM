"""T2 — THE feasibility gate: full E2B SFT train step on the 5090.

Real SFTDiffusion + hackable_diffusion (same graph as sft_smoke.py) at FULL E2B
scale and the internal geometry: vocab 262144, prompt 512, canvas 2x256, bf16
params, the B5 adafactor chain, donated buffers. Synthetic batch (random-init
weights) — this measures memory/throughput, not learning.

Usage: python e2b_feas_t2_train_step.py --batch N [--remat]
Exit codes: 0 = pass, 2 = OOM (caught), 1 = other error.
Emits /home/kaiwen/data/dgemma_e2b/feas/t2_train_b{N}{_remat}.json.
"""

import argparse
import functools
import json
import os
import sys
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.93")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2b_feas_common as common  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--remat", action="store_true")
args = parser.parse_args()

tag = f"b{args.batch}" + ("_remat" if args.remat else "")
out_path = os.path.join(common.RESULTS_DIR, f"t2_train_{tag}.json")
OUT = {"batch": args.batch, "remat": args.remat}

if args.remat:
  common.enable_block_remat()  # must run BEFORE model class is used

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from gemma.diffusion.hackable_diffusion_adapter.hd import (  # noqa: E402
    hd_gemma_network, sft_model,
)
from hackable_diffusion import hd  # noqa: E402
from hackable_diffusion.lib.training import discrete_loss  # noqa: E402

VOCAB = 262_144
PROMPT_LEN, CANVAS, NUM_CANVASES = 512, 256, 2  # internal geometry [internal·reply]
TOTAL_CANVAS = CANVAS * NUM_CANVASES
FULL = PROMPT_LEN + TOTAL_CANVAS
B = args.batch

print(f"T2[{tag}]: backend={jax.default_backend()} geometry prompt={PROMPT_LEN} canvas={NUM_CANVASES}x{CANVAS}")
assert jax.default_backend() == "gpu"


def main() -> int:
  gemma_model = common.DiffusionGemma_E2B(text_only=True, dtype=jnp.bfloat16)
  network = hd_gemma_network.WrappedDiffusionGemmaNetwork(gemma_model=gemma_model)
  corruption = hd.corruption.CategoricalProcess.uniform_process(
      num_categories=VOCAB, schedule=hd.corruption.RFSchedule()
  )
  time_sampler = hd.training.time_sampling.UniformTimeSampler(
      span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4)
  )
  model = sft_model.SFTDiffusion(
      x0="batch.canvas", prompt="batch.prompt", canvas_id="batch.canvas_id",
      canvas_mask="batch.canvas_mask", encoder_target="batch.encoder_target",
      encoder_target_mask="batch.encoder_target_mask",
      corruption_process=corruption, time_sampler=time_sampler,
      gemma_network=network, prompt_len=PROMPT_LEN, canvas_size=CANVAS,
      num_canvases=NUM_CANVASES,
  )

  ks = jax.random.split(jax.random.PRNGKey(0), 6)
  batch = dict(
      x0=jax.random.randint(ks[1], (B, TOTAL_CANVAS, 1), 1, VOCAB),
      prompt=jax.random.randint(ks[0], (B, PROMPT_LEN), 1, VOCAB),
      canvas_id=jnp.broadcast_to(
          jnp.repeat(jnp.arange(NUM_CANVASES, dtype=jnp.int32), CANVAS)[None, :],
          (B, TOTAL_CANVAS),
      ),
      canvas_mask=jnp.ones((B, TOTAL_CANVAS), bool),
      encoder_target=jax.random.randint(ks[2], (B, FULL), 1, VOCAB),
      encoder_target_mask=jnp.ones((B, FULL), jnp.float32),
  )

  t = time.time()
  variables = model.init({"params": ks[3], "sampling": ks[4]}, **batch, is_training=True)
  params = variables["params"]
  jax.block_until_ready(params)
  OUT["init_s"] = round(time.time() - t, 1)
  n = sum(x.size for x in jax.tree_util.tree_leaves(params))
  OUT["n_params"] = n
  OUT["mem_after_init"] = common.gpu_memory_stats()
  print(f"T2[{tag}]: SFT init OK {n/1e9:.3f}B params in {OUT['init_s']}s | mem {OUT['mem_after_init']}")

  def _scalar(x):
    return x.loss if hasattr(x, "loss") else jnp.asarray(x)

  def loss_fn(p, rng):
    preds = model.apply({"params": p}, **batch, is_training=True, rngs={"sampling": rng})
    dloss = _scalar(discrete_loss.compute_discrete_diffusion_loss(
        preds=preds["output"], targets=preds["target"],
        time=preds["noise_info"]["time"], use_mask=True, mask_key="target_mask",
    )).mean()
    eloss = sft_model.EncoderARLoss().get_values(
        preds["encoder_logits"], preds["encoder_target"], preds["encoder_target_mask"],
    ).mean()
    return dloss + eloss, (dloss, eloss)

  # B5 optimizer chain (LR constant 3e-5 for the smoke)
  opt = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.scale_by_factored_rms(),
      optax.add_decayed_weights(1e-4),
      optax.scale_by_learning_rate(3e-5),
  )
  opt_state = opt.init(params)

  @functools.partial(jax.jit, donate_argnums=(0, 1))
  def train_step(p, st, rng):
    (tot, (dl, el)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, rng)
    updates, st = opt.update(grads, st, p)
    p = optax.apply_updates(p, updates)
    return p, st, tot, dl, el

  rng = ks[5]
  step_times, losses = [], []
  for i in range(3):
    rng, sub = jax.random.split(rng)
    t = time.time()
    params, opt_state, tot, dl, el = train_step(params, opt_state, sub)
    jax.block_until_ready(tot)
    dt_ = time.time() - t
    step_times.append(round(dt_, 2))
    losses.append(round(float(tot), 4))
    print(f"T2[{tag}]: step {i}: total={float(tot):.4f} (diff={float(dl):.4f} enc={float(el):.4f}) {dt_:.1f}s")

  OUT["compile_step_s"] = step_times[0]
  OUT["steady_step_s"] = step_times[-1]
  OUT["losses"] = losses
  OUT["mem_peak"] = common.gpu_memory_stats()
  print(f"T2[{tag}]: PASS | steady {OUT['steady_step_s']}s/step | mem {OUT['mem_peak']}")
  return 0


try:
  code = main()
  OUT["status"] = "pass"
except Exception as e:  # noqa: BLE001
  msg = repr(e)
  oom = "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg or "OOM" in msg
  OUT["status"] = "oom" if oom else "error"
  OUT["error"] = msg[:2000]
  traceback.print_exc()
  code = 2 if oom else 1
  print(f"T2[{tag}]: {'OOM' if oom else 'ERROR'}")

os.makedirs(common.RESULTS_DIR, exist_ok=True)
with open(out_path, "w") as f:
  json.dump(OUT, f, indent=2)
print(f"T2[{tag}]: wrote {out_path}")
sys.exit(code)
