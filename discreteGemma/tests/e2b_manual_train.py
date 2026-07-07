"""Manual E2B ChartQA SFT training loop (T2-proven memory profile).

Fallback for the kauldron trainstep's ~43.75GB allocation (BLOCKERS.md #7):
same model / data / losses / optimizer as configs/sft_chartqa_e2b.py, but the
step is the hand-rolled donate-buffers loop that the T2 feasibility PASSED at
peak 17.9GB. Saves RAW orbax params at milestone steps.

Usage: python e2b_manual_train.py --variant it [--steps 1600]
Emits: /home/kaiwen/data/dgemma_e2b/xp_manual_{variant}/params_{step}/ (orbax)
       results/{variant}/train_log.jsonl (losses)
"""

import argparse
import functools
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.93")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", "/home/kaiwen/data/dgemma_e2b/xla_cache")

REPO = "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma"
sys.path.insert(0, f"{REPO}/gemma")
DATA = "/home/kaiwen/data/dgemma_e2b"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from kauldron import konfig  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variant", choices=["pt", "it"], required=True)
  ap.add_argument("--steps", type=int, default=1600)
  ap.add_argument("--peak_lr", type=float, default=3e-5)
  ap.add_argument("--warmup", type=int, default=20)
  args = ap.parse_args()
  save_steps = sorted({args.steps // 4, args.steps // 2, args.steps})

  os.environ["DGEMMA_E2B_VARIANT"] = args.variant
  os.environ["DGEMMA_E2B_BATCH"] = "1"
  os.environ["DGEMMA_E2B_ACCUM"] = "1"

  # Remat BEFORE model resolution (config-scoped pattern).
  from gemma.diffusion.hackable_diffusion_adapter.configs import sft_chartqa as _base
  _base._enable_gradient_checkpointing()  # pylint: disable=protected-access

  from gemma.diffusion.hackable_diffusion_adapter.configs import sft_chartqa_e2b
  from gemma.diffusion.hackable_diffusion_adapter.data.chartqa import chartqa_data as cq
  from gemma.diffusion.hackable_diffusion_adapter.hd import gemma_checkpointer
  from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model as sm
  from hackable_diffusion.lib.training import discrete_loss
  import flax

  cfg = sft_chartqa_e2b.get_config()
  model = konfig.resolve(cfg.model)

  train_paths = tuple(sorted(
      __import__("glob").glob(f"{DATA}/chartqa/human_train-*.arrayrecord")))
  ds = konfig.resolve(cq.make_chartqa_records_ds(
      training=True, batch_size=1, paths=train_paths, fmt="arrayrecord",
      num_workers=2))
  it = iter(ds)

  def clean(b):
    return {k: jnp.asarray(np.asarray(v)) for k, v in b.items()}

  batch0 = clean(next(it))
  print(f"[{args.variant}] batch keys: {sorted(batch0.keys())}")

  # ---- init (multimodal trace via the REAL batch) + production loader chain ----
  t = time.time()
  variables = model.init(
      {"params": jax.random.PRNGKey(0), "sampling": jax.random.PRNGKey(1)},
      **batch0, is_training=True,
  )
  params = variables["params"]
  print(f"[{args.variant}] init {sum(x.size for x in jax.tree_util.tree_leaves(params))/1e9:.3f}B in {time.time()-t:.0f}s")

  gm_params = params["gemma_network"]["gemma_model"]
  merged = gemma_checkpointer.cheaply_load_params(
      params_from_state=gm_params,
      checkpoint_path=f"{DATA}/ckpts/gemma4-e2b-{args.variant}",
      expected_missing=("self_conditioner",),
      coverage_json_path=f"{REPO}/results/{args.variant}/load_coverage_train.json",
  )
  params = dict(params)
  params["gemma_network"] = dict(params["gemma_network"])
  params["gemma_network"]["gemma_model"] = merged
  flat = flax.traverse_util.flatten_dict(params, sep="/")
  n_zero = 0
  for k in list(flat):
    if "self_conditioner/ffw/linear" in k:
      flat[k] = jnp.zeros_like(flat[k])
      n_zero += 1
  assert n_zero >= 1
  params = flax.traverse_util.unflatten_dict(flat, sep="/")
  print(f"[{args.variant}] AR ckpt loaded + sc W3 zeroed ({n_zero} leaves)")

  # ---- losses (identical pieces to the config) ----
  def _scalar(x):
    return x.loss if hasattr(x, "loss") else jnp.asarray(x)

  def loss_fn(p, batch, rng):
    preds = model.apply({"params": p}, **batch, is_training=True,
                        rngs={"sampling": rng})
    d = _scalar(discrete_loss.compute_discrete_diffusion_loss(
        preds=preds["output"], targets=preds["target"],
        time=preds["noise_info"]["time"], use_mask=True,
        mask_key="target_mask")).mean()
    e = sm.EncoderARLoss().get_values(
        preds["encoder_logits"], preds["encoder_target"],
        preds["encoder_target_mask"]).mean()
    return d + e, (d, e)

  schedule = optax.warmup_cosine_decay_schedule(
      0.0, args.peak_lr, args.warmup, args.steps, args.peak_lr / 10)
  opt = optax.chain(
      optax.clip_by_global_norm(1.0),
      optax.scale_by_factored_rms(),
      optax.add_decayed_weights(1e-4),
      optax.scale_by_learning_rate(schedule),
  )
  opt_state = opt.init(params)

  @functools.partial(jax.jit, donate_argnums=(0, 1))
  def train_step(p, st, batch, rng):
    (tot, (d, e)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, batch, rng)
    updates, st = opt.update(grads, st, p)
    p = optax.apply_updates(p, updates)
    return p, st, tot, d, e

  # ---- loop ----
  import orbax.checkpoint as ocp
  workdir = f"{DATA}/xp_manual_{args.variant}"
  os.makedirs(workdir, exist_ok=True)
  log_path = f"{REPO}/results/{args.variant}/train_log.jsonl"
  os.makedirs(os.path.dirname(log_path), exist_ok=True)
  logf = open(log_path, "w")

  rng = jax.random.PRNGKey(42)
  batch = batch0
  t0 = time.time()
  for step in range(1, args.steps + 1):
    rng, sub = jax.random.split(rng)
    params, opt_state, tot, d, e = train_step(params, opt_state, batch, sub)
    if step % 25 == 0 or step == 1:
      jax.block_until_ready(tot)
      rec = {"step": step, "total": float(tot), "diffusion": float(d),
             "encoder": float(e), "elapsed_s": round(time.time() - t0, 1)}
      logf.write(json.dumps(rec) + "\n"); logf.flush()
      print(f"[{args.variant}] {rec}")
    if step in save_steps:
      jax.block_until_ready(params)
      path = f"{workdir}/params_{step}"
      ocp.PyTreeCheckpointer().save(path, jax.device_get(params))
      print(f"[{args.variant}] saved {path}")
    batch = clean(next(it))

  # sc liveness (C2 post-train)
  flat = flax.traverse_util.flatten_dict(params, sep="/")
  lin = sum(float(jnp.abs(v).sum()) for k, v in flat.items()
            if "self_conditioner/ffw/linear" in k)
  with open(f"{REPO}/results/{args.variant}/sc_liveness.json", "w") as f:
    json.dump({"w3_abs_sum_after_train": lin, "alive": bool(lin > 0)}, f)
  print(f"[{args.variant}] sc W3 |sum| after train = {lin} (alive={lin>0})")
  print(f"[{args.variant}] TRAIN DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
  main()
