"""Step 3: test the OFFICIAL DiffusionGemma SFT path (hackable_diffusion).

Uses the real SFTDiffusion model + hackable_diffusion uniform categorical
corruption + UniformTimeSampler + NoWeightDiscreteLoss, on a TINY gemma config
with synthetic data. Runs init -> forward -> (diffusion loss + encoder AR loss)
-> one Adam step, all on the RTX 5090. No weights, no bagz dataset, no kauldron Trainer.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma")

import jax, jax.numpy as jnp, optax
from gemma.gm.nn.gemma4 import _config, _modules
from gemma.diffusion import _models as dm
from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model, hd_gemma_network
from hackable_diffusion import hd
from hackable_diffusion.lib.training import discrete_loss

print("jax", jax.__version__, "|", jax.default_backend(), "|", jax.devices())

VOCAB, B, PROMPT_LEN, CANVAS, NUM_CANVASES = 256, 2, 8, 8, 1
TOTAL_CANVAS = CANVAS * NUM_CANVASES
FULL = PROMPT_LEN + TOTAL_CANVAS

# ---- tiny gemma backbone (dense, no MoE, no vision) ----
attn = _config.make_attention_layers_types((_modules.AttentionType.LOCAL_SLIDING,), num_layers=2)
cfg = _config.TransformerConfig(
    num_embed=VOCAB, embed_dim=64, hidden_dim=128, num_heads=2, head_dim=16, num_kv_heads=1,
    final_logit_softcap=None, use_post_attn_norm=True, use_post_ffw_norm=True,
    attention_types=attn, sliding_window_size=32, qk_norm_with_scale=True,
    global_rope_proportion=0.25, local_rope_proportion=1.0, per_layer_input_dim=0,
    enable_moe=False, vision_encoder=None, audio_encoder=None,
)
gemma_model = dm.DiffusionGemma_26B_A4B(config=cfg, dtype=jnp.float32)
network = hd_gemma_network.WrappedDiffusionGemmaNetwork(gemma_model=gemma_model)

# ---- REAL hackable_diffusion pieces (same as sft_sudoku.py) ----
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
    corruption_process=corruption, time_sampler=time_sampler, gemma_network=network,
    prompt_len=PROMPT_LEN, canvas_size=CANVAS, num_canvases=NUM_CANVASES,
)

# ---- synthetic SFT batch (prompt + target canvas) ----
ks = jax.random.split(jax.random.PRNGKey(0), 6)
batch = dict(
    x0=jax.random.randint(ks[1], (B, TOTAL_CANVAS, 1), 1, VOCAB),   # target canvas tokens [B,L,1] (HD convention)
    prompt=jax.random.randint(ks[0], (B, PROMPT_LEN), 1, VOCAB),    # 1..V-1 (0 = pad)
    canvas_id=jnp.zeros((B, TOTAL_CANVAS), jnp.int32),
    canvas_mask=jnp.ones((B, TOTAL_CANVAS), bool),
    encoder_target=jax.random.randint(ks[2], (B, FULL), 1, VOCAB),
    encoder_target_mask=jnp.ones((B, FULL), jnp.float32),
)

# ---- init (needs 'params' + 'sampling' rngs) ----
variables = model.init({"params": ks[3], "sampling": ks[4]}, **batch, is_training=True)
params = variables["params"]
nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
print(f"SFT model built: {nparams:,} params")

def _to_scalar(x):
    return x.loss if hasattr(x, "loss") else jnp.asarray(x)

def loss_fn(p, rng):
    preds = model.apply({"params": p}, **batch, is_training=True, rngs={"sampling": rng})
    dloss = _to_scalar(discrete_loss.compute_discrete_diffusion_loss(
        preds=preds["output"], targets=preds["target"],
        time=preds["noise_info"]["time"], use_mask=True, mask_key="target_mask",
    )).mean()
    eloss = sft_model.EncoderARLoss().get_values(
        preds["encoder_logits"], preds["encoder_target"], preds["encoder_target_mask"],
    ).mean()
    return dloss + eloss, (dloss, eloss)

opt = optax.adam(1e-2)
opt_state = opt.init(params)

@jax.jit
def train_step(p, st, rng):
    (tot, (dl, el)), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, rng)
    updates, st = opt.update(grads, st)
    p = optax.apply_updates(p, updates)
    return p, st, tot, dl, el, optax.global_norm(grads)

rng = ks[5]
for i in range(3):
    rng, sub = jax.random.split(rng)
    params, opt_state, tot, dl, el, gn = train_step(params, opt_state, sub)
    print(f"  step {i}: total={float(tot):.4f}  diffusion={float(dl):.4f}  encoder={float(el):.4f}  grad_norm={float(gn):.4f}")

dev = jax.tree_util.tree_leaves(params)[0].devices().pop().platform
print("device:", dev)
print("PASS: official DiffusionGemma SFT trains on GPU" if dev == "gpu" else "FAIL: not on GPU")
