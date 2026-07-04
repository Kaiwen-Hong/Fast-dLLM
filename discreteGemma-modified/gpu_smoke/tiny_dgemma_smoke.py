"""Step 2: instantiate a TINY DiffusionGemma (real DeepMind code, shrunk config),
random-init on GPU, run inference forward + one training grad step. No weights.

Proves: THIS implementation (gemma.diffusion) trains & infers on the RTX 5090 in JAX.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")  # silence tiny-shape autotune noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys, functools
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma")

import jax, jax.numpy as jnp
import optax

from gemma.gm.nn.gemma4 import _config, _modules
from gemma.diffusion import _models as dm

print("jax", jax.__version__, "| backend", jax.default_backend(), "|", jax.devices())

# ---- tiny config: 2 local-sliding layers, d=64, vocab=256, dense (no MoE), no vision ----
attn_types = _config.make_attention_layers_types(
    (_modules.AttentionType.LOCAL_SLIDING,), num_layers=2
)
cfg = _config.TransformerConfig(
    num_embed=256,
    embed_dim=64,
    hidden_dim=128,
    num_heads=2,
    head_dim=16,
    num_kv_heads=1,
    final_logit_softcap=None,
    use_post_attn_norm=True,
    use_post_ffw_norm=True,
    attention_types=attn_types,
    sliding_window_size=32,
    qk_norm_with_scale=True,
    global_rope_proportion=0.25,
    local_rope_proportion=1.0,
    per_layer_input_dim=0,
    enable_moe=False,          # dense FeedForward path -> pure standard XLA ops
    vision_encoder=None,
    audio_encoder=None,
)

model = dm.DiffusionGemma_26B_A4B(config=cfg, dtype=jnp.float32)

B, L, V = 2, 16, cfg.num_embed
key = jax.random.PRNGKey(0)
tokens = jax.random.randint(key, (B, L), 0, V)

# Inputs for the diffusion-specific forward (bidirectional canvas + self-conditioning).
positions = jnp.broadcast_to(jnp.arange(L), (B, L))
attn_mask = jnp.ones((B, L, L), dtype=bool)                    # bidirectional over the canvas
sc_embeddings = jnp.zeros((B, L, cfg.embed_dim), jnp.float32)  # 1st denoising step: zeros

def diff_apply(p, toks):
    """The real DiffusionGemma forward: self-conditioning FFW + bidir attention."""
    return model.apply(
        p, toks,
        sc_embeddings=sc_embeddings, positions=positions,
        attention_mask=attn_mask, sliding_attention_mask=attn_mask,
        method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning,
    )

# ---- init random params on GPU (via the diffusion method -> creates self_conditioner too) ----
params = model.init(
    jax.random.PRNGKey(1), tokens,
    sc_embeddings=sc_embeddings, positions=positions,
    attention_mask=attn_mask, sliding_attention_mask=attn_mask,
    method=dm.DiffusionGemma_26B_A4B.call_with_self_conditioning,
)
nparams = sum(x.size for x in jax.tree_util.tree_leaves(params))
has_sc = any('self_conditioner' in '/'.join(map(str, p)) for p, _ in
             jax.tree_util.tree_flatten_with_path(params)[0])
print(f"tiny model built: {nparams:,} params  (dtype float32) | self_conditioner present: {has_sc}")

# ---- INFERENCE: base AR forward (shares the transformer params) ----
logits = model.apply(params, tokens).logits
logits.block_until_ready()
print("inference OK (base AR forward)   : logits", logits.shape, "device:", list(logits.devices()))
assert logits.shape == (B, L, V)

# ---- INFERENCE: diffusion-specific forward (self-cond + bidirectional) ----
logits_sc = diff_apply(params, tokens).logits
logits_sc.block_until_ready()
print("inference OK (diffusion forward) : logits", logits_sc.shape, "device:", list(logits_sc.devices()))

# ---- TRAINING: masked-diffusion-style CE through the diffusion forward + grad step ----
def loss_fn(p, toks):
    lg = diff_apply(p, toks).logits            # trains the FULL diffusion model incl. self_conditioner
    return optax.softmax_cross_entropy_with_integer_labels(lg, toks).mean()

opt = optax.sgd(1e-1)
opt_state = opt.init(params)

@jax.jit
def train_step(p, st, toks):
    loss, grads = jax.value_and_grad(loss_fn)(p, toks)
    updates, st = opt.update(grads, st)
    p = optax.apply_updates(p, updates)
    gnorm = optax.global_norm(grads)
    return p, st, loss, gnorm

l0 = loss_fn(params, tokens)
for i in range(3):
    params, opt_state, loss, gnorm = train_step(params, opt_state, tokens)
    print(f"  step {i}: loss={float(loss):.4f}  grad_norm={float(gnorm):.4f}")
l1 = loss_fn(params, tokens)
print(f"training OK: loss {float(l0):.4f} -> {float(l1):.4f} (decreased: {float(l1) < float(l0)})")

print("device of updated param:",
      list(jax.tree_util.tree_leaves(params)[0].devices()))
print("PASS: tiny DiffusionGemma trains & infers on GPU"
      if jax.tree_util.tree_leaves(params)[0].devices().pop().platform == "gpu"
      else "FAIL: not on GPU")
