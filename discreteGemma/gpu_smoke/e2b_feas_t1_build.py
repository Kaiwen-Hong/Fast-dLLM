"""T1 — DiffusionGemma_E2B assembly on the 5090: param counts, bf16 init, forward.

Real E2B config (PLE-256, KV-sharing 20/35, sliding 512, vocab 262144), text_only,
random init. Answers: does the full-size model fit + run a diffusion forward?
Emits /home/kaiwen/data/dgemma_e2b/feas/t1_build.json.
"""

import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.93")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2b_feas_common as common  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from gemma.diffusion import _models as dm  # noqa: E402  (import check: 26B class)

OUT = {"jax_backend": jax.default_backend(), "device": str(jax.devices()[0])}
print(f"T1: backend={jax.default_backend()} {jax.devices()}")
assert jax.default_backend() == "gpu", "not on GPU — check LD_LIBRARY_PATH hook"

model = common.DiffusionGemma_E2B(text_only=True, dtype=jnp.bfloat16)
B, L = 1, 1024  # prompt 512 + canvas 512 worth of context, full-bidir canvas smoke
tokens = jnp.zeros((B, L), jnp.int32)
positions = jnp.broadcast_to(jnp.arange(L), (B, L))
attn = jnp.ones((B, L, L), dtype=bool)
sc = jnp.zeros((B, L, model.config.embed_dim), jnp.bfloat16)

init_kwargs = dict(
    sc_embeddings=sc, positions=positions, attention_mask=attn,
    sliding_attention_mask=attn,
    method=common.DiffusionGemma_E2B.call_with_self_conditioning,
)

# ---- abstract param counting (no device memory) ----
t = time.time()
shapes = jax.eval_shape(
    lambda: model.init(
        {"params": jax.random.PRNGKey(0), "sampling": jax.random.PRNGKey(1)},
        tokens, **init_kwargs,
    )
)
counts = common.subtree_param_counts(shapes["params"])
OUT["param_counts"] = counts
OUT["eval_shape_s"] = round(time.time() - t, 1)
print(f"T1: total params = {counts['__total__']/1e9:.3f}B")
for k, v in sorted(counts.items()):
  if k != "__total__":
    print(f"T1:   {k}: {v/1e6:.1f}M")

# ---- real bf16 init on GPU ----
t = time.time()
variables = model.init(
    {"params": jax.random.PRNGKey(0), "sampling": jax.random.PRNGKey(1)},
    tokens, **init_kwargs,
)
jax.block_until_ready(variables)
OUT["init_s"] = round(time.time() - t, 1)
OUT["mem_after_init"] = common.gpu_memory_stats()
print(f"T1: init OK in {OUT['init_s']}s | mem {OUT['mem_after_init']}")

# ---- diffusion forward (jitted): compile + steady ----
@jax.jit
def fwd(v, tok):
  return model.apply(
      v, tok, sc_embeddings=sc, positions=positions, attention_mask=attn,
      sliding_attention_mask=attn,
      method=common.DiffusionGemma_E2B.call_with_self_conditioning,
  ).logits

t = time.time()
logits = fwd(variables, tokens)
logits.block_until_ready()
OUT["fwd_compile_s"] = round(time.time() - t, 1)
t = time.time()
logits = fwd(variables, tokens)
logits.block_until_ready()
OUT["fwd_steady_s"] = round(time.time() - t, 3)
OUT["logits_shape"] = list(logits.shape)
OUT["mem_after_fwd"] = common.gpu_memory_stats()
print(
    f"T1: forward OK {logits.shape} compile={OUT['fwd_compile_s']}s"
    f" steady={OUT['fwd_steady_s']}s | mem {OUT['mem_after_fwd']}"
)

os.makedirs(common.RESULTS_DIR, exist_ok=True)
with open(os.path.join(common.RESULTS_DIR, "t1_build.json"), "w") as f:
  json.dump(OUT, f, indent=2)
print("T1: PASS")
