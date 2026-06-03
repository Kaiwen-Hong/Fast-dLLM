"""CPU unit tests for LoRA (audit finding #1/#4): forward correctness + pure-LoRA freeze.
Run: JAX_PLATFORMS=cpu python jax_ddrive/tests/test_lora.py"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.models.qwen2_5_text import Linear, RMSNorm
from ddrive_jax.lora import LoRAConfig, LoRALinear, apply_lora, freeze_non_lora_params


def test_lora_forward():
    rngs = nnx.Rngs(0)
    base = Linear(8, 6, use_bias=True, dtype=jnp.float32, rngs=rngs)
    base.bias.value = jnp.asarray(np.random.default_rng(3).standard_normal(6).astype(np.float32))
    x = jnp.asarray(np.random.default_rng(1).standard_normal((3, 8)).astype(np.float32))
    y_base = np.asarray(base(x))
    lo = LoRALinear(base, rank=4, alpha=8.0, rngs=nnx.Rngs(2))
    # zero-init lora_B -> exact pass-through of the base linear
    assert np.abs(np.asarray(lo(x)) - y_base).max() < 1e-5
    # non-zero lora_B -> base + (x@A@B)*scale
    lo.lora_B.value = jnp.asarray(np.random.default_rng(4).standard_normal((4, 6)).astype(np.float32))
    A, B = np.asarray(lo.lora_A.value), np.asarray(lo.lora_B.value)
    ref = y_base + (np.asarray(x) @ A @ B) * (8.0 / 4)
    assert np.abs(np.asarray(lo(x)) - ref).max() < 1e-4
    # no-bias base branch
    nb = Linear(8, 6, use_bias=False, dtype=jnp.float32, rngs=nnx.Rngs(5))
    lo2 = LoRALinear(nb, rank=2, alpha=4.0, rngs=nnx.Rngs(6))
    assert np.abs(np.asarray(lo2(x)) - np.asarray(nb(x))).max() < 1e-5
    print("[ok] test_lora_forward")


def test_lora_freeze_structure():
    class Blk(nnx.Module):
        def __init__(s, rngs):
            s.q_proj = Linear(8, 8, use_bias=True, dtype=jnp.float32, rngs=rngs)
            s.norm = RMSNorm(8, 1e-6, dtype=jnp.float32, rngs=rngs)
    m = Blk(nnx.Rngs(0))
    assert apply_lora(m, LoRAConfig(rank=4, target_modules=("q_proj",)), rngs=nnx.Rngs(1)) == 1
    freeze_non_lora_params(m)
    leaves = jax.tree_util.tree_leaves(nnx.state(m, nnx.Param))
    cnt = sum(int(np.prod(x.shape)) for x in leaves)
    assert cnt == 4 * (8 + 8), cnt          # only lora_A[8,4]+lora_B[4,8] remain trainable
    print("[ok] test_lora_freeze_structure")


if __name__ == "__main__":
    test_lora_forward()
    test_lora_freeze_structure()
    print("\nALL LORA TESTS PASS")
