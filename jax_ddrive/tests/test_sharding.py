"""CPU test: ShardedLinear/ShardedEmbedding (out_sharding plumbing) are identical to the
plain Linear/Embed at mesh size 1 — proving the model is shardable without changing results.
Run: JAX_PLATFORMS=cpu python jax_ddrive/tests/test_sharding.py"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.models.sharded import ShardedLinear, ShardedEmbedding
from ddrive_jax.models.qwen2_5_text import Linear
from ddrive_jax.sharding import make_mesh

_mesh_ctx = getattr(jax.sharding, "use_mesh", None) or getattr(jax.sharding, "set_mesh")


def test_sharded_linear():
    mesh = make_mesh(1, 1)
    with _mesh_ctx(mesh):
        sl = ShardedLinear(8, 6, use_bias=True, kernel_sharding=P("fsdp", "tp"),
                           bias_sharding=P("tp"), dtype=jnp.float32, rngs=nnx.Rngs(0))
        pl = Linear(8, 6, use_bias=True, dtype=jnp.float32, rngs=nnx.Rngs(1))
        pl.kernel.value = jnp.asarray(np.asarray(sl.kernel.value))
        pl.bias.value = jnp.asarray(np.asarray(sl.bias.value))
        x = jnp.asarray(np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32))
        ys = np.asarray(sl(x, out_sharding=P(None, "tp")))
        yp = np.asarray(pl(x))
        assert np.abs(ys - yp).max() < 1e-5, np.abs(ys - yp).max()
    print("[ok] test_sharded_linear")


def test_sharded_embedding():
    mesh = make_mesh(1, 1)
    with _mesh_ctx(mesh):
        se = ShardedEmbedding(20, 6, dtype=jnp.float32, rngs=nnx.Rngs(0))
        pe = nnx.Embed(20, 6, dtype=jnp.float32, rngs=nnx.Rngs(1))
        pe.embedding.value = jnp.asarray(np.asarray(se.embedding.value))
        ids = jnp.asarray(np.array([[1, 5, 19, 0]]))
        assert np.abs(np.asarray(se(ids, out_sharding=P())) - np.asarray(pe(ids))).max() < 1e-6
        q = jnp.asarray(np.random.default_rng(2).standard_normal((1, 4, 6)).astype(np.float32))
        assert np.abs(np.asarray(se.attend(q, out_sharding=P())) - np.asarray(pe.attend(q))).max() < 1e-4
    print("[ok] test_sharded_embedding")


if __name__ == "__main__":
    test_sharded_linear()
    test_sharded_embedding()
    print("\nALL SHARDING TESTS PASS")
