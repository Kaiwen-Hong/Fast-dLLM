"""Sharding-aware Linear/Embedding for TPU tensor-parallelism (Phase 5b).

JAX 0.10 "sharding-in-types" requires `out_sharding=` on matmul/gather when operands are
sharded (this is why a plain device_put of the model's params doesn't work — see sharding.py).
These mirror the verified `qwen2_5_text.Linear` / `nnx.Embed` but thread `out_sharding`,
following the reference `jax-mdlm-handoff/code/models/llada.py:122-158`. Swapping the text
model's Linear→ShardedLinear and Embed→ShardedEmbedding (mechanical, ~80 LOC) makes the whole
model shardable on a 2D ('fsdp','tp') mesh; at mesh size 1 they are identical to the plain ones.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P
from jaxtyping import Array


class ShardedLinear(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, use_bias: bool,
                 kernel_sharding=P(), bias_sharding=P(), dtype=jnp.float32, rngs: nnx.Rngs):
        self.kernel = nnx.Param(jax.nn.initializers.lecun_normal()(
            rngs.params(), (in_dim, out_dim), dtype=dtype, out_sharding=kernel_sharding))
        self.bias = (nnx.Param(jnp.zeros((out_dim,), dtype=dtype, out_sharding=bias_sharding))
                     if use_bias else None)

    def __call__(self, x: Array, *, out_sharding=None) -> Array:
        y = jnp.matmul(x, self.kernel, out_sharding=out_sharding)
        return y + self.bias if self.bias is not None else y


class ShardedEmbedding(nnx.Embed):
    def __call__(self, inputs: Array, *, out_sharding=None) -> Array:
        emb = self.embedding[...]
        return emb.at[inputs].get(out_sharding=out_sharding)

    def attend(self, query: Array, *, out_sharding=None) -> Array:
        return jnp.dot(query, self.embedding[...].T, out_sharding=out_sharding)
