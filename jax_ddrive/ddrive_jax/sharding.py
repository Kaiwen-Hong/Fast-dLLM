"""TPU sharding specs for the NNX model (Phase 5). See docs/02_tpu_plan.md.

JAX 0.10 uses "sharding-in-types": a vocab-sharded embedding gather and sharded
matmuls require explicit `out_sharding=` (the reference's `ShardedEmbedding`/
`ShardedLinear`, llada.py:122-158). So physical FSDP/TP needs that ~80-LOC port
(option 2 in the plan). What we CAN provide cleanly is the canonical mesh + the
PartitionSpec pytree those sharded layers (or a MaxText fork) consume — the actual
artifact needed to scale, validated to match `nnx.state` structure.
"""
from __future__ import annotations

import jax
from jax.sharding import PartitionSpec as P
from flax import nnx


def make_mesh(n_fsdp: int, n_tp: int = 1):
    """2D mesh ('fsdp','tp'). Start pure-FSDP (N,1)."""
    return jax.make_mesh((n_fsdp, n_tp), ("fsdp", "tp"))


def fsdp_pspec(shape) -> P:
    """FSDP: shard the largest axis on 'fsdp', replicate the rest. Embeddings are
    replicated (sharding the vocab axis needs ShardedEmbedding.attend out_sharding)."""
    if len(shape) < 2:
        return P()
    ax = max(range(len(shape)), key=lambda i: shape[i])
    return P(*[("fsdp" if i == ax else None) for i in range(len(shape))])


def param_pspecs(model, *, replicate_embedding: bool = True):
    """Return a pytree of PartitionSpecs matching nnx.state(model, nnx.Param), assigning
    an FSDP spec per param. Feed to jit(in_shardings=...) / shard_map once the model uses
    sharding-aware layers, or hand to a MaxText fork's pyconfig."""
    state = nnx.state(model, nnx.Param)
    flat = nnx.to_flat_state(state) if hasattr(nnx, "to_flat_state") else None

    def spec_for(path, leaf):
        shape = getattr(leaf, "shape", None) or getattr(getattr(leaf, "value", None), "shape", ())
        if replicate_embedding and any("embed" in str(p) for p in path):
            return P()
        return fsdp_pspec(tuple(shape))

    # Generic: map over the state tree with paths.
    specs = jax.tree_util.tree_map_with_path(
        lambda path, leaf: spec_for([k.key if hasattr(k, "key") else k for k in path], leaf),
        state)
    return specs
