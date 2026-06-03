"""RoPE — standalone copy of jax-mdlm-handoff/code/utils/rope.py.

Standard HF rotate-half convention (matches Qwen2/2.5/3). For Fast-dDrive text-only
training M-RoPE collapses to this (all 3 mrope sections share position_ids = arange).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


def default_rope_params(positions: Array, head_dim: int, rope_theta: float = 1_000_000.0,
                        factor: float = 1.0) -> tuple[Array, float]:
    fraction = jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim
    timescale = rope_theta ** fraction
    rotational_frequency = 1.0 / timescale / factor
    return rotational_frequency, 1.0


rope_functions = dict(default=default_rope_params)


def apply_rope(x: Array, sin: Array, cos: Array) -> Array:
    """x: [B, T, H, Dh]; sin,cos: [B, T, Dh/2]. Returns rotated x (same dtype)."""
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


class RoPE(nnx.Module):
    def __init__(self, *, rope_type: str, **rope_kwargs):
        self.rope_kwargs = rope_kwargs
        self.rope_fn = partial(rope_functions[rope_type], **rope_kwargs)

    def __call__(self, positions: Array) -> tuple[Array, Array]:
        rotational_frequency, attention_factor = self.rope_fn(positions)
        # HIGHEST precision: avoid bf16 rounding 257->256 in sin/cos.
        sinusoid_inp = jnp.einsum("BT,k->BTk", positions.astype(jnp.float32), rotational_frequency,
                                  precision=jax.lax.Precision.HIGHEST)
        sin = jnp.sin(sinusoid_inp) * attention_factor
        cos = jnp.cos(sinusoid_inp) * attention_factor
        return sin, cos
