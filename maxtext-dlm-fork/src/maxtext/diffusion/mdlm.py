"""Masked Diffusion LM (Sahoo 2024) — JAX implementation.

References:
    - dllm/core/trainers/mdlm.py:118-212  (PyTorch spec)
    - md4/models/diffusion/md4.py:203-240 (JAX numerical ground truth)

Math (linear schedule, ε≈0):
    L = E_{t∼U(0,1)} E_{q(x_t|x_0)} [ (1/t) · Σ_l 1[x_t^l = MASK] · CE(logits_l, x_0^l) ]

Normalization: 'token' (dllm config: loss_norm_type='token'). Numer is the
weighted CE summed over masked positions; denom is total non-pad positions
(== |maskable|). At full mask + no pad this reduces to (1/t) · mean_per_pos_CE.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp


_TIME_EPS = 1e-3   # matches dllm.MDLMConfig.time_epsilon


def forward_mask(
    x_0: jax.Array,            # [B, L] int32
    t: jax.Array,              # [B] float32 in (0, 1]
    mask_token_id: int,
    rng_key: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Sample x_t ~ q(x_t | x_0) under a linear schedule.

    Each token is independently replaced by `mask_token_id` with probability
    t[b] (per-sequence). Returns (x_t, mask_bool) where mask is True for
    positions that were masked.
    """
    u = jax.random.uniform(rng_key, x_0.shape, dtype=jnp.float32)   # [B, L]
    mask = u < t[:, None].astype(jnp.float32)                       # [B, L] bool
    x_t = jnp.where(mask, jnp.asarray(mask_token_id, dtype=x_0.dtype), x_0)
    return x_t, mask


def mdlm_loss_inner(
    logits: jax.Array,         # [B, L, V] float
    x_0: jax.Array,            # [B, L] int32
    mask: jax.Array,           # [B, L] bool — True at positions counted in the loss
    t: jax.Array,              # [B] float32
    pad_token_id: int,
    loss_valid: jax.Array | None = None,  # [B, L] bool — overrides (x_0 != pad)
) -> jax.Array:                # scalar
    """MDLM weighted CE with explicit per-position mask. Token-normalized.

    `mask` is the indicator that a position contributes to the loss; this
    typically equals `(x_t == MASK) AND loss_valid`. Tests pass it directly.

    `loss_valid` (optional) marks which positions are *eligible* for the
    loss at all — used by SFT to exclude prompt tokens. If `None`, defaults
    to `(x_0 != pad_token_id)`. The denominator of the per-token mean is
    `sum(loss_valid)`, NOT `sum(mask)`: this matches dllm's `loss_norm_type
    = 'token'` and keeps the rate scale-invariant to mask ratio.

    Returns:
        scalar loss = (Σ_{b,l} mask · (1/t_b) · -log p_θ(x_0|x_t)) / |loss_valid|
    """
    log_p = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)             # [B, L, V]
    log_p_x0 = jnp.take_along_axis(log_p, x_0[..., None], axis=-1).squeeze(-1)  # [B, L]

    weight = 1.0 / jnp.clip(t.astype(jnp.float32), _TIME_EPS, None)             # [B]
    per_pos = weight[:, None] * (-log_p_x0)                                     # [B, L] >=0

    masked_f = mask.astype(jnp.float32)
    numer = jnp.sum(masked_f * per_pos)
    if loss_valid is None:
        valid_f = (x_0 != pad_token_id).astype(jnp.float32)
    else:
        valid_f = loss_valid.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(valid_f), 1.0)
    return numer / denom


def mdlm_loss(
    logits: jax.Array,         # [B, L, V]
    x_0: jax.Array,            # [B, L]
    x_t: jax.Array,            # [B, L]
    t: jax.Array,              # [B]
    mask_token_id: int,
    pad_token_id: int,
    loss_valid: jax.Array | None = None,  # [B, L] bool — SFT response mask
) -> jax.Array:                # scalar
    """Sahoo 2024 weighted CE on positions where `x_t == MASK` AND the
    position is `loss_valid` (defaults to `x_0 != pad_token_id`).

    For SFT, pass `loss_valid = response_mask` so the loss only sees
    response tokens.
    """
    if loss_valid is None:
        valid_b = (x_0 != pad_token_id)
    else:
        valid_b = loss_valid
    mask = (x_t == mask_token_id) & valid_b
    return mdlm_loss_inner(logits, x_0, mask, t, pad_token_id, loss_valid=valid_b)
