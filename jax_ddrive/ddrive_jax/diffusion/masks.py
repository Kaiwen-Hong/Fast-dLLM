"""Hybrid block-causal attention masks for Fast-dDrive block diffusion.

Exact port of modeling.py:178-234 `hybrid_block_causal_mask_multiturn` (the training
mask over the doubled [noisy(0..n) | clean(n..2n)] sequence) and modeling.py:248-279
`eval_hybrid_block_causal_mask` (inference). True = may attend.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array


def hybrid_block_causal_mask_dense(response_block_idx: Array, turn_idx: Array, n: int) -> Array:
    """Dense [2n, 2n] bool mask for the doubled training sequence.

    response_block_idx, turn_idx: int arrays [n] (per original position; -1 = prompt).
    Mirrors modeling.py:199-234 exactly.
    """
    idx = jnp.arange(2 * n)
    x0 = idx >= n                                   # [2n] clean-half flag
    pos = jnp.where(x0, idx - n, idx)               # [2n] original position
    block = response_block_idx[pos]                 # [2n]
    turn = turn_idx[pos]                            # [2n]

    x0_q, x0_kv = x0[:, None], x0[None, :]
    pos_q, pos_kv = pos[:, None], pos[None, :]
    turn_q, turn_kv = turn[:, None], turn[None, :]

    block_diagonal = (~x0_q) & (~x0_kv) & (turn_q == turn_kv)
    offset_block_causal = (turn_q > turn_kv) & x0_kv & (~x0_q)
    x0_causal = x0_q & x0_kv & (pos_q >= pos_kv)
    return block_diagonal | offset_block_causal | x0_causal      # [2n, 2n] bool


def to_attn_mask4d(mask2d: Array) -> Array:
    """[Q, K] bool -> [1, 1, Q, K] bool for Qwen25Attention (True = attend)."""
    return mask2d[None, None, :, :]


def eval_hybrid_block_causal_mask_dense(response_block_idx: Array) -> Array:
    """Dense [L, L] inference mask (modeling.py:248-279). True = attend."""
    L = response_block_idx.shape[0]
    qi = jnp.arange(L)[:, None]
    ki = jnp.arange(L)[None, :]
    bq = response_block_idx[:, None]
    bk = response_block_idx[None, :]
    is_p_q, is_p_kv = bq < 0, bk < 0
    prompt_causal = is_p_q & is_p_kv & (qi >= ki)
    resp_sees_prompt = (~is_p_q) & is_p_kv
    resp_block_causal = (~is_p_q) & (~is_p_kv) & (bq >= bk)
    return prompt_causal | resp_sees_prompt | resp_block_causal


def compute_response_block_idx_simple(labels: Array, bd_size: int) -> tuple[Array, Array, int]:
    """Non-deep fallback (modeling.py compute_response_block_idx): contiguous response
    runs split into blocks of bd_size; prompt = -1. Returns (rbi[L], turn[L], n_blocks).

    NumPy-style (host) computation — used for tests/data prep, not inside jit.
    """
    import numpy as np
    lab = np.asarray(labels)
    L = lab.shape[-1] if lab.ndim == 1 else lab.shape[1]
    lab = lab.reshape(-1)[:L] if lab.ndim == 1 else lab[0]
    rbi = np.full(L, -1, dtype=np.int32)
    resp = lab != -100
    cur_block = 0
    i = 0
    while i < L:
        if resp[i]:
            j = i
            while j < L and resp[j]:
                j += 1
            seg = j - i
            for k in range(seg):
                rbi[i + k] = cur_block + (k // bd_size)
            cur_block += (seg + bd_size - 1) // bd_size
            i = j
        else:
            i += 1
    turn = np.zeros(L, dtype=np.int32)
    for i in range(1, L):
        turn[i] = turn[i - 1] + (1 if rbi[i] != rbi[i - 1] else 0)
    return jnp.asarray(rbi), jnp.asarray(turn), int(cur_block)
