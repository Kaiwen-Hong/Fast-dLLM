"""SASD loss in JAX — exact ports of the Fast-dDrive PyTorch loss.

- section_weighted_ce  <- modeling.py:2123-2157 compute_section_weighted_loss
- causal_ce            <- HF ForCausalLMLoss (the complementary/clean-half term, modeling.py:2764)
- section_weight_vector<- modeling.py:2104-2121 _build_section_weight_tensor

Total training loss (modeling.py:2709-2766), with the doubled [noisy|clean] seq and the
complementary batch stacked:
    loss = section_weighted_ce(noisy_half_logits, labels, weights, num_items)
         + causal_ce(clean_half_logits[mdm rows], original_labels, num_items)
where num_items = 2 * num_items_in_batch (modeling.py:2727).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def _ce_per_token(logits2d: Array, labels1d: Array, ignore: int = -100):
    """fp32 per-token NLL; returns (nll[N] with 0 at ignored, valid_bool[N])."""
    logp = jax.nn.log_softmax(logits2d.astype(jnp.float32), axis=-1)
    valid = labels1d != ignore
    safe = jnp.where(valid, labels1d, 0)
    nll = -jnp.take_along_axis(logp, safe[:, None], axis=-1)[:, 0]
    return jnp.where(valid, nll, 0.0), valid


def section_weighted_ce(logits: Array, labels: Array, weights: Array,
                        num_items: float | None = None, vocab_size: int | None = None) -> Array:
    """logits [B,L,V], labels [B,L] (-100 ignore), weights [B,L]. Causal shift by 1."""
    V = logits.shape[-1]
    sl = logits[:, :-1, :].reshape(-1, V)
    lab = labels[:, 1:].reshape(-1)
    w = weights[:, 1:].reshape(-1)
    nll, valid = _ce_per_token(sl, lab)
    weighted = nll * w
    if num_items is not None:
        return weighted.sum() / num_items
    return weighted.sum() / jnp.maximum(valid.sum(), 1)


def causal_ce(logits: Array, labels: Array, num_items: float | None = None) -> Array:
    """Standard next-token CE (HF ForCausalLMLoss): shift by 1, ignore -100."""
    V = logits.shape[-1]
    sl = logits[:, :-1, :].reshape(-1, V)
    lab = labels[:, 1:].reshape(-1)
    nll, valid = _ce_per_token(sl, lab)
    if num_items is not None:
        return nll.sum() / num_items
    return nll.sum() / jnp.maximum(valid.sum(), 1)


def section_weight_vector(response_block_idx, block_to_section: dict,
                          section_loss_weights: dict) -> Array:
    """Per-position weight [L] from block->section->weight (host-side; dicts)."""
    rbi = np.asarray(response_block_idx).reshape(-1)
    w = np.ones(rbi.shape[0], dtype=np.float32)
    for i, b in enumerate(rbi):
        b = int(b)
        if b >= 0:
            s = block_to_section.get(b)
            if s is not None and s in section_loss_weights:
                w[i] = float(section_loss_weights[s])
    return jnp.asarray(w)


def sasd_total_loss(noisy_logits: Array, labels: Array, weights: Array,
                    clean_logits: Array | None, original_labels: Array | None,
                    num_items: float) -> dict:
    """Combine the two terms exactly as modeling.py:2745-2766.

    noisy_logits  [B', L, V]  (B' includes the complementary batch rows)
    clean_logits  [B'/2, L, V] (mdm rows only) or None
    num_items     = 2 * num_items_in_batch
    """
    mdm = section_weighted_ce(noisy_logits, labels, weights, num_items=num_items)
    out = {"mdm": mdm, "total": mdm}
    if clean_logits is not None:
        comp = causal_ce(clean_logits, original_labels, num_items=num_items)
        out["complementary"] = comp
        out["total"] = mdm + comp
    return out
