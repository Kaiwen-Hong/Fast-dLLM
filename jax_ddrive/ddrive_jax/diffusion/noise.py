"""Per-step stochastic noising for SASD training (numpy host-side).

Port of modeling.py:2251-2346 (text-only path): per-block Beta noise -> p_mask ->
random masking, scaffold freeze, always-mask-im_end, doubled [noisy|clean] seq,
complementary batch. The hybrid mask + weights are fixed per sample (computed once).
"""
from __future__ import annotations

import numpy as np

MASK_ID, IM_END, EPS = 151665, 151645, 1e-3


def make_batch(s: dict, rng: np.random.Generator, *, fixed_mask: np.ndarray | None = None):
    """s: one sample's arrays from prep (input_ids, labels, rbi, scaffold, weight_vec,
    block_alpha, block_beta). Returns numpy (input_final[2,2L], labels_final[2,L],
    original_labels[1,L], weights[2,L]).  fixed_mask: optional [L] bool for deterministic eval."""
    ids = s["input_ids"]; labels = s["labels"]; rbi = s["rbi"]; scaff = s["scaffold"]
    L = ids.shape[0]
    resp = labels != -100

    if fixed_mask is not None:
        mask_indices = fixed_mask.copy()
    else:
        t = rng.beta(s["block_alpha"], s["block_beta"])            # [n_blocks]
        p = (1 - EPS) * t + EPS
        pmask = np.zeros(L, np.float32)
        valid = rbi >= 0
        pmask[valid] = p[rbi[valid]]
        mask_indices = (rng.random(L) < pmask) & resp & (~scaff)
    mask_indices = mask_indices | ((ids == IM_END) & resp)         # always_mask_im_end

    noisy = ids.copy(); noisy[mask_indices] = MASK_ID
    lab_m = labels.copy(); lab_m[~mask_indices] = -100
    doubled = np.concatenate([noisy, ids])                         # [2L]

    comp = (resp & ~mask_indices) | ((ids == IM_END) & resp)
    comp = comp & (~scaff)
    cnoisy = ids.copy(); cnoisy[comp] = MASK_ID
    clab = labels.copy(); clab[~comp] = -100
    cdoubled = np.concatenate([cnoisy, ids])

    input_final = np.stack([doubled, cdoubled]).astype(np.int64)   # [2, 2L]
    labels_final = np.stack([lab_m, clab]).astype(np.int64)        # [2, L]
    original_labels = labels[None].astype(np.int64)                # [1, L]
    weights = np.stack([s["weight_vec"], s["weight_vec"]]).astype(np.float32)  # [2, L]
    return input_final, labels_final, original_labels, weights


def num_items(s: dict) -> float:
    return float(2 * int((s["labels"] != -100).sum()))
