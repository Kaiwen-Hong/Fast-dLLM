"""Phase 6 hard gate: 3 unit tests for the MDLM implementation.

Test 1 — mask ratio matches the diffusion time t.
Test 2 — at t→1 with all positions masked, MDLM loss reduces to (1/t) · CE.
Test 3 — numerical match against the md4 reference loss (ported from
         md4/models/diffusion/md4.py:203-240, linear schedule, eps=1e-4).

Run with:
    cd ~/jax-dlm-baseline/maxtext-dlm
    PYTHONPATH=. python -m pytest MaxText/tests/test_mdlm.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.diffusion.mdlm import forward_mask, mdlm_loss, mdlm_loss_inner


# ----------------------------------------------------------------------------
# md4 reference loss (linear schedule, eps=1e-4) — ported from
# references/md4/md4/models/diffusion/md4.py:203-240. Self-contained: avoids
# md4's tensorflow/distrax dependencies. See references/MD4_NOTES.md.
# ----------------------------------------------------------------------------

def md4_ref_loss_linear(
    logits, x_0, x_t, t, mask_token_id, vocab_size, eps=1e-4,
):
    """Match md4.diffusion_loss(...).mean() exactly for linear cont-time schedule."""
    log_p = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    one_hot = jax.nn.one_hot(x_0, vocab_size, dtype=jnp.float32)
    log_p_x0 = (one_hot * log_p).sum(axis=-1)                                  # [B, L]
    mask_f = (x_t == mask_token_id).astype(jnp.float32)                        # [B, L]
    sum_per_seq = jnp.sum(mask_f * log_p_x0, axis=-1)                          # [B]
    one_minus_alpha = eps + t.astype(jnp.float32) * (1.0 - 2.0 * eps)          # 1 - α(t), [B]
    dgamma_alpha = -(1.0 - 2.0 * eps) / one_minus_alpha                        # ≈ -1/t, [B]
    loss_per_seq = dgamma_alpha * sum_per_seq                                  # [B]
    return loss_per_seq.mean()                                                 # scalar


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

MASK_ID = 99_999      # any int outside the ranges we use
PAD_ID = -1


def test_mask_ratio():
    """forward_mask should respect the per-sequence rate t (Bernoulli law of large numbers)."""
    rng = jax.random.PRNGKey(42)
    B, L = 1000, 128
    x_0 = jnp.zeros((B, L), dtype=jnp.int32)
    t = jnp.full((B,), 0.30, dtype=jnp.float32)

    x_t, mask = forward_mask(x_0, t, mask_token_id=MASK_ID, rng_key=rng)

    actual = float(mask.mean())
    assert 0.28 < actual < 0.32, f"expected ~0.30, got {actual}"
    # mask True ↔ x_t == MASK_ID
    assert bool(((mask) == (x_t == MASK_ID)).all())
    # x_0 was zeros, so unmasked positions should still be zero
    assert bool((jnp.where(mask, 0, x_t) == 0).all())


def test_loss_reduces_to_ce_at_full_mask():
    """At t→1, all positions masked, MDLM loss = (1/t) · standard CE."""
    rng = jax.random.PRNGKey(0)
    B, L, V = 4, 16, 1000
    x_0 = jax.random.randint(rng, (B, L), 0, V - 1)
    logits = jax.random.normal(jax.random.PRNGKey(1), (B, L, V))

    t = jnp.full((B,), 0.999, dtype=jnp.float32)
    x_t = jnp.full_like(x_0, MASK_ID)
    mask_all = jnp.ones_like(x_0, dtype=bool)
    mdlm_l = mdlm_loss_inner(logits, x_0, mask_all, t, pad_token_id=PAD_ID)

    # standard mean per-position CE
    ce = -jnp.mean(jnp.sum(
        jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        * jax.nn.one_hot(x_0, V, dtype=jnp.float32),
        axis=-1,
    ))

    diff = float(jnp.abs(mdlm_l - ce / 0.999))
    assert diff < 1e-4, f"MDLM={mdlm_l}, CE/t={ce/0.999}, diff={diff}"


def test_md4_match():
    """Numerical match (within fp32 noise) against md4 reference loss."""
    B, L, V = 2, 32, 256
    mask_id = V                           # md4 convention: id == vocab_size

    rng = jax.random.PRNGKey(123)
    rng, k1 = jax.random.split(rng)
    x_0 = jax.random.randint(k1, (B, L), 0, V)
    t = jnp.array([0.3, 0.7], dtype=jnp.float32)

    rng, k2 = jax.random.split(rng)
    x_t, _mask = forward_mask(x_0, t, mask_token_id=mask_id, rng_key=k2)

    rng, k3 = jax.random.split(rng)
    logits = jax.random.normal(k3, (B, L, V), dtype=jnp.float32)

    ours_per_pos = float(mdlm_loss(
        logits, x_0, x_t, t,
        mask_token_id=mask_id, pad_token_id=PAD_ID,
    ))
    md4_per_seq_batchmean = float(md4_ref_loss_linear(
        logits, x_0, x_t, t, mask_token_id=mask_id, vocab_size=V,
    ))

    # Convert: dllm 'token' norm divides by total maskable (= B*L when no pad);
    # md4's mean-over-batch divides only by B. So md4 = our * L (in eps→0 limit).
    # We use eps=1e-3 in our weight clip and md4 uses eps=1e-4 in alpha schedule;
    # the discrepancy on weight (1/t vs (1-2e)/(e + t(1-2e))) at t=0.3,0.7 is
    #   t=0.3: (1/0.3=3.3333) vs ((0.9998)/(0.0001+0.3*0.9998)=3.3325) → 2.4e-4 rel
    #   t=0.7: (1/0.7=1.4286) vs ((0.9998)/(0.0001+0.7*0.9998)=1.4283) → 2.0e-4 rel
    # so a ~3e-4 relative difference is structural; we allow 1e-3 absolute.
    ours_scaled_to_md4 = ours_per_pos * L
    abs_diff = abs(ours_scaled_to_md4 - md4_per_seq_batchmean)
    rel_diff = abs_diff / max(abs(md4_per_seq_batchmean), 1e-9)
    assert rel_diff < 1e-3, (
        f"ours_per_pos*L={ours_scaled_to_md4:.6f}, md4={md4_per_seq_batchmean:.6f}, "
        f"abs_diff={abs_diff:.3e}, rel_diff={rel_diff:.3e}"
    )

    # Also store the numbers for the LOG.md report.
    print(
        f"\n  test_md4_match numerical report:"
        f"\n    ours (per-token MDLM, ε_w=1e-3) = {ours_per_pos:.6f}"
        f"\n    ours × L (= per-seq batchmean) = {ours_scaled_to_md4:.6f}"
        f"\n    md4 (per-seq batchmean, ε_α=1e-4) = {md4_per_seq_batchmean:.6f}"
        f"\n    abs_diff = {abs_diff:.3e}"
        f"\n    rel_diff = {rel_diff:.3e}"
    )


def test_loss_valid_excludes_prompt_positions():
    """SFT path: with loss_valid masking out the prompt half, the loss
    should match a recomputation that ignores those positions entirely.
    """
    rng = jax.random.PRNGKey(7)
    B, L, V = 4, 32, 200
    x_0 = jax.random.randint(rng, (B, L), 0, V)
    logits = jax.random.normal(jax.random.PRNGKey(8), (B, L, V))

    # Loss-valid = response half (last L/2 positions only).
    loss_valid = jnp.concatenate(
        [jnp.zeros((B, L // 2), dtype=bool), jnp.ones((B, L // 2), dtype=bool)],
        axis=1,
    )

    t = jnp.full((B,), 0.5, dtype=jnp.float32)
    x_t = jnp.full_like(x_0, MASK_ID)   # all masked, so the only filter is loss_valid

    loss_full = mdlm_loss(
        logits, x_0, x_t, t,
        mask_token_id=MASK_ID, pad_token_id=PAD_ID,
        loss_valid=loss_valid,
    )

    # Recompute manually: per-position weighted CE on response half only,
    # divided by sum(loss_valid).
    log_p = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    log_p_x0 = jnp.take_along_axis(log_p, x_0[..., None], axis=-1).squeeze(-1)
    weight = 1.0 / 0.5     # t=0.5, no clip needed
    per_pos = weight * (-log_p_x0)
    expected = (per_pos * loss_valid.astype(jnp.float32)).sum() / loss_valid.sum()

    diff = float(jnp.abs(loss_full - expected))
    assert diff < 1e-4, f"loss={float(loss_full):.6f} expected={float(expected):.6f}"

    # Sanity: a permutation of `x_0` over the prompt half (loss_valid=False)
    # should NOT change the loss.
    rng2 = jax.random.PRNGKey(9)
    x_0_alt = x_0.at[:, :L // 2].set(jax.random.randint(rng2, (B, L // 2), 0, V))
    loss_alt = mdlm_loss(
        logits, x_0_alt, x_t, t,
        mask_token_id=MASK_ID, pad_token_id=PAD_ID,
        loss_valid=loss_valid,
    )
    # Note: we permuted x_0 in the prompt half, so log_p_x0 also changes
    # there. But those positions have loss_valid=False, so they don't enter
    # numerator OR denominator. The response half is identical in both runs
    # (we only changed prompt positions), so loss should match exactly.
    assert float(jnp.abs(loss_alt - loss_full)) < 1e-5, "prompt permutation leaked into loss"
