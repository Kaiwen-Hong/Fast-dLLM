"""Bit-exact parity: MaxText SASD port vs the validated NNX reference.

Uses a REAL Fast-dDrive SASD sample (parquet) + seeded RNGs and asserts the ported
functions in ``maxtext.diffusion.sasd`` equal the NNX originals in
``ddrive_jax.diffusion.{noise,masks,sasd_loss}`` for-bit.

Run:
  PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive:\
/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/src \
  JAX_PLATFORMS=cpu /home/kaiwen/jax-dlm-baseline/.venv/bin/python <this file>
"""
import numpy as np
import jax
import jax.numpy as jnp

# --- NNX reference (port FROM) -------------------------------------------------
from ddrive_jax.data.parquet_dataset import load_all
from ddrive_jax.diffusion import noise as ref_noise
from ddrive_jax.diffusion import masks as ref_masks
from ddrive_jax.diffusion import sasd_loss as ref_loss

# --- MaxText port (port TO) ----------------------------------------------------
from maxtext.diffusion import sasd as mxt

DATA_DIR = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd"


def _eq(a, b, name):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {a.shape} vs {b.shape}")
    if a.dtype != b.dtype:
        raise AssertionError(f"{name}: dtype mismatch {a.dtype} vs {b.dtype}")
    if not np.array_equal(a, b):
        diff = int(np.sum(a != b))
        raise AssertionError(f"{name}: NOT bit-exact ({diff} differing elements)")
    print(f"  [OK] {name}: bit-exact ({a.shape} {a.dtype})")


def main():
    sample = load_all(DATA_DIR)[0]
    L = int(sample["L"])
    n_blocks = int(sample["n_blocks"])
    print(f"sample_id={sample['sample_id']} L={L} n_blocks={n_blocks} "
          f"resp_tokens={int((sample['labels'] != -100).sum())}")

    # Sanity: ported constants match the reference.
    assert (mxt.MASK_ID, mxt.IM_END, mxt.EPS) == (ref_noise.MASK_ID, ref_noise.IM_END, ref_noise.EPS), \
        "constants differ"

    # ---- (a) make_batch: identical stochastic noising with the SAME seed --------
    # Reference and port must draw the SAME rng stream -> separate Generators, same seed.
    rng_ref = np.random.default_rng(12345)
    rng_mxt = np.random.default_rng(12345)
    ref_out = ref_noise.make_batch(sample, rng_ref)
    mxt_out = mxt.make_batch(sample, rng_mxt)
    for nm, r, m in zip(
        ["input_final", "labels_final", "original_labels", "weights"], ref_out, mxt_out
    ):
        _eq(r, m, f"make_batch.{nm}")

    # also exercise the fixed_mask (deterministic eval) branch
    fixed = np.zeros(L, dtype=bool)
    resp = sample["labels"] != -100
    fixed[np.where(resp)[0][:5]] = True
    ref_fx = ref_noise.make_batch(sample, np.random.default_rng(0), fixed_mask=fixed)
    mxt_fx = mxt.make_batch(sample, np.random.default_rng(0), fixed_mask=fixed)
    for nm, r, m in zip(
        ["input_final", "labels_final", "original_labels", "weights"], ref_fx, mxt_fx
    ):
        _eq(r, m, f"make_batch[fixed_mask].{nm}")

    # num_items
    ri, mi = ref_noise.num_items(sample), mxt.num_items(sample)
    assert ri == mi, f"num_items differ: {ri} vs {mi}"
    print(f"  [OK] num_items: {mi}")
    n_items = mi

    # ---- (b) hybrid_block_causal_mask_dense: identical ------------------------
    rbi = jnp.asarray(sample["rbi"])
    turn = jnp.asarray(sample["turn"])
    ref_mask = ref_masks.hybrid_block_causal_mask_dense(rbi, turn, L)
    mxt_mask = mxt.hybrid_block_causal_mask_dense(rbi, turn, L)
    _eq(ref_mask, mxt_mask, "hybrid_block_causal_mask_dense")
    # to_attn_mask4d
    _eq(ref_masks.to_attn_mask4d(ref_mask), mxt.to_attn_mask4d(mxt_mask), "to_attn_mask4d")

    # ---- (c) section_weighted_ce + causal_ce on seeded random logits -----------
    # The loss math is vocab-agnostic, so we remap real token ids (up to ~151936)
    # into a tiny contiguous vocab to keep the random-logits tensors small (the
    # 30 GB host has no swap). -100 (ignore) is preserved as-is; valid labels are
    # taken modulo V so every gather index is in-range for BOTH stacks identically.
    input_final, labels_final, original_labels, weights = mxt_out
    V = 256  # tiny vocab keeps logits small; math is vocab-agnostic
    Bn = labels_final.shape[0]            # 2 (mdm + complementary rows)

    def _remap(lab_np):
        lab_np = np.asarray(lab_np)
        valid = lab_np != -100
        out = lab_np.copy()
        out[valid] = lab_np[valid] % V    # in-range gather index for both stacks
        return out

    labels_final = _remap(labels_final)
    original_labels = _remap(original_labels)

    # noisy-half logits over the [B, L, V] label window (section_weighted_ce + causal_ce
    # both consume [B, L, V] and shift by 1 internally).
    klog = jax.random.PRNGKey(777)
    k1, k2 = jax.random.split(klog)
    logits_sw = jax.random.normal(k1, (Bn, L, V), dtype=jnp.float32)
    logits_cz = jax.random.normal(k2, (original_labels.shape[0], L, V), dtype=jnp.float32)

    lab = jnp.asarray(labels_final)
    w = jnp.asarray(weights)
    olab = jnp.asarray(original_labels)

    ref_sw = ref_loss.section_weighted_ce(logits_sw, lab, w, num_items=n_items)
    mxt_sw = mxt.section_weighted_ce(logits_sw, lab, w, num_items=n_items)
    _eq(ref_sw, mxt_sw, "section_weighted_ce[num_items]")

    # also the unnormalized (num_items=None -> divide by valid count) path
    ref_sw0 = ref_loss.section_weighted_ce(logits_sw, lab, w)
    mxt_sw0 = mxt.section_weighted_ce(logits_sw, lab, w)
    _eq(ref_sw0, mxt_sw0, "section_weighted_ce[None]")

    ref_cz = ref_loss.causal_ce(logits_cz, olab, num_items=n_items)
    mxt_cz = mxt.causal_ce(logits_cz, olab, num_items=n_items)
    _eq(ref_cz, mxt_cz, "causal_ce[num_items]")

    ref_cz0 = ref_loss.causal_ce(logits_cz, olab)
    mxt_cz0 = mxt.causal_ce(logits_cz, olab)
    _eq(ref_cz0, mxt_cz0, "causal_ce[None]")

    print("SASD_PARITY_PASS")


if __name__ == "__main__":
    main()
