"""Tests for the ArrayRecord (dataset v2) path of the SASD input pipeline.

Asserts the AR source is a drop-in replacement for the parquet source:
  (a) source equality: ArRecordSource record == parquet decode_row, field-by-field
      BIT-EXACT (the 12 shared arrays + scalars), and v2 records carry image_embeds
      [168, 2048] bfloat16
  (b) loader equality: make_sasd_loader(AR dir) and make_sasd_loader(parquet dir) with the
      same seed produce IDENTICAL batches (every shared key bit-exact — shuffle order,
      noising and collation all unchanged), with image_embeds additionally present
      (B, 168, 2048) on the AR side
  (c) embeds doubling contract: concat([ie, ie], 0) -> (336, D) == sasd_num_image_tokens
  (d) resumability on the AR loader: state()/set_state() -> bit-identical continuation

Run (CPU; does not touch the GPU):
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
  export JAX_PLATFORMS=cpu
  PY=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
  $PY jax_ddrive/tests/test_ar_pipeline.py

Prints exactly AR_PIPELINE_TESTS_PASS on success.
"""
import numpy as np
import ml_dtypes

from ddrive_jax.data import ar_dataset
from ddrive_jax.data import grain_pipeline as gp
from ddrive_jax.data import parquet_dataset

DATA_PQ = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd"
DATA_AR = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd_v2_ar"
SPLIT = "train"
NTOTAL = 400
N_IMG_TOKENS = 168          # single copy; doubled -> 336 == sasd_num_image_tokens
D_MODEL = 2048

ARRAY_KEYS = ["input_ids", "labels", "rbi", "turn", "scaffold", "weight_vec",
              "block_alpha", "block_beta", "position_ids", "vision_mask",
              "pixel_values", "image_grid_thw"]


def test_source_equality():
    ar = ar_dataset.ArRecordSource(ar_dataset.shard_paths(DATA_AR, SPLIT))
    pq_rows = parquet_dataset.load_all(DATA_PQ, SPLIT)
    assert len(ar) == len(pq_rows) == NTOTAL, (len(ar), len(pq_rows))
    for i in [0, 1, 63, 64, 257, NTOTAL - 1]:
        a, p = ar[i], pq_rows[i]
        assert a["sample_id"] == p["sample_id"], i
        assert a["L"] == p["L"] and a["n_blocks"] == p["n_blocks"], i
        for k in ARRAY_KEYS:
            assert a[k].dtype == p[k].dtype, (i, k, a[k].dtype, p[k].dtype)
            assert a[k].shape == p[k].shape, (i, k)
            assert a[k].tobytes() == p[k].tobytes(), (i, k, "payload mismatch")
        e = a["image_embeds"]
        assert e.shape == (N_IMG_TOKENS, D_MODEL) and e.dtype == ml_dtypes.bfloat16, e.shape
        assert np.isfinite(e.astype(np.float32)).all(), i
    print(f"  [a] source equality OK ({len(ar)} records, 6 spot-checked bit-exact + embeds)")


def test_loader_equality():
    B = 4
    ld_pq = gp.make_sasd_loader(DATA_PQ, SPLIT, per_host_batch=B, seed=11)
    ld_ar = gp.make_sasd_loader(DATA_AR, SPLIT, per_host_batch=B, seed=11)
    assert ld_ar.L == ld_pq.L and ld_ar.N == ld_pq.N and ld_ar.n_img == ld_pq.n_img
    it_pq, it_ar = iter(ld_pq), iter(ld_ar)
    for step in range(3):
        bp, ba = next(it_pq), next(it_ar)
        assert bp["sample_id"] == ba["sample_id"], f"order diverged at step {step}"
        assert bp["step"] == ba["step"]
        for k, v in bp.items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(v, ba[k]), (step, k)
        e = ba["image_embeds"]
        assert e.shape == (B, N_IMG_TOKENS, D_MODEL) and e.dtype == ml_dtypes.bfloat16
        assert "image_embeds" not in bp
    print("  [b] loader equality OK (3 batches bit-exact across sources; embeds on AR side)")


def test_embeds_doubling():
    ld = gp.make_sasd_loader(DATA_AR, SPLIT, per_host_batch=2, seed=0)
    b = next(iter(ld))
    ie = b["image_embeds"]                                  # (B, N, D)
    doubled = np.concatenate([ie, ie], axis=1)              # (B, 2N, D)
    assert doubled.shape == (2, 2 * N_IMG_TOKENS, D_MODEL)
    assert np.array_equal(doubled[:, :N_IMG_TOKENS], doubled[:, N_IMG_TOKENS:])
    print(f"  [c] doubling contract OK ((B,{N_IMG_TOKENS},{D_MODEL}) -> (B,{2*N_IMG_TOKENS},{D_MODEL}))")


def test_resume():
    B = 3
    ld = gp.make_sasd_loader(DATA_AR, SPLIT, per_host_batch=B, seed=42)
    it = iter(ld)
    next(it), next(it)
    st = ld.state()
    expect = next(it)

    ld2 = gp.make_sasd_loader(DATA_AR, SPLIT, per_host_batch=B, seed=42)
    iter(ld2)
    ld2.set_state(st)
    got = next(ld2)
    assert got["sample_id"] == expect["sample_id"] and got["step"] == expect["step"]
    for k, v in expect.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, got[k]), k
    print("  [d] AR resume OK (continuation bit-identical, incl. image_embeds)")


if __name__ == "__main__":
    print(f"running AR pipeline tests: {DATA_AR} vs {DATA_PQ}")
    test_source_equality()
    test_loader_equality()
    test_embeds_doubling()
    test_resume()
    print("AR_PIPELINE_TESTS_PASS")
