"""Tests for ddrive_jax/data/grain_pipeline.py (multi-host SASD input pipeline).

Covers, per the design's "what tests must assert":
  (a) batch shapes/dtypes match the contract
  (b) determinism: two fresh loaders, same seed -> identical first 3 batches (bit-exact)
  (c) multi-host disjoint + complete: process_count=2 -> sample_id sets disjoint, union == 400
  (d) noising sanity: noised positions subset of (labels!=-100)&~scaffold | im_end;
      im_end always masked in the noised half
  (e) resumability: checkpoint state, restore in a fresh loader, next batch identical
  + rng-fold reproducibility: independently re-deriving _fold_rng + noise.make_batch
    reproduces the loader's input_final bit-for-bit
  + uniform assertion: non-uniform L w/o max_length raises

Run (CPU 8-device emulation as multi-host proxy):
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
  export JAX_PLATFORMS=cpu
  export XLA_FLAGS="--xla_force_host_platform_device_count=8"
  PY=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
  $PY jax_ddrive/tests/test_grain_pipeline.py

Prints exactly GRAIN_PIPELINE_TESTS_PASS on success.
"""
import sys
import numpy as np

from ddrive_jax.data import grain_pipeline as gp
from ddrive_jax.data import parquet_dataset
from ddrive_jax.diffusion import noise

DATA = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd"
SPLIT = "train"
L_EXPECT = 1184
N_EXPECT = 672
NIMG_EXPECT = 3
NTOTAL_EXPECT = 400

# expected (key -> (ndim_after_B, dtype)) for the batch dict array entries.
EXPECT = {
    "input_final": (np.int64, lambda B: (B, 2, 2 * L_EXPECT)),
    "labels_final": (np.int64, lambda B: (B, 2, L_EXPECT)),
    "original_labels": (np.int64, lambda B: (B, 1, L_EXPECT)),
    "weights": (np.float32, lambda B: (B, 2, L_EXPECT)),
    "position_ids": (np.int32, lambda B: (B, 3, L_EXPECT)),
    "rbi": (np.int32, lambda B: (B, L_EXPECT)),
    "turn": (np.int32, lambda B: (B, L_EXPECT)),
    "scaffold": (np.bool_, lambda B: (B, L_EXPECT)),
    "pixel_values": (np.float16, lambda B: (B, N_EXPECT, 1176)),
    "image_grid_thw": (np.int64, lambda B: (B, NIMG_EXPECT, 3)),
    "num_items": (np.float32, lambda B: (B,)),
}


def _arrays_equal(a, b):
    return all(np.array_equal(a[k], b[k]) for k in EXPECT) and \
        a["sample_id"] == b["sample_id"] and a["step"] == b["step"]


def test_shapes_dtypes():
    B = 4
    ld = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=0)
    it = iter(ld)
    batch = next(it)
    for k, (dt, shp) in EXPECT.items():
        assert k in batch, f"missing key {k}"
        assert batch[k].dtype == dt, f"{k}: dtype {batch[k].dtype} != {dt}"
        assert batch[k].shape == shp(B), f"{k}: shape {batch[k].shape} != {shp(B)}"
    assert isinstance(batch["sample_id"], list) and len(batch["sample_id"]) == B
    assert all(isinstance(x, str) for x in batch["sample_id"])
    assert isinstance(batch["step"], int)
    print("  [a] shapes/dtypes OK "
          f"(B={B}, L={L_EXPECT}, N={N_EXPECT}, n_img={NIMG_EXPECT}, step={batch['step']})")


def test_determinism():
    B = 3
    ld1 = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=7)
    ld2 = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=7)
    it1, it2 = iter(ld1), iter(ld2)
    for i in range(3):
        b1, b2 = next(it1), next(it2)
        assert _arrays_equal(b1, b2), f"seed-determinism mismatch at batch {i}"
    # different seed -> the stream differs (sample order and/or noise)
    ld3 = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=999)
    ld1b = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=7)
    a = next(iter(ld1b))
    c = next(iter(ld3))
    differs = (a["sample_id"] != c["sample_id"]) or \
        (not np.array_equal(a["input_final"], c["input_final"]))
    assert differs, "different seed produced identical first batch"
    print("  [b] determinism OK (same seed bit-exact x3; different seed differs)")


def test_multihost_disjoint_complete():
    B = 4
    H = 2
    # one full epoch = NTOTAL_EXPECT samples across H hosts => NTOTAL/(B*H) steps each.
    steps = NTOTAL_EXPECT // (B * H)
    assert steps * B * H == NTOTAL_EXPECT, "choose B,H dividing NTOTAL for the epoch test"
    ids = {0: set(), 1: set()}
    gidx = {0: [], 1: []}
    for h in range(H):
        ld = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=3,
                                 process_index=h, process_count=H)
        it = iter(ld)
        for _ in range(steps):
            b = next(it)
            ids[h].update(b["sample_id"])
    assert ids[0].isdisjoint(ids[1]), "hosts share sample_ids within one epoch"
    union = ids[0] | ids[1]
    full = {r["sample_id"]
            for r in gp._RowSource(parquet_dataset.shard_paths(DATA, SPLIT))._rows}
    assert union == full, (
        f"union != full set: |union|={len(union)} |full|={len(full)} "
        f"missing={len(full - union)} extra={len(union - full)}")
    assert len(union) == NTOTAL_EXPECT
    print(f"  [c] multi-host OK (H={H}, B={B}, {steps} steps/host: "
          f"disjoint, union=={NTOTAL_EXPECT})")


def test_noising_sanity():
    B = 4
    ld = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=11)
    it = iter(ld)
    batch = next(it)
    # need raw input_ids/labels/scaffold per sample_id to verify the noise invariant.
    rows = {r["sample_id"]: r
            for r in gp._RowSource(parquet_dataset.shard_paths(DATA, SPLIT))._rows}
    checked = 0
    for bi, sid in enumerate(batch["sample_id"]):
        r = rows[sid]
        ids = r["input_ids"]
        labels = r["labels"]
        scaff = r["scaffold"].astype(bool)
        resp = labels != -100
        imend = (ids == noise.IM_END) & resp
        # masked-by-noising set == positions kept in labels_final[noisy half] (lab_m).
        mask_by_noise = batch["labels_final"][bi, 0] != -100
        allowed = (resp & ~scaff) | imend
        assert (mask_by_noise & ~allowed).sum() == 0, \
            f"{sid}: noised positions escape (labels!=-100)&~scaffold | im_end"
        if imend.sum():
            assert mask_by_noise[imend].all(), f"{sid}: im_end not masked in noised half"
        # the noised half actually substitutes MASK_ID at every newly-masked, non-preexisting pos.
        noisy = batch["input_final"][bi, 0, :L_EXPECT]
        newly = mask_by_noise & (ids != noise.MASK_ID)
        assert (noisy[newly] == noise.MASK_ID).all(), f"{sid}: noised positions not MASK_ID"
        checked += 1
    print(f"  [d] noising sanity OK ({checked} samples: subset + im_end-always-masked)")


def test_rng_fold_reproducible():
    """Independently re-derive _fold_rng + noise.make_batch -> matches loader bit-for-bit."""
    B = 2
    H = 3
    pidx = 1
    ld = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=5,
                             process_index=pidx, process_count=H)
    it = iter(ld)
    batch = next(it)
    rows = {r["sample_id"]: r
            for r in gp._RowSource(parquet_dataset.shard_paths(DATA, SPLIT))._rows}
    # gidx of sample j in this host's first batch = pidx + j*H (host stride);
    # step = gidx // H.  Re-derive and compare input_final/labels_final.
    for j, sid in enumerate(batch["sample_id"]):
        gidx = pidx + j * H
        step = gidx // H
        rng = gp._fold_rng(5, step, gidx)
        ifn, lfn, ol, w = noise.make_batch(rows[sid], rng)
        assert np.array_equal(ifn, batch["input_final"][j]), \
            f"rng-fold input_final mismatch at j={j} sid={sid} gidx={gidx}"
        assert np.array_equal(lfn, batch["labels_final"][j]), \
            f"rng-fold labels_final mismatch at j={j} sid={sid}"
    assert batch["step"] == pidx // H, f"step {batch['step']} != {pidx // H}"
    print(f"  [rng] fold reproducible OK (H={H}, pidx={pidx}: re-derived noise bit-exact)")


def test_resumability():
    B = 3
    # uninterrupted reference: consume 5 batches.
    ref = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=42)
    itr = iter(ref)
    ref_batches = [next(itr) for _ in range(5)]
    # interrupted run: consume 2, checkpoint, build fresh loader, restore, consume 3 more.
    ld = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=42)
    it = iter(ld)
    for _ in range(2):
        next(it)
    st = ld.state()
    assert "grain" in st and "next_index" in st["grain"], f"bad state {st}"
    ld2 = gp.make_sasd_loader(DATA, SPLIT, per_host_batch=B, seed=42)
    ld2.set_state(st)
    for k in range(2, 5):
        b = next(ld2)
        assert _arrays_equal(b, ref_batches[k]), f"resume mismatch at batch {k}"
    print(f"  [e] resumability OK (state={st['grain']}; resumed 3 batches bit-exact)")


def test_uniform_assertion():
    """Non-uniform L without max_length must raise (padding path is a guarded TODO)."""
    raised = False
    try:
        # force the assertion by faking a _RowSource with two L values.
        orig = gp._RowSource
        class _FakeSrc:
            def __init__(self, paths):
                base = orig(paths)
                self._rows = list(base._rows)
                # truncate one row's input_ids to a different L
                r = dict(self._rows[0])
                r["input_ids"] = r["input_ids"][:-bd_off]
                self._rows[0] = r
            def __len__(self):
                return len(self._rows)
            def __getitem__(self, i):
                return self._rows[i]
        bd_off = 32
        gp._RowSource = _FakeSrc
        try:
            gp.make_sasd_loader(DATA, SPLIT, per_host_batch=2, seed=0)
        finally:
            gp._RowSource = orig
    except AssertionError:
        raised = True
    assert raised, "non-uniform L without max_length did not raise"
    print("  [f] uniform assertion OK (non-uniform L raises without max_length)")


def main():
    print("running grain_pipeline tests on", DATA)
    test_shapes_dtypes()
    test_determinism()
    test_multihost_disjoint_complete()
    test_noising_sanity()
    test_rng_fold_reproducible()
    test_resumability()
    test_uniform_assertion()
    print("GRAIN_PIPELINE_TESTS_PASS")


if __name__ == "__main__":
    main()
