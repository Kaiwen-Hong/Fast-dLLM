"""Independent sampled audit of a dataset-v2 ArrayRecord set against its source parquet.

Checks (CPU):
  1. COUNT: total AR records == dataset_info num_samples == total parquet rows.
  2. FIELDS: for --samples random global indices (deterministic seed), every of the 12
     arrays + sample_id/L/n_blocks in the AR record is BYTE-IDENTICAL to the parquet row
     at the same global index; image_embeds present with shape (168, 2048) bf16, finite.
Optional (--embeds_check K, needs GPU + snapshot): recompute fp32 embeds from the stored
pixel_values via the eager (validated) ViT and compare to stored bf16 (cast-compare:
exact-match fraction reported, requires allclose rtol 2e-2 at bf16 resolution).

  python verify_ar_v2_dataset.py <AR_DIR> <PQ_DIR> [--split train] [--samples 64]
                                 [--embeds_check 0] [--seed 0]
Prints VERIFY_AR_V2_PASS on success.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

import ml_dtypes
import pyarrow.parquet as pq
from array_record.python.array_record_data_source import ArrayRecordDataSource

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.data.ar_dataset import decode_example
from ddrive_jax.data.parquet_dataset import decode_row
from ddrive_jax.convert.prep_to_parquet import ARRAY_FIELDS

SNAP_DEFAULT = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
                "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ar_dir")
    ap.add_argument("pq_dir")
    ap.add_argument("--split", default="train")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--embeds_check", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap", default=SNAP_DEFAULT)
    args = ap.parse_args()

    ar_paths = sorted(glob.glob(os.path.join(args.ar_dir, f"{args.split}-*.arrayrecord")))
    pq_paths = sorted(glob.glob(os.path.join(args.pq_dir, f"{args.split}-*.parquet")))
    assert ar_paths and pq_paths, (args.ar_dir, args.pq_dir)
    ds = ArrayRecordDataSource(ar_paths)

    # --- 1. counts -----------------------------------------------------------------
    pq_counts = [pq.read_metadata(p).num_rows for p in pq_paths]
    n_pq, n_ar = sum(pq_counts), len(ds)
    info_path = os.path.join(args.ar_dir, f"dataset_info_{args.split}.json")
    n_info = json.load(open(info_path))["num_samples"] if os.path.exists(info_path) else n_ar
    print(f"[count] AR={n_ar} parquet={n_pq} dataset_info={n_info}")
    assert n_ar == n_pq == n_info, "COUNT MISMATCH"

    # --- 2. sampled bit-exact field compare ----------------------------------------
    rng = np.random.default_rng(args.seed)
    idxs = sorted(set([0, n_ar - 1] + rng.integers(0, n_ar, args.samples).tolist()))
    # group by parquet file so each file is read at most once
    bounds = np.cumsum([0] + pq_counts)
    by_file = {}
    for gi in idxs:
        fi = int(np.searchsorted(bounds, gi, side="right") - 1)
        by_file.setdefault(fi, []).append(gi)
    checked = 0
    for fi, gis in sorted(by_file.items()):
        table = pq.read_table(pq_paths[fi])
        for gi in gis:
            row = decode_row(table.slice(gi - bounds[fi], 1).to_pylist()[0])
            rec = decode_example(ds[gi])
            assert rec["sample_id"] == row["sample_id"], (gi, "sample_id")
            assert rec["L"] == row["L"] and rec["n_blocks"] == row["n_blocks"], gi
            for k in ARRAY_FIELDS:
                assert rec[k].dtype == row[k].dtype and rec[k].shape == row[k].shape, (gi, k)
                assert rec[k].tobytes() == row[k].tobytes(), (gi, k, "payload")
            e = rec["image_embeds"]
            assert e.shape == (168, 2048) and e.dtype == ml_dtypes.bfloat16, (gi, e.shape)
            assert np.isfinite(e.astype(np.float32)).all(), (gi, "non-finite embeds")
            checked += 1
    print(f"[fields] {checked} sampled records bit-exact vs parquet (+embeds well-formed)")

    # --- 3. optional embeds recompute (GPU) -----------------------------------------
    if args.embeds_check > 0:
        import jax
        jax.config.update("jax_default_matmul_precision", "highest")
        import jax.numpy as jnp
        from flax import nnx  # noqa: F401  (vit construction)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from parquet_to_ar_with_embeds import load_vit_streaming
        vit, _ = load_vit_streaming(args.snap)
        sub = idxs[:: max(1, len(idxs) // args.embeds_check)][: args.embeds_check]
        worst_rel, exact_min = 0.0, 1.0
        for gi in sub:
            rec = decode_example(ds[gi])
            ref32 = np.asarray(vit(jnp.asarray(rec["pixel_values"], jnp.float32),
                                   rec["image_grid_thw"]))
            ref16 = ref32.astype(ml_dtypes.bfloat16)
            got = rec["image_embeds"]
            exact = float((got == ref16).mean())
            rel = float(np.max(np.abs(got.astype(np.float32) - ref32))
                        / (np.max(np.abs(ref32)) + 1e-12))
            worst_rel, exact_min = max(worst_rel, rel), min(exact_min, exact)
            assert rel < 2e-2, (gi, rel)
        print(f"[embeds] {len(sub)} recomputed: exact-match frac >= {exact_min:.4f}, "
              f"max rel diff {worst_rel:.2e} (eager fp32 reference)")

    print("VERIFY_AR_V2_PASS")


if __name__ == "__main__":
    main()
