"""Stage 1 (ddrive env): drive the NEW msgpack dataloader and persist the SASD batch dicts.

Data loading is always host-side (grain runs on the CPU host, then feeds the accelerator); this
stage IS that host step. It runs the new `make_msgpack_sasd_loader` for N steps and writes each
emitted batch dict to an npz, so the JAX step driver (`train_jax_driver.py`, jax env, GPU/TPU)
can consume identical batches. Because the batches are a deterministic function of (seed, step),
materializing them here vs on the TPU host yields byte-identical inputs — verified by an optional
row-level bit-exact gate against the `prep_train_jax` npz oracle.

Run (ddrive env, `unset LD_LIBRARY_PATH`):
    python materialize_batches.py \
        --ar_path /home/kaiwen/data/flume_pipeline/parity/wod_e2e_sasd.array_record \
        --out_dir /home/kaiwen/data/flume_pipeline/batches_samplejson \
        --steps 10 --batch 1 --seed 0 [--oracle_prep_dir <prep npz dir to gate against>]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sasd_loader_msgpack import make_msgpack_sasd_loader, MsgpackArSource  # noqa: E402

# the grain batch keys train_tpu.run_step / prepare_batch consume
BATCH_ARRAYS = ["input_final", "labels_final", "original_labels", "weights",
                "position_ids", "rbi", "turn", "scaffold",
                "pixel_values", "image_grid_thw", "num_items"]
# per-sample prep fields to gate against the oracle npz (bit-exact)
PREP_FIELDS = ["input_ids", "labels", "rbi", "turn", "scaffold", "weight_vec",
               "block_alpha", "block_beta", "n_blocks", "pixel_values",
               "image_grid_thw", "position_ids", "vision_mask"]


def gate_rows_vs_oracle(ar_path, oracle_prep_dir):
    """Bit-exact gate: MsgpackArSource rows (TokenizeSASD) == prep_train_jax npz, all 13 fields."""
    src = MsgpackArSource(ar_path)
    npzs = sorted(os.path.join(oracle_prep_dir, f) for f in os.listdir(oracle_prep_dir)
                  if f.endswith(".npz"))
    assert len(npzs) == len(src), f"{len(npzs)} oracle npz != {len(src)} rows"
    all_ok = True
    for i, npz in enumerate(npzs):
        ref = np.load(npz)
        row = src[i]
        for f in PREP_FIELDS:
            a, b = np.asarray(row[f]), np.asarray(ref[f])
            ok = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
            if not ok:
                all_ok = False
                print(f"   ROW {i} field {f}: MISMATCH ({a.dtype}{a.shape} vs {b.dtype}{b.shape})")
    print(f"[materialize] row-level oracle gate: {'PASS (13/13 bit-exact)' if all_ok else 'FAIL'}")
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", default=False,
                    help="default off: deterministic round-robin so bs=1 cycles samples in order")
    ap.add_argument("--oracle_prep_dir", default=None, help="gate rows vs prep_train_jax npz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.oracle_prep_dir:
        # ENFORCE (not just report): a failing bit-exact gate must abort, so a divergent new
        # dataloader can never silently produce batches. (Codex double-check 2026-06-23.)
        if not gate_rows_vs_oracle(args.ar_path, args.oracle_prep_dir):
            raise SystemExit("ABORT: row-level oracle gate FAILED — new dataloader output != "
                             "prep_train_jax npz; refusing to write batches.")

    loader = make_msgpack_sasd_loader(
        args.ar_path, per_host_batch=args.batch, seed=args.seed,
        process_index=0, process_count=1, shuffle=args.shuffle)
    it = iter(loader)

    meta = {"ar_path": args.ar_path, "steps": args.steps, "batch": args.batch,
            "seed": args.seed, "L": loader.L, "N": loader.N, "n_img": loader.n_img, "batches": []}
    for st in range(args.steps):
        batch = next(it)
        out = os.path.join(args.out_dir, f"batch_{st:02d}.npz")
        np.savez(out, **{k: np.asarray(batch[k]) for k in BATCH_ARRAYS},
                 step=np.int64(batch["step"]),
                 sample_id=np.array(batch["sample_id"], dtype=object))
        meta["batches"].append({"step": st, "npz": os.path.basename(out),
                                "sample_id": list(batch["sample_id"])})
        print(f"[materialize] step {st}: sample_id={batch['sample_id']} "
              f"input_final{batch['input_final'].shape} num_items={batch['num_items'].tolist()}")
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"[materialize] wrote {args.steps} batches -> {args.out_dir}")
    print("MATERIALIZE_BATCHES_DONE")


if __name__ == "__main__":
    main()
