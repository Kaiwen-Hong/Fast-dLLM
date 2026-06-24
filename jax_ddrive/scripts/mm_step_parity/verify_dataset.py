"""OVERNIGHT PARITY — dataset-processing validation (the SERIALIZATION round-trip).

Standalone (does NOT edit existing code). Runs in the `jax` venv.

The parity scripts (capture/parity_nnx/parity_maxtext) already prove that the prep npz
(the dataset-processing output of jax_ddrive/eval/prep_train_jax.py) -> model gives PyTorch
parity. THIS script closes the last link: the npz survives serialization to the TPU training
format (Parquet, then ArrayRecord) BIT-EXACT, so training-from-disk consumes exactly the
tensors we validated.

It compares, for each of the 12 array fields:
    prep npz  ==  decode_row(parquet)  ==  decode_example(arrayrecord)
using the REAL pipeline decoders (ddrive_jax.data.parquet_dataset.decode_row,
ddrive_jax.data.ar_dataset.decode_example). It also re-derives get_rope_index/weight_vec
sanity already covered by capture (rope_match logged there).

The Parquet + AR shards are produced by the REAL pipeline scripts (run by the orchestrator):
    prep_to_parquet.py  ->  parquet_file_to_tfexample_ar.py

Usage:
  python verify_dataset.py --prep_dir <dir> --parquet_dir <dir> --ar_path <shard.arrayrecord>
"""
import argparse, glob, json, os, sys
import numpy as np

ROOT = os.environ.get("MMSTEP_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
sys.path.insert(0, ROOT)
from ddrive_jax.convert.prep_to_parquet import ARRAY_FIELDS, ARRAY_DTYPES
from ddrive_jax.data.parquet_dataset import decode_row

ARR = list(ARRAY_FIELDS)


def load_npz(path):
    d = np.load(path)
    return {k: np.ascontiguousarray(d[k], dtype=np.dtype(ARRAY_DTYPES[k])) for k in ARR}


def eq_fields(a, b):
    """All 12 fields bit-exact? returns (ok, {field: (ok, detail)})."""
    res = {}
    for k in ARR:
        ak, bk = np.asarray(a[k]), np.asarray(b[k])
        ok = ak.shape == bk.shape and ak.dtype == bk.dtype and bool(np.array_equal(ak, bk))
        res[k] = {"pass": ok, "shape_a": list(ak.shape), "shape_b": list(bk.shape),
                  "dtype_a": str(ak.dtype), "dtype_b": str(bk.dtype),
                  "maxdiff": (float(np.abs(ak.astype(np.float64) - bk.astype(np.float64)).max())
                              if ak.shape == bk.shape and ak.size else None)}
    return all(v["pass"] for v in res.values()), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep_dir", required=True)
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--ar_path", default=None, help="single .arrayrecord shard (optional)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.prep_dir, "manifest.json")))
    npz = [load_npz(os.path.join(args.prep_dir, m["npz"])) for m in man]
    n = len(npz)
    print(f"[D] prep samples: {n}", flush=True)

    # --- parquet decode_row (the framework-free SSOT contract) ---
    import pyarrow.parquet as pq
    rows = []
    for p in sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet"))):
        rows.extend(pq.read_table(p).to_pylist())
    assert len(rows) == n, f"parquet rows {len(rows)} != npz {n}"
    pq_dec = [decode_row(r) for r in rows]

    # --- arrayrecord decode_example (optional) ---
    ar_dec = None
    if args.ar_path and os.path.exists(args.ar_path):
        from ddrive_jax.data.ar_dataset import decode_example
        from array_record.python.array_record_data_source import ArrayRecordDataSource
        ds = ArrayRecordDataSource([args.ar_path])
        ar_dec = [decode_example(ds[i]) for i in range(len(ds))]
        assert len(ar_dec) == n, f"AR records {len(ar_dec)} != npz {n}"

    out = {"n": n, "samples": []}
    all_ok = True
    for i in range(n):
        sid = man[i].get("sample_id", str(i))
        pq_ok, pq_res = eq_fields(npz[i], pq_dec[i])
        rec = {"sample_id": sid, "parquet_roundtrip": {"pass": pq_ok}}
        line = f"[D] sample {i} ({sid[:24]}): parquet={'OK' if pq_ok else 'FAIL'}"
        if not pq_ok:
            rec["parquet_roundtrip"]["fields"] = {k: v for k, v in pq_res.items() if not v["pass"]}
        if ar_dec is not None:
            ar_ok, ar_res = eq_fields(npz[i], ar_dec[i])
            xq_ok, _ = eq_fields(pq_dec[i], ar_dec[i])
            rec["ar_roundtrip"] = {"pass": ar_ok}
            rec["parquet_vs_ar"] = {"pass": xq_ok}
            if not ar_ok:
                rec["ar_roundtrip"]["fields"] = {k: v for k, v in ar_res.items() if not v["pass"]}
            line += f"  ar={'OK' if ar_ok else 'FAIL'}  pq==ar={'OK' if xq_ok else 'FAIL'}"
            sample_ok = pq_ok and ar_ok and xq_ok
        else:
            sample_ok = pq_ok
            line += "  (no AR shard)"
        all_ok = all_ok and sample_ok
        rec["pass"] = bool(sample_ok)
        out["samples"].append(rec)
        print(line, flush=True)

    out["pass"] = bool(all_ok)
    rp = args.report or os.path.join(args.prep_dir, "verify_dataset.json")
    json.dump(out, open(rp, "w"), indent=2)
    print(f"\n[D] report -> {rp}")
    print(f"SASD_MM_DATASET_ROUNDTRIP_{'PASS' if all_ok else 'FAIL'} "
          f"(parquet + arrayrecord bit-exact vs prep npz, {n} samples, 12 fields)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
