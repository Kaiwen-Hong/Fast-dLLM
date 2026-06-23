"""Bit-exact parity gate (doc 09 §8): TokenizeSASD output == prep_train_jax npz oracle.

Chain under test (the whole new pipeline, end to end):
    sample.json --build_record--> msgpack --pack--> ArrayRecord --read_all--> unpack
        --TokenizeSASD.map--> 13-field dict     ==(bit-exact)==     prep_train_jax npz

`prep_train_jax.py` is the ORACLE: we run it fresh as a subprocess on the same sample.json
so the comparison is apples-to-apples with the current code. The gate asserts every one of
the 13 fields is byte-identical (incl. the float16 pixel_values and the train-time-expanded
weight_vec/block_alpha/block_beta), plus a separate ArrayRecord round-trip check (raw bytes
in == raw bytes out). Emits FLUME_PIPELINE_PARITY_PASS iff everything matches.

Run (ddrive env, `unset LD_LIBRARY_PATH`):
    python parity_test.py \
        --train_json /home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/sample.json \
        --image_root /home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from etl_record import build_record_from_json, pack, unpack    # noqa: E402
from etl_local import write_array_record                       # noqa: E402
from grain_loader import read_raw_records                      # noqa: E402

REPO = os.environ.get("FASTDDRIVE_REPO", "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
ORACLE = os.path.join(REPO, "eval", "prep_train_jax.py")

# 13-field contract (doc 09 §1). dtype recorded so a mismatch reports type drift, not just values.
FIELDS = ["input_ids", "labels", "rbi", "turn", "scaffold", "weight_vec",
          "block_alpha", "block_beta", "n_blocks", "pixel_values",
          "image_grid_thw", "position_ids", "vision_mask"]


def _clean_env():
    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)               # ddrive env requires LD_LIBRARY_PATH unset
    return env


def run_oracle(train_json, image_root, out_dir):
    """Run prep_train_jax.py fresh -> per-sample npz in out_dir. Returns sorted npz paths."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, ORACLE, "--train_json", train_json,
           "--image_root", image_root, "--out_dir", out_dir]
    print(f"[parity] running oracle: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=_clean_env(), capture_output=True, text=True)
    if r.returncode != 0 or "PREP_TRAIN_JAX_DONE" not in r.stdout:
        sys.stderr.write(r.stdout + "\n" + r.stderr + "\n")
        raise RuntimeError("oracle prep_train_jax.py failed")
    return sorted(p for p in (os.path.join(out_dir, f) for f in os.listdir(out_dir))
                  if p.endswith(".npz"))


def compare(got, npz_path):
    """Compare a TokenizeSASD output dict against one oracle npz. Returns list of (field, ok, msg)."""
    ref = np.load(npz_path)
    rows = []
    for f in FIELDS:
        a, b = np.asarray(got[f]), np.asarray(ref[f])
        if a.shape != b.shape:
            rows.append((f, False, f"shape {a.shape} != {b.shape}"))
        elif a.dtype != b.dtype:
            rows.append((f, False, f"dtype {a.dtype} != {b.dtype}"))
        elif not np.array_equal(a, b):
            n = int(np.sum(a != b))
            # first differing index for a quick diagnostic
            idx = np.argwhere(a != b)
            where = tuple(idx[0]) if len(idx) else ()
            rows.append((f, False, f"{n} elem differ; first@{where} got={a[where]} ref={b[where]}"))
        else:
            rows.append((f, True, f"ok {a.dtype}{a.shape}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json",
                    default="/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/sample.json")
    ap.add_argument("--image_root",
                    default="/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive")
    ap.add_argument("--oracle_dir", default=None,
                    help="reuse existing oracle npz dir instead of regenerating (must match this sample.json)")
    ap.add_argument("--workdir", default=None, help="where to put AR + oracle (default: temp)")
    args = ap.parse_args()

    work = args.workdir or tempfile.mkdtemp(prefix="flume_parity_")
    os.makedirs(work, exist_ok=True)
    import json
    data = json.load(open(args.train_json))
    print(f"[parity] {len(data)} samples from {args.train_json}")

    # ── ETL: build records -> msgpack -> ArrayRecord ────────────────────────────────────
    records = [build_record_from_json(item, args.image_root) for item in data]
    blobs = [pack(r) for r in records]
    ar_path = os.path.join(work, "wod_e2e_sasd.array_record")
    write_array_record(blobs, ar_path)

    # ── GATE 1: ArrayRecord round-trip (raw bytes lossless) ─────────────────────────────
    back = read_raw_records(ar_path)
    rt_ok = len(back) == len(blobs) and all(x == y for x, y in zip(back, blobs))
    rt_field_ok = True
    for orig, b in zip(records, back):
        u = unpack(b)
        if not (u["sample_id"] == orig["sample_id"] and u["images"] == orig["images"]
                and u["prompt_text"] == orig["prompt_text"] and u["target_text"] == orig["target_text"]):
            rt_field_ok = False
    print(f"[parity] GATE 1 ArrayRecord round-trip: bytes={'OK' if rt_ok else 'FAIL'} "
          f"fields={'OK' if rt_field_ok else 'FAIL'}")

    # ── oracle npz ──────────────────────────────────────────────────────────────────────
    if args.oracle_dir and os.path.isdir(args.oracle_dir):
        npzs = sorted(os.path.join(args.oracle_dir, f) for f in os.listdir(args.oracle_dir)
                      if f.endswith(".npz"))
        print(f"[parity] reusing oracle dir {args.oracle_dir} ({len(npzs)} npz)")
    else:
        npzs = run_oracle(args.train_json, args.image_root, os.path.join(work, "oracle"))
    if len(npzs) != len(records):
        raise RuntimeError(f"oracle produced {len(npzs)} npz but {len(records)} records")

    # ── GATE 2: TokenizeSASD vs oracle, 13 fields bit-exact ─────────────────────────────
    from tokenize_sasd import TokenizeSASD
    print("[parity] building TokenizeSASD (loads Qwen processor)...", flush=True)
    tk = TokenizeSASD()

    all_ok = rt_ok and rt_field_ok
    for k, (blob, npz_path) in enumerate(zip(back, npzs)):
        got = tk.map(blob)                                  # decode msgpack bytes -> 13-field dict
        rows = compare(got, npz_path)
        sample_ok = all(ok for _, ok, _ in rows)
        all_ok = all_ok and sample_ok
        print(f"\n[parity] sample {k} ({records[k]['sample_id']}): "
              f"{'PASS' if sample_ok else 'FAIL'}")
        for f, ok, msg in rows:
            print(f"    {'OK  ' if ok else 'FAIL'} {f:16s} {msg}")

    print()
    if all_ok:
        print("FLUME_PIPELINE_PARITY_PASS")
        return 0
    print("FLUME_PIPELINE_PARITY_FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
