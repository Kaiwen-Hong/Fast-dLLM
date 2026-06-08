"""WOD-E2E Fast-dDrive SASD prepped npz -> sharded Apache Parquet (TPU/HF-ready).

Repackages the per-sample SASD npz produced by ``jax_ddrive/eval/prep_train_jax.py``
into sharded Parquet -- the portable, HF-native, grain-readable format the TPU training
pipeline consumes. Every array is stored as an EXACT little-endian binary blob plus a
``<name>_shape`` column, so reconstruction is bit-exact and lossless (pixel_values stay
fp16). Variable per-row ``L``/``N`` are fine; the grain loader pads to a fixed bucket.

Full-scale chain (docs/03_scaleup_tpu_spec.md):
  1. fast_ddrive/data/convert_wod_e2e.py  (autovla env)  tfrecord  -> train JSON + JPEGs
  2. jax_ddrive/eval/prep_train_jax.py     (ddrive env)   JSON+JPEG -> per-sample npz
  3. THIS SCRIPT                           (pyarrow)      npz       -> Parquet shards

Run:
  python jax_ddrive/ddrive_jax/convert/prep_to_parquet.py \
      --npz_dir /home/kaiwen/data/fast-ddrive/train/prep_train \
      --out_dir /home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd \
      --shard_size 64 --split train
"""
import argparse, glob, json, os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Exact stored dtype of each array (drives np.frombuffer reconstruction on the TPU side).
ARRAY_DTYPES = {
    "input_ids": "int64", "labels": "int64", "rbi": "int32", "turn": "int32",
    "scaffold": "bool", "weight_vec": "float32", "block_alpha": "float32",
    "block_beta": "float32", "position_ids": "int32", "vision_mask": "bool",
    "pixel_values": "float16", "image_grid_thw": "int64",
}
ARRAY_FIELDS = list(ARRAY_DTYPES)


def row_from_npz(path: str, sample_id: str) -> dict:
    d = np.load(path)
    row = {"sample_id": str(sample_id),
           "L": int(np.asarray(d["input_ids"]).shape[0]),
           "n_blocks": int(d["n_blocks"])}
    for name in ARRAY_FIELDS:
        a = np.ascontiguousarray(d[name], dtype=np.dtype(ARRAY_DTYPES[name]))
        row[name] = a.tobytes()                    # native (x86 little-endian) bytes
        row[name + "_shape"] = [int(x) for x in a.shape]
    return row


def parquet_schema() -> pa.Schema:
    fields = [pa.field("sample_id", pa.string()),
              pa.field("L", pa.int32()),
              pa.field("n_blocks", pa.int32())]
    for name in ARRAY_FIELDS:
        fields.append(pa.field(name, pa.large_binary()))
        fields.append(pa.field(name + "_shape", pa.list_(pa.int64())))
    return pa.schema(fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=64)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_samples", type=int, default=-1)
    args = ap.parse_args()

    man_path = os.path.join(args.npz_dir, "manifest.json")
    if os.path.exists(man_path):
        manifest = json.load(open(man_path))
    else:
        manifest = [{"npz": os.path.basename(p),
                     "sample_id": os.path.splitext(os.path.basename(p))[0]}
                    for p in sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))]
    if args.max_samples > 0:
        manifest = manifest[: args.max_samples]

    os.makedirs(args.out_dir, exist_ok=True)
    sch = parquet_schema()
    n = len(manifest)
    nshards = max(1, (n + args.shard_size - 1) // args.shard_size)
    written, shard_files = 0, []
    for si in range(nshards):
        chunk = manifest[si * args.shard_size:(si + 1) * args.shard_size]
        if not chunk:
            break
        rows = [row_from_npz(os.path.join(args.npz_dir, m["npz"]),
                             m.get("sample_id", m["npz"])) for m in chunk]
        table = pa.Table.from_pylist(rows, schema=sch)
        fn = f"{args.split}-{si:05d}-of-{nshards:05d}.parquet"
        pq.write_table(table, os.path.join(args.out_dir, fn), compression="zstd")
        shard_files.append(fn)
        written += len(rows)
        print(f"  shard {si + 1}/{nshards}: {len(rows)} rows -> {fn}", flush=True)

    info = {"split": args.split, "num_samples": written, "num_shards": len(shard_files),
            "shard_size": args.shard_size, "array_dtypes": ARRAY_DTYPES,
            "shape_suffix": "_shape", "byte_order": "little",
            "reconstruct": "np.frombuffer(row[name], dtype=ARRAY_DTYPES[name]).reshape(row[name+'_shape'])",
            "files": shard_files}
    json.dump(info, open(os.path.join(args.out_dir, f"dataset_info_{args.split}.json"), "w"), indent=2)
    print(f"[prep_to_parquet] {written} rows in {len(shard_files)} shards -> {args.out_dir}")
    print("PREP_TO_PARQUET_DONE")


if __name__ == "__main__":
    main()
