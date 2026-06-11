"""Consolidate many tiny SASD parquet shards into fewer large ones (upload-friendly).

The chunked converter writes ~64-row shards (6,499 files for the full set) which trips HF's
per-file API rate limit (429) on upload. This repacks them into ~GROUP*64-row files
(~1.4 GB each) WITHOUT touching the row contents (pure pa.concat_tables -> same decode
contract). Resumable: skips already-written outputs.

  python pack_parquet.py <SRC_DIR> <DST_DIR> [GROUP=50]
"""
import sys, glob, os, json
import pyarrow as pa, pyarrow.parquet as pq

SRC, DST = sys.argv[1], sys.argv[2]
GROUP = int(sys.argv[3]) if len(sys.argv) > 3 else 50
os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC, "train-*.parquet")))
n = len(files)
nout = (n + GROUP - 1) // GROUP
print(f"packing {n} files -> {nout} files (group={GROUP})", flush=True)

written, rows_total = [], 0
for oi in range(nout):
    fn = f"train-{oi:05d}-of-{nout:05d}.parquet"
    outp = os.path.join(DST, fn)
    grp = files[oi * GROUP:(oi + 1) * GROUP]
    if os.path.exists(outp):                      # resume
        written.append(fn); rows_total += pq.read_metadata(outp).num_rows
        continue
    t = pa.concat_tables([pq.read_table(f) for f in grp])
    pq.write_table(t, outp, compression="zstd")
    written.append(fn); rows_total += t.num_rows
    print(f"[{oi+1}/{nout}] {fn} <- {len(grp)} files, {t.num_rows} rows", flush=True)

# carry over dataset_info with the new file list
info_src = os.path.join(SRC, "dataset_info_train.json")
info = json.load(open(info_src)) if os.path.exists(info_src) else {"split": "train"}
info.update({"num_samples": rows_total, "num_shards": len(written), "files": written})
json.dump(info, open(os.path.join(DST, "dataset_info_train.json"), "w"), indent=2)
print(f"PACK_DONE {len(written)} files, {rows_total} rows", flush=True)
