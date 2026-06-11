"""Parallel full-dataset converter: packed SASD parquet -> ArrayRecord(tf.train.Example).

Each of the N parquet files -> one arrayrecord shard, via the single-file worker
(parquet_file_to_tfexample_ar.py) run as a fresh subprocess (TF on CPU, ~3 GB RAM each).
Resumable: skips shards whose output already exists (worker writes atomically .tmp->replace).

  python full_to_tfexample_ar_driver.py <SRC_PARQUET_DIR> <DST_AR_DIR> [WORKERS=7]
"""
import glob, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC, DST = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 7
JAXPY = "/home/kaiwen/jax-dlm-baseline/.venv/bin/python"
REPO = "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive"
CONV = REPO + "/scripts/parquet_file_to_tfexample_ar.py"
os.makedirs(DST, exist_ok=True)
pfiles = sorted(glob.glob(os.path.join(SRC, "train-*.parquet")))
nsh = len(pfiles)
DTYPES = {"input_ids": "int64", "labels": "int64", "rbi": "int32", "turn": "int32",
          "scaffold": "bool", "weight_vec": "float32", "block_alpha": "float32",
          "block_beta": "float32", "position_ids": "int32", "vision_mask": "bool",
          "pixel_values": "float16", "image_grid_thw": "int64"}


def conv(i, pf):
    out = os.path.join(DST, f"train-{i:05d}-of-{nsh:05d}.arrayrecord")
    if os.path.exists(out):
        return (i, "skip", out)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=REPO, TF_CPP_MIN_LOG_LEVEL="3")
    r = subprocess.run([JAXPY, CONV, pf, out], env=env, capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(out)
    return (i, "ok" if ok else "FAIL:" + (r.stderr or "")[-300:], out)


print(f"[driver] {nsh} parquet -> {nsh} arrayrecord shards, {W} workers", flush=True)
t0, done, fails = time.time(), 0, []
with ThreadPoolExecutor(max_workers=W) as ex:
    futs = {ex.submit(conv, i, pf): i for i, pf in enumerate(pfiles)}
    for f in as_completed(futs):
        i, st, out = f.result()
        done += 1
        if st.startswith("FAIL"):
            fails.append(i)
        print(f"[{done}/{nsh}] shard {i:03d}: {st[:60]}  ({(time.time()-t0)/60:.1f}min)", flush=True)

total = sum(1 for _ in glob.glob(os.path.join(DST, "train-*.arrayrecord")))
json.dump({"format": "arrayrecord/tf.train.Example", "split": "train", "num_shards": total,
           "record_encoding": "tf.train.Example; array fields = tf.io.serialize_tensor bytes; "
                              "decode via tf.io.parse_tensor(bytes, dtype)",
           "array_fields": list(DTYPES.keys()), "array_dtypes": DTYPES,
           "scalar_fields": {"sample_id": "bytes", "L": "int64", "n_blocks": "int64"},
           "files": sorted(os.path.basename(p) for p in glob.glob(os.path.join(DST, "train-*.arrayrecord")))},
          open(os.path.join(DST, "dataset_info_train.json"), "w"), indent=2)
print(f"FULL_TFEXAMPLE_AR_DONE shards={total} fails={fails} elapsed={(time.time()-t0)/60:.1f}min", flush=True)
