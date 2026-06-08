"""Parallel, shard-streaming converter: RAW WOD-E2E train tfrecords -> TPU-ready Parquet.

Memory-safe for the 30 GB / no-swap box:
  - bounded worker pool (each worker runs stage1+stage2 as subprocesses, ~2 GB RAM each)
  - per-shard JPEG cleanup IMMEDIATELY after stage2 (Parquet holds the processed pixel_values,
    so the raw JPEGs are disposable) -> scratch never accumulates the ~1 TB full-set JPEG peak
  - incremental Parquet writing (64 rows/chunk) -> RAM stays flat regardless of total size

Each shard ~1597 frames; 32 shards ~= 51k frames (the "~50k subset"). Stages reuse the exact
validated scripts (convert_wod_e2e.py, prep_train_jax.py) and decode contract (prep_to_parquet).

Run (after the harness test frees RAM; ~10-15 min for 32 shards on this box):
  /home/kaiwen/miniconda3/envs/ddrive/bin/python \
    jax_ddrive/scripts/convert_subset_parallel.py \
    --shards 32 --workers 10 \
    --out_dir /home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd_50k \
    --upload --hf_repo kaiwen2/wod-e2e-fast-ddrive-sasd-50k
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "/home/kaiwen/Desktop/research/Fast-dLLM"
AUTOVLA = "/home/kaiwen/miniconda3/envs/autovla/bin/python"   # stage1: TF + E2E proto
DDRIVE = "/home/kaiwen/miniconda3/envs/ddrive/bin/python"     # stage2: transformers processor
TRAIN_GLOB = "/home/kaiwen/data/fast-ddrive/waymo/train/training_*.tfrecord-*"
sys.path.insert(0, REPO + "/jax_ddrive")


def process_shard(shard_path, work, idx):
    """stage1 (tfrecord->JSON+JPEG) -> stage2 (JSON+JPEG->npz) -> drop JPEGs. Returns rows or error."""
    base = os.path.join(work, f"shard_{idx:03d}")
    img, js, npz, log = base + "_img", base + ".json", base + "_npz", base + ".log"
    os.makedirs(npz, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=REPO + "/jax_ddrive")
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.run(
            [AUTOVLA, REPO + "/fast_ddrive/data/convert_wod_e2e.py",
             "--tfrecords", shard_path, "--out_json", js, "--image_root", img, "--with_target"],
            stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0 or not os.path.exists(js):
            shutil.rmtree(img, ignore_errors=True)
            return {"idx": idx, "n": 0, "err": "stage1_fail", "rows": []}
        rc = subprocess.run(
            [DDRIVE, REPO + "/jax_ddrive/eval/prep_train_jax.py",
             "--train_json", js, "--image_root", img, "--out_dir", npz],
            stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    man = os.path.join(npz, "manifest.json")
    shutil.rmtree(img, ignore_errors=True)              # drop JPEGs immediately
    if os.path.exists(js):
        os.remove(js)
    if rc != 0 or not os.path.exists(man):
        return {"idx": idx, "n": 0, "err": "stage2_fail", "rows": []}
    rows = [(os.path.join(npz, e["npz"]), e.get("sample_id", e["npz"]))
            for e in json.load(open(man))]
    return {"idx": idx, "n": len(rows), "err": None, "rows": rows, "sec": round(time.time() - t0, 1)}


def write_parquet(all_rows, out_dir, split, shard_size=64):
    import pyarrow as pa, pyarrow.parquet as pq
    from ddrive_jax.convert.prep_to_parquet import row_from_npz, parquet_schema, ARRAY_DTYPES
    os.makedirs(out_dir, exist_ok=True)
    sch = parquet_schema()
    n = len(all_rows)
    nsh = max(1, (n + shard_size - 1) // shard_size)
    files = []
    for si in range(nsh):
        chunk = all_rows[si * shard_size:(si + 1) * shard_size]
        if not chunk:
            break
        rows = [row_from_npz(p, sid) for p, sid in chunk]     # incremental: 64 rows in RAM
        t = pa.Table.from_pylist(rows, schema=sch)
        fn = f"{split}-{si:05d}-of-{nsh:05d}.parquet"
        pq.write_table(t, os.path.join(out_dir, fn), compression="zstd")
        files.append(fn)
        if (si + 1) % 20 == 0:
            print(f"  parquet {si + 1}/{nsh} shards written", flush=True)
    json.dump({"split": split, "num_samples": n, "num_shards": len(files),
               "shard_size": shard_size, "array_dtypes": ARRAY_DTYPES, "shape_suffix": "_shape",
               "byte_order": "little",
               "reconstruct": "np.frombuffer(row[name], ARRAY_DTYPES[name]).reshape(row[name+'_shape'])",
               "files": files},
              open(os.path.join(out_dir, f"dataset_info_{split}.json"), "w"), indent=2)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=32, help="# of train tfrecord shards (~1597 frames each)")
    ap.add_argument("--workers", type=int, default=10, help="bounded for 30GB RAM (~2GB/worker)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--work_dir", default="/home/kaiwen/data/fast-ddrive/dataset_build/work_50k")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--hf_repo", default="kaiwen2/wod-e2e-fast-ddrive-sasd-50k")
    ap.add_argument("--keep_npz", action="store_true", help="keep per-shard npz after parquet")
    args = ap.parse_args()

    shards = sorted(glob.glob(TRAIN_GLOB))[: args.shards]
    assert shards, f"no shards under {TRAIN_GLOB}"
    os.makedirs(args.work_dir, exist_ok=True)
    print(f"[convert] {len(shards)} shards x ~1597 frames, {args.workers} workers", flush=True)

    all_rows, done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, s, args.work_dir, i): i for i, s in enumerate(shards)}
        for fu in as_completed(futs):
            r = fu.result()
            done += 1
            if r["err"]:
                print(f"  [shard {r['idx']:03d}] ERROR {r['err']} ({done}/{len(shards)})", flush=True)
            else:
                all_rows.extend(r["rows"])
                print(f"  [shard {r['idx']:03d}] +{r['n']} frames in {r.get('sec','?')}s "
                      f"({done}/{len(shards)}, total {len(all_rows)})", flush=True)

    print(f"[convert] stage1+2 done: {len(all_rows)} frames in {time.time()-t0:.0f}s. Writing Parquet...", flush=True)
    files = write_parquet(all_rows, args.out_dir, args.split)
    sz = sum(os.path.getsize(os.path.join(args.out_dir, f)) for f in files)
    print(f"[convert] {len(all_rows)} frames -> {len(files)} parquet shards, {sz/1e9:.1f} GB -> {args.out_dir}", flush=True)

    if not args.keep_npz:
        shutil.rmtree(args.work_dir, ignore_errors=True)
        print(f"[convert] cleaned work dir {args.work_dir}", flush=True)

    if args.upload:
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)  # PRIVATE (WOD license)
        api.upload_folder(folder_path=args.out_dir, repo_id=args.hf_repo, repo_type="dataset",
                          commit_message=f"WOD-E2E SASD ~{len(all_rows)}-frame subset (private)")
        print(f"[convert] uploaded PRIVATE -> https://huggingface.co/datasets/{args.hf_repo}", flush=True)
    print("CONVERT_SUBSET_DONE")


if __name__ == "__main__":
    main()
