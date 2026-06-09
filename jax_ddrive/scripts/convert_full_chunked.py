"""Chunked, resumable, disk-bounded full WOD-E2E -> TPU-ready Parquet converter.

Wraps the VALIDATED per-shard stage1+2 (`convert_subset_parallel.process_shard`) and the
validated Parquet decode contract (`prep_to_parquet.row_from_npz/parquet_schema`), adding only
robust orchestration:

  * Process shards in CHUNKS (default 32). After each chunk: write that chunk's Parquet,
    delete the chunk's npz immediately -> disk peak ~= chunk_npz (+final parquet), not all-263.
  * RESUMABLE: a `_progress.json` records completed chunks + running global parquet-file index +
    frame count. Re-running skips done chunks -> a kill/restart never wastes more than one chunk.
  * Parquet files use a running GLOBAL index `train-NNNNN.parquet` (no per-chunk collision).

Why this exists: the single-pass `convert_subset_parallel.py` accumulates ALL npz and writes
Parquet only at the very end, so a mid-run kill (e.g. a session/host restart) loses everything
AND peaks at ~650 GB npz. This driver fixes both while reusing the exact validated conversion.

Run (ddrive env; detached so it survives a session restart):
  setsid nohup /home/kaiwen/miniconda3/envs/ddrive/bin/python \
    jax_ddrive/scripts/convert_full_chunked.py \
    --out_dir /home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd_full \
    --work_dir /home/kaiwen/data/fast-ddrive/dataset_build/work_full \
    --workers 16 --chunk 32 > /tmp/full_chunked.log 2>&1 &
"""
import argparse, glob, json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "/home/kaiwen/Desktop/research/Fast-dLLM"
sys.path.insert(0, REPO + "/jax_ddrive")
sys.path.insert(0, REPO + "/jax_ddrive/scripts")

from convert_subset_parallel import process_shard, TRAIN_GLOB   # validated stage1+2 per shard
import pyarrow as pa, pyarrow.parquet as pq
from ddrive_jax.convert.prep_to_parquet import row_from_npz, parquet_schema, ARRAY_DTYPES


def write_chunk_parquet(rows, out_dir, start_file_idx, split, shard_size=64):
    """Write `rows` [(npz_path, sample_id)] to {split}-{global_idx}.parquet (running index)."""
    sch = parquet_schema()
    files = []
    n = len(rows)
    nsh = max(1, (n + shard_size - 1) // shard_size)
    for k in range(nsh):
        chunk = rows[k * shard_size:(k + 1) * shard_size]
        if not chunk:
            break
        recs = [row_from_npz(p, sid) for p, sid in chunk]   # incremental: 64 rows in RAM
        t = pa.Table.from_pylist(recs, schema=sch)
        fn = f"{split}-{start_file_idx + k:05d}.parquet"
        pq.write_table(t, os.path.join(out_dir, fn), compression="zstd")
        files.append(fn)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=32, help="shards per chunk (disk = chunk*~2.5GB npz)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_shards", type=int, default=0, help="0=all; >0 limits total shards (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    shards = sorted(glob.glob(TRAIN_GLOB))
    assert shards, f"no shards under {TRAIN_GLOB}"
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    nchunks = (len(shards) + args.chunk - 1) // args.chunk

    state_path = os.path.join(args.out_dir, "_progress.json")
    if os.path.exists(state_path):
        state = json.load(open(state_path))
        print(f"[resume] {len(state['done_chunks'])} chunks already done, "
              f"{state['frames']} frames, {state['file_idx']} parquet files", flush=True)
    else:
        state = {"done_chunks": [], "file_idx": 0, "frames": 0, "split": args.split}

    print(f"[chunked] {len(shards)} shards, chunk={args.chunk} -> {nchunks} chunks, "
          f"{args.workers} workers", flush=True)
    t0 = time.time()
    for c in range(nchunks):
        if c in state["done_chunks"]:
            print(f"[chunk {c}/{nchunks-1}] skip (done)", flush=True)
            continue
        sl = shards[c * args.chunk:(c + 1) * args.chunk]
        cw = os.path.join(args.work_dir, f"chunk_{c}")
        shutil.rmtree(cw, ignore_errors=True)   # clear any partial orphan from a prior killed attempt
        os.makedirs(cw, exist_ok=True)

        rows, ce = [], time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_shard, s, cw, i): i for i, s in enumerate(sl)}
            for fu in as_completed(futs):
                r = fu.result()
                if r["err"]:
                    print(f"  [chunk {c} shard {r['idx']:02d}] ERROR {r['err']}", flush=True)
                else:
                    rows.extend(r["rows"])
                    print(f"  [chunk {c} shard {r['idx']:02d}] +{r['n']} frames ({r.get('sec','?')}s)", flush=True)
        files = write_chunk_parquet(rows, args.out_dir, state["file_idx"], args.split)
        shutil.rmtree(cw, ignore_errors=True)   # free this chunk's npz immediately

        state["done_chunks"].append(c)
        state["file_idx"] += len(files)
        state["frames"] += len(rows)
        json.dump(state, open(state_path, "w"))
        el = (time.time() - t0) / 60
        print(f"[chunk {c}/{nchunks-1}] DONE +{len(rows)} frames in {(time.time()-ce):.0f}s "
              f"-> total {state['frames']} frames / {state['file_idx']} parquet files; {el:.1f} min elapsed",
              flush=True)

    parquet_files = sorted(f for f in os.listdir(args.out_dir) if f.endswith(".parquet"))
    json.dump({"split": args.split, "num_samples": state["frames"], "num_shards": len(parquet_files),
               "array_dtypes": ARRAY_DTYPES, "shape_suffix": "_shape", "byte_order": "little",
               "reconstruct": "np.frombuffer(row[name], ARRAY_DTYPES[name]).reshape(row[name+'_shape'])",
               "files": parquet_files},
              open(os.path.join(args.out_dir, f"dataset_info_{args.split}.json"), "w"), indent=2)
    shutil.rmtree(args.work_dir, ignore_errors=True)
    print(f"ALL_DONE total_frames={state['frames']} parquet_files={len(parquet_files)}", flush=True)


if __name__ == "__main__":
    main()
