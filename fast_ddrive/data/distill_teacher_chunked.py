#!/usr/bin/env python3
"""Resumable, chunked teacher inference for large-scale distillation (Stage 2).

Runs ``eval/batch_inference.py`` over a Fast-dDrive JSON in CHUNK-sized pieces so a
multi-hour run survives crashes: each chunk's predictions are saved independently
(``pred_chunkNNN.json``) and completed chunks are skipped on restart. Per-chunk
timeout + one retry; a chunk that still fails is left for a later pass (the job keeps
going). At the end, all chunk predictions are concatenated into ``predictions_all.json``.

Idempotent: rerun the exact same command to resume. Writes ``STATUS.json`` for at-a-glance
progress.

Usage::

    python fast_ddrive/data/distill_teacher_chunked.py \
        --json   /home/kaiwen/data/fast-ddrive/train/distill_50k/train_targets_50k.json \
        --images /home/kaiwen/data/fast-ddrive/train/distill_50k/images_50k \
        --out    /home/kaiwen/data/fast-ddrive/train/distill_50k/teacher \
        --chunk 2500
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PYTHON = "/home/kaiwen/miniconda3/envs/ddrive/bin/python"
BATCH = os.path.join(REPO, "fast_ddrive", "eval", "batch_inference.py")
MODEL = "Efficient-Large-Model/Fast-dDrive"
HF_HOME = "/home/kaiwen/data/huggingface"


def load_preds(path):
    d = json.load(open(path))
    return d["predictions"] if isinstance(d, dict) and "predictions" in d else d


def chunk_complete(path, expected):
    if not os.path.exists(path):
        return False
    try:
        return len(load_preds(path)) == expected
    except Exception:
        return False


def write_status(out, **kw):
    kw["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(kw, open(os.path.join(out, "STATUS.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=2500)
    ap.add_argument("--mode", default="scaffold_spec")
    ap.add_argument("--timeout", type=int, default=7200, help="per-chunk seconds before kill+retry")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "chunks"), exist_ok=True)
    data = json.load(open(args.json))
    n = len(data)
    nchunks = (n + args.chunk - 1) // args.chunk
    print(f"[distill] {n} samples, chunk={args.chunk} -> {nchunks} chunks", flush=True)

    t0 = time.time()
    failed = []
    for i in range(nchunks):
        lo, hi = i * args.chunk, min((i + 1) * args.chunk, n)
        chunk = data[lo:hi]
        pred_path = os.path.join(args.out, f"pred_chunk{i:03d}.json")
        done = sum(1 for j in range(i) if chunk_complete(os.path.join(args.out, f"pred_chunk{j:03d}.json"),
                                                          min((j + 1) * args.chunk, n) - j * args.chunk))
        if chunk_complete(pred_path, len(chunk)):
            print(f"[skip] chunk {i:03d} ({lo}:{hi}) already complete", flush=True)
            continue

        cj = os.path.join(args.out, "chunks", f"chunk{i:03d}.json")
        json.dump(chunk, open(cj, "w"))
        run_dir = os.path.join(args.out, f"run_chunk{i:03d}")
        ok = False
        for attempt in range(2):
            elapsed = time.time() - t0
            write_status(args.out, total=n, nchunks=nchunks, current_chunk=i,
                         chunks_done=done, samples_done=done * args.chunk,
                         elapsed_min=round(elapsed / 60, 1),
                         note=f"running chunk {i} attempt {attempt}")
            print(f"[run ] chunk {i:03d} ({lo}:{hi}) attempt {attempt} "
                  f"| {done}/{nchunks} chunks done | {round(elapsed/60,1)} min elapsed", flush=True)
            try:
                subprocess.run(
                    [PYTHON, BATCH, "--model_path", MODEL, "--eval_json", cj,
                     "--image_root", args.images, "--output_dir", run_dir,
                     "--mode", args.mode, "--confidence_threshold", "0.0", "--num_gpus", "1"],
                    env={**os.environ, "HF_HOME": HF_HOME},
                    timeout=args.timeout, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                )
                src = os.path.join(run_dir, "predictions.json")
                if chunk_complete(src, len(chunk)):
                    shutil.move(src, pred_path)
                    shutil.rmtree(run_dir, ignore_errors=True)
                    ok = True
                    print(f"[done] chunk {i:03d}", flush=True)
                    break
                print(f"[warn] chunk {i:03d} attempt {attempt}: predictions incomplete", flush=True)
            except subprocess.TimeoutExpired:
                print(f"[warn] chunk {i:03d} attempt {attempt}: TIMEOUT", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"[warn] chunk {i:03d} attempt {attempt}: exit {e.returncode}", flush=True)
            shutil.rmtree(run_dir, ignore_errors=True)
        if not ok:
            failed.append(i)
            print(f"[FAIL] chunk {i:03d} after retries — continuing", flush=True)

    # Concatenate all completed chunk predictions.
    all_preds, missing = [], []
    for i in range(nchunks):
        p = os.path.join(args.out, f"pred_chunk{i:03d}.json")
        lo, hi = i * args.chunk, min((i + 1) * args.chunk, n)
        if chunk_complete(p, hi - lo):
            all_preds.extend(load_preds(p))
        else:
            missing.append(i)
    out_all = os.path.join(args.out, "predictions_all.json")
    json.dump({"metadata": {"n": len(all_preds), "source": args.json}, "predictions": all_preds},
              open(out_all, "w"))
    write_status(args.out, total=n, nchunks=nchunks, chunks_done=nchunks - len(missing),
                 samples_done=len(all_preds), elapsed_min=round((time.time() - t0) / 60, 1),
                 failed_chunks=failed, missing_chunks=missing,
                 note="DONE" if not missing else "INCOMPLETE — rerun to fill missing chunks")
    print(f"[distill] concatenated {len(all_preds)}/{n} preds -> {out_all}", flush=True)
    if missing:
        print(f"[distill] MISSING chunks {missing} — rerun same command to fill. EXIT_INCOMPLETE", flush=True)
        sys.exit(2)
    print("STAGE2_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
