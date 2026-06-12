#!/usr/bin/env python3
"""Self-contained, resumable finalizer for the 50k teacher-distill (Stages 3+4).

Waits for the chunked teacher run (Stage 2) to finish, then:
  Stage 3  merge_distilled_labels.py (hybrid) -> train_targets_distilled_50k.json
  Stage 4a prep_train_jax.py         -> per-sample npz (variable L)
  Stage 4b pad_npz_uniform.py        -> uniform L (auto-max), loss-zero verified contract
  Stage 4c prep_to_parquet.py        -> uniform Parquet (the trainable distilled dataset)
  cleanup intermediate npz; write FINALIZE_REPORT.json

Design for unattended overnight runs: ALL subprocess env (HF_HOME, PYTHONPATH) and ALL
cleanup (shutil.rmtree) happen INSIDE this process, so launching it is a single clean
command that needs no approval, and no sub-step prompts. Resumable: each stage is skipped
if its output already exists, so a crash/rerun continues.

Launch (clean, allowlisted):
    /home/kaiwen/miniconda3/envs/ddrive/bin/python fast_ddrive/data/finalize_distill_50k.py
"""
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PY = "/home/kaiwen/miniconda3/envs/ddrive/bin/python"
HF_HOME = "/home/kaiwen/data/huggingface"
D = "/home/kaiwen/data/fast-ddrive/train/distill_50k"

TEACHER_PRED = f"{D}/teacher/predictions_all.json"
ORIG_JSON = f"{D}/train_targets_50k.json"
IMAGES = f"{D}/images_50k"
DISTILLED = f"{D}/train_targets_distilled_50k.json"
NPZ_DIR = f"{D}/prep_npz"
PAD_DIR = f"{D}/prep_npz_pad"
PARQUET = f"{D}/parquet_L1280"
REPORT = f"{D}/FINALIZE_REPORT.json"
LOG = f"{D}/finalize.log"

N_EXPECTED = 50331


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def env():
    return {**os.environ, "HF_HOME": HF_HOME, "PYTHONPATH": os.path.join(REPO, "jax_ddrive")}


def run(cmd, **kw):
    log("RUN " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env(), cwd=REPO, **kw)


def run_nocheck(cmd, **kw):
    log("RUN " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=False, env=env(), cwd=REPO, **kw).returncode


def teacher_running():
    """True if a distill_teacher_chunked orchestrator is alive (external or our own resume)."""
    r = subprocess.run(["pgrep", "-f", "distill_teacher_chunked"],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def preds_complete():
    if not os.path.exists(TEACHER_PRED):
        return False
    try:
        d = json.load(open(TEACHER_PRED))
        preds = d["predictions"] if isinstance(d, dict) else d
        return len(preds) >= N_EXPECTED
    except Exception:
        return False


def count_json(path):
    return len(json.load(open(path)))


def count_npz(d):
    return len([f for f in os.listdir(d) if f.endswith(".npz")]) if os.path.isdir(d) else 0


def main():
    log("=== finalize_distill_50k start (wait+selfheal teacher -> merge -> stage4) ===")

    # ---- Stage 2: wait for the running teacher; self-heal if it dies incomplete ------
    # The standalone teacher orchestrator runs independently. We don't disrupt it: we
    # wait. If it ever exits while predictions are still incomplete, we resume it
    # ourselves (distill_teacher_chunked is idempotent: completed chunks are skipped).
    heals = 0
    waited = 0
    while not preds_complete():
        if teacher_running():
            if waited % 1800 == 0:
                try:
                    st = json.load(open(f"{D}/teacher/STATUS.json"))
                    log(f"waiting: chunks_done={st.get('chunks_done')}/{st.get('nchunks')} "
                        f"samples~{st.get('samples_done')}")
                except Exception:
                    log("waiting for teacher")
            time.sleep(300)
            waited += 300
            continue
        # not running and not complete -> resume it (blocks until it exits)
        if heals >= 6:
            log("teacher kept failing after 6 resumes — STOPPING before merge")
            sys.exit(2)
        heals += 1
        log(f"teacher not running and incomplete -> resume #{heals}")
        run_nocheck([PY, "fast_ddrive/data/distill_teacher_chunked.py",
                     "--json", ORIG_JSON, "--images", IMAGES,
                     "--out", f"{D}/teacher", "--chunk", "2500", "--timeout", "7200"])
    log(f"teacher predictions complete ({TEACHER_PRED})")

    # ---- Stage 3: merge -------------------------------------------------------------
    if not os.path.exists(DISTILLED) or count_json(DISTILLED) < N_EXPECTED:
        run([PY, "fast_ddrive/data/merge_distilled_labels.py",
             "--teacher_pred", TEACHER_PRED, "--orig_json", ORIG_JSON,
             "--out_json", DISTILLED])
    log(f"Stage 3 done: {count_json(DISTILLED)} distilled samples")

    # ---- Stage 4a: prep_train_jax (variable-L npz) ----------------------------------
    if count_npz(NPZ_DIR) < count_json(DISTILLED):
        run([PY, "jax_ddrive/eval/prep_train_jax.py",
             "--train_json", DISTILLED, "--image_root", IMAGES, "--out_dir", NPZ_DIR])
    log(f"Stage 4a done: {count_npz(NPZ_DIR)} npz")

    # ---- Stage 4b: pad to uniform L (auto-detect global max) ------------------------
    if count_npz(PAD_DIR) < count_npz(NPZ_DIR):
        run([PY, "jax_ddrive/scripts/pad_npz_uniform.py",
             "--in_dir", NPZ_DIR, "--out_dir", PAD_DIR])
    log(f"Stage 4b done: {count_npz(PAD_DIR)} padded npz")

    # ---- Stage 4c: pack to Parquet --------------------------------------------------
    import glob
    if not glob.glob(f"{PARQUET}/**/*.parquet", recursive=True):
        run([PY, "jax_ddrive/ddrive_jax/convert/prep_to_parquet.py",
             "--npz_dir", PAD_DIR, "--out_dir", PARQUET, "--shard_size", "64", "--split", "train"])

    # ---- verify + report ------------------------------------------------------------
    import pyarrow.parquet as pq
    shards = sorted(glob.glob(f"{PARQUET}/**/*.parquet", recursive=True))
    rows = sum(pq.ParquetFile(f).metadata.num_rows for f in shards)
    t = pq.ParquetFile(shards[0]).read()
    Ls = sorted(set(t["L"].to_pylist()))
    NBs = sorted(set(t["n_blocks"].to_pylist()))
    report = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "distilled_json": DISTILLED, "n_distilled": count_json(DISTILLED),
        "parquet_dir": PARQUET, "parquet_rows": rows, "parquet_shards": len(shards),
        "uniform_L_shard0": Ls, "uniform_n_blocks_shard0": NBs,
        "ok": rows >= N_EXPECTED and len(Ls) == 1 and len(NBs) == 1,
    }
    json.dump(report, open(REPORT, "w"), indent=2)
    log("REPORT " + json.dumps(report))

    # ---- cleanup intermediate npz (keep JSON + parquet) -----------------------------
    if report["ok"]:
        shutil.rmtree(NPZ_DIR, ignore_errors=True)
        shutil.rmtree(PAD_DIR, ignore_errors=True)
        log("cleaned intermediate npz")
    log("=== FINALIZE_DONE ===")


if __name__ == "__main__":
    main()
