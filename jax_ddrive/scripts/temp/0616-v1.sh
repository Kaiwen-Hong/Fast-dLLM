#!/usr/bin/env bash
# Owner-side ONE-SHOT publisher for the STEP 3 track (data-processing sanity + released-ckpt
# val parity + new-split processing). Executable form of docs/6for_internal/00_owner_publish.md
# (STEP 0 §1-2 code bundle + STEP 3 §6 artifacts). Run on the LOCAL 5090 box — it has the raw
# val, the release snapshot, and an authed gsutil.
#
# SAFETY (after the 0616-v0 accidental-upload incident): DRY-RUN by default — prints the exact
# plan and uploads NOTHING. Pass --go to actually build + upload. Artifacts already on disk are
# reused (idempotent), so re-running is cheap.
#
#   bash jax_ddrive/scripts/temp/0616-v1.sh         # dry run — show the plan, touch nothing
#   bash jax_ddrive/scripts/temp/0616-v1.sh --go    # really build (if missing) + upload to GCS
set -euo pipefail

GO=0; [ "${1:-}" = "--go" ] && GO=1
run() { echo "+ $*"; if [ "$GO" = 1 ]; then "$@"; fi; }   # echo always; execute only with --go

SRC="${BKT:-gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd}"
REPO=/home/kaiwen/Desktop/research/Fast-dLLM
AV=/home/kaiwen/miniconda3/envs/autovla/bin/python
SNAP=/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f
VAL_GLOB='/home/kaiwen/data/fast-ddrive/waymo/val/val_*.tfrecord*'
EVAL=/home/kaiwen/data/fast-ddrive/eval
cd "$REPO"
mkdir -p "$EVAL"

if [ "$GO" = 1 ]; then
  echo "########## EXECUTE (--go): building + uploading to $SRC ##########"
else
  echo "########## DRY RUN — nothing is built or uploaded. Re-run with --go to execute. ##########"
fi

# ---- 0) git cleanliness (the bundle SHA only describes the code if the tree is committed) ----
if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: working tree is DIRTY — the code bundle will be tagged -dirty and its SHA will not"
  echo "         fully describe it. Commit first if you want a clean provenance SHA."
fi

# ---- 1+2) code bundle (now packs fast_ddrive/ too: convert + official metric) ----
echo "## STEP 0 §1-2 — code bundle (maxtext-dlm-fork + jax_ddrive + fast_ddrive)"
run bash jax_ddrive/scripts/upload_code_to_gcs.sh

# ---- 6a) released NVIDIA fp32 snapshot (internal has none → ship it, ~16GB) ----
echo "## STEP 3 §6a — release snapshot -> $SRC/release_fast_ddrive_snapshot"
[ -d "$SNAP" ] || { echo "ERROR: release snapshot not found at $SNAP" >&2; exit 1; }
run gsutil -m rsync -r "$SNAP" "$SRC/release_fast_ddrive_snapshot"

# ---- 6b) 479-rated val subset (build if missing, then upload; ~1.1GB) ----
echo "## STEP 3 §6b — val_rated_479.tfrecord"
if [ -s "$EVAL/val_rated_479.tfrecord" ]; then
  echo "   reuse existing ($(du -h "$EVAL/val_rated_479.tfrecord" | cut -f1)) — skip rebuild"
else
  run "$AV" jax_ddrive/scripts/extract_rated_val_subset.py \
      --val_tfrecords "$VAL_GLOB" --out_tfrecord "$EVAL/val_rated_479.tfrecord"
fi
run gsutil cp "$EVAL/val_rated_479.tfrecord" "$SRC/eval/val_rated_479.tfrecord"

# ---- 6c) GT pkl for the official metric (build if missing, then upload; ~0.55MB) ----
echo "## STEP 3 §6c — rated_val_gt.pkl"
if [ -s "$EVAL/rated_val_gt.pkl" ]; then
  echo "   reuse existing ($(du -h "$EVAL/rated_val_gt.pkl" | cut -f1)) — skip rebuild"
else
  run "$AV" jax_ddrive/scripts/build_rated_val_gt.py \
      --val_tfrecords "$VAL_GLOB" --out_pkl "$EVAL/rated_val_gt.pkl" --rated_only
fi
run gsutil cp "$EVAL/rated_val_gt.pkl" "$SRC/eval/rated_val_gt.pkl"

# ---- 6d) provenance manifest for the shipped snapshot ----
echo "## STEP 3 §6d — DATA_MANIFEST for the snapshot (crc32c, no download)"
run "$AV" jax_ddrive/scripts/data_manifest.py "$SRC/release_fast_ddrive_snapshot" --upload

# ---- self-check (only meaningful after --go) ----
echo "## self-check"
run gsutil cat "$SRC/code/fastddrive-LATEST.txt"
if [ "$GO" = 1 ]; then
  for f in release_fast_ddrive_snapshot/ eval/val_rated_479.tfrecord eval/rated_val_gt.pkl; do
    gsutil ls "$SRC/$f" >/dev/null 2>&1 && echo "OK       $f" || echo "MISSING  $f"
  done
fi

echo "########## DONE (GO=$GO).  Next: internal agent runs docs/6for_internal/03_data_and_inference_parity.md ##########"
