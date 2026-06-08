#!/usr/bin/env bash
# Build the TPU-ready Fast-dDrive SASD Parquet dataset from RAW WOD-E2E tfrecords, end-to-end.
# This is the "full = one command" chain referenced in docs/03_scaleup_tpu_spec.md §1.1.
# The overnight subset (400 samples) used the SAME three stages; here they are parameterized so
# the real 263-shard train split (or any shard range) runs unchanged on this box or Waymo infra.
#
# Stages:
#   1. convert_wod_e2e.py  (autovla env, TF + compiled E2E proto): tfrecord -> train JSON + JPEGs
#   2. prep_train_jax.py   (ddrive env, CPU, no weights):          JSON+JPEGs -> per-sample SASD npz
#   3. prep_to_parquet.py  (pyarrow):                              npz -> sharded Parquet (+ schema)
#   4. (optional) upload:  private HF repo + GCS bucket
#
# SMOKE (validate the whole chain on 1 shard, minutes):
#   bash build_full_dataset.sh --smoke
# FULL (the real 263-shard train split; hours, image-heavy, hundreds of GB of JPEGs):
#   bash build_full_dataset.sh --full
# Upload after building:
#   bash build_full_dataset.sh --full --upload-hf --upload-gcs gs://YOUR_BUCKET/wod_e2e_sasd
set -euo pipefail

REPO=/home/kaiwen/Desktop/research/Fast-dLLM
AUTOVLA=/home/kaiwen/miniconda3/envs/autovla/bin/python   # TF + end_to_end_driving_data_pb2
DDRIVE=/home/kaiwen/miniconda3/envs/ddrive/bin/python     # transformers + section_utils + pyarrow
DATA=/home/kaiwen/data/fast-ddrive
TRAIN_TFR="$DATA/waymo/train/training_*.tfrecord-*"       # 263 raw train shards (local)
SPLIT=train
HF_REPO=kaiwen2/wod-e2e-fast-ddrive-sasd

MODE=""; UPLOAD_HF=0; GCS=""
MAX_SHARDS=-1; MAX_FRAMES=-1; TAG=full
while [ $# -gt 0 ]; do case "$1" in
  --smoke) MODE=smoke; MAX_SHARDS=1; MAX_FRAMES=64; TAG=smoke;;
  --full)  MODE=full;;
  --max-shards) MAX_SHARDS="$2"; shift;;
  --max-frames) MAX_FRAMES="$2"; shift;;
  --upload-hf) UPLOAD_HF=1;;
  --upload-gcs) GCS="$2"; shift;;
  *) echo "unknown arg: $1"; exit 2;;
esac; shift; done
[ -z "$MODE" ] && { echo "pass --smoke or --full"; exit 2; }

OUT=$DATA/dataset_build/$TAG
JSON=$OUT/train_targets.json
IMGROOT=$OUT/train_images
NPZ=$OUT/prep_train
PQ=$OUT/parquet
mkdir -p "$OUT"
export PYTHONPATH=$REPO/jax_ddrive
cd "$REPO"

echo "== Stage 1: tfrecord -> JSON + JPEGs (autovla env) =="
"$AUTOVLA" fast_ddrive/data/convert_wod_e2e.py \
  --tfrecords "$TRAIN_TFR" --out_json "$JSON" --image_root "$IMGROOT" \
  --with_target --max_shards "$MAX_SHARDS" --max_frames "$MAX_FRAMES"

echo "== Stage 2: JSON+JPEGs -> per-sample SASD npz (ddrive env) =="
"$DDRIVE" jax_ddrive/eval/prep_train_jax.py \
  --train_json "$JSON" --image_root "$IMGROOT" --out_dir "$NPZ"

echo "== Stage 3: npz -> sharded Parquet =="
"$DDRIVE" jax_ddrive/ddrive_jax/convert/prep_to_parquet.py \
  --npz_dir "$NPZ" --out_dir "$PQ" --split "$SPLIT" --shard_size 64

echo "== Parquet built at: $PQ =="
ls -la "$PQ" | tail -n +2

if [ "$UPLOAD_HF" = 1 ]; then
  echo "== Stage 4a: upload to PRIVATE HF repo $HF_REPO =="
  cp -n "$REPO/jax_ddrive/docs/03_scaleup_tpu_spec.md" "$PQ/" 2>/dev/null || true
  "$DDRIVE" - "$PQ" "$HF_REPO" <<'PY'
import sys, os
from huggingface_hub import HfApi, create_repo
folder, repo = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ.get("HF_TOKEN"))
create_repo(repo, repo_type="dataset", private=True, exist_ok=True)  # PRIVATE — WOD license
api.upload_folder(folder_path=folder, repo_id=repo, repo_type="dataset",
                  commit_message="full WOD-E2E SASD Parquet (private)")
print("uploaded ->", f"https://huggingface.co/datasets/{repo}")
PY
fi

if [ -n "$GCS" ]; then
  echo "== Stage 4b: stage to GCS $GCS (TPU-native) =="
  gsutil -m rsync -r "$PQ" "$GCS"
  echo "gs:// staged -> $GCS"
fi

echo "DONE ($MODE): $(ls "$PQ"/*.parquet 2>/dev/null | wc -l) parquet shards"
