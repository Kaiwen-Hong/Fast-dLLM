#!/usr/bin/env bash
# INFERENCE — run section_diffusion inference on a real v6e-1 TPU over 1 example, then score
#   ADE_3s / ADE_5s / RFS against the Waymo GT.
#   input : 1-example AR -> scaffold prep (ddrive env)  +  the raw tfrecord (for GT)
#   output: GCS .../inference/<sample>/infer/predictions.json  +  waymo_eval_results.json
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPY="${DDRIVE_PY:-/home/kaiwen/miniconda3/envs/ddrive/bin/python}"
AV="${AUTOVLA_PY:-/home/kaiwen/miniconda3/envs/autovla/bin/python}"
AR="${1:-/home/kaiwen/data/flume_pipeline/example0/example0.array_record}"
JOB="${2:-/home/kaiwen/data/flume_pipeline/example0}"
GT_TFREC="${GT_TFREC:-/home/kaiwen/data/flume_pipeline/raw_track/two_examples.tfrecord}"
GCS_OUT="${GCS_OUT:-gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd/flume/deliverable2/inference/example0}"
EVAL=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/eval
# (a) scaffold prep (ddrive)
( unset LD_LIBRARY_PATH; export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
  [ -d "$JOB/prep" ] || { "$DPY" "$HERE/ar_to_eval_json.py" --ar_path "$AR" --out_json "$JOB/eval.json" --image_root "$JOB/imgroot"
    "$DPY" "$EVAL/prep_jax_eval.py" --eval_json "$JOB/eval.json" --image_root "$JOB/imgroot" --out_dir "$JOB/prep"; } )
# (b) TPU inference
JOB_DIR="$JOB" GCS_OUT="$GCS_OUT" MODE=infer NAME="${NAME:-flume-inf}" bash "$HERE/tpu_run.sh"
# (c) Waymo metrics (autovla) on the TPU predictions
gsutil -q cp "$GCS_OUT/infer/predictions.json" "$JOB/tpu_predictions.json"
"$AV" /home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/eval/evaluate_waymo_metrics.py \
  --pred_json "$JOB/tpu_predictions.json" --gt "$GT_TFREC" --output_dir "$JOB/metrics"
gsutil -q cp "$JOB/metrics/waymo_eval_results.json" "$GCS_OUT/waymo_eval_results.json"
echo "=== ADE_3s / ADE_5s / RFS ==="; cat "$JOB/metrics/waymo_eval_results.json"
