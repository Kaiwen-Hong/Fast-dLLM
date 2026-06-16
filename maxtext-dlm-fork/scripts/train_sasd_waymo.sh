#!/usr/bin/env bash
# Deployable SASD (Fast-dDrive) training entry through MaxText's STANDARD train loop.
#
# Two steps:
#   (1) ONCE: save the Fast-dDrive Qwen2.5-3B weights as a MaxText Orbax PARAM checkpoint
#       (scanned/MaxText format, built with the SAME scan_layers=false config the train run
#       uses so the param tree matches). The train run then restores via load_parameters_path
#       and freshly inits the optimizer (correct fine-tune start).
#   (2) TRAIN: run trainers/pre_train/train.py with the SASD config + dataset_type=waymo_sasd.
#
# GPU (single RTX 5090) — run FOREGROUND with a timeout (no bg poll-loops):
#   source /home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/jax_gpu_env.sh
#   bash scripts/train_sasd_waymo.sh
# On a TPU pod: same commands, set OUT/PARAMS to gs://..., opt_type=adamw (FSDP shards opt
# state), per_device_batch_size / steps to taste; the mesh/FSDP/Orbax-to-GCS come for free.

set -euo pipefail

# --- paths (override via env) ---------------------------------------------------------------
REPO=/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork
CONFIG="${CONFIG:-$REPO/src/maxtext/configs/sasd_waymo.yml}"
SNAP="${SNAP:-/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f}"
PARAMS="${PARAMS:-/home/kaiwen/data/fast-ddrive/maxtext_sasd_params/fast_ddrive_qwen25_3b_params}"
OUT="${OUT:-/home/kaiwen/data/fast-ddrive/maxtext_sasd_run}"
RUN="${RUN:-sasd_waymo_v1}"
STEPS="${STEPS:-200}"
CKPT_PERIOD="${CKPT_PERIOD:-50}"

# --- env ------------------------------------------------------------------------------------
export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive:$REPO/src
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
PY="${JAXPY:-python}"

# --- (1) one-time param checkpoint (skip if it already exists) ------------------------------
if [ ! -d "$PARAMS" ]; then
  echo "=== [1/2] saving Fast-dDrive params -> $PARAMS ==="
  "$PY" "$REPO/scripts/save_fast_ddrive_params_ckpt.py" "$CONFIG" \
      model_name=qwen2.5-3b \
      out_ckpt_dir="$PARAMS" \
      snapshot_dir="$SNAP"
else
  echo "=== [1/2] param checkpoint already present at $PARAMS (skipping) ==="
fi

# --- (2) train through the standard MaxText loop --------------------------------------------
# HARDWARE: single-GPU sets hardware=gpu (skips the multi-host jax.distributed coordinator and
# uses the GPU init path). On a TPU pod, drop HARDWARE (default tpu) so the pod's distributed
# init + FSDP mesh kick in, and set opt_type=adamw (FSDP shards the fp32 moments), OUT/PARAMS
# to gs://..., async_checkpointing=true. Everything else is identical.
HARDWARE="${HARDWARE:-gpu}"
echo "=== [2/2] training: objective=sasd dataset_type=waymo_sasd hardware=$HARDWARE steps=$STEPS ==="
"$PY" -m maxtext.trainers.pre_train.train "$CONFIG" \
    model_name=qwen2.5-3b \
    hardware="$HARDWARE" \
    load_parameters_path="$PARAMS" \
    steps="$STEPS" \
    checkpoint_period="$CKPT_PERIOD" \
    base_output_directory="$OUT" \
    run_name="$RUN"
