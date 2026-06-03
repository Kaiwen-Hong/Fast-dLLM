#!/usr/bin/env bash
# Overnight Fast-dDrive runner: full-val eval on BOTH stacks + JAX real-data training.
# Each stage writes a marker; a watcher greps the log. Order puts the long JAX-479 eval last.
set -uo pipefail
unset LD_LIBRARY_PATH || true
ROOT=/home/kaiwen/Desktop/research/Fast-dLLM
PT=/home/kaiwen/miniconda3/envs/ddrive/bin/python
AV=/home/kaiwen/miniconda3/envs/autovla/bin/python
JX=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
SNAP=/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f
E=/home/kaiwen/data/fast-ddrive/eval
T=/home/kaiwen/data/fast-ddrive/train
L=/home/kaiwen/data/fast-ddrive/logs
VALGLOB='/home/kaiwen/data/fast-ddrive/waymo/val/val_*.tfrecord*'
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.92
cd "$ROOT"
echo "OVERNIGHT_START $(date)"

# ── 1. PyTorch full-479 eval (scaffold_spec) + official metric ──
echo "=== [1/3] PyTorch full-479 eval $(date) ==="
$PT fast_ddrive/eval/batch_inference.py --model_path "$SNAP" \
  --eval_json $E/val_rated.json --image_root $E/val_images \
  --output_dir $E/pt_val_full_ss --mode scaffold_spec --num_gpus 1 > $L/pt_eval_full.log 2>&1
$AV fast_ddrive/eval/evaluate_waymo_metrics.py --pred_json $E/pt_val_full_ss/predictions.json \
  --gt "$VALGLOB" --output_dir $E/pt_val_full_ss >> $L/pt_eval_full.log 2>&1
echo "PT_FULL_DONE $(date)"

# ── 2. JAX SASD training on real Waymo data ──
echo "=== [2/3] JAX real-data training $(date) ==="
$JX jax_ddrive/ddrive_jax/train_waymo_sasd_jax.py --prep_dir $T/prep_train \
  --samples 200 --steps 400 --lr 2e-5 --ckpt_dir $T/ckpt_jax > $L/train_jax.log 2>&1
echo "TRAIN_DONE $(date)"

# ── 3. JAX full-479 eval (section_diffusion) + official metric ──
echo "=== [3/3] JAX full-479 eval $(date) ==="
$JX jax_ddrive/eval/jax_batch_inference.py --prep_dir $E/prep_val_full \
  --out_dir $E/jax_val_full_sd > $L/jax_eval_full.log 2>&1
$AV fast_ddrive/eval/evaluate_waymo_metrics.py --pred_json $E/jax_val_full_sd/predictions.json \
  --gt "$VALGLOB" --output_dir $E/jax_val_full_sd >> $L/jax_eval_full.log 2>&1
echo "JAX_FULL_DONE $(date)"
echo "OVERNIGHT_ALL_DONE $(date)"
