#!/bin/bash
# E2B ChartQA SFT training (design doc Phase D). Usage:
#   run_chartqa_e2b_train.sh <variant pt|it> [batch] [steps]
set -u
V=${1:?variant pt|it}
B=${2:-2}
S=${3:-200}
A=${4:-4}   # grad-accum k (global batch = B*A)
source /home/kaiwen/miniconda3/etc/profile.d/conda.sh && conda activate dgemma-jax
cd /home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.93
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
export JAX_COMPILATION_CACHE_DIR=/home/kaiwen/data/dgemma_e2b/xla_cache
export DGEMMA_E2B_VARIANT=$V DGEMMA_E2B_BATCH=$B DGEMMA_E2B_STEPS=$S DGEMMA_E2B_ACCUM=$A
WORK=/home/kaiwen/data/dgemma_e2b/xp_chartqa_$V
LOGD=/home/kaiwen/data/dgemma_e2b/logs
mkdir -p "$WORK" "$LOGD"
python -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_chartqa_e2b.py \
  --cfg.workdir="$WORK" \
  > "$LOGD/train_$V.log" 2>&1
echo "train[$V] exit=$?  (log: $LOGD/train_$V.log)"
