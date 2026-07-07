#!/bin/bash
# E2B ChartQA offline eval via the official eval_main (D5 protocol; steps=32
# gate; run 64/96 on the best ckpt separately). Usage:
#   run_chartqa_e2b_eval.sh <variant pt|it> <step> [eval_batch] [eval_names]
set -u
V=${1:?variant pt|it}
STEP=${2:?ckpt step}
B=${3:-4}
NAMES=${4:-sample_ar_steps32}
source /home/kaiwen/miniconda3/etc/profile.d/conda.sh && conda activate dgemma-jax
cd /home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
export JAX_COMPILATION_CACHE_DIR=/home/kaiwen/data/dgemma_e2b/xla_cache
export DGEMMA_E2B_VARIANT=$V DGEMMA_E2B_BATCH=$B
WORK=/home/kaiwen/data/dgemma_e2b/xp_chartqa_$V
LOGD=/home/kaiwen/data/dgemma_e2b/logs
mkdir -p "$LOGD"
python -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_chartqa_e2b.py \
  --task=chartqa \
  --step="$STEP" \
  --eval_names="$NAMES" \
  --cfg.workdir="$WORK" \
  > "$LOGD/eval_${V}_step${STEP}_${NAMES}.log" 2>&1
echo "eval[$V step=$STEP $NAMES] exit=$?"
grep -oE 'chartqa_[a-z_]+=[0-9.]+' "$LOGD/eval_${V}_step${STEP}_${NAMES}.log" | tail -4
