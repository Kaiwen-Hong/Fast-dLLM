#!/bin/bash
# Offline eval of sudoku checkpoints at 10/25/50/100 via the official eval_main.
set -u
source /home/kaiwen/miniconda3/etc/profile.d/conda.sh && conda activate dgemma-jax
cd /home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export XLA_FLAGS="--xla_disable_hlo_passes=constant_folding --xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
WORK=/home/kaiwen/data/overnight_dgemma/xp_sudoku
LOGD=/home/kaiwen/data/overnight_dgemma/logs
for STEP in 10 25 50 100; do
  echo "======== EVAL step $STEP ========"
  python -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
    --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku_tiny.py \
    --task=sudoku \
    --step=$STEP \
    --eval_names=sample_ar_steps32 \
    --cfg.workdir=$WORK \
    --cfg.eval_ds.batch_size=4 \
    --cfg.aux.eval_num_batches=2 \
    > $LOGD/eval_step${STEP}.log 2>&1
  echo "step $STEP exit=$?"
done
echo "ALL EVALS DONE"
