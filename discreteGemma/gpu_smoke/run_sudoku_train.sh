#!/bin/bash
# Small-scale end-to-end (train side): the OFFICIAL kauldron.main Trainer on a
# TINY random-init DiffusionGemma + real sudoku bagz, 100 steps, ckpts @10/25/50/100.
# Unlike gpu_smoke/sft_smoke.py (a hand-rolled loop), this exercises the full released
# training harness (Trainer -> orbax checkpoints), on one RTX 5090. Then run
# run_sudoku_eval.sh for the eval_main AR-sampling side.
set -u
source /home/kaiwen/miniconda3/etc/profile.d/conda.sh && conda activate dgemma-jax
cd /home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2
WORK=${1:-/home/kaiwen/data/dgemma_clean_e2e/xp_sudoku}
LOGD=/home/kaiwen/data/dgemma_clean_e2e/logs
mkdir -p "$WORK" "$LOGD"
python -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_sudoku_tiny.py \
  --cfg.workdir="$WORK" \
  > "$LOGD/train_sudoku.log" 2>&1
echo "train exit=$?  (log: $LOGD/train_sudoku.log)"
