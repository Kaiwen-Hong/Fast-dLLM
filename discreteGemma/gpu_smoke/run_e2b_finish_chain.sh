#!/bin/bash
# Post-training finish chain: export manual params -> kauldron ckpt, run
# eval_main (C4, steps=32 gate at the final ckpt), C6 vision tests.
# Usage: run_e2b_finish_chain.sh [it_step] [pt_step]   (default 1600 1600)
set -u
IT_STEP=${1:-1600}
PT_STEP=${2:-1600}
source /home/kaiwen/miniconda3/etc/profile.d/conda.sh && conda activate dgemma-jax
TESTS=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/tests
SMOKE=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gpu_smoke
LOGD=/home/kaiwen/data/dgemma_e2b/logs
export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export JAX_COMPILATION_CACHE_DIR=/home/kaiwen/data/dgemma_e2b/xla_cache
mkdir -p "$LOGD"

for V in it pt; do
  STEP=$([ "$V" = it ] && echo "$IT_STEP" || echo "$PT_STEP")
  if [ -d "/home/kaiwen/data/dgemma_e2b/xp_manual_$V/params_$STEP" ]; then
    echo "=== EXPORT $V step=$STEP ==="
    (cd "$TESTS" && python e2b_export_ckpt.py --variant "$V" --step "$STEP" \
       > "$LOGD/export_${V}_$STEP.log" 2>&1)
    echo "export $V exit=$?"
    echo "=== EVAL $V step=$STEP (steps=32, chartqa metrics) ==="
    bash "$SMOKE/run_chartqa_e2b_eval.sh" "$V" "$STEP" 1 sample_ar_steps32
  else
    echo "!!! no trained params for $V step=$STEP — skip"
  fi
done

echo "=== VISION TESTS (C6) ==="
cd "$TESTS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
python e2b_vision_test.py --check plumbing --variant it > "$LOGD/vision_plumbing.log" 2>&1
echo "vision plumbing exit=$?"
python e2b_vision_test.py --check padding --variant it > "$LOGD/vision_padding.log" 2>&1
echo "vision padding exit=$?"
if [ -d "/home/kaiwen/data/dgemma_e2b/xp_manual_it/params_$IT_STEP" ]; then
  python e2b_vision_test.py --check gap --variant it \
    --params_dir "/home/kaiwen/data/dgemma_e2b/xp_manual_it/params_$IT_STEP" \
    > "$LOGD/vision_gap_it.log" 2>&1
  echo "vision gap(it) exit=$?"
fi
if [ -d "/home/kaiwen/data/dgemma_e2b/xp_manual_pt/params_$PT_STEP" ]; then
  python e2b_vision_test.py --check gap --variant pt \
    --params_dir "/home/kaiwen/data/dgemma_e2b/xp_manual_pt/params_$PT_STEP" \
    > "$LOGD/vision_gap_pt.log" 2>&1
  echo "vision gap(pt) exit=$?"
fi
echo "FINISH CHAIN DONE"
