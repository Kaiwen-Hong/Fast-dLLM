#!/bin/bash
# Overnight Phase-D chain: train it -> train pt -> eval final ckpts -> vision
# tests. Falls back batch 2/accum4 -> 1/accum8 on train failure. Sequential
# (one GPU). Logs under /home/kaiwen/data/dgemma_e2b/logs/.
set -u
cd /home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gpu_smoke
LOGD=/home/kaiwen/data/dgemma_e2b/logs; mkdir -p "$LOGD"
RES=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/results

train_variant () {
  local V=$1
  echo "=== TRAIN $V (b2/accum4) ==="
  bash run_chartqa_e2b_train.sh "$V" 2 200 4
  if ! grep -q 'train_complete' /home/kaiwen/data/dgemma_e2b/xp_chartqa_$V/train_complete.txt 2>/dev/null \
     && ! tail -5 "$LOGD/train_$V.log" | grep -q '\[800\]'; then
    if grep -qE 'RESOURCE_EXHAUSTED|Out of memory' "$LOGD/train_$V.log"; then
      echo "=== TRAIN $V OOM -> fallback b1/accum8 ==="
      rm -rf /home/kaiwen/data/dgemma_e2b/xp_chartqa_$V
      bash run_chartqa_e2b_train.sh "$V" 1 200 8
    fi
  fi
}

steps_of () {  # final micro-step = 200 * accum actually used
  local V=$1
  ls /home/kaiwen/data/dgemma_e2b/xp_chartqa_$V/checkpoints 2>/dev/null \
    | grep -oE '[0-9]+' | sort -n | tail -1
}

train_variant it
train_variant pt

for V in it pt; do
  FINAL=$(steps_of "$V")
  if [ -n "$FINAL" ]; then
    echo "=== EVAL $V step=$FINAL (steps=32 gate) ==="
    bash run_chartqa_e2b_eval.sh "$V" "$FINAL" 4 sample_ar_steps32
  else
    echo "!!! no ckpt for $V — skipping eval"
  fi
done

echo "=== VISION TESTS ==="
source /home/kaiwen/miniconda3/etc/profile.d/conda.sh && conda activate dgemma-jax
cd /home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/tests
export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COMPILATION_CACHE_DIR=/home/kaiwen/data/dgemma_e2b/xla_cache
python e2b_vision_test.py --check plumbing --variant it > "$LOGD/vision_plumbing.log" 2>&1
echo "vision plumbing exit=$?"
python e2b_vision_test.py --check padding --variant it > "$LOGD/vision_padding.log" 2>&1
echo "vision padding exit=$?"
FINAL_IT=$(steps_of it)
if [ -n "$FINAL_IT" ]; then
  python e2b_vision_test.py --check gap --variant it \
    --params_dir /home/kaiwen/data/dgemma_e2b/xp_chartqa_it/checkpoints/ckpt_$FINAL_IT \
    > "$LOGD/vision_gap_it.log" 2>&1
  echo "vision gap(it,trained) exit=$?"
fi
echo "OVERNIGHT CHAIN DONE"
