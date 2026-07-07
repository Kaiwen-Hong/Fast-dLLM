#!/bin/bash
# Phase-B load acceptance driver: ref+diff per variant, sequential (GPU-exclusive).
set -u
source /home/kaiwen/miniconda3/etc/profile.d/conda.sh && conda activate dgemma-jax
cd /home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/tests
export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma
LOGD=/home/kaiwen/data/dgemma_e2b/loadtest
mkdir -p "$LOGD"
overall=0
for V in pt it; do
  # ref mode retired: the diff worker now bit-compares against the raw ckpt
  # (ocp ground truth) itself and does class-equivalence for the AR forward.
  echo "=== load-test diff $V ==="
  python e2b_load_test.py --mode diff --variant "$V" > "$LOGD/diff_$V.log" 2>&1 || { echo "diff $V FAILED"; overall=1; }
  grep -hE ': (PASS|FAIL)$' "$LOGD/diff_$V.log" | tail -1
done
echo "LOAD TEST SUITE exit=$overall"
exit $overall
