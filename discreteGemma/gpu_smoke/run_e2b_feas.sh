#!/bin/bash
# E2B feasibility suite driver (GPU part). T0 is run separately (CPU).
# Runs T1 (build+forward), then T2 train-step at batch 1,2,4,8 — with a remat
# retry whenever the plain run OOMs. Each run is its own process so an OOM
# can't poison the next. Logs + JSON under /home/kaiwen/data/dgemma_e2b/feas/.
set -u
source /home/kaiwen/miniconda3/etc/profile.d/conda.sh && conda activate dgemma-jax
cd /home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gpu_smoke
LOGD=/home/kaiwen/data/dgemma_e2b/feas/logs
mkdir -p "$LOGD"

echo "=== T1 build ==="
python e2b_feas_t1_build.py > "$LOGD/t1.log" 2>&1
echo "T1 exit=$?"

for BS in 1 2 4 8; do
  echo "=== T2 batch=$BS ==="
  python e2b_feas_t2_train_step.py --batch "$BS" > "$LOGD/t2_b${BS}.log" 2>&1
  rc=$?
  echo "T2 b$BS exit=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "=== T2 batch=$BS --remat (retry) ==="
    python e2b_feas_t2_train_step.py --batch "$BS" --remat > "$LOGD/t2_b${BS}_remat.log" 2>&1
    rc2=$?
    echo "T2 b$BS remat exit=$rc2"
    # if even remat b1 fails, larger batches are pointless
    if [ "$BS" -eq 1 ] && [ "$rc2" -ne 0 ]; then
      echo "FATAL: batch-1 fails even with remat — stopping suite"
      break
    fi
  fi
done
echo "E2B FEASIBILITY SUITE DONE"
