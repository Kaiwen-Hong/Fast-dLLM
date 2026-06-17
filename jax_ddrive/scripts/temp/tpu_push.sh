#!/usr/bin/env bash
# Iteration helper: re-ship the local parity code (fork src/maxtext incl. fixed eval_sasd + harness)
# to the already-up TPU, so tpu_run.sh re-runs the FIXED inference code. Fast (code only, no model).
set -uo pipefail
PROJECT=project-8a53f5ab-2ea2-4892-a78
ZONE="${ZONE:-us-east5-a}"
NAME="${NAME:-sasd-parity}"
SRC=gs://${PROJECT}-ddrive-sasd
FORK=/home/kaiwen/Desktop/research/Fast-dLLM/maxtext-dlm-fork
W=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/temp
tar czf /tmp/parity_code.tgz -C "$FORK/src" maxtext -C "$W" parity_eval.py
gsutil -q cp /tmp/parity_code.tgz "$SRC/parity/parity_code.tgz"
echo "$(date +%T) pushed code tgz to GCS; extracting on $NAME ..."
gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --worker=all --command '
  set -e
  gsutil -q cp '"$SRC"'/parity/parity_code.tgz ~ && rm -rf ~/parity && mkdir -p ~/parity && tar xzf ~/parity_code.tgz -C ~/parity && echo CODE_PUSHED
' 2>&1 | grep -E "CODE_PUSHED|ERROR" || true
echo "TPU_PUSH_DONE"
