#!/usr/bin/env bash
# One-shot parity gate for the Flume/Grain data pipeline (doc 09 §8).
# Runs the ETL -> ArrayRecord -> read-back -> TokenizeSASD chain and asserts all 13 fields are
# bit-exact vs a freshly-run prep_train_jax.py oracle. Emits FLUME_PIPELINE_PARITY_PASS on success.
#
# Usage: bash run_parity.sh            # uses the 2-sample toy sample.json
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${FASTDDRIVE_PY:-/home/kaiwen/miniconda3/envs/ddrive/bin/python}"
TRAIN_JSON="${1:-/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive/data/example/sample.json}"
IMAGE_ROOT="${2:-/home/kaiwen/Desktop/research/Fast-dLLM/fast_ddrive}"
WORKDIR="${WORKDIR:-/home/kaiwen/data/flume_pipeline/parity}"

unset LD_LIBRARY_PATH                      # ddrive (torch-cu128) env requirement
mkdir -p "$WORKDIR"

echo "[run_parity] python   = $PY"
echo "[run_parity] train    = $TRAIN_JSON"
echo "[run_parity] workdir  = $WORKDIR"

"$PY" "$HERE/parity_test.py" \
    --train_json "$TRAIN_JSON" \
    --image_root "$IMAGE_ROOT" \
    --workdir "$WORKDIR"
