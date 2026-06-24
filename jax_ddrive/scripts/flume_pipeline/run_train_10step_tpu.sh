#!/usr/bin/env bash
# TRAINING — train the new dataloader's batches for 10 steps on a real v6e-1 TPU.
#   input : 1-example AR -> 10 SASD batches (materialized + gated 13/13 bit-exact, ddrive env)
#   output: GCS .../training/<sample>/train/{results.json,logits_sig.npz}  (loss + logits)
# Two stages: (a) materialize batches locally (ddrive: torch+grain), (b) run the real-3B SASD
# step on TPU via tpu_run.sh (MODE=train). Pass an AR path as $1 (default example0).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPY="${DDRIVE_PY:-/home/kaiwen/miniconda3/envs/ddrive/bin/python}"
AR="${1:-/home/kaiwen/data/flume_pipeline/example0/example0.array_record}"
JOB="${2:-/home/kaiwen/data/flume_pipeline/example0}"
GCS_OUT="${GCS_OUT:-gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd/flume/deliverable2/training/example0}"
( unset LD_LIBRARY_PATH; export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
  [ -d "$JOB/batches" ] || "$DPY" "$HERE/materialize_batches.py" --ar_path "$AR" --out_dir "$JOB/batches" --steps 10 --batch 1 --seed 0 )
JOB_DIR="$JOB" GCS_OUT="$GCS_OUT" MODE=train STEPS=10 NAME="${NAME:-flume-tr}" bash "$HERE/tpu_run.sh"
