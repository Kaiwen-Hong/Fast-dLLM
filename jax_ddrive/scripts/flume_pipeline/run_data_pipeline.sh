#!/usr/bin/env bash
# DATA PIPELINE — raw WOD-E2E tfrecord -> validated msgpack ArrayRecord.
#   input  ($1): a WOD-E2E tfrecord of E2EDFrame protos   (default: the 2 raw examples)
#   output ($2): <out>.array_record
# Runs in the autovla env (tensorflow + waymo proto). The AR is validated 13/13 bit-exact
# vs prep_train_jax by materialize_batches.py --oracle_prep_dir (see run_train_10step_tpu.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AV="${AUTOVLA_PY:-/home/kaiwen/miniconda3/envs/autovla/bin/python}"
IN="${1:-/home/kaiwen/data/flume_pipeline/raw_track/two_examples.tfrecord}"
OUT="${2:-/home/kaiwen/data/flume_pipeline/raw_track/raw.array_record}"
echo "[data-pipeline] $IN  ->  $OUT"
"$AV" "$HERE/convert_raw_to_ar.py" --tfrecords "$IN" --out "$OUT"
