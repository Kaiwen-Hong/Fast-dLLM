#!/usr/bin/env bash
# Round-2 semantic verification of wod_e2e_sasd_full_tfexample_ar (10 samples).
# Re-runs the exact build chain (stage1 converter + stage2 prep) on the first 10
# frames of train tfrecord shard 0, then compares against the ArrayRecord rows.
set -euo pipefail
unset LD_LIBRARY_PATH || true

REPO=/home/kaiwen/Desktop/research/Fast-dLLM
AV=/home/kaiwen/miniconda3/envs/autovla/bin/python      # stage1: TF + E2E proto
PT=/home/kaiwen/miniconda3/envs/ddrive/bin/python       # stage2 + compare: transformers
JX=/home/kaiwen/jax-dlm-baseline/.venv/bin/python       # locate/extract: tf + array_record

D=/home/kaiwen/data/fast-ddrive
OUT=$D/verify_round2
PACKED=$D/hf/wod_e2e_sasd_full_packed
AR=$D/hf/wod_e2e_sasd_full_tfexample_ar
N=${N:-10}

SHARD0=$(ls $D/waymo/train/training_*.tfrecord-* | sort | head -1)
mkdir -p $OUT
echo "== tfrecord shard 0: $SHARD0"

echo "== [1/5] stage1: tfrecord -> JSON + JPEG (first $N frames, autovla env)"
PYTHONPATH=$REPO/jax_ddrive $AV $REPO/fast_ddrive/data/convert_wod_e2e.py \
    --tfrecords "$SHARD0" --out_json $OUT/src.json --image_root $OUT/src_images \
    --with_target --max_frames $N

$PT - <<PY
import json
src = json.load(open("$OUT/src.json"))
json.dump([s["sample_id"] for s in src], open("$OUT/ids.json", "w"), indent=2)
print("sample_ids:", [s["sample_id"] for s in src])
PY

echo "== [2/5] stage2: JSON+JPEG -> reference npz (exact build prep, ddrive env)"
PYTHONPATH=$REPO/jax_ddrive $PT $REPO/jax_ddrive/eval/prep_train_jax.py \
    --train_json $OUT/src.json --image_root $OUT/src_images --out_dir $OUT/ref_npz

echo "== [3/5] locate the 10 sample_ids in the packed parquet (row map for ar)"
$JX $REPO/jax_ddrive/scripts/verify_ar_round2.py locate \
    --packed_dir $PACKED --ids_json $OUT/ids.json --out $OUT/locations.json

echo "== [4/5] extract those records from the ArrayRecord dataset"
$JX $REPO/jax_ddrive/scripts/verify_ar_round2.py extract \
    --ar_dir $AR --locations $OUT/locations.json --out_dir $OUT/ar_npz

echo "== [5/6] compare + render review packets"
PYTHONPATH=$REPO/jax_ddrive $PT $REPO/jax_ddrive/scripts/verify_ar_round2.py compare \
    --src_json $OUT/src.json --image_root $OUT/src_images \
    --ref_npz $OUT/ref_npz --ar_npz $OUT/ar_npz \
    --locations $OUT/locations.json --out_dir $OUT/review

echo "== [6/6] visualize the non-image model inputs (BEV / layout / M-RoPE / mask)"
$AV $REPO/jax_ddrive/scripts/verify_ar_round2.py viz \
    --ar_npz $OUT/ar_npz --locations $OUT/locations.json --out_dir $OUT/review

echo; echo "== report: $OUT/review/report.md"
