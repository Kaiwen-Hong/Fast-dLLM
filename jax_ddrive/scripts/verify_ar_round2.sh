#!/usr/bin/env bash
# Round-2 semantic verification of the dataset-v2 ArrayRecord sets
# (train: wod_e2e_sasd_full_v2_ar, val: wod_e2e_sasd_val_v2_ar).
# Re-runs the exact build chain (stage1 converter + stage2 prep) on NT train frames
# (tfrecord shard 0) + NV rated val frames, compares all 12 columns bit-exact against
# the AR rows, and re-derives image_embeds from the stored pixels (two-tier criterion).
set -euo pipefail
unset LD_LIBRARY_PATH || true

REPO=/home/kaiwen/Desktop/research/Fast-dLLM
AV=/home/kaiwen/miniconda3/envs/autovla/bin/python      # stage1 + viz: TF proto + matplotlib
PT=/home/kaiwen/miniconda3/envs/ddrive/bin/python       # stage2 + compare: transformers
JX=/home/kaiwen/jax-dlm-baseline/.venv/bin/python       # locate/extract/embeds: tf+ar+jax

D=/home/kaiwen/data/fast-ddrive
OUT=$D/verify_round2_v2
NT=${NT:-10}   # train samples (first NT frames of train tfrecord shard 0)
NV=${NV:-2}    # val samples (first NV rated frames of the val tfrecords)

TRAIN_SHARD0=$(ls $D/waymo/train/training_*.tfrecord-* | sort | head -1)
mkdir -p $OUT
echo "== train tfrecord shard 0: $TRAIN_SHARD0"

echo "== [1/8] stage1 train: tfrecord -> JSON + JPEG (first $NT frames, autovla env)"
PYTHONPATH=$REPO/jax_ddrive $AV $REPO/fast_ddrive/data/convert_wod_e2e.py \
    --tfrecords "$TRAIN_SHARD0" --out_json $OUT/src_train.json \
    --image_root $OUT/src_images --with_target --max_frames $NT

echo "== [2/8] stage1 val: rated frames -> JSON + JPEG (first $NV, autovla env)"
PYTHONPATH=$REPO/jax_ddrive $AV $REPO/fast_ddrive/data/convert_wod_e2e.py \
    --tfrecords "$D/waymo/val/val_*.tfrecord*" --out_json $OUT/src_val.json \
    --image_root $OUT/src_images --with_target --rated_only --max_frames $NV

$PT - <<PY
import json
tr = json.load(open("$OUT/src_train.json"))
va = json.load(open("$OUT/src_val.json"))
json.dump(tr + va, open("$OUT/src.json", "w"))
ids = [{"sid": s["sample_id"], "split": "train"} for s in tr] + \
      [{"sid": s["sample_id"], "split": "val"} for s in va]
json.dump(ids, open("$OUT/ids.json", "w"), indent=2)
print(f"merged: {len(tr)} train + {len(va)} val ->", [e["sid"] for e in ids])
PY

echo "== [3/8] stage2: JSON+JPEG -> reference npz (exact build prep, ddrive env)"
PYTHONPATH=$REPO/jax_ddrive $PT $REPO/jax_ddrive/eval/prep_train_jax.py \
    --train_json $OUT/src.json --image_root $OUT/src_images --out_dir $OUT/ref_npz

echo "== [4/8] locate sample_ids in the source parquet (row map for the v2 AR)"
$JX $REPO/jax_ddrive/scripts/verify_ar_round2.py locate \
    --ids_json $OUT/ids.json --out $OUT/locations.json

echo "== [5/8] extract those records from the v2 ArrayRecord datasets"
PYTHONPATH=$REPO/jax_ddrive $JX $REPO/jax_ddrive/scripts/verify_ar_round2.py extract \
    --locations $OUT/locations.json --out_dir $OUT/ar_npz

echo "== [6/8] image_embeds: recompute from stored pixels (frozen ViT fp32 highest, GPU)"
PYTHONPATH=$REPO/jax_ddrive XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $JX $REPO/jax_ddrive/scripts/verify_ar_round2.py embeds-verify \
    --locations $OUT/locations.json --ar_npz $OUT/ar_npz --out_dir $OUT/embeds_check

echo "== [7/8] compare + render review packets"
PYTHONPATH=$REPO/jax_ddrive $PT $REPO/jax_ddrive/scripts/verify_ar_round2.py compare \
    --src_json $OUT/src.json --image_root $OUT/src_images \
    --ref_npz $OUT/ref_npz --ar_npz $OUT/ar_npz --embeds_check $OUT/embeds_check \
    --locations $OUT/locations.json --out_dir $OUT/review

echo "== [8/8] visualize (BEV / layout / M-RoPE / mask / embeds-diff)"
$AV $REPO/jax_ddrive/scripts/verify_ar_round2.py viz \
    --ar_npz $OUT/ar_npz --embeds_check $OUT/embeds_check \
    --locations $OUT/locations.json --out_dir $OUT/review

echo; echo "== report: $OUT/review/report.md"
