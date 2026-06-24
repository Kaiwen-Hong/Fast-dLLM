#!/usr/bin/env bash
# OVERNIGHT PARITY — single-command orchestrator for the MULTIMODAL SASD training-step
# three-way parity (PyTorch oracle <-> ddrive_jax NNX <-> maxtext-dlm-fork), plus the
# dataset-processing round-trip. All scripts are STANDALONE (no existing code is edited).
#
# Stages (each prints its own PASS/FAIL sentinel; failures do NOT mask one another):
#   prep        prep_train_jax.py  (REAL dataset processing) -> per-sample npz       [ddrive,CPU]
#   dataset     prep_to_parquet -> parquet_file_to_tfexample_ar -> verify_dataset.py [jax,CPU]
#   capture     capture_oracle_sasd_mm.py  (frozen inputs + full logits + loss)      [ddrive,GPU]
#   nnx         parity_nnx.py  (Layer2 strict + Layer1 + ctrl + determinism)         [jax,GPU]
#   maxtext     parity_maxtext.py  (mrope/mask/vocab-tiled-CE math)                  [jax,CPU]
#   maxtext_ff  parity_maxtext_fullfwd.py  (REAL-3B full decoder forward)            [jax,CPU]
#
# Run:  bash jax_ddrive/scripts/mm_step_parity/run_overnight_parity.sh
set -uo pipefail
unset LD_LIBRARY_PATH || true

ROOT=/home/kaiwen/Desktop/research/Fast-dLLM
PT=/home/kaiwen/miniconda3/envs/ddrive/bin/python                 # PyTorch oracle env
JX=/home/kaiwen/jax-dlm-baseline/.venv/bin/python                 # JAX env
MMP=$ROOT/jax_ddrive/scripts/mm_step_parity
OUT=/home/kaiwen/data/parity_overnight
PREP=$OUT/bundles/prep PQ=$OUT/bundles/parquet AR=$OUT/bundles/ar BND=$OUT/bundles LOG=$OUT/logs
mkdir -p "$PREP" "$PQ" "$AR" "$BND" "$LOG"
BOTH=$ROOT/jax_ddrive:$ROOT/maxtext-dlm-fork/src
GPU_ENV="XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.9"
SAMPLE_JSON=$ROOT/fast_ddrive/data/example/sample.json
IMG_ROOT=$ROOT/fast_ddrive
RES=${RES:-200704}                 # toy resolution (200704 px -> 720 image tokens, released-faithful)
NSAMP=${NSAMP:-2}
cd "$ROOT"
declare -A R
clean(){ grep -viE "external/|cuda_|warn|Unrecognized|rope_param|layer_type|PyTorch was not|tokenizers|libtinfo|allocat|fragment|TF_GPU|hlo_remat|XlaRuntime|^I[0-9]|^INFO|Platform|oneDNN|TF_CPP|convert-|🚨"; }
gate(){ local name="$1" marker="$2" log="$3"; if grep -qE "$marker" "$log"; then R[$name]=PASS; else R[$name]=FAIL; fi; echo "  -> $name = ${R[$name]}"; }

echo "================ STAGE: prep (real dataset processing) ================"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$ROOT/jax_ddrive $PT jax_ddrive/eval/prep_train_jax.py \
  --train_json "$SAMPLE_JSON" --image_root "$IMG_ROOT" --out_dir "$PREP" \
  --max_samples "$NSAMP" --min_pixels "$RES" --max_pixels "$RES" 2>&1 | clean | tee "$LOG/prep.log" | tail -3
gate prep "PREP_TRAIN_JAX_DONE" "$LOG/prep.log"

echo "================ STAGE: dataset round-trip (parquet + arrayrecord) ================"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$ROOT/jax_ddrive $JX jax_ddrive/ddrive_jax/convert/prep_to_parquet.py \
  --npz_dir "$PREP" --out_dir "$PQ" --shard_size 64 --split train 2>&1 | clean | tee "$LOG/parquet.log" | tail -2
PQFILE=$(ls "$PQ"/train-*.parquet | head -1)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$ROOT/jax_ddrive $JX jax_ddrive/scripts/parquet_file_to_tfexample_ar.py \
  "$PQFILE" "$AR/train-00000.arrayrecord" 2>&1 | clean | tee "$LOG/ar.log" | tail -2
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$ROOT/jax_ddrive $JX $MMP/verify_dataset.py \
  --prep_dir "$PREP" --parquet_dir "$PQ" --ar_path "$AR/train-00000.arrayrecord" 2>&1 | clean | tee "$LOG/verify_dataset.log" | tail -5
gate dataset_roundtrip "SASD_MM_DATASET_ROUNDTRIP_PASS" "$LOG/verify_dataset.log"

# per-sample parity (tags = npz stems)
for npz in "$PREP"/[0-9]*.npz; do
  tag=$(basename "$npz" .npz)
  echo "================ SAMPLE $tag : capture (PyTorch oracle, GPU) ================"
  PYTHONPATH=$ROOT/jax_ddrive $PT $MMP/capture_oracle_sasd_mm.py \
    --prep_npz "$npz" --out_dir "$BND" 2>&1 | clean | tee "$LOG/capture_$tag.log" | grep -E "^\[A\]|ORACLE_SASD_MM_DONE"
  gate "capture_$tag" "ORACLE_SASD_MM_DONE" "$LOG/capture_$tag.log"

  echo "---------------- SAMPLE $tag : NNX parity (GPU) ----------------"
  env $GPU_ENV PYTHONPATH=$ROOT/jax_ddrive $JX $MMP/parity_nnx.py \
    --bundle "$BND/$tag.bundle.npz" 2>&1 | clean | tee "$LOG/nnx_$tag.log" | grep -E "determinism|=== |relmax|loss primary|->|SASD_MM_NNX"
  gate "nnx_$tag" "SASD_MM_NNX_PARITY_PASS" "$LOG/nnx_$tag.log"

  echo "---------------- SAMPLE $tag : MaxText math parity (CPU) ----------------"
  JAX_PLATFORMS=cpu PYTHONPATH=$ROOT/jax_ddrive $JX $MMP/parity_maxtext.py \
    --bundle "$BND/$tag.bundle.npz" 2>&1 | clean | tee "$LOG/mt_$tag.log" | grep -E "mrope|mask|sasd_loss|SASD_MM_MAXTEXT_MATH"
  gate "maxtext_math_$tag" "SASD_MM_MAXTEXT_MATH_PARITY_PASS" "$LOG/mt_$tag.log"

  echo "---------------- SAMPLE $tag : MaxText REAL-3B full forward (CPU) ----------------"
  JAX_PLATFORMS=cpu PYTHONPATH=$BOTH $JX $MMP/parity_maxtext_fullfwd.py \
    --bundle "$BND/$tag.bundle.npz" 2>&1 | clean | tee "$LOG/mtff_$tag.log" | grep -E "filled|logits:|loss primary|SASD_MM_MAXTEXT_FULLFWD"
  gate "maxtext_fullfwd_$tag" "SASD_MM_MAXTEXT_FULLFWD_PARITY_PASS" "$LOG/mtff_$tag.log"
done

echo
echo "================ OVERNIGHT MM-SASD-STEP PARITY SUMMARY ================"
ok=0; n=0
for k in "${!R[@]}"; do :; done
ordered=(prep dataset_roundtrip)
for npz in "$PREP"/[0-9]*.npz; do tag=$(basename "$npz" .npz)
  ordered+=("capture_$tag" "nnx_$tag" "maxtext_math_$tag" "maxtext_fullfwd_$tag"); done
for k in "${ordered[@]}"; do v=${R[$k]:-MISSING}; printf "  %-26s %s\n" "$k" "$v"; n=$((n+1)); [ "$v" = PASS ] && ok=$((ok+1)); done
echo "----------------------------------------------------------------------"
echo "  $ok/$n gates PASS"
[ "$ok" = "$n" ] && echo "SASD_MM_STEP_PARITY_PASS" || echo "SASD_MM_STEP_PARITY_SOME_FAILED"
