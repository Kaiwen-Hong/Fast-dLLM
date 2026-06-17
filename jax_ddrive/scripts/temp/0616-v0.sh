#!/usr/bin/env bash
# temp/0616-v0.sh — one-off: write DATA_MANIFEST.json for each GCS data artifact (provenance).
# This is STEP 0 §3 of docs/6for_internal/00_owner_publish.md, packaged as a runnable script.
# Reads GCS crc32c metadata ONLY (no download) → fast even for the 369 GB full dataset.
#
#   bash jax_ddrive/scripts/temp/0616-v0.sh
#   BKT=gs://other-bucket bash jax_ddrive/scripts/temp/0616-v0.sh    # override bucket
#   PY=/path/to/python   bash jax_ddrive/scripts/temp/0616-v0.sh     # override python (stdlib-only tool)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$SCRIPT_DIR/../data_manifest.py"             # jax_ddrive/scripts/data_manifest.py
PY="${PY:-python3}"
SRC="${BKT:-gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd}"

ARTIFACTS=(
  maxtext_sasd_params_base                          # base init weights (Orbax)
  wod_e2e_sasd_distilled_0613-400_baseViT_v2_ar     # distilled-400 dataset (overfit milestone)
  wod_e2e_sasd_full_v2_ar                           # full 415,663-frame dataset (production)
  base_qwen25vl_3b_snapshot                         # base HF snapshot (export ref + tokenizer)
  eval_inputs                                       # 40 inference-input npz
)

# ---- preflight ----
[ -f "$TOOL" ] || { echo "ERROR: $TOOL not found" >&2; exit 1; }
command -v gsutil >/dev/null || { echo "ERROR: gsutil not on PATH (need owner gcloud auth)" >&2; exit 1; }
gsutil ls "$SRC/" >/dev/null 2>&1 || { echo "ERROR: cannot list $SRC — run 'gcloud auth login' first" >&2; exit 1; }

echo "bucket: $SRC"
echo "tool  : $TOOL"
ok=0; fail=0
for A in "${ARTIFACTS[@]}"; do
  echo "=== $A ==="
  if "$PY" "$TOOL" "$SRC/$A" --upload; then
    ok=$((ok + 1))
  else
    echo "  !! FAILED for $A (skipping)" >&2
    fail=$((fail + 1))
  fi
done
echo "-------------------------------------------"
echo "DATA_MANIFEST written: $ok ok, $fail failed (of ${#ARTIFACTS[@]})"
[ "$fail" = 0 ] && echo "ALL_DATA_MANIFESTS_DONE"
