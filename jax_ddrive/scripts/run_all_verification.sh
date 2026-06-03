#!/usr/bin/env bash
# Single-command verification: capture every PyTorch oracle (ddrive env) then run every
# JAX parity gate (jax venv), and print a PASS/FAIL summary. ~5-10 min (loads the 16GB
# checkpoint several times). Run from the repo root:  bash jax_ddrive/scripts/run_all_verification.sh
set -uo pipefail
unset LD_LIBRARY_PATH || true
ROOT=/home/kaiwen/Desktop/research/Fast-dLLM
PT=/home/kaiwen/miniconda3/envs/ddrive/bin/python                 # PyTorch oracle env
JX=/home/kaiwen/jax-dlm-baseline/.venv/bin/python                 # JAX env
export PYTHONPATH=$ROOT/jax_ddrive
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.9
cd "$ROOT"
declare -A R
clean(){ grep -viE "external/|cuda_|warn|Unrecognized|rope_param|layer_type|PyTorch was not|tokenizers|libtinfo|allocat|fragment|summary|TF_GPU|\*\*\*|hlo_remat"; }
gate(){ # name  marker  cmd...
  local name="$1" marker="$2"; shift 2
  echo "=== $name ===" >&2
  if "$@" 2>&1 | clean | grep -qE "$marker"; then R[$name]=PASS; else R[$name]=FAIL; fi
}

# CPU unit tests
JAX_PLATFORMS=cpu $JX jax_ddrive/tests/test_mask_loss.py 2>&1 | clean | grep -q "ALL CPU TESTS PASS" && R[cpu_mask_loss]=PASS || R[cpu_mask_loss]=FAIL
JAX_PLATFORMS=cpu $JX jax_ddrive/tests/test_lora.py 2>&1 | clean | grep -q "ALL LORA TESTS PASS" && R[cpu_lora]=PASS || R[cpu_lora]=FAIL
JAX_PLATFORMS=cpu $JX jax_ddrive/tests/test_noising.py 2>&1 | clean | grep -q "ALL NOISING TESTS PASS" && R[cpu_noise]=PASS || R[cpu_noise]=FAIL
JAX_PLATFORMS=cpu $JX jax_ddrive/tests/test_sharding.py 2>&1 | clean | grep -q "ALL SHARDING TESTS PASS" && R[cpu_sharding]=PASS || R[cpu_sharding]=FAIL
JAX_PLATFORMS=cpu $JX jax_ddrive/tests/test_eval_ports.py 2>&1 | clean | grep -q "ALL EVAL PORT TESTS PASS" && R[cpu_eval_ports]=PASS || R[cpu_eval_ports]=FAIL

# Phase 1 text
$PT jax_ddrive/scripts/capture_oracle_text.py >/dev/null 2>&1
gate phase1_text "PHASE1_PARITY_PASS" $JX jax_ddrive/scripts/parity_text.py
# Phase 2 SASD loss
$PT jax_ddrive/scripts/capture_oracle_sasd.py >/dev/null 2>&1
gate phase2_sasd "PHASE2_PARITY_PASS" $JX jax_ddrive/scripts/parity_sasd.py
# Phase 4 ViT
$PT jax_ddrive/scripts/capture_oracle_vit.py >/dev/null 2>&1
$PT jax_ddrive/scripts/debug_vit.py >/dev/null 2>&1
gate phase4_vit "PHASE4_VIT_PASS" $JX jax_ddrive/scripts/parity_vit.py
# Phase 4b multimodal forward
$PT jax_ddrive/scripts/capture_oracle_mm.py >/dev/null 2>&1
gate phase4b_mm_fwd "PHASE4b_MM_PASS" $JX jax_ddrive/scripts/parity_mm.py
# Phase 3 default training path (LoRA) smoke gate: trains the LoRA adapters a few steps and
# asserts the fixed-eval loss decreases (no NaN). Needs prep_overfit_data.npz from prep step.
$PT jax_ddrive/scripts/prep_overfit_data.py >/dev/null 2>&1
gate phase3_lora_train "PHASE3_PASS" $JX jax_ddrive/ddrive_jax/train_overfit.py --source trained --fixed_batch --steps 30 --lr 1e-4

echo
echo "================ VERIFICATION SUMMARY ================"
ok=0; n=0
for k in cpu_mask_loss cpu_lora cpu_noise cpu_sharding cpu_eval_ports phase1_text phase2_sasd phase4_vit phase4b_mm_fwd phase3_lora_train; do
  v=${R[$k]:-MISSING}; printf "  %-16s %s\n" "$k" "$v"; n=$((n+1)); [ "$v" = PASS ] && ok=$((ok+1))
done
echo "-----------------------------------------------------"
echo "  $ok/$n gates PASS"
echo "  (training loss-decrease is verified separately: train_overfit.py / train_overfit_mm.py)"
[ "$ok" = "$n" ] && echo "ALL_VERIFICATION_PASS" || echo "SOME_VERIFICATION_FAILED"
