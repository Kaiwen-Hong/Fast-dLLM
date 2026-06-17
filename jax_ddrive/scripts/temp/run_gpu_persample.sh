set -uo pipefail
cd /home/kaiwen/Desktop/research/Fast-dLLM
unset LD_LIBRARY_PATH || true
source jax_ddrive/scripts/jax_gpu_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.9
export PYTHONPATH=maxtext-dlm-fork/src
W=jax_ddrive/scripts/temp; PS=$W/persample
mkdir -p $PS $W/tokens_gpu
FILT='libtinfo|deprecated|FutureWarning|warnings.warn|rope_|layer_type|Unrecognized|PyTorch was not|standardize_rope|got `key='
for i in $(seq -w 0 19); do
  S=val_s$i
  for DT in fp32 bf16; do
    echo "=== $S $DT ==="
    timeout 240 "$JAXPY" $W/parity_eval.py --snapshot "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f" --npz_dir /home/kaiwen/data/fast-ddrive/eval_inputs \
      --samples $S --dtype $DT --out $PS/${S}_${DT}.json --tokens_dir $W/tokens_gpu 2>&1 \
      | grep -vE "$FILT" || echo "  [$S $DT] nonzero/timeout"
  done
done
"$JAXPY" - <<'PY'
import json, glob
W="jax_ddrive/scripts/temp/persample"
for dt in ("fp32","bf16"):
    out=[]
    for f in sorted(glob.glob(f"{W}/val_s*_{dt}.json")): out+=json.load(open(f))
    json.dump(out, open(f"jax_ddrive/scripts/temp/gpu_{dt}.json","w"), indent=2)
    print(f"merged gpu_{dt}.json: {len(out)} ok={sum('error' not in r for r in out)}")
PY
echo "GPU_PERSAMPLE_DONE"
