set -uo pipefail
cd /home/kaiwen/Desktop/research/Fast-dLLM
unset LD_LIBRARY_PATH || true
export JAX_PLATFORMS=cpu
export PYTHONPATH=maxtext-dlm-fork/src
JXPY=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
for S in val_s01 val_s02 val_s03 val_s04 val_s05 val_s07 val_s08 val_s09 val_s10 val_s11 val_s12 val_s14 val_s15 val_s16 val_s18 val_s19; do
  echo "### CPU fp32 $S ###"
  "$JXPY" jax_ddrive/scripts/temp/parity_eval.py --snapshot "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f" --npz_dir /home/kaiwen/data/fast-ddrive/eval_inputs \
    --samples $S --dtype fp32 --out jax_ddrive/scripts/temp/cpu_${S}_fp32.json --tokens_dir jax_ddrive/scripts/temp/tokens_cpu 2>&1 | grep -E 'fp32:|FAILED' || echo "  $S issue"
done
echo "CPU_REST_DONE"
