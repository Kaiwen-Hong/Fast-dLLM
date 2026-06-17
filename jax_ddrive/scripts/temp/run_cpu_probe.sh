set -uo pipefail
cd /home/kaiwen/Desktop/research/Fast-dLLM
unset LD_LIBRARY_PATH || true
export JAX_PLATFORMS=cpu
export PYTHONPATH=maxtext-dlm-fork/src
JXPY=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
T0=$(date +%s)
"$JXPY" jax_ddrive/scripts/temp/parity_eval.py --snapshot "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f" --npz_dir /home/kaiwen/data/fast-ddrive/eval_inputs \
  --samples val_s00 --dtype fp32 --out jax_ddrive/scripts/temp/cpu_val_s00_fp32.json --tokens_dir jax_ddrive/scripts/temp/tokens_cpu
echo "CPU_PROBE_SEC=$(( $(date +%s) - T0 )) CPU_PROBE_EXIT=$?"
