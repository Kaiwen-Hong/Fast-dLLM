set -uo pipefail
cd /home/kaiwen/Desktop/research/Fast-dLLM
unset LD_LIBRARY_PATH || true
export JAX_PLATFORMS=cpu
export PYTHONPATH=maxtext-dlm-fork/src
JXPY=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
for S in val_s13 val_s17 val_s06; do
  echo "### CPU fp32 $S ###"
  "$JXPY" jax_ddrive/scripts/temp/parity_eval.py --snapshot "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f" --npz_dir /home/kaiwen/data/fast-ddrive/eval_inputs \
    --samples $S --dtype fp32 --out jax_ddrive/scripts/temp/cpu_${S}_fp32.json --tokens_dir jax_ddrive/scripts/temp/tokens_cpu
done
echo "CPU_HARD_DONE"
