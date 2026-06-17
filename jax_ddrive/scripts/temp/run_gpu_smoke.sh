set -uo pipefail
cd /home/kaiwen/Desktop/research/Fast-dLLM
unset LD_LIBRARY_PATH || true
source jax_ddrive/scripts/jax_gpu_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.9
export PYTHONPATH=maxtext-dlm-fork/src
"$JAXPY" jax_ddrive/scripts/temp/parity_eval.py \
  --snapshot "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f" --npz_dir /home/kaiwen/data/fast-ddrive/eval_inputs \
  --samples val_s00 --dtype fp32 --out jax_ddrive/scripts/temp/smoke_gpu_fp32.json --tokens_dir jax_ddrive/scripts/temp/tokens_gpu
echo "SMOKE_EXIT=$?"
