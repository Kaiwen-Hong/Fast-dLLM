set -uo pipefail
cd /home/kaiwen/Desktop/research/Fast-dLLM
unset LD_LIBRARY_PATH || true
source jax_ddrive/scripts/jax_gpu_env.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.9
export PYTHONPATH=maxtext-dlm-fork/src
W=jax_ddrive/scripts/temp
for DT in fp32 bf16; do
  echo "###### GPU $DT ######"
  "$JAXPY" $W/parity_eval.py --snapshot "/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f" \
    --npz_dir /home/kaiwen/data/fast-ddrive/eval_inputs --samples "val_s00,val_s01,val_s02,val_s03,val_s04,val_s05,val_s06,val_s07,val_s08,val_s09,val_s10,val_s11,val_s12,val_s13,val_s14,val_s15,val_s16,val_s17,val_s18,val_s19" \
    --dtype $DT --out $W/gpu_$DT.json --tokens_dir $W/tokens_gpu
done
echo "GPU_FULL_DONE"
