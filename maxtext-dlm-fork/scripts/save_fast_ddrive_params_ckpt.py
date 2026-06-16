"""One-time: save the Fast-dDrive Qwen2.5-3B weights as a MaxText Orbax PARAM checkpoint.

The deployable SASD train entry gets its initial weights via MaxText's STANDARD
``load_parameters_path`` path: ``setup_initial_state`` -> ``load_state_if_possible`` ->
``load_params_from_path`` restores the params into the (FSDP-sharded on TPU) abstract train
state, optimizer state is freshly initialized (correct for a fine-tune start). This script
produces that checkpoint ONCE.

It builds the qwen2.5-3b Linen model with the EXACT config the train run uses (so the param
tree -- unstacked per-layer leaves under ``scan_layers=false`` -- matches the train-time
abstract tree), calls the verified 100%-top-1 loader
``build_maxtext_params_from_fast_ddrive``, casts leaves to the train weight_dtype (default
bf16, so the restore is a direct sharded read with no host materialization of the fp32
tree), and writes them with ``checkpointing.save_params_to_path``.

Usage (GPU):
  source /home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/jax_gpu_env.sh
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive:\
/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/src
  "$JAXPY" /home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/scripts/save_fast_ddrive_params_ckpt.py \
      src/maxtext/configs/sasd_waymo.yml model_name=qwen2.5-3b \
      out_ckpt_dir=/home/kaiwen/data/fast-ddrive/maxtext_sasd_params/fast_ddrive_qwen25_3b_params \
      snapshot_dir=<HF snapshot of Efficient-Large-Model/Fast-dDrive>

The same script runs unchanged on a TPU pod with ``out_ckpt_dir=gs://.../params``.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

SNAP_DEFAULT = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
                "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")


def main():
  from maxtext.configs import pyconfig
  from maxtext.common import checkpointing
  from maxtext.diffusion.load_fast_ddrive_maxtext import build_maxtext_params_from_fast_ddrive

  # Pull our two extra args out of argv (pyconfig would reject unknown keys otherwise).
  argv = list(sys.argv)
  out_ckpt_dir = None
  snapshot_dir = SNAP_DEFAULT
  kept = []
  for a in argv:
    if a.startswith("out_ckpt_dir="):
      out_ckpt_dir = a.split("=", 1)[1]
    elif a.startswith("snapshot_dir="):
      snapshot_dir = a.split("=", 1)[1]
    else:
      kept.append(a)
  assert out_ckpt_dir, "pass out_ckpt_dir=<dir or gs://...> on the command line."

  # Build config EXACTLY as the train run will (same config file + overrides) so the abstract
  # param tree matches. Force the SASD-required model overrides (scan_layers=false etc. are in
  # the yml; we only need a valid, non-checkpointing init here).
  config = pyconfig.initialize(kept, enable_checkpointing=False)
  assert config.scan_layers is False, "SASD requires scan_layers=false; param tree must match the train run."

  weight_dtype = jnp.dtype(config.weight_dtype)
  print(f"[save-ckpt] building params: model={config.model_name} scan_layers={config.scan_layers} "
        f"weight_dtype={weight_dtype}  out={out_ckpt_dir}", flush=True)

  params = build_maxtext_params_from_fast_ddrive(snapshot_dir, config, verbose=True)
  # Cast every leaf to the train weight_dtype and drop the fp32 buffers. Saving in the
  # train-time dtype makes load_params_from_path a direct sharded read (no cast / no fp32
  # materialization on restore).
  params = jax.tree.map(lambda x: x.astype(weight_dtype), params)
  n_el = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
  print(f"[save-ckpt] cast to {weight_dtype}: {n_el/1e9:.3f}B params", flush=True)

  # save_params_to_path wraps as {"params": params}; load_params_from_path restores
  # item={"params": abstract_unboxed_state.params}. params here is freeze({"params": tree}),
  # so the on-disk structure is {"params": {"params": tree}} matching the abstract
  # {"params": {"params": tree}} the restore expects (state.params == {"params": tree}).
  checkpointing.save_params_to_path(out_ckpt_dir, params)
  print(f"[save-ckpt] DONE -> {out_ckpt_dir}", flush=True)
  print("SASD_PARAM_CKPT_SAVED", flush=True)


if __name__ == "__main__":
  main()
