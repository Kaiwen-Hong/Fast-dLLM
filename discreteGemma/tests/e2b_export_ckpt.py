"""Export manual-loop params into a kauldron checkpoint eval_main can restore.

Bridges the manual trainer (e2b_manual_train.py — raw orbax params) to the
official eval path: build the trainer state via trainer.init_state() (random
init; init_transform disabled like eval_main does), swap in the trained
params, save via trainer.checkpointer at the given step.

Usage: python e2b_export_ckpt.py --variant it --step 1600
"""

import argparse
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", "/home/kaiwen/data/dgemma_e2b/xla_cache")

REPO = "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma"
sys.path.insert(0, f"{REPO}/gemma")
DATA = "/home/kaiwen/data/dgemma_e2b"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from kauldron import konfig  # noqa: E402

def _hide_gpu_from_tf():
  try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
  except Exception:
    pass


_hide_gpu_from_tf()



def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variant", choices=["pt", "it"], required=True)
  ap.add_argument("--step", type=int, required=True)
  args = ap.parse_args()

  os.environ["DGEMMA_E2B_VARIANT"] = args.variant
  os.environ["DGEMMA_E2B_BATCH"] = "1"
  os.environ["DGEMMA_E2B_ACCUM"] = "1"

  from gemma.diffusion.hackable_diffusion_adapter.configs import sft_chartqa_e2b

  cfg = sft_chartqa_e2b.get_config()
  cfg.workdir = f"{DATA}/xp_chartqa_{args.variant}"
  cfg.init_transform = None  # same as eval_main: params come from the swap

  trainer = konfig.resolve(cfg)
  state = trainer.init_state()

  import orbax.checkpoint as ocp
  trained = ocp.PyTreeCheckpointer().restore(
      f"{DATA}/xp_manual_{args.variant}/params_{args.step}")
  # cast/put to the state's param dtypes (identical tree; belt-and-braces)
  new_params = jax.tree.map(
      lambda cur, new: jnp.asarray(new, cur.dtype), state.params, trained)
  state = state.replace(params=new_params, step=args.step)
  ok = trainer.checkpointer.save(state, step=args.step, force=True)
  # block until the async save lands before the process exits
  if hasattr(trainer.checkpointer, "wait_until_finished"):
    trainer.checkpointer.wait_until_finished()
  print(f"EXPORT[{args.variant} step={args.step}]: saved={ok} -> {cfg.workdir}")


if __name__ == "__main__":
  main()
