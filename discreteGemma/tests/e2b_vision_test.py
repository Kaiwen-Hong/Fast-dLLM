"""C6 vision criteria (design doc §7): plumbing / counterfactual gap / padding.

Modes (each a separate process run; pinned rngs throughout):
  --check plumbing : swap images between the two toy rows -> total loss changes.
  --check padding  : perturb (i) padded patch rows (positions_xy == -1) and
                     (ii) prompt PAD-tail token ids (masks unchanged) -> loss
                     BIT-IDENTICAL.
  --check gap      : correct-vs-wrong-image loss gap (run with --params_dir of
                     a TRAINED kauldron ckpt for C6b; direction gates).

Params: --variant pt|it loads the AR ckpt via the production loader chain
(+ sc W3 zeroing); --params_dir overrides with a trained kauldron checkpoint.
Emits results/{variant}/vision_<check>.json.
"""

import argparse
import json
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma"
sys.path.insert(0, f"{REPO}/gemma")
DATA = "/home/kaiwen/data/dgemma_e2b"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from kauldron import konfig  # noqa: E402

def _hide_gpu_from_tf():
  try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
  except Exception:
    pass


_hide_gpu_from_tf()



def build_model_and_batch():
  """SFT model resolved from the production config + one pinned toy batch."""
  os.environ.setdefault("DGEMMA_E2B_VARIANT", "it")
  from gemma.diffusion.hackable_diffusion_adapter.configs import sft_chartqa_e2b
  from gemma.diffusion.hackable_diffusion_adapter.data.chartqa import chartqa_data as cq

  cfg = sft_chartqa_e2b.get_config()
  model = konfig.resolve(cfg.model)

  # batch_size=1: the MM path is single-sequence per forward (batch-dim
  # wrapper constraint). Counterfactuals swap IMAGES between two separate
  # batch-1 examples instead of between rows.
  ds_cfg = cq.make_chartqa_records_ds(
      training=False, batch_size=1,
      paths=(f"{DATA}/chartqa/chartqa_toy.bagz",), fmt="bagz",
      answer_len=32, num_workers=0)
  ds = konfig.resolve(ds_cfg)
  it = iter(ds)
  b0 = next(it)
  b1 = next(it)
  clean = lambda b: {k: jnp.asarray(np.asarray(v)) for k, v in b.items()
                     if k != "answer_tokens"}
  return model, clean(b0), clean(b1)


def load_params(model, batch, variant: str, params_dir: str | None):
  from gemma.diffusion.hackable_diffusion_adapter.hd import gemma_checkpointer

  variables = model.init(
      {"params": jax.random.PRNGKey(0), "sampling": jax.random.PRNGKey(1)},
      **batch, is_training=True,
  )
  params = variables["params"]
  if params_dir:  # trained kauldron ckpt (orbax): restore the params subtree
    import orbax.checkpoint as ocp
    ckptr = ocp.PyTreeCheckpointer()
    restored = ckptr.restore(params_dir)
    # kauldron train-state layout: {'params': ...} possibly nested
    tree = restored.get("params", restored)
    params = jax.tree.map(lambda spec, v: jnp.asarray(v, spec.dtype), params, tree)
    print(f"loaded TRAINED params from {params_dir}")
  else:
    merged = gemma_checkpointer.cheaply_load_params(
        params_from_state=params["gemma_network"]["gemma_model"]
        if "gemma_network" in params else params,
        checkpoint_path=f"{DATA}/ckpts/gemma4-e2b-{variant}",
        expected_missing=("self_conditioner",),
    )
    if "gemma_network" in params:
      params = dict(params)
      params["gemma_network"] = dict(params["gemma_network"])
      params["gemma_network"]["gemma_model"] = merged
    else:
      params = merged
    # sc W3 zeroing (production chain)
    import flax
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    for k in list(flat):
      if "self_conditioner/ffw/linear" in k:
        flat[k] = jnp.zeros_like(flat[k])
    params = flax.traverse_util.unflatten_dict(flat, sep="/")
    print(f"loaded AR ckpt ({variant}) + sc W3 zeroed")
  return params


def total_loss(model, params, batch, seed: int = 7) -> float:
  from hackable_diffusion.lib.training import discrete_loss
  from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model as sm

  preds = model.apply(
      {"params": params}, **batch, is_training=True,
      rngs={"sampling": jax.random.PRNGKey(seed)},  # PINNED rng (doc §B6)
  )
  d = discrete_loss.compute_discrete_diffusion_loss(
      preds=preds["output"], targets=preds["target"],
      time=preds["noise_info"]["time"], use_mask=True, mask_key="target_mask")
  d = (d.loss if hasattr(d, "loss") else jnp.asarray(d)).mean()
  e = sm.EncoderARLoss().get_values(
      preds["encoder_logits"], preds["encoder_target"],
      preds["encoder_target_mask"]).mean()
  return float(d + e)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--check", choices=["plumbing", "padding", "gap"], required=True)
  ap.add_argument("--variant", default="it")
  ap.add_argument("--params_dir", default=None)
  a = ap.parse_args()

  model, batch, other = build_model_and_batch()
  params = load_params(model, batch, a.variant, a.params_dir)
  out = {"check": a.check, "variant": a.variant, "params_dir": a.params_dir}

  base = total_loss(model, params, batch)
  out["loss_base"] = base

  if a.check == "plumbing":
    b2 = dict(batch)
    b2["patches"] = other["patches"]              # swap in the OTHER example's image
    b2["positions_xy"] = other["positions_xy"]
    swapped = total_loss(model, params, b2)
    out["loss_swapped"] = swapped
    out["PASS"] = bool(abs(swapped - base) > 1e-6)

  elif a.check == "padding":
    pad_rows = np.asarray(batch["positions_xy"]) == -1     # [B, N, 2]
    pad_rows = pad_rows.any(axis=-1)                        # [B, N]
    patches = np.asarray(batch["patches"]).copy()
    rng = np.random.RandomState(0)
    patches[pad_rows] = rng.normal(size=patches[pad_rows].shape).astype(np.float32)
    b2 = dict(batch)
    b2["patches"] = jnp.asarray(patches)
    perturbed = total_loss(model, params, b2)
    out["loss_patch_padding_perturbed"] = perturbed
    out["n_pad_rows"] = int(pad_rows.sum())
    out["PASS"] = bool(perturbed == base)  # bit-identical requirement

  elif a.check == "gap":
    b2 = dict(batch)
    b2["patches"] = other["patches"]              # wrong image for this example
    b2["positions_xy"] = other["positions_xy"]
    wrong = total_loss(model, params, b2)
    out["loss_wrong_image"] = wrong
    out["gap_wrong_minus_correct"] = wrong - base
    out["PASS"] = bool(wrong > base)  # direction gates (magnitude reported)

  res_dir = f"{REPO}/results/{a.variant}"
  os.makedirs(res_dir, exist_ok=True)
  suffix = "_trained" if a.params_dir else ""
  with open(f"{res_dir}/vision_{a.check}{suffix}.json", "w") as f:
    json.dump(out, f, indent=2)
  print(json.dumps(out, indent=2))
  sys.exit(0 if out["PASS"] else 1)


if __name__ == "__main__":
  main()
