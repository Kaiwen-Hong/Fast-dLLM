"""SASD train-step gate: native MaxText `objective:"sasd"` on a TINY qwen2.5 model.

This is the PLUMBING + loss-math acceptance test for wiring Fast-dDrive's SASD
objective into MaxText's `trainers/pre_train/train.loss_fn`. It runs on a tiny
qwen2.5 decoder (2 layers, emb 128, head_dim 32 so 2*sum([4,6,6]) == 32) with the
FULL vocab, fed a REAL SASD batch from `make_sasd_loader`, and asserts:

  (a) one train step runs end-to-end (loss FINITE, grads non-NaN/non-inf);
  (b) the SASD loss computed INSIDE MaxText equals the NNX reference
      section_weighted_ce + causal_ce on the SAME logits (the bit-exact sasd.py
      vs NNX equality is reused: identical logits -> identical loss).

How M-RoPE + the hybrid mask are injected (no MaxText RoPE / mask used):
  - 3D M-RoPE: cos/sin are computed host-side by `maxtext.diffusion.sasd.mrope_cos_sin`
    (bit-identical to ddrive_jax.models.qwen2_5_text.mrope_cos_sin, CONTIGUOUS-chunk
    layout) on the DOUBLED positions, threaded as attention_metadata["sasd_rope_cos"/
    "sasd_rope_sin"], and applied in Attention.__call__ via HF full-form rope in fp32.
    config.use_mrope stays FALSE (MaxText's interleaved M-RoPE class is never built).
  - hybrid block-causal 4D mask: built host-side by sasd.hybrid_block_causal_mask_dense,
    threaded as attention_metadata["sasd_attn_mask"], and applied inside
    apply_attention_dot by bypassing generate_attention_mask and doing
    jnp.where(mask, logits, float32.min) -- bit-identical to the NNX masking.

Run:
  unset LD_LIBRARY_PATH; export JAX_PLATFORMS=cpu
  BOTH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive:/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/src
  PYTHONPATH=$BOTH JAX_PLATFORMS=cpu /home/kaiwen/jax-dlm-baseline/.venv/bin/python <this file>
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import Mesh

from maxtext.configs import pyconfig
from maxtext.models.models import transformer_as_linen
from maxtext.utils import maxtext_utils
from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR
from maxtext.trainers.pre_train import train as train_mod
from maxtext.diffusion import sasd as mxt_sasd

# --- NNX reference (the loss we must reproduce) -------------------------------
from ddrive_jax.diffusion import sasd_loss as ref_loss
from ddrive_jax.data.grain_pipeline import make_sasd_loader

DATA_DIR = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd"

# tiny qwen2.5: head_dim 32 == 2*sum([4,6,6]); 2 layers; emb 128; FULL vocab.
MROPE_SECTION = [4, 6, 6]
HEAD_DIM = 32
ROPE_THETA = 1_000_000.0
SEED = 0


def _make_config():
  base = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  overrides = dict(
      run_name="sasd_train_step_test",
      objective="sasd",
      sasd_mrope_section=MROPE_SECTION,
      sasd_rope_theta=ROPE_THETA,
      decoder_block="qwen2",
      enable_checkpointing=False,
      scan_layers=False,            # simple per-layer loop carries attention_metadata
      attention="dot_product",      # the kernel whose additive mask we intercept
      use_mrope=False,              # we inject contiguous-chunk cos/sin ourselves
      logits_via_embedding=True,    # qwen2.5 ties embeddings
      logits_dot_in_fp32=True,      # fp32 logit projection (NNX parity)
      cast_logits_to_fp32=True,
      float32_logits=True,          # fp32 softmax (NNX parity)
      float32_qk_product=True,      # fp32 qk product (NNX parity)
      normalize_embedding_logits=False,
      use_qk_norm=False,
      attention_bias=True,          # qwen2.5: q/k/v bias
      dtype="float32",
      weight_dtype="float32",
      per_device_batch_size=1.0,
      max_target_length=2400,       # >= 2L = 2368
      base_emb_dim=128,
      base_num_query_heads=4,
      base_num_kv_heads=2,
      base_mlp_dim=256,
      base_num_decoder_layers=2,
      head_dim=HEAD_DIM,
      vocab_size=151936,
      mlp_activations=["silu", "linear"],
      normalization_layer_epsilon=1e-6,
      rope_max_timescale=ROPE_THETA,
      enable_dropout=False,
  )
  return pyconfig.initialize([sys.argv[0], base], override_model_config=True, **overrides)


def _make_mesh(cfg):
  devices_array = maxtext_utils.create_device_mesh(cfg)
  return Mesh(devices_array, cfg.mesh_axes)


def main():
  cfg = _make_config()
  mesh = _make_mesh(cfg)

  # --- 1. real SASD batch (B=1 sample -> 2 doubled rows = 2B) ------------------
  loader = make_sasd_loader(DATA_DIR, "train", per_host_batch=1, seed=SEED)
  batch = next(iter(loader))
  B = int(batch["input_final"].shape[0])
  L = int(batch["rbi"].shape[1])
  print(f"loaded SASD batch: B={B} L={L} 2L={2*L} sample={batch['sample_id'][0]}")

  # --- 2. host precompute: doubled cos/sin (contiguous-chunk M-RoPE) + 4D mask -
  prepped = mxt_sasd.prepare_sasd_inputs(batch, HEAD_DIM, MROPE_SECTION, ROPE_THETA)
  twoB, twoL = prepped["inputs"].shape
  assert twoB == 2 * B and twoL == 2 * L, (twoB, twoL, B, L)
  assert prepped["cos"].shape == (2 * B, 2 * L, HEAD_DIM), prepped["cos"].shape
  assert prepped["attn_mask"].shape == (2 * B, 2 * L, 2 * L), prepped["attn_mask"].shape
  print(f"prepared: inputs {prepped['inputs'].shape}, cos {prepped['cos'].shape}, "
        f"attn_mask {prepped['attn_mask'].shape}")

  # --- 3. build the tiny model + params ---------------------------------------
  model = transformer_as_linen(cfg, mesh, quant=None)
  init_rng = jax.random.PRNGKey(SEED)
  dummy_pos = jnp.broadcast_to(jnp.arange(2 * L, dtype=jnp.int32)[None, :], (2 * B, 2 * L))
  with mesh:
    variables = model.init(
        {"params": init_rng, "dropout": init_rng},
        prepped["inputs"],
        dummy_pos,
        decoder_segment_ids=None,
        enable_dropout=False,
    )
  params = variables  # Linen: {"params": {...}}
  n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
  print(f"model built: {n_params/1e6:.1f}M params (tiny qwen2.5, FULL vocab)")

  # --- 4. assemble the `data` dict the SASD loss_fn branch expects -------------
  # Higher-rank SASD keys are sasd_* prefixed (the loss_fn branch pops them before
  # the generic [B,L] decimation/slicing that would crash on them).
  data = {
      "inputs": prepped["inputs"],                       # [2B, 2L] i32
      "inputs_position": dummy_pos,                       # [2B, 2L] i32 (dummy; real pos -> M-RoPE)
      "sasd_cos": prepped["cos"],                         # [2B, 2L, hd]
      "sasd_sin": prepped["sin"],                         # [2B, 2L, hd]
      "sasd_attn_mask": prepped["attn_mask"],            # [2B, 2L, 2L] bool
      "sasd_labels_final": prepped["labels_final"],      # [B, 2, L]
      "sasd_original_labels": prepped["original_labels"],  # [B, 1, L]
      "sasd_weights": prepped["weights"],                # [B, 2, L]
      "sasd_num_items": prepped["num_items"],            # [B]
      "sasd_B": jnp.full((2 * B,), B, dtype=jnp.int32),  # broadcast scalar so slicing is safe
      "sasd_L": jnp.full((2 * B,), L, dtype=jnp.int32),
  }

  # --- 5. ONE train step end-to-end (value_and_grad) — mirrors train.train_step
  grad_func = jax.value_and_grad(train_mod.loss_fn, argnums=4, has_aux=True)
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    (loss, aux), grads = grad_func(
        model, cfg, dict(data), init_rng, params,
        sparsity_state={}, is_train=True,
    )
  loss = float(loss)
  print(f"\n(a) END-TO-END train step:")
  print(f"    loss            = {loss:.6f}")
  print(f"    xent_sum        = {float(aux['xent_sum']):.6f}")
  print(f"    total_weights   = {float(aux['total_weights']):.1f}")
  assert np.isfinite(loss), f"loss not finite: {loss}"
  leaves = jax.tree_util.tree_leaves(grads)
  grad_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in leaves)))
  all_finite = all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
  print(f"    grad global norm= {grad_norm:.6f}  (all finite: {all_finite})")
  assert all_finite, "grads contain NaN/Inf"
  assert grad_norm > 0.0, "grads are all zero"
  print("    [OK] loss finite, grads non-NaN/non-zero")

  # --- 6. bit-exact: MaxText in-loss SASD == NNX reference on the SAME logits --
  # Recompute the forward to capture the exact logits MaxText fed to its loss,
  # then run BOTH the NNX reference (ddrive_jax.diffusion.sasd_loss) and the
  # MaxText-port (maxtext.diffusion.sasd) loss on those identical logits.
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    logits, _ = model.apply(
        params,
        prepped["inputs"],
        dummy_pos,
        decoder_segment_ids=None,
        enable_dropout=False,
        rngs={"dropout": init_rng, "params": init_rng},
        mutable=["intermediates"],
        attention_metadata={
            "sasd_rope_cos": prepped["cos"],
            "sasd_rope_sin": prepped["sin"],
            "sasd_attn_mask": prepped["attn_mask"],
        },
    )
  logits = jnp.asarray(logits, jnp.float32)  # [2B, 2L, V]
  V = logits.shape[-1]

  # MaxText port path (exactly what loss_fn used)
  mxt_loss, mxt_sec, mxt_cau, mxt_tw = mxt_sasd.sasd_loss_from_logits(
      logits, prepped["labels_final"], prepped["original_labels"],
      prepped["weights"], prepped["num_items"], B, L)

  # NNX reference path on the SAME logits, reproducing per_sample_loss_sums layout
  lg = np.asarray(logits).reshape(B, 2, 2 * L, V)
  noisy = jnp.asarray(lg[:, :, :L, :].reshape(2 * B, L, V))   # both rows
  clean = jnp.asarray(lg[:, 0, L:, :])                        # mdm row only -> [B, L, V]
  ref_sec = ref_loss.section_weighted_ce(
      noisy, prepped["labels_final"].reshape(2 * B, L),
      prepped["weights"].reshape(2 * B, L), num_items=1.0)
  ref_cau = ref_loss.causal_ce(clean, prepped["original_labels"][:, 0, :], num_items=1.0)
  ref_total_weights = float(jnp.sum(prepped["num_items"]))
  ref_loss_val = float((ref_sec + ref_cau) / max(ref_total_weights, 1.0))

  print(f"\n(b) SASD loss math (MaxText port vs NNX reference, SAME logits):")
  print(f"    section sum : maxtext={float(mxt_sec):.6f}  nnx={float(ref_sec):.6f}")
  print(f"    causal  sum : maxtext={float(mxt_cau):.6f}  nnx={float(ref_cau):.6f}")
  print(f"    total_weights: maxtext={float(mxt_tw):.1f}  nnx={ref_total_weights:.1f}")
  print(f"    final loss  : maxtext={float(mxt_loss):.6f}  nnx={ref_loss_val:.6f}")

  def _eq(a, b, name):
    a, b = np.asarray(a), np.asarray(b)
    if not np.array_equal(a, b):
      raise AssertionError(f"{name}: NOT bit-exact: {a} vs {b} (diff {float(np.abs(a-b))})")
    print(f"    [OK] {name}: bit-exact")

  _eq(mxt_sec, ref_sec, "section_weighted_ce sum")
  _eq(mxt_cau, ref_cau, "causal_ce sum")
  _eq(float(mxt_tw), ref_total_weights, "total_weights")
  _eq(float(mxt_loss), ref_loss_val, "global SASD loss")

  # The in-loss xent_sum (numerator) must equal sec+cau (what flows to the metric tail).
  _eq(float(aux["xent_sum"]), float(mxt_sec + mxt_cau), "loss_fn xent_sum == sec+cau")

  # Sanity: the loss_fn's own loss == numerator/(total_weights+EPS) within fp tolerance.
  from maxtext.utils.globals import EPS
  expect_loss = float((mxt_sec + mxt_cau) / (float(mxt_tw) + EPS))
  assert abs(loss - expect_loss) < 1e-4, (loss, expect_loss)
  print(f"    [OK] loss_fn loss ({loss:.6f}) == xent_sum/(total_weights+EPS) ({expect_loss:.6f})")

  print("\nSASD_TRAIN_STEP_PASS")


if __name__ == "__main__":
  main()
