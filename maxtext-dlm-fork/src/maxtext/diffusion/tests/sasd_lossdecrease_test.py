"""SASD LOSS-DECREASE gate (the training-loop capstone): the native MaxText
``objective:"sasd"`` TRAINING LOOP must REDUCE loss on the REAL Fast-dDrive 3B model
+ real Waymo SASD data, on the single RTX 5090. Overfit ONE fixed real batch for
~40 steps -> loss strictly decreases, no NaN. This is the MaxText analog of the NNX
``gpu_real_smoke.py`` smoke (which went 0.985 -> 0.598).

EVERYTHING UPSTREAM IS ALREADY VERIFIED:
  - sasd_train_step_test    : objective:"sasd" loss bit-exact vs NNX GIVEN logits.
  - sasd_weight_parity_test : Fast-dDrive text weights -> MaxText forward 100% top-1 vs NNX.
  - sasd_vla_parity_test    : FULL VLA forward+loss (ViT embeds + M-RoPE + mask + loss)
                              matches NNX harness loss rel ~1.7e-4 on a real Waymo batch.
So the FORWARD + LOSS is correct; this driver adds ONLY the OPTIMIZER LOOP (hand-built
optax.adafactor, since MaxText's get_optimizer has no adafactor) and proves loss goes DOWN.

MEMORY (critical -- mirror the NNX smoke that fit the 5090):
  - optimizer = ADAFACTOR (NOT adamw -- adamw's fp32 moments ~24GB won't fit).
  - bf16 params/compute; loss/log-softmax stay fp32 (sasd.py _ce_per_token upcasts).
  - REMAT every decoder layer (scan_layers=False + remat_policy="full" -> per-layer full
    activation checkpointing AND carries the SASD cos/sin/mask/image_embeds metadata).
  - per_device_batch_size=1 (SASD doubling -> the model sees [2B=2, 2L]).
  - ViT is FROZEN, run ONCE in a SEPARATE subprocess (so 3B + ViT never coexist in 30GB host
    RAM), saved to an npz; the train loop never re-runs it (stop_gradient already inside).

Two phases, each a SUBPROCESS (the VLA-parity pattern):
  PHASE ref_vit (--phase=ref_vit): stream-load ONLY the frozen ViT, run it on one real
    batch's pixel_values -> [B,2N,D] doubled embeds (EXACT train_tpu._real_image_embeds_fn);
    save the raw batch arrays + embeds to an npz; exit (frees ViT + GPU).
  PHASE mxt (--phase=mxt): build qwen2.5-3b Linen (bf16 + remat), stream-load the real
    Fast-dDrive text weights, cast to bf16; load the saved batch + ViT embeds; assemble the
    FIXED objective:"sasd" data dict (incl. sasd_image_embeds/sasd_image_pos); build
    optax.adafactor; loop value_and_grad(train.loss_fn, argnums=4) -> adafactor update for
    ~40 steps on the SAME batch; assert last-3 mean < first-3 mean by a clear margin and all
    finite. Prints SASD_TRAIN_LOSSDECREASE_PASS + the loss trajectory + peak VRAM.

Run (GPU)
---------
  source /home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/jax_gpu_env.sh
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive:/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/src
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
  "$JAXPY" -m maxtext.diffusion.tests.sasd_lossdecrease_test   # runs both phases as subprocesses
"""
import argparse
import gc
import os
import subprocess
import sys
from functools import partial

import numpy as np

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
DATA_DIR = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd"
ARTIFACT = "/home/kaiwen/data/fast-ddrive/sasd_lossdecrease_batch.npz"

MROPE_SECTION = (16, 24, 24)   # 2*(16+24+24) == 128 == head_dim
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
SEED = 0

# training-loop hyperparams (mirror gpu_real_smoke.py HarnessConfig)
LR = 1e-4
TOTAL_STEPS = 40
WARMUP_STEPS = 4
CLIP = 1.0
MARGIN = 0.05  # last-3 mean must be at least this much below first-3 mean


# ===========================================================================================
# PHASE ref_vit: frozen ViT image embeds (streaming ViT; saves batch + embeds; frees GPU).
# Mirror sasd_vla_parity_test._load_nnx_vit_streaming + run_ref_vit EXACTLY.
# ===========================================================================================
def _load_nnx_vit_streaming(cfg):
  """Streaming ViT load: mirror hf_to_jax.load_fast_ddrive_vit's exact name map / transpose
  rule, one tensor at a time (never holds the full state dict). Returns the loaded ViT."""
  import jax.numpy as jnp
  from flax import nnx
  from safetensors import safe_open
  from ddrive_jax.convert.hf_to_jax import _set
  from ddrive_jax.models.vision_qwen25vl import VisionTransformer

  vit = VisionTransformer(cfg, rngs=nnx.Rngs(0))

  want = {"visual.patch_embed.proj.weight": ("patch_embed/kernel", "conv")}
  for i in range(cfg.depth):
    p, q = f"visual.blocks.{i}.", f"blocks/{i}/"
    want[p + "norm1.weight"] = (q + "norm1/weight", None)
    want[p + "norm2.weight"] = (q + "norm2/weight", None)
    want[p + "attn.qkv.weight"] = (q + "attn/qkv/kernel", "T")
    want[p + "attn.qkv.bias"] = (q + "attn/qkv/bias", None)
    want[p + "attn.proj.weight"] = (q + "attn/proj/kernel", "T")
    want[p + "attn.proj.bias"] = (q + "attn/proj/bias", None)
    for m in ("gate_proj", "up_proj", "down_proj"):
      want[p + f"mlp.{m}.weight"] = (q + f"mlp/{m}/kernel", "T")
      want[p + f"mlp.{m}.bias"] = (q + f"mlp/{m}/bias", None)
  want["visual.merger.ln_q.weight"] = ("merger/ln_q/weight", None)
  want["visual.merger.mlp.0.weight"] = ("merger/fc1/kernel", "T")
  want["visual.merger.mlp.0.bias"] = ("merger/fc1/bias", None)
  want["visual.merger.mlp.2.weight"] = ("merger/fc2/kernel", "T")
  want["visual.merger.mlp.2.bias"] = ("merger/fc2/bias", None)

  n = 0
  for fname in sorted(os.listdir(SNAP)):
    if not fname.endswith(".safetensors"):
      continue
    with safe_open(os.path.join(SNAP, fname), framework="numpy") as f:
      for k in f.keys():
        if k not in want:
          continue
        path, tf = want[k]
        arr = f.get_tensor(k).astype(np.float32)
        if tf == "T":
          arr = arr.T
        elif tf == "conv":
          arr = arr.reshape(arr.shape[0], -1).T
        _set(vit, path, jnp.asarray(arr))
        del arr
        n += 1
    gc.collect()
  return vit, n


def run_ref_vit():
  import jax.numpy as jnp
  from ddrive_jax.models.vision_qwen25vl import VisionConfig
  from ddrive_jax.data.grain_pipeline import make_sasd_loader
  from maxtext.diffusion.sasd import compute_fast_ddrive_image_embeds, _image_positions

  loader = make_sasd_loader(DATA_DIR, "train", per_host_batch=1, seed=SEED)
  batch = next(iter(loader))
  B = int(batch["input_final"].shape[0])
  L = int(batch["rbi"].shape[1])
  print(f"[ref_vit] batch: B={B} L={L} 2L={2*L} sample={batch['sample_id'][0]}", flush=True)

  print("[ref_vit] streaming-loading frozen ViT ...", flush=True)
  vit, n = _load_nnx_vit_streaming(VisionConfig(dtype=jnp.float32))
  print(f"[ref_vit] ViT loaded ({n} tensors).", flush=True)

  # EXACT mirror of train_tpu._real_image_embeds_fn (the production module helper).
  image_embeds = np.asarray(
      compute_fast_ddrive_image_embeds(batch, vit, jnp.float32), np.float32)  # [B,2N,D]
  img_pos, twoN = _image_positions(batch, B)                                   # [B,2N], int
  print(f"[ref_vit] frozen ViT embeds {image_embeds.shape}  img_pos {img_pos.shape}  "
        f"(IMAGE_TOK count/sample 2N = {twoN})", flush=True)

  np.savez(
      ARTIFACT,
      input_final=np.asarray(batch["input_final"], np.int64),         # [B,2,2L]
      labels_final=np.asarray(batch["labels_final"], np.int64),       # [B,2,L]
      original_labels=np.asarray(batch["original_labels"], np.int64), # [B,1,L]
      weights=np.asarray(batch["weights"], np.float32),               # [B,2,L]
      position_ids=np.asarray(batch["position_ids"], np.int32),       # [B,3,L]
      rbi=np.asarray(batch["rbi"], np.int32),                         # [B,L]
      turn=np.asarray(batch["turn"], np.int32),                       # [B,L]
      num_items=np.asarray(batch["num_items"], np.float32),           # [B]
      image_embeds=image_embeds,                                      # [B,2N,D]  frozen ViT
      img_pos_ref=img_pos.astype(np.int64),                          # [B,2N]
      B=np.int64(B), L=np.int64(L),
      sample_id=np.array(str(batch["sample_id"][0])),
  )
  print(f"[ref_vit] saved {ARTIFACT}", flush=True)


# ===========================================================================================
# PHASE mxt: 3B + remat + bf16 + adafactor train loop on the fixed batch.
# ===========================================================================================
def _make_config(L):
  from maxtext.configs import pyconfig
  from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

  base = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  overrides = dict(
      run_name="sasd_lossdecrease_test",
      objective="sasd",
      sasd_mrope_section=list(MROPE_SECTION),
      sasd_rope_theta=ROPE_THETA,
      decoder_block="qwen2",
      enable_checkpointing=False,
      scan_layers=False,                # REQUIRED for SASD (carries attention_metadata)
      remat_policy="full",              # per-layer full activation checkpointing
      attention="dot_product",
      use_mrope=False,                  # inject contiguous-chunk cos/sin ourselves
      logits_via_embedding=True,        # qwen2.5 ties embeddings
      logits_dot_in_fp32=True,
      cast_logits_to_fp32=True,
      float32_logits=True,              # fp32 softmax
      float32_qk_product=True,          # fp32 qk product
      matmul_precision="default",       # bf16 matmuls (memory)
      normalize_embedding_logits=False,
      use_qk_norm=False,
      attention_bias=True,              # qwen2.5: q/k/v bias
      dtype="bfloat16",                 # bf16 compute (MEMORY)
      weight_dtype="bfloat16",          # bf16 params (MEMORY)
      per_device_batch_size=1.0,
      max_target_length=2 * L + 8,      # >= 2L (doubled sequence)
      base_emb_dim=2048,
      base_num_query_heads=16,
      base_num_kv_heads=2,
      base_mlp_dim=11008,
      base_num_decoder_layers=36,
      head_dim=HEAD_DIM,
      vocab_size=151936,
      mlp_activations=["silu", "linear"],
      normalization_layer_epsilon=1e-6,
      rope_max_timescale=ROPE_THETA,
      enable_dropout=False,
      gradient_clipping_threshold=CLIP,
  )
  return pyconfig.initialize([sys.argv[0], base], override_model_config=True, **overrides)


def run_mxt():
  import jax
  import jax.numpy as jnp
  import optax
  from flax import linen as nn
  from jax.sharding import Mesh

  from maxtext.models.models import transformer_as_linen
  from maxtext.utils import maxtext_utils
  from maxtext.diffusion import sasd as mxt_sasd
  from maxtext.diffusion.load_fast_ddrive_maxtext import build_maxtext_params_from_fast_ddrive
  from maxtext.trainers.pre_train import train as train_mod

  ref = np.load(ARTIFACT, allow_pickle=True)
  B = int(ref["B"]); L = int(ref["L"])
  print(f"[mxt] sample {ref['sample_id']}  B={B} L={L} 2L={2*L}", flush=True)

  # Rebuild the grain-style batch dict (numpy) from the saved raw arrays.
  batch = {
      "input_final": ref["input_final"],
      "labels_final": ref["labels_final"],
      "original_labels": ref["original_labels"],
      "weights": ref["weights"],
      "position_ids": ref["position_ids"],
      "rbi": ref["rbi"],
      "turn": ref["turn"],
      "num_items": ref["num_items"],
  }
  # bf16 image embeds (frozen ViT, computed once in the ref_vit phase).
  image_embeds = jnp.asarray(ref["image_embeds"], jnp.bfloat16)   # [B, 2N, D]

  # Host precompute ONCE: doubled cos/sin + hybrid mask + the ViT-embed scatter inputs.
  prepped = mxt_sasd.prepare_sasd_inputs(
      batch, HEAD_DIM, MROPE_SECTION, ROPE_THETA, image_embeds=image_embeds)
  twoB, twoL = prepped["inputs"].shape
  assert twoB == 2 * B and twoL == 2 * L, (twoB, twoL, B, L)
  assert "image_embeds" in prepped and "img_pos" in prepped, "image path keys missing"
  twoN = prepped["image_embeds"].shape[1]
  print(f"[mxt] prepared: inputs{prepped['inputs'].shape} cos{prepped['cos'].shape} "
        f"mask{prepped['attn_mask'].shape} image_embeds{prepped['image_embeds'].shape} "
        f"img_pos{prepped['img_pos'].shape} (2N={twoN})", flush=True)

  cfg = _make_config(L)
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)

  model = transformer_as_linen(cfg, mesh, quant=None)

  # --- load real weights (streaming -> device, fp32), then cast leaves to bf16 + drop fp32 -
  print("[mxt] loading Fast-dDrive 3B text weights (streaming) ...", flush=True)
  params = build_maxtext_params_from_fast_ddrive(SNAP, cfg, verbose=True)
  # cast every leaf fp32 -> bf16 and free the fp32 buffers (MEMORY: ~12.4GB fp32 -> ~6.2GB bf16)
  # keep the {"params": ...} variables wrapper so model.apply(model_vars=params) works for the
  # tied-embedding output head (shared_embedding.variables["params"]["embedding"]); optax
  # optimizes the nested pytree fine either way.
  pure_params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
  del params
  gc.collect()
  n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(pure_params))
  print(f"[mxt] params cast to bf16: {n_params/1e9:.3f}B leaves", flush=True)

  dummy_pos = jnp.broadcast_to(
      jnp.arange(2 * L, dtype=jnp.int32)[None, :], (2 * B, 2 * L))

  # Assemble the FIXED `data` dict the objective:"sasd" loss_fn branch expects.
  data = {
      "inputs": prepped["inputs"],                       # [2B, 2L] i32
      "inputs_position": dummy_pos,                       # dummy; real pos -> M-RoPE
      "sasd_cos": prepped["cos"],
      "sasd_sin": prepped["sin"],
      "sasd_attn_mask": prepped["attn_mask"],
      "sasd_labels_final": prepped["labels_final"],
      "sasd_original_labels": prepped["original_labels"],
      "sasd_weights": prepped["weights"],
      "sasd_num_items": prepped["num_items"],
      "sasd_image_embeds": prepped["image_embeds"],      # [2B, 2N, D]  <-- the image path
      "sasd_image_pos": prepped["img_pos"],              # [2B, 2N] i32
      "sasd_B": jnp.full((2 * B,), B, dtype=jnp.int32),
      "sasd_L": jnp.full((2 * B,), L, dtype=jnp.int32),
  }

  # --- hand-built ADAFACTOR (MaxText's get_optimizer has no adafactor) ----------------------
  # mirror train_tpu.build_tx: warmup_cosine_decay + clip_by_global_norm + adafactor.
  sched = optax.warmup_cosine_decay_schedule(
      0.0, LR, WARMUP_STEPS, TOTAL_STEPS - WARMUP_STEPS, LR * 0.1)
  tx = optax.chain(
      optax.clip_by_global_norm(CLIP),
      optax.adafactor(learning_rate=sched, multiply_by_parameter_scale=True,
                      min_dim_size_to_factor=128),
  )
  opt_state = tx.init(pure_params)
  print("[mxt] adafactor optimizer initialized (factored 2nd moment; no fp32 adamw moments)",
        flush=True)

  grad_fn = jax.value_and_grad(train_mod.loss_fn, argnums=4, has_aux=True)
  init_rng = jax.random.PRNGKey(SEED)

  # The batch is FIXED -> close over `data` as a jit constant (so `int(data["sasd_B"][0])`
  # in loss_fn resolves to a concrete value at trace time, not a tracer). Only params/
  # opt_state are threaded through jit + donated to free old buffers each step.
  @partial(jax.jit, donate_argnums=(0, 1))
  def step(pp, ostate, rng):
    (loss, aux), grads = grad_fn(model, cfg, dict(data), rng, pp,
                                 sparsity_state={}, is_train=True)
    # mirror train.py:460 grad cast to grad_dtype (here bf16) for memory
    grads = jax.tree.map(
        lambda g: g.astype(jnp.bfloat16) if g.dtype == jnp.float32 else g, grads)
    updates, ostate = tx.update(grads, ostate, pp)
    pp = optax.apply_updates(pp, updates)
    return pp, ostate, loss

  # --- the loss-decrease loop: SAME fixed batch every step ----------------------------------
  losses = []
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    for st in range(1, TOTAL_STEPS + 1):
      pure_params, opt_state, loss = step(pure_params, opt_state, init_rng)
      lv = float(loss)
      losses.append(lv)
      if st == 1 or st % 5 == 0:
        print(f"  step {st:3d}  loss {lv:.6f}", flush=True)

  # --- peak VRAM ----------------------------------------------------------------------------
  peak_gb = None
  try:
    stats = jax.devices()[0].memory_stats()
    peak_gb = stats.get("peak_bytes_in_use", 0) / 1e9
  except Exception:
    pass

  # --- verdict ------------------------------------------------------------------------------
  first3 = float(np.mean(losses[:3]))
  last3 = float(np.mean(losses[-3:]))
  all_finite = bool(np.all(np.isfinite(losses)))
  decreased = all_finite and (last3 < first3 - MARGIN)

  print("\n=== SASD TRAIN LOSS-DECREASE (real Fast-dDrive 3B + Waymo, single 5090) ===")
  print(f"  sample / B / L / 2N         : {ref['sample_id']} / {B} / {L} / {twoN}")
  print(f"  optimizer                   : adafactor (lr {LR}, warmup {WARMUP_STEPS}, "
        f"cosine -> {LR*0.1:g}, clip {CLIP})")
  print(f"  dtype / remat               : bf16 params+compute / remat_policy=full (per-layer)")
  print(f"  steps                       : {TOTAL_STEPS}")
  print(f"  loss[0] -> loss[-1]         : {losses[0]:.6f} -> {losses[-1]:.6f}")
  print(f"  first-3 mean -> last-3 mean : {first3:.6f} -> {last3:.6f}  (margin {first3-last3:+.6f})")
  print(f"  all finite (no NaN/Inf)     : {all_finite}")
  if peak_gb is not None:
    print(f"  peak VRAM                   : {peak_gb:.2f} GB")
  print(f"  full trajectory             :")
  print("    " + " ".join(f"{i+1}:{v:.4f}" for i, v in enumerate(losses)))

  print(f"\n  gate: all finite AND last-3 mean < first-3 mean - {MARGIN}")
  if decreased:
    print("\nSASD_TRAIN_LOSSDECREASE_PASS")
  else:
    print("\nSASD_TRAIN_LOSSDECREASE_FAIL")
  return decreased


# ===========================================================================================
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", choices=["ref_vit", "mxt", "both"], default="both")
  args, _ = ap.parse_known_args()

  if args.phase == "ref_vit":
    run_ref_vit()
    return
  if args.phase == "mxt":
    ok = run_mxt()
    sys.exit(0 if ok else 1)

  # both: run ref_vit in a child (frees its GPU + host dict on exit), then mxt in a child.
  env = dict(os.environ)
  print("=== launching PHASE ref_vit (frozen ViT image embeds) in subprocess ===", flush=True)
  r = subprocess.run([sys.executable, __file__, "--phase=ref_vit"], env=env)
  if r.returncode != 0:
    print("PHASE ref_vit FAILED")
    sys.exit(r.returncode)
  print("\n=== launching PHASE mxt (3B + adafactor train loop) in subprocess ===", flush=True)
  r = subprocess.run([sys.executable, __file__, "--phase=mxt"], env=env)
  sys.exit(r.returncode)


if __name__ == "__main__":
  main()
