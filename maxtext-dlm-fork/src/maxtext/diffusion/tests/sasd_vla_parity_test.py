"""SASD FULL-VLA PARITY gate (the capstone): the COMPLETE MaxText SASD forward+loss
-- real Fast-dDrive text weights + FROZEN ViT image embeds injected at the image-token
positions in the doubled sequence + 3D M-RoPE + hybrid block-causal mask + section/causal
loss -- must MATCH the validated NNX harness reference loss on a REAL Waymo SASD batch
(make_sasd_loader, incl pixel_values).

This closes the last gap after:
  - sasd_train_step_test : objective:"sasd" loss bit-exact vs NNX GIVEN logits.
  - sasd_weight_parity_test : Fast-dDrive text weights -> MaxText forward 100% top-1 vs NNX.
Here we add the IMAGE-EMBEDS path and prove the WHOLE thing matches end to end.

How (phases, each a SUBPROCESS so the 3B + ViT never coexist in the 30GB host RAM)
----------------------------------------------------------------------------------
The NNX reference is computed in TWO sub-phases (text decoder and ViT are loaded by
SEPARATE streaming loaders in SEPARATE processes -- the stock build_harness loads the
full ~16GB fp32 state dict + ~12GB model + AdamW state and OOMs the 30GB host, so we
sidestep it and reproduce its EXACT per-sample math directly):

PHASE ref_vit (--phase=ref_vit): frozen ViT image embeds.
  - one batch from make_sasd_loader(.../wod_e2e_sasd, "train", per_host_batch=1).
  - STREAMING-load ONLY the visual.* ViT tensors (one at a time), run the ViT per sample
    on pixel_values + image_grid_thw -> [N,D], double to [2N,D] (concatenate([ie,ie])) ==
    train_tpu._real_image_embeds_fn EXACTLY; stack -> [B,2N,D]; save raw batch + embeds.
    Exit (frees ViT host dict + GPU).

PHASE ref_text (--phase=ref_text): the NNX VLA loss.
  - STREAMING-load ONLY the Fast-dDrive text decoder (like sasd_weight_parity_test's
    _load_nnx_text_streaming -- never holds the whole state dict).
  - load the saved batch + ViT embeds; run train_tpu.per_sample_loss_sums (the harness's
    EXACT per-sample SASD forward: embed_tokens(ifn[r]).at[img_pos].set(image_embeds),
    hidden_forward_mrope_cs, attend, section_weighted_ce + causal_ce) vmapped over B, then
    the harness's global normaliser loss = (sum sec + sum cau)/sum num_items. This is
    numerically identical to build_harness+run_step's loss. Save the reference scalars.
    Exit (frees the 3B + GPU before MaxText loads).

PHASE 2 (--phase=mxt): MaxText.
  - load the npz (batch arrays + the SAME frozen ViT embeds the NNX harness used).
  - build qwen2.5-3b Linen (scan_layers=False), load the SAME Fast-dDrive text weights
    in-process (build_maxtext_params_from_fast_ddrive).
  - prepare_sasd_inputs(batch, ..., image_embeds=ie) -> doubled inputs + cos/sin + hybrid
    mask + the [2B,2N,D] image embeds + [2B,2N] img_pos (repeat x2, flat=2b+r).
  - run the objective:"sasd" loss_fn (the real train.py branch): model.apply scatters the
    ViT embeds at img_pos in _apply_embedding, then sasd_loss_from_logits -> MaxText loss.
  - assert MaxText loss == NNX loss within a small relative tolerance.

Tolerance: the 7.3 weight-parity forward is 100% top-1 / mean logit |d| ~1.9e-3, so the
scalar loss should match to well under 1e-2 relative. Gate: rel-diff on the total loss
< 1e-2 (report achieved). Prints SASD_VLA_PARITY_PASS.

Run (GPU)
---------
  source /home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/jax_gpu_env.sh
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive:/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/src
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
  "$JAXPY" -m maxtext.diffusion.tests.sasd_vla_parity_test   # runs both phases as subprocesses
"""
import argparse
import os
import subprocess
import sys

import numpy as np

SNAP = ("/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/"
        "snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f")
DATA_DIR = "/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd"
ARTIFACT = "/home/kaiwen/data/fast-ddrive/sasd_vla_parity_ref.npz"

MROPE_SECTION = (16, 24, 24)   # 2*(16+24+24) == 128 == head_dim
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
SEED = 0
REL_TOL = 1e-2                  # relative diff on the total loss


# ===========================================================================================
# Streaming loaders (mirror sasd_weight_parity_test._load_nnx_text_streaming + the ViT map
# in hf_to_jax.load_fast_ddrive_vit, but read ONE tensor at a time -> never hold the full
# ~16GB state dict; the 30GB host can't fit dict + model together).
# ===========================================================================================
def _load_nnx_text_streaming(cfg):
  import gc
  import jax.numpy as jnp
  from flax import nnx
  from safetensors import safe_open
  from ddrive_jax.convert.hf_to_jax import _name_map_text, _set
  from ddrive_jax.models.qwen2_5_text import Qwen25TextModel

  model = Qwen25TextModel(cfg, rngs=nnx.Rngs(0))
  want = {}
  for i in range(cfg.n_layers):
    for hf, (path, tr) in _name_map_text(i).items():
      want[hf] = (path, tr)
  want["model.embed_tokens.weight"] = ("embed_tokens/embedding", False)
  want["model.norm.weight"] = ("norm/weight", False)

  n = 0
  for fname in sorted(os.listdir(SNAP)):
    if not fname.endswith(".safetensors"):
      continue
    with safe_open(os.path.join(SNAP, fname), framework="numpy") as f:
      for k in f.keys():
        if k not in want:
          continue
        path, tr = want[k]
        arr = f.get_tensor(k).astype(np.float32)
        if tr:
          arr = arr.T
        _set(model, path, jnp.asarray(arr))
        del arr
        n += 1
    gc.collect()
  return model, n


def _load_nnx_vit_streaming(cfg):
  """Streaming ViT load: mirror hf_to_jax.load_fast_ddrive_vit's exact name map / transpose
  rule, one tensor at a time (never holds the full state dict). Returns the loaded ViT."""
  import gc
  import jax.numpy as jnp
  from flax import nnx
  from safetensors import safe_open
  from ddrive_jax.convert.hf_to_jax import _set
  from ddrive_jax.models.vision_qwen25vl import VisionTransformer

  vit = VisionTransformer(cfg, rngs=nnx.Rngs(0))

  # hf_key -> (path, transform) ; transform in {"T", "conv", None}.
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
        elif tf == "conv":          # Conv3d [out,in,T,H,W] -> Lin kernel [in_flat, out]
          arr = arr.reshape(arr.shape[0], -1).T
        _set(vit, path, jnp.asarray(arr))
        del arr
        n += 1
    gc.collect()
  return vit, n


# ===========================================================================================
# PHASE ref_vit: frozen ViT image embeds (streaming ViT; saves batch + embeds; frees GPU).
# ===========================================================================================
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

  print(f"[ref_vit] streaming-loading frozen ViT ...", flush=True)
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
# PHASE ref_text: the NNX VLA loss (streaming text; reads saved batch+embeds; frees GPU).
# ===========================================================================================
def run_ref_text():
  import jax
  import jax.numpy as jnp
  from flax import nnx
  from ddrive_jax.models.qwen2_5_text import Qwen25TextConfig
  from ddrive_jax.diffusion.masks import hybrid_block_causal_mask_dense, to_attn_mask4d
  from ddrive_jax.models.qwen2_5_text import mrope_cos_sin as nnx_mrope_cos_sin
  from ddrive_jax.train.train_tpu import per_sample_loss_sums

  ref = np.load(ARTIFACT, allow_pickle=True)
  B = int(ref["B"]); L = int(ref["L"])
  print(f"[ref_text] sample {ref['sample_id']}  B={B} L={L}", flush=True)

  cfg = Qwen25TextConfig.fast_ddrive(dtype=jnp.float32)
  print(f"[ref_text] streaming-loading Fast-dDrive text decoder ...", flush=True)
  model, n = _load_nnx_text_streaming(cfg)
  print(f"[ref_text] text loaded ({n} tensors; tied -> lm_head not loaded).", flush=True)

  # Per-sample host precompute: doubled cos/sin + hybrid 4D mask + img_pos (EXACT
  # train_tpu.prepare_batch, B-loop). Then vmap per_sample_loss_sums (harness math).
  input_final = jnp.asarray(ref["input_final"])             # [B,2,2L] i64
  labels_final = jnp.asarray(ref["labels_final"])           # [B,2,L]
  original_labels = jnp.asarray(ref["original_labels"])     # [B,1,L]
  weights = jnp.asarray(ref["weights"])                     # [B,2,L]
  image_embeds = jnp.asarray(ref["image_embeds"], jnp.float32)  # [B,2N,D]
  img_pos = jnp.asarray(ref["img_pos_ref"].astype(np.int32))   # [B,2N]
  num_items = np.asarray(ref["num_items"], np.float32)

  cos_list, sin_list, mask_list = [], [], []
  for b in range(B):
    pos2 = np.concatenate([ref["position_ids"][b], ref["position_ids"][b]], axis=1)  # [3,2L]
    cos_b, sin_b = nnx_mrope_cos_sin(pos2, HEAD_DIM, ROPE_THETA, MROPE_SECTION)      # [2L,hd]
    mask_b = to_attn_mask4d(hybrid_block_causal_mask_dense(
        jnp.asarray(ref["rbi"][b]), jnp.asarray(ref["turn"][b]), L))                 # [1,1,2L,2L]
    cos_list.append(np.asarray(cos_b)); sin_list.append(np.asarray(sin_b))
    mask_list.append(np.asarray(mask_b))
  cos = jnp.asarray(np.stack(cos_list))         # [B,2L,hd]
  sin = jnp.asarray(np.stack(sin_list))         # [B,2L,hd]
  mask4d = jnp.asarray(np.stack(mask_list))     # [B,1,1,2L,2L]

  # The harness's EXACT per-sample SASD forward (incl. image scatter), vmapped over B.
  sec_sums, cau_sums = jax.vmap(
      per_sample_loss_sums, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
      model, input_final, labels_final, original_labels, weights,
      cos, sin, mask4d, img_pos, image_embeds)
  g_sec = float(jnp.sum(sec_sums)); g_cau = float(jnp.sum(cau_sums))
  g_den = float(np.sum(num_items))
  loss = (g_sec + g_cau) / max(g_den, 1.0)
  print(f"[ref_text] NNX VLA loss={loss:.8f}  sec_sum={g_sec:.6f}  cau_sum={g_cau:.6f}  "
        f"den(num_items)={g_den:.1f}", flush=True)

  # Append the reference scalars to the artifact (keep the batch arrays + embeds).
  saved = {k: ref[k] for k in ref.files}
  saved["nnx_loss"] = np.float64(loss)
  saved["nnx_sec"] = np.float64(g_sec)
  saved["nnx_cau"] = np.float64(g_cau)
  saved["nnx_den"] = np.float64(g_den)
  np.savez(ARTIFACT, **saved)
  print(f"[ref_text] saved reference scalars to {ARTIFACT}", flush=True)


# ===========================================================================================
# PHASE 2: MaxText full SASD VLA (real text weights + injected frozen ViT embeds).
# ===========================================================================================
def _make_config(L):
  from maxtext.configs import pyconfig
  from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR

  base = os.path.join(MAXTEXT_CONFIGS_DIR, "base.yml")
  overrides = dict(
      run_name="sasd_vla_parity_test",
      objective="sasd",
      sasd_mrope_section=list(MROPE_SECTION),
      sasd_rope_theta=ROPE_THETA,
      decoder_block="qwen2",
      enable_checkpointing=False,
      scan_layers=False,                # per-layer loop carries attention_metadata
      attention="dot_product",
      use_mrope=False,                  # inject contiguous-chunk cos/sin ourselves
      logits_via_embedding=True,        # qwen2.5 ties embeddings
      logits_dot_in_fp32=True,
      cast_logits_to_fp32=True,
      float32_logits=True,              # fp32 softmax (NNX parity)
      float32_qk_product=True,          # fp32 qk product (NNX parity)
      matmul_precision="highest",       # full fp32 matmuls
      normalize_embedding_logits=False,
      use_qk_norm=False,
      attention_bias=True,              # qwen2.5: q/k/v bias
      dtype="float32",
      weight_dtype="float32",
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
  )
  return pyconfig.initialize([sys.argv[0], base], override_model_config=True, **overrides)


def run_mxt():
  import jax
  import jax.numpy as jnp
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
  image_embeds = jnp.asarray(ref["image_embeds"], jnp.float32)   # [B, 2N, D]  (frozen ViT)

  # Host precompute: doubled cos/sin + hybrid mask + the ViT-embed scatter inputs.
  prepped = mxt_sasd.prepare_sasd_inputs(
      batch, HEAD_DIM, MROPE_SECTION, ROPE_THETA, image_embeds=image_embeds)
  twoB, twoL = prepped["inputs"].shape
  assert twoB == 2 * B and twoL == 2 * L, (twoB, twoL, B, L)
  assert "image_embeds" in prepped and "img_pos" in prepped, "image path keys missing"
  twoN = prepped["image_embeds"].shape[1]
  print(f"[mxt] prepared: inputs{prepped['inputs'].shape} cos{prepped['cos'].shape} "
        f"mask{prepped['attn_mask'].shape} image_embeds{prepped['image_embeds'].shape} "
        f"img_pos{prepped['img_pos'].shape} (2N={twoN})", flush=True)

  # CROSS-CHECK: our row-0-derived img_pos must equal the NNX-derived one (repeat x2).
  nnx_imgpos = np.repeat(ref["img_pos_ref"].astype(np.int32), 2, axis=0)  # [2B, 2N]
  assert np.array_equal(np.asarray(prepped["img_pos"]), nnx_imgpos), \
      "img_pos mismatch vs NNX-derived positions"
  print(f"[mxt] img_pos == NNX img_pos (repeat x2): OK", flush=True)

  cfg = _make_config(L)
  devices_array = maxtext_utils.create_device_mesh(cfg)
  mesh = Mesh(devices_array, cfg.mesh_axes)

  model = transformer_as_linen(cfg, mesh, quant=None)
  params = build_maxtext_params_from_fast_ddrive(SNAP, cfg, verbose=True)

  dummy_pos = jnp.broadcast_to(
      jnp.arange(2 * L, dtype=jnp.int32)[None, :], (2 * B, 2 * L))

  # Assemble the `data` dict the objective:"sasd" loss_fn branch expects (sasd_* keys are
  # popped before the generic [B,L] decimation; the image keys flow via attention_metadata).
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

  init_rng = jax.random.PRNGKey(SEED)
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    loss, aux = train_mod.loss_fn(
        model, cfg, dict(data), init_rng, params, sparsity_state={}, is_train=True)
  mxt_loss = float(loss)
  mxt_xent = float(aux["xent_sum"])           # = sec_sum + cau_sum
  mxt_tw = float(aux["total_weights"])
  print(f"[mxt] MaxText loss={mxt_loss:.8f}  xent_sum(sec+cau)={mxt_xent:.6f}  "
        f"total_weights={mxt_tw:.1f}", flush=True)

  # --- compare to the NNX reference -------------------------------------------------------
  nnx_loss = float(ref["nnx_loss"])
  nnx_sec = float(ref["nnx_sec"]); nnx_cau = float(ref["nnx_cau"]); nnx_den = float(ref["nnx_den"])
  nnx_xent = nnx_sec + nnx_cau

  rel = abs(mxt_loss - nnx_loss) / max(abs(nnx_loss), 1e-12)
  abs_d = abs(mxt_loss - nnx_loss)
  rel_xent = abs(mxt_xent - nnx_xent) / max(abs(nnx_xent), 1e-12)

  print("\n=== SASD FULL-VLA PARITY (NNX harness reference vs MaxText objective:sasd) ===")
  print(f"  sample / B / L              : {ref['sample_id']} / {B} / {L}")
  print(f"  image tokens (2N)/sample    : {twoN}")
  print(f"  NNX   loss                  : {nnx_loss:.8f}")
  print(f"  MaxText loss                : {mxt_loss:.8f}")
  print(f"  abs diff (loss)             : {abs_d:.3e}")
  print(f"  REL diff (loss)             : {rel:.3e}   (tol {REL_TOL:.0e})")
  print(f"  NNX   sec+cau (numerator)   : {nnx_xent:.6f}   (sec={nnx_sec:.4f} cau={nnx_cau:.4f})")
  print(f"  MaxText sec+cau (numerator) : {mxt_xent:.6f}")
  print(f"  REL diff (numerator)        : {rel_xent:.3e}")
  print(f"  denom (num_items)           : nnx={nnx_den:.1f}  maxtext={mxt_tw:.1f}")

  den_ok = abs(nnx_den - mxt_tw) < 1e-3
  ok = (rel < REL_TOL) and den_ok
  if not den_ok:
    print(f"  [!] denominator mismatch: nnx={nnx_den} maxtext={mxt_tw}")
  if ok:
    print("\nSASD_VLA_PARITY_PASS")
  else:
    print("\nSASD_VLA_PARITY_FAIL")
  return ok


# ===========================================================================================
def _run_phase(name, label, env):
  print(f"\n=== launching PHASE {label} in subprocess ===", flush=True)
  r = subprocess.run([sys.executable, __file__, f"--phase={name}"], env=env)
  if r.returncode != 0:
    print(f"PHASE {label} FAILED (exit {r.returncode})")
    sys.exit(r.returncode)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--phase", choices=["ref_vit", "ref_text", "mxt", "both"], default="both")
  args, _ = ap.parse_known_args()

  if args.phase == "ref_vit":
    run_ref_vit()
    return
  if args.phase == "ref_text":
    run_ref_text()
    return
  if args.phase == "mxt":
    ok = run_mxt()
    sys.exit(0 if ok else 1)

  # both: each model loads in its OWN subprocess (text/ViT/3B never coexist in 30GB host).
  env = dict(os.environ)
  _run_phase("ref_vit", "1a (frozen ViT image embeds)", env)
  _run_phase("ref_text", "1b (NNX VLA reference loss)", env)
  print("\n=== launching PHASE 2 (MaxText full VLA) in subprocess ===", flush=True)
  r = subprocess.run([sys.executable, __file__, "--phase=mxt"], env=env)
  sys.exit(r.returncode)


if __name__ == "__main__":
  main()
