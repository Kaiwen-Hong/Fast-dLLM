"""Export a MaxText SASD param checkpoint BACK to a Hugging Face safetensors snapshot.

This is the inverse of ``save_fast_ddrive_params_ckpt.py`` /
``build_maxtext_params_from_fast_ddrive`` — it closes blocker **B1** (the missing
"MaxText -> HF" direction). The forward map (HF -> MaxText, 434/434 text leaves) is
the verified ``QWEN_MAXTEXT_TO_HF_PARAM_*`` functions; here we reuse the SAME mapping
with ``saving_to_hf=True`` hooks (which already implement the reverse reshape/transpose)
to write a snapshot that the validated NNX sampler loaders (``load_fast_ddrive_text`` /
``load_fast_ddrive_vit``) read verbatim.

What it writes (key set identical to the source HF snapshot, 824 keys for Qwen2.5-VL-3B):
  - 434 TEXT tensors: pulled from the trained MaxText param tree, inverse-mapped to HF
    layout (kernels: reshape+transpose; biases: [heads,hd]->[hidden]; embedding: identity).
  - 390 ``visual.*`` tensors: copied VERBATIM from the reference snapshot — the ViT is
    FROZEN during SASD training (embeds are precomputed), so vision weights never change.
  - ``lm_head.weight`` is NOT written (tie_word_embeddings=True; logits via embedding),
    matching the source snapshot which also lacks it.
Output dtype is bf16 (matches the bf16 param ckpt; an f32 export OOM-killed the ~30 GB host).
The NNX sampler loaders read it via a raw-byte ml_dtypes reader (safetensors' numpy framework
cannot decode bf16). config.json / tokenizer files are copied from the reference snapshot.

Round-trip identity test (the B1 gate): export FROM the base param ckpt (built by
save_fast_ddrive_params_ckpt.py from the base snapshot), then every exported tensor must
equal the base snapshot tensor (bf16 upcast to f32) bitwise. A wrong transpose/reshape in
the inverse map would permute values and fail this. Pass verify_against=<base ref> to do the check inline —
ONLY meaningful when exporting the BASE param ckpt; a trained ckpt's text weights changed, so
it would spuriously fail the bitwise round-trip (drop verify_against for trained exports).

Usage (GPU or CPU):
  source /home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/jax_gpu_env.sh
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive:\
/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/src
  "$JAXPY" /home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/scripts/maxtext_to_hf_export.py \
      src/maxtext/configs/sasd_waymo.yml model_name=qwen2.5-3b \
      param_ckpt_dir=<MaxText Orbax param ckpt> \
      ref_snapshot=<HF snapshot for vision + shapes + config> \
      out_dir=<HF snapshot to write> \
      [verify_against=<HF snapshot, usually == ref_snapshot for round-trip>]
"""
import gc
import json
import os
import shutil
import struct
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

SHARD_BYTES = 5 * 1024**3  # ~5 GB per safetensors shard


# ---- safetensors header helpers (numpy framework can't decode bf16) --------------
def _st_header(path):
  with open(path, "rb") as fh:
    n = int.from_bytes(fh.read(8), "little")
    hdr = json.loads(fh.read(n))
  return hdr, 8 + n


def _st_tensor_f32(path, key, hdr=None, data_start=None):
  if hdr is None:
    hdr, data_start = _st_header(path)
  m = hdr[key]
  b, e = m["data_offsets"]
  with open(path, "rb") as fh:
    fh.seek(data_start + b)
    raw = fh.read(e - b)
  npdt = {"F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16}[m["dtype"]]
  return np.frombuffer(raw, dtype=npdt).reshape(m["shape"]).astype(np.float32)


def _st_tensor_native(path, key, hdr=None, data_start=None):
  """Read one safetensors tensor preserving its stored dtype (bf16/f16/f32)."""
  if hdr is None:
    hdr, data_start = _st_header(path)
  m = hdr[key]
  b, e = m["data_offsets"]
  with open(path, "rb") as fh:
    fh.seek(data_start + b)
    raw = fh.read(e - b)
  npdt = {"F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16}[m["dtype"]]
  return np.frombuffer(raw, dtype=npdt).reshape(m["shape"]).copy()


def _build_ref_index(snapshot_dir):
  """key -> (shard_path, shape, dtype). Reads headers only (no tensor data)."""
  index = {}
  for fname in sorted(os.listdir(snapshot_dir)):
    if not fname.endswith(".safetensors"):
      continue
    p = os.path.join(snapshot_dir, fname)
    hdr, _ = _st_header(p)
    for k, m in hdr.items():
      if k == "__metadata__":
        continue
      index[k] = (p, tuple(m["shape"]), m["dtype"])
  return index


def _shard_and_write(tensors, out_dir, split_prefix="model"):
  """Write {key: np.ndarray} as sharded safetensors + index.json (HF convention)."""
  from safetensors.numpy import save_file

  os.makedirs(out_dir, exist_ok=True)
  items = list(tensors.items())
  shards, cur, cur_bytes = [], {}, 0
  for k, v in items:
    nb = v.nbytes
    if cur and cur_bytes + nb > SHARD_BYTES:
      shards.append(cur)
      cur, cur_bytes = {}, 0
    cur[k] = v
    cur_bytes += nb
  if cur:
    shards.append(cur)

  nsh = len(shards)
  weight_map, total_size = {}, 0
  for i, sh in enumerate(shards):
    fname = f"{split_prefix}-{i+1:05d}-of-{nsh:05d}.safetensors"
    save_file(sh, os.path.join(out_dir, fname), metadata={"format": "pt"})
    for k, v in sh.items():
      weight_map[k] = fname
      total_size += v.nbytes
  index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
  with open(os.path.join(out_dir, f"{split_prefix}.safetensors.index.json"), "w") as f:
    json.dump(index, f, indent=2)
  return nsh, len(weight_map)


def main():
  from maxtext.configs import pyconfig
  from maxtext.common import checkpointing
  from maxtext.diffusion.load_fast_ddrive_maxtext import _model_for
  from maxtext.checkpoint_conversion.utils.param_mapping import (
      QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
      QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
  )
  from maxtext.checkpoint_conversion.utils.utils import (
      apply_hook_fns,
      validate_and_filter_param_map_keys,
  )
  from maxtext.utils import max_utils, maxtext_utils

  # pull our extra kwargs out of argv (pyconfig rejects unknown keys)
  argv = list(sys.argv)
  param_ckpt_dir = ref_snapshot = out_dir = verify_against = None
  kept = []
  for a in argv:
    if a.startswith("param_ckpt_dir="):
      param_ckpt_dir = a.split("=", 1)[1]
    elif a.startswith("ref_snapshot="):
      ref_snapshot = a.split("=", 1)[1]
    elif a.startswith("out_dir="):
      out_dir = a.split("=", 1)[1]
    elif a.startswith("verify_against="):
      verify_against = a.split("=", 1)[1]
    else:
      kept.append(a)
  assert param_ckpt_dir and ref_snapshot and out_dir, \
      "need param_ckpt_dir=, ref_snapshot=, out_dir="

  config = pyconfig.initialize(kept, enable_checkpointing=False)
  assert config.scan_layers is False, "SASD exporter assumes scan_layers=false (per-layer leaves)."

  # ---- 1. abstract MaxText param tree + restore the trained values ----------------
  # load_params_from_path needs an UNBOXED abstract tree (ShapeDtypeStruct leaves); passing
  # the LogicallyPartitioned-boxed tree makes the restore return abstract structs, not data.
  abstract_vars = maxtext_utils.get_abstract_param(_model_for(config), config)
  abstract_unboxed = max_utils.unbox_logicallypartioned(abstract_vars)
  concurrent_gb = getattr(config, "checkpoint_storage_concurrent_gb", 96)
  restored = checkpointing.load_params_from_path(param_ckpt_dir, abstract_unboxed, concurrent_gb)

  # defensively unwrap any number of {"params": X} nestings down to the real tree
  import flax
  from flax import linen as nn
  def _plain(x):
    return flax.core.unfreeze(x) if isinstance(x, flax.core.FrozenDict) else x
  tree = _plain(restored)
  while isinstance(tree, dict) and set(tree.keys()) == {"params"}:
    tree = _plain(tree["params"])

  # flatten to mt_key ("params-...") -> value, matching the forward builder's key scheme.
  # Leaves are nn.LogicallyPartitioned boxes (the MaxText abstract tree's sharding wrapper);
  # treat each box as a leaf and unbox via .value to get the concrete jax.Array.
  flat, _ = jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned))
  mt_vals = {}
  for path_tuple, leaf in flat:
    if isinstance(leaf, nn.LogicallyPartitioned):
      leaf = leaf.value
    parts = [k.key for k in path_tuple if hasattr(k, "key")]
    mt_vals["params-" + "-".join(parts)] = leaf
  print(f"[export] restored {len(mt_vals)} MaxText leaves from {param_ckpt_dir}", flush=True)

  # ---- 2. mapping + REVERSE hooks (saving_to_hf=True) -----------------------------
  hf_cfg = {"num_hidden_layers": 36, "num_experts": 0}
  param_map = QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, config, scan_layers=config.scan_layers)
  hook_map = QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN(
      hf_cfg, config, scan_layers=config.scan_layers, saving_to_hf=True)
  filtered = validate_and_filter_param_map_keys(param_map.keys(), set(mt_vals.keys()))

  ref_index = _build_ref_index(ref_snapshot)

  # ---- 3. text tensors: MaxText -> HF layout (bf16 out; free source as we go) ------
  # Output bf16 (matches the bf16 param ckpt — lossless, ~half the host RAM of f32, which
  # OOM-killed the 30 GB box). reshape/transpose preserve values; del each source leaf.
  hf_tensors = {}
  for mt_key in filtered:
    hf_key = param_map[mt_key]
    if isinstance(hf_key, list):
      raise RuntimeError(f"unexpected scanned/MoE mapping for {mt_key}; dense unscanned only.")
    if hf_key not in ref_index:
      raise KeyError(f"HF key {hf_key!r} (for {mt_key}) not in reference snapshot {ref_snapshot}")
    target_shape = ref_index[hf_key][1]
    arr = np.asarray(mt_vals[mt_key])  # bf16 (ml_dtypes)
    hook = hook_map.get(mt_key)
    arr = apply_hook_fns(arr, target_shape, hook)
    if tuple(arr.shape) != target_shape:
      raise ValueError(f"{hf_key}: exported shape {arr.shape} != ref {target_shape}")
    hf_tensors[hf_key] = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16))
    mt_vals[mt_key] = None  # free the source leaf
  print(f"[export] mapped {len(hf_tensors)} TEXT tensors MaxText->HF (bf16)", flush=True)
  del mt_vals, restored, tree, flat
  gc.collect()

  # ---- 4. vision tensors: verbatim copy (frozen ViT), native dtype ----------------
  n_vis = 0
  for hf_key, (shard_path, shape, dt) in ref_index.items():
    if not hf_key.startswith("visual."):
      continue
    hf_tensors[hf_key] = np.ascontiguousarray(_st_tensor_native(shard_path, hf_key))
    n_vis += 1
  print(f"[export] copied {n_vis} VISION tensors verbatim (frozen ViT)", flush=True)

  # sanity: exported key set must equal ref key set (minus any lm_head, which neither has)
  ref_keys = set(ref_index.keys())
  exp_keys = set(hf_tensors.keys())
  missing = ref_keys - exp_keys
  extra = exp_keys - ref_keys
  if missing or extra:
    raise RuntimeError(f"key-set mismatch: missing={sorted(missing)[:6]} extra={sorted(extra)[:6]}")

  # ---- 5. write snapshot + copy config/tokenizer ----------------------------------
  os.makedirs(out_dir, exist_ok=True)
  nsh, nk = _shard_and_write(hf_tensors, out_dir, split_prefix="model")
  for fn in os.listdir(ref_snapshot):
    if fn.endswith((".json", ".txt", ".py")) and not fn.startswith("model-") \
        and fn != "model.safetensors.index.json":
      src = os.path.join(ref_snapshot, fn)
      if os.path.isfile(src) or os.path.islink(src):
        shutil.copyfile(src, os.path.join(out_dir, fn))  # follows symlink -> real bytes
  print(f"[export] wrote {nk} tensors in {nsh} shards + config/tokenizer -> {out_dir}", flush=True)
  print("MAXTEXT_TO_HF_EXPORT_DONE", flush=True)

  # ---- 6. optional round-trip identity verify -------------------------------------
  if verify_against:
    vidx = _build_ref_index(verify_against)
    n_ok = n_text_ok = n_vis_ok = 0
    diffs = []
    for hf_key, arr in hf_tensors.items():
      ref = _st_tensor_f32(vidx[hf_key][0], hf_key)          # f32 (bf16/f32 source upcast)
      arr = arr.astype(np.float32)                            # exported bf16 -> f32 (lossless)
      if arr.shape != ref.shape:
        diffs.append((hf_key, "shape", arr.shape, ref.shape)); continue
      md = float(np.max(np.abs(arr - ref)))
      if md == 0.0:
        n_ok += 1
        if hf_key.startswith("visual."): n_vis_ok += 1
        else: n_text_ok += 1
      else:
        diffs.append((hf_key, f"max_abs={md:.3e}"))
    tot = len(hf_tensors)
    print(f"[verify] round-trip vs {verify_against}: bitwise-identical {n_ok}/{tot} "
          f"(text {n_text_ok}, vision {n_vis_ok}); differing {len(diffs)}", flush=True)
    for d in diffs[:15]:
      print("   DIFF", d, flush=True)
    print("B1_ROUNDTRIP_PASS" if not diffs else "B1_ROUNDTRIP_FAIL", flush=True)


if __name__ == "__main__":
  main()
