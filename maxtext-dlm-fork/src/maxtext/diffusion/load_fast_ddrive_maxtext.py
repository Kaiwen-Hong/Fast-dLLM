"""Load Fast-dDrive (Qwen2.5-VL-3B) TEXT decoder weights into MaxText's native
qwen2.5-3b Linen model -- IN-PROCESS (no Orbax round trip), forward-parity ready.

Why in-process and not `to_maxtext.py`?
  - `qwen2.5-3b` is not registered in HF_IDS / HF_MODEL_CONFIGS / PARAM_MAPPING /
    HOOK_FNS. But the mapping/hook FUNCTIONS (QWEN_MAXTEXT_TO_HF_PARAM_MAPPING /
    QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN) are generic over num_hidden_layers and work
    verbatim for 36 layers. We reuse them directly, fill the abstract param tree's
    leaves from the local safetensors, and `model.apply` -- no registry edits, no
    Orbax, no disk checkpoint.

VL -> text extraction:
  Fast-dDrive stores FLAT keys: model.embed_tokens.weight, model.norm.weight,
  model.layers.{i}.*, lm_head.weight. The 390 `visual.*` keys are skipped (we never
  read them). There is NO `language_model.` prefix.

MASK / extended vocab / tied embeddings:
  vocab_size=151936 is the full embedding table; MASK_ID=151665 / NULL=151666 are
  rows of that table -> copied verbatim (no mean-init). In the RELEASE Fast-dDrive
  snapshot they are already-trained; in the BASE Qwen2.5-VL snapshot (from-base,
  constraint C2) they are UNTRAINED rows (V3) -- still copied verbatim, learned
  during overfit (fallback: mean-init if convergence stalls). tie_word_embeddings=True;
  MaxText's logits_via_embedding=True uses the embedding table for logits, so
  lm_head.weight is NOT loaded (matches the NNX tied `embed_tokens.attend`).

dtype: source may be F32 (release) or BF16 (base); both are read to float32 for
  the parity isolation (see `_st_tensor_f32`).

The hooks applied per leaf are exactly the converter's:
  - reshape_kernel(x, target) = x.T.reshape(target)   (HF [out,in] -> MaxText kernel)
  - reshape_bias(x, target)   = x.reshape(target)      (HF [hidden] -> [heads, head_dim])
  - pad_embedding_layer       = identity (source vocab == target vocab == 151936)
"""
from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from flax import linen as nn
from flax.core import freeze
from safetensors import safe_open

from maxtext.checkpoint_conversion.utils.param_mapping import (
    QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN,
    QWEN_MAXTEXT_TO_HF_PARAM_MAPPING,
)
from maxtext.checkpoint_conversion.utils.utils import (
    apply_hook_fns,
    validate_and_filter_param_map_keys,
)
from maxtext.utils import maxtext_utils


def _st_tensor_f32(path: str, key: str) -> np.ndarray:
  """Read one safetensors tensor as fp32 numpy, tolerant of BF16/F16/F32.
  safetensors' numpy framework cannot decode bf16, so parse the header and
  decode raw bytes via ml_dtypes. Bit-identical to
  ``f.get_tensor(key).astype(np.float32)`` for F32/F16 sources (the release
  Fast-dDrive snapshot is F32); the BASE Qwen2.5-VL snapshot is BF16, hence
  this path (from-base training, constraint C2)."""
  with open(path, "rb") as fh:
    n = int.from_bytes(fh.read(8), "little")
    hdr = json.loads(fh.read(n))
    data_start = 8 + n
    m = hdr[key]
    b, e = m["data_offsets"]
    fh.seek(data_start + b)
    raw = fh.read(e - b)
  npdt = {"F32": np.float32, "F16": np.float16, "BF16": ml_dtypes.bfloat16}[m["dtype"]]
  return np.frombuffer(raw, dtype=npdt).reshape(m["shape"]).astype(np.float32)


def _build_text_key_index(snapshot_dir: str) -> dict[str, str]:
  """Map each NON-visual tensor key -> the safetensors filename holding it. Reading
  only the headers keeps host RAM ~0 (no tensor data is materialized here)."""
  index: dict[str, str] = {}
  for fname in sorted(os.listdir(snapshot_dir)):
    if not fname.endswith(".safetensors"):
      continue
    with safe_open(os.path.join(snapshot_dir, fname), framework="numpy") as f:
      for k in f.keys():
        if k.startswith("visual."):
          continue
        index[k] = fname
  return index


class _StreamingTextGetter:
  """On-demand fp32 numpy getter for one HF text tensor at a time (visual.* excluded).

  Avoids holding the full ~9GB fp32 text state dict alongside the ~12GB MaxText param
  tree (which would OOM the 31GB host). Each get_tensor opens the right shard, reads
  exactly one tensor as fp32, and returns it; the caller moves it to device and drops it.
  """

  def __init__(self, snapshot_dir: str):
    self.dir = snapshot_dir
    self.index = _build_text_key_index(snapshot_dir)

  def __contains__(self, key: str) -> bool:
    return key in self.index

  def get_tensor(self, key: str) -> np.ndarray:
    if key not in self.index:
      raise KeyError(f"text tensor {key!r} not found (visual.* excluded).")
    return _st_tensor_f32(os.path.join(self.dir, self.index[key]), key)


def _hf_config_dict_3b() -> dict:
  """The minimal HF-config dict the mapping/hook functions read (num_hidden_layers,
  num_experts). Mirrors the Fast-dDrive text_config (36 dense layers, no experts)."""
  return {"num_hidden_layers": 36, "num_experts": 0}


def build_maxtext_params_from_fast_ddrive(
    snapshot_dir: str,
    config,
    *,
    verbose: bool = True,
):
  """Return a Linen params pytree {"params": {...}} for the qwen2.5-3b model with
  Fast-dDrive text weights loaded. `config` must be a MaxText pyconfig built for the
  3B dims with scan_layers=False (see sasd_weight_parity_test._make_config).

  Strategy:
    1. abstract param tree -> flat (path_tuple, AbstractValue) -> {params-...: (idx, shape)}
    2. QWEN mapping (mt_key -> hf_key) + hooks (mt_key -> reshape fn), generic over 36 L.
    3. for each filtered mt_key: arr = apply_hook_fns(hf_tensor, target_shape, hook).
    4. tree_unflatten into the abstract treedef -> freeze -> {"params": tree}.
  """
  # --- 1. abstract MaxText param tree -------------------------------------------
  abstract_vars = maxtext_utils.get_abstract_param(_model_for(config), config)
  abstract_params_tree = abstract_vars["params"]
  abstract_params_flat, _ = jax.tree_util.tree_flatten_with_path(abstract_params_tree)
  abstract_treedef = jax.tree_util.tree_structure(
      jax.tree.map(
          lambda _: 0,
          abstract_params_tree,
          is_leaf=lambda x: isinstance(x, nn.LogicallyPartitioned),
      )
  )

  mt_dict: dict[str, tuple[int, tuple]] = {}
  for idx, (path_tuple, leaf) in enumerate(abstract_params_flat):
    # str() each path key: text params use str DictKeys, but the bridged ViT subtree
    # (sasd_vit_trainable) introduces INT dict keys (nnx-style) that break "-".join.
    key_parts = [str(k.key) for k in path_tuple if hasattr(k, "key")]
    mt_dict["params-" + "-".join(key_parts)] = (idx, tuple(leaf.shape))

  # --- 2. mapping + hooks (generic Qwen functions, 36 layers, dense) ------------
  hf_cfg = _hf_config_dict_3b()
  param_map = QWEN_MAXTEXT_TO_HF_PARAM_MAPPING(hf_cfg, config, scan_layers=config.scan_layers)
  hook_map = QWEN_MAXTEXT_TO_HF_PARAM_HOOK_FN(hf_cfg, config, scan_layers=config.scan_layers, saving_to_hf=False)

  # Validate/fill ONLY the TEXT leaves with the QWEN mapping. When sasd_vit_trainable adds the
  # bridged-ViT subtree to mt_dict, those keys aren't in the text param_map (and would trip the
  # "state must be a subset of param_map" check) — they are filled separately below by the in-graph
  # ViT snapshot-init. So restrict the validated state keys to mt_dict ∩ param_map's atomic domain.
  _pm_atomic = set()
  for _k in param_map.keys():
    _pm_atomic.update(_k if isinstance(_k, tuple) else (_k,))
  _text_state_keys = set(mt_dict.keys()) & _pm_atomic
  filtered = validate_and_filter_param_map_keys(param_map.keys(), _text_state_keys)

  # --- 3. fill leaves (streaming: one HF tensor at a time -> device -> drop) ----
  getter = _StreamingTextGetter(snapshot_dir)
  if verbose:
    print(f"[mxt-convert] indexed {len(getter.index)} text tensors "
          f"(visual.* excluded); streaming into MaxText param tree")
  leaves: list = [None] * len(mt_dict)
  n_loaded = 0
  for mt_key in filtered:
    idx, target_shape = mt_dict[mt_key]
    hf_src = param_map[mt_key]
    hook = hook_map.get(mt_key)
    if isinstance(hf_src, list):
      # scanned layer stacking (only if config.scan_layers=True)
      slice_shape = list(target_shape)
      del slice_shape[config.param_scan_axis]
      stacked = [apply_hook_fns(getter.get_tensor(k), tuple(slice_shape), hook) for k in hf_src]
      arr = np.stack(stacked, axis=config.param_scan_axis)
    else:
      if hf_src not in getter:
        raise KeyError(f"HF tensor {hf_src!r} (for {mt_key}) not in snapshot.")
      arr = apply_hook_fns(getter.get_tensor(hf_src), target_shape, hook)
    if tuple(arr.shape) != tuple(target_shape):
      raise ValueError(f"shape mismatch for {mt_key}: got {arr.shape}, want {target_shape}")
    leaves[idx] = jax.device_put(jnp.asarray(arr, dtype=jnp.float32))
    del arr
    n_loaded += 1

  missing = [k for k, (i, _) in mt_dict.items() if leaves[i] is None]
  if missing and getattr(config, "sasd_vit_trainable", False):
    # TRAINABLE in-graph ViT: the QWEN text mapping does not cover the bridged ViT subtree
    # (SasdInGraphViT/ToLinen). Fill those remaining leaves IN ORDER from the snapshot ViT —
    # validated 1:1 in-order correspondence (390<->390, shapes match) in loader_map_test; the
    # per-leaf shape check below + the step-0 frozen-vs-trainable parity guard against any
    # ordering drift. The HF visual.* -> NNX name-map lives inside load_fast_ddrive_vit.
    from maxtext.diffusion.sasd_vit_ingraph import load_sasd_vit_leaves_in_order

    idx_to_shape = {i: s for _, (i, s) in mt_dict.items()}
    miss_idx = [i for k, (i, _) in mt_dict.items() if leaves[i] is None]
    vit_leaves = load_sasd_vit_leaves_in_order(snapshot_dir, dtype=jnp.float32)
    if len(miss_idx) != len(vit_leaves):
      raise ValueError(
          f"ViT subtree has {len(miss_idx)} unfilled leaves but snapshot ViT has {len(vit_leaves)}")
    for i, vl in zip(miss_idx, vit_leaves):
      want = idx_to_shape[i]
      if tuple(np.shape(vl)) != tuple(want):
        raise ValueError(f"ViT leaf shape mismatch at idx {i}: got {np.shape(vl)}, want {want} "
                         "(bridged-ViT subtree order != snapshot ViT order)")
      leaves[i] = jax.device_put(jnp.asarray(vl, dtype=jnp.float32))
      n_loaded += 1
    if verbose:
      print(f"[mxt-convert] filled {len(vit_leaves)} in-graph ViT leaves from snapshot (in order)")
    missing = [k for k, (i, _) in mt_dict.items() if leaves[i] is None]
  if missing:
    raise ValueError(f"{len(missing)} MaxText leaves left unfilled: {missing[:8]} ...")

  params_tree = jax.tree_util.tree_unflatten(abstract_treedef, leaves)
  if verbose:
    n_el = sum(int(np.prod(p.shape)) for p in leaves)
    print(f"[mxt-convert] filled {n_loaded}/{len(mt_dict)} MaxText leaves "
          f"({n_el/1e9:.3f}B params); lm_head.weight NOT loaded (tied embeddings)")
  return freeze({"params": params_tree})


# transformer_as_linen needs a mesh; build a tiny helper that the caller can override.
def _model_for(config):
  """Build the Linen model (single-device mesh) used only to derive abstract shapes."""
  from jax.sharding import Mesh
  from maxtext.models.models import transformer_as_linen

  devices_array = maxtext_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)
  return transformer_as_linen(config, mesh, quant=None)
