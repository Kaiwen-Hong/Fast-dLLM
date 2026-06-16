# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Waymo SASD (Fast-dDrive Section-Aware Stochastic Diffusion) input pipeline.

This is the data path for ``dataset_type="waymo_sasd"`` + ``objective="sasd"``. It is the
"production" equivalent of the verified driver in
``maxtext.diffusion.tests.sasd_lossdecrease_test`` (which assembles the SASD ``data`` dict
by hand): per training step it yields the exact dict the ``objective=="sasd"`` loss_fn
branch in ``trainers/pre_train/train.py`` consumes.

Per step, on the HOST:
  1. ``batch = next(loader)``  -- numpy dict from ``ddrive_jax.data.grain_pipeline``'s
     ``make_sasd_loader`` (already PER-HOST sharded via process_index/process_count and
     resumable).
  2. (optional image path) run the FROZEN Fast-dDrive ViT on the batch's pixel_values ->
     ``[B, 2N, D]`` doubled image embeds (``compute_fast_ddrive_image_embeds``).
  3. ``prepare_sasd_inputs`` -> doubled [2B,2L] inputs, 3D M-RoPE cos/sin, hybrid
     block-causal 4D mask, (+ image_embeds/img_pos scatter inputs).
  4. assemble the SASD ``data`` dict (inputs/inputs_position/sasd_*),
  5. form per-host -> global sharded ``jax.Array`` s via ``_form_global_array`` (all SASD
     keys are batch-major on axis 0, so axis-0 batch sharding is consistent).

The frozen ViT is loaded ONCE at iterator construction (option 2a in the design): on a TPU
pod host RAM is ample so the 3B + ViT can coexist (the single-GPU test ran the ViT in a
separate subprocess only because 3B + ViT won't fit in 31GB host RAM on the 5090; for the
GPU SMOKE we set per_device_batch_size=1 and the ViT is small/bf16 so it fits).
"""

import gc
import os

import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from functools import partial

from maxtext.input_pipeline.multihost_dataloading import _form_global_array


# ---------------------------------------------------------------------------
# Frozen ViT streaming loader (mirror of sasd_lossdecrease_test._load_nnx_vit_streaming
# / hf_to_jax.load_fast_ddrive_vit: same name map / transpose rule, one tensor at a time).
# ---------------------------------------------------------------------------
def _load_nnx_vit_streaming(snapshot_dir, cfg):
  """Stream-load the frozen Fast-dDrive ViT (never holds the full state dict). Returns vit."""
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
  for fname in sorted(os.listdir(snapshot_dir)):
    if not fname.endswith(".safetensors"):
      continue
    with safe_open(os.path.join(snapshot_dir, fname), framework="numpy") as f:
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


class WaymoSasdDataIterator:
  """Per-step iterator yielding the SASD ``data`` dict as global sharded ``jax.Array`` s."""

  def __init__(self, config, mesh):
    # Vendored, self-contained data path (no ddrive_jax dependency for v2 AR data;
    # ddrive_jax is only imported lazily below for the v1 pixels-only ViT fallback).
    from maxtext.input_pipeline.sasd_data.grain_pipeline import make_sasd_loader

    self.config = config
    self.mesh = mesh

    data_dir = config.sasd_data_dir or config.dataset_path
    assert data_dir, "dataset_type=waymo_sasd requires sasd_data_dir (or dataset_path) to be set."

    self.head_dim = int(config.head_dim)
    self.mrope_section = tuple(int(x) for x in config.sasd_mrope_section)
    self.rope_theta = float(config.sasd_rope_theta)
    # Embedding dtype for the image embeds (matches the model compute dtype so the scatter
    # in _apply_embedding stays in one dtype).
    self.embed_dtype = jnp.dtype(config.dtype)

    # PER-HOST sharding of the shuffled global stream (grain_pipeline guarantee #1).
    process_index = jax.process_index()
    process_count = jax.process_count()
    per_host_batch = int(config.global_batch_size_to_load) // process_count
    assert per_host_batch >= 1, (
        f"global_batch_size_to_load={config.global_batch_size_to_load} < process_count={process_count}")
    self.per_host_batch = per_host_batch

    self.loader = make_sasd_loader(
        data_dir,
        "train",
        per_host_batch=per_host_batch,
        seed=int(config.data_shuffle_seed),
        process_index=process_index,
        process_count=process_count,
    )
    self._it = iter(self.loader)

    # Sanity-check the configured L / N against the dataset (these drive get_shaped_batch's
    # static abstract shapes; a mismatch would surface as a confusing compile error later).
    L = int(self.loader.L)
    if int(config.sasd_seq_len) and int(config.sasd_seq_len) != L:
      raise ValueError(f"sasd_seq_len={config.sasd_seq_len} != dataset L={L}")
    self.L = L

    # Frozen ViT image path (gated on sasd_num_image_tokens>0 == there is an image path).
    # Dataset v2 carries precomputed frozen-ViT image_embeds (single copy [B,N,D] bf16 in
    # the batch) — then the ViT is never loaded and the data loop is pure IO.
    self.use_image_path = int(config.sasd_num_image_tokens) > 0
    self.data_has_embeds = bool(getattr(self.loader, "has_embeds", False))
    self.vit = None
    if self.use_image_path and not self.data_has_embeds:
      from ddrive_jax.models.vision_qwen25vl import VisionConfig

      snap = config.sasd_vit_snapshot
      assert snap, "sasd_num_image_tokens>0 requires sasd_vit_snapshot (the frozen ViT HF snapshot dir)."
      self.vit, n = _load_nnx_vit_streaming(snap, VisionConfig(dtype=jnp.float32))
      from maxtext.utils import max_logging

      max_logging.log(f"[waymo_sasd] frozen ViT loaded ({n} tensors) from {snap}")
    elif self.use_image_path:
      from maxtext.utils import max_logging

      max_logging.log("[waymo_sasd] dataset carries precomputed image_embeds — ViT not loaded")

  @property
  def local_iterator(self):
    """The underlying grain DatasetIterator — what MaxText's GrainCheckpointHandler
    saves/restores (json get_state/set_state). Lets dataset_type='waymo_sasd' join the
    grain checkpoint path so resume continues the data stream instead of replaying it."""
    return self.loader.grain_iterator

  def reset(self):
    self._it = iter(self.loader)

  def __iter__(self):
    return self

  def __next__(self):
    from maxtext.diffusion import sasd as mxt_sasd

    batch = next(self._it)
    B = int(batch["input_final"].shape[0])

    image_embeds = None
    if self.use_image_path:
      if self.data_has_embeds:
        # dataset v2: precomputed frozen-ViT embeds, single copy [B,N,D] bf16. Double via
        # concat (first N rows = noisy-half image positions, second N = clean) — exactly
        # what compute_fast_ddrive_image_embeds does after its per-sample ViT calls.
        ie = np.asarray(batch["image_embeds"])                       # [B, N, D] bf16
        image_embeds = jnp.asarray(np.concatenate([ie, ie], axis=1), self.embed_dtype)
      else:
        ie = mxt_sasd.compute_fast_ddrive_image_embeds(batch, self.vit, jnp.float32)  # [B,2N,D]
        image_embeds = jnp.asarray(ie, self.embed_dtype)

    prepped = mxt_sasd.prepare_sasd_inputs(
        batch, self.head_dim, self.mrope_section, self.rope_theta, image_embeds=image_embeds)
    twoB, twoL = prepped["inputs"].shape
    dummy_pos = np.broadcast_to(
        np.arange(twoL, dtype=np.int32)[None, :], (twoB, twoL))

    # Assemble the SASD `data` dict (numpy, host-side). dtypes match get_shaped_batch's
    # SASD branch so the standard `p_train_step.lower(...).compile()` matches exactly.
    local = {
        "inputs": np.asarray(prepped["inputs"], np.int32),                          # [2B, 2L]
        "inputs_position": np.asarray(dummy_pos, np.int32),                          # [2B, 2L] dummy
        "sasd_cos": np.asarray(prepped["cos"], np.float32),                          # [2B, 2L, hd]
        "sasd_sin": np.asarray(prepped["sin"], np.float32),                          # [2B, 2L, hd]
        "sasd_attn_mask": np.asarray(prepped["attn_mask"], np.bool_),                # [2B, 2L, 2L]
        "sasd_labels_final": np.asarray(prepped["labels_final"], np.int32),          # [B, 2, L]
        "sasd_original_labels": np.asarray(prepped["original_labels"], np.int32),    # [B, 1, L]
        "sasd_weights": np.asarray(prepped["weights"], np.float32),                  # [B, 2, L]
        "sasd_num_items": np.asarray(prepped["num_items"], np.float32),              # [B]
    }
    if image_embeds is not None:
      local["sasd_image_embeds"] = np.asarray(prepped["image_embeds"])              # [2B, 2N, D]
      local["sasd_image_pos"] = np.asarray(prepped["img_pos"], np.int32)           # [2B, 2N]

    # Per-host local -> global sharded jax.Array (axis-0 batch sharding). All SASD keys are
    # batch-major on axis 0 (2B for the doubled rows, B for the [B,2,L]/[B,1,L]/[B] keys),
    # so a single axis-0 partition is consistent across the dict.
    return jtu.tree_map_with_path(partial(_form_global_array, global_mesh=self.mesh), local)
