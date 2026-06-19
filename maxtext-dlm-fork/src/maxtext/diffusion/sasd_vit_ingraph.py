"""In-graph TRAINABLE Fast-dDrive ViT for SASD training (`sasd_vit_trainable=true`).

Today SASD training consumes frozen pre-baked `image_embeds`. This module instead runs the
Qwen2.5-VL ViT IN-GRAPH on `pixel_values` every step, with the ViT TRAINABLE: the (validated,
NNX) ``ddrive_jax`` ViT *body* is wrapped as a Linen submodule via ``flax.nnx.bridge.ToLinen``,
so its params live in the MaxText train state (trainable / sharded / checkpointed through the
standard path). The host-only ViT geometry (window partition / 2D-RoPE / segment masks) is
precomputed by the data iterator (``precompute_sasd_structural``, depends only on grid_thw) and
threaded into the jitted forward; the pure-jax ``body`` is what runs + differentiates in-graph.

At init from the **release** snapshot the in-graph embeds reproduce the pre-baked embeds
(cosine ~1.0) — the step-0 sanity for the whole path. See docs/1plans/06_trainable_vit_plan.md.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax import linen as nn
from flax.nnx import bridge

from ddrive_jax.models.vision_qwen25vl import VisionConfig, VisionTransformer

__all__ = [
    "VisionConfig",
    "sasd_vision_config",
    "precompute_sasd_structural",
    "SasdInGraphViT",
    "load_sasd_vit_leaves_in_order",
]


class _ViTBody(nnx.Module):
    """Thin NNX wrapper whose ``__call__`` IS the differentiable ViT body (pixels + structural).
    Bridged to Linen via ``ToLinen`` so the ViT params land in the Linen train state."""

    def __init__(self, cfg: VisionConfig, *, rngs):
        self.vit = VisionTransformer(cfg, rngs=rngs)

    def __call__(self, pixel_values, structural):
        return self.vit.body(pixel_values, structural)


def _as_jnp_dtype(dt):
    """Accept a jnp dtype OR a MaxText config string like 'bfloat16' / 'float32'."""
    if isinstance(dt, str):
        return jnp.dtype(dt)
    return dt


def sasd_vision_config(dtype=jnp.bfloat16) -> VisionConfig:
    """The fixed Fast-dDrive ViT config (depth 32 / hidden 1280 / out 2048 / ...).
    ``dtype`` may be a jnp dtype or a config string ('bfloat16')."""
    return VisionConfig(dtype=_as_jnp_dtype(dtype))


def precompute_sasd_structural(grid_thw, *, dtype=jnp.bfloat16) -> dict:
    """Host-side ViT geometry for ONE sample's ``grid_thw`` (constant across the batch at a fixed
    resolution). Returns a dict of jnp arrays (window_index / cos / sin / mask_full / mask_win /
    rev) to thread into the jitted step. Param-independent (a throwaway ViT builds it)."""
    vit = VisionTransformer(VisionConfig(dtype=_as_jnp_dtype(dtype)), rngs=nnx.Rngs(0))
    return vit.precompute_structural(grid_thw)


# Fixed WOD-E2E image grid (3 imgs x (t,h,w)=(1,16,14) -> 672 patches -> 168 tokens). Verified
# constant across the dataset, so the ViT geometry (structural) is a compile-time CONSTANT.
SASD_GRID_THW = ((1, 16, 14), (1, 16, 14), (1, 16, 14))


class SasdInGraphViT(nn.Module):
    """Linen submodule: ``pixel_values`` [B, N, 1176] -> DOUBLED image embeds [2B, 2N, D],
    matching ``compute_fast_ddrive_image_embeds`` (per-sample ViT -> concat [ie,ie] -> stack over
    B -> repeat x2 for the doubled [main|complementary] rows).

    The ViT geometry (``structural``: window partition / 2D-RoPE / segment masks) depends ONLY on
    the (fixed) image grid, so it is built on the host at TRACE time and embeds as graph constants
    — it is NOT threaded through the data pipeline (which shards on the batch axis and would
    mis-shard these batch-shared arrays). B is static (per_device_batch_size); the bridged ViT is
    instantiated once and applied per sample with SHARED params (standard Linen call-reuse)."""

    vit_cfg: VisionConfig
    grid_thw: tuple = SASD_GRID_THW

    @nn.compact
    def __call__(self, pixel_values):
        structural = precompute_sasd_structural(np.asarray(self.grid_thw), dtype=self.vit_cfg.dtype)
        lin = bridge.ToLinen(_ViTBody, args=(self.vit_cfg,))
        B = int(pixel_values.shape[0])
        per = [lin(pixel_values[b], structural) for b in range(B)]   # each [N_tok, D]
        ies = jnp.stack(per, axis=0)                                 # [B, N_tok, D]
        ie_doubled = jnp.concatenate([ies, ies], axis=1)            # [B, 2N, D]  (doubling rule)
        return jnp.repeat(ie_doubled, 2, axis=0)                     # [2B, 2N, D]


def load_sasd_vit_leaves_in_order(snapshot_dir, *, dtype=jnp.bfloat16):
    """Snapshot ViT params as leaves IN THE ORDER ``ToLinen`` lays them out, for an in-order
    substitution into the bridged params (validated 390<->390, shapes match in order). The
    HF ``visual.*`` -> NNX name-map lives inside ``load_fast_ddrive_vit``."""
    from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_vit

    dt = _as_jnp_dtype(dtype)
    vit = load_fast_ddrive_vit(snapshot_dir, VisionConfig(dtype=dt), dtype=dt)
    vit = vit[0] if isinstance(vit, tuple) else vit
    body = _ViTBody(VisionConfig(dtype=dt), rngs=nnx.Rngs(0))
    body.vit = vit
    leaves = jax.tree_util.tree_leaves(
        nnx.state(body, nnx.Param), is_leaf=lambda x: hasattr(x, "value")
    )
    return [np.asarray(getattr(l, "value", l)) for l in leaves]
