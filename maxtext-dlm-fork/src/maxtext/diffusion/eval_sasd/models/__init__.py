"""Vendored NNX model defs for SASD inference (B2): Qwen2.5 text decoder + frozen ViT.
Vendored from Fast-dLLM ddrive_jax/models @ 4b0f4f2 — DO NOT EDIT logic; sync from source.
Note: ddrive_jax/models/sharded.py is intentionally NOT vendored (training-only, unused at
inference); these modules import only `.rope` internally."""
from .qwen2_5_text import Qwen25TextConfig, Qwen25TextModel, mrope_cos_sin
from .rope import RoPE, apply_rope
from .vision_qwen25vl import VisionConfig, VisionTransformer

__all__ = [
    "Qwen25TextConfig", "Qwen25TextModel", "mrope_cos_sin",
    "RoPE", "apply_rope", "VisionConfig", "VisionTransformer",
]
