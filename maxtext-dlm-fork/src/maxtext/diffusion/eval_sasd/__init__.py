"""Self-contained SASD inference for the internal-TPU fork (blocker B2).

Vendored from Fast-dLLM ddrive_jax @ 4b0f4f2 (see ../PATCHES.md). Runs the validated
multimodal section-diffusion sampler with PYTHONPATH=fork/src only — NO ddrive_jax on the
internal side. Weights come from a B1-exported bf16 HF snapshot via the bf16-hardened loaders.

Public API:
  run_eval(npz, snapshot, dtype=...)          -> generate + T2 scalar metrics (driver.py)
  run_parity(npz, snapshot, ...)              -> ViT embedding fp32/bf16 parity (embedding_parity.py)
  mm_section_diffusion_sample / decode_generation   (sampler_sasd.py)
  load_fast_ddrive_text / load_fast_ddrive_vit      (hf_to_jax.py, bf16-hardened)
"""
from .driver import run_eval
from .embedding_parity import run_parity
from .hf_to_jax import load_fast_ddrive_text, load_fast_ddrive_vit
from .models.qwen2_5_text import Qwen25TextConfig, Qwen25TextModel
from .models.vision_qwen25vl import VisionConfig, VisionTransformer
from .sampler_sasd import (
    block_ranges_from_rbi,
    decode_generation,
    mm_section_diffusion_sample,
)

__all__ = [
    "run_eval", "run_parity",
    "mm_section_diffusion_sample", "decode_generation", "block_ranges_from_rbi",
    "load_fast_ddrive_text", "load_fast_ddrive_vit",
    "Qwen25TextConfig", "Qwen25TextModel", "VisionConfig", "VisionTransformer",
]
