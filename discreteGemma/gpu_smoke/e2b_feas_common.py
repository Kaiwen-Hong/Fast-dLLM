"""Shared pieces for the E2B feasibility smokes (Phase-A/B0 pre-flight).

Defines DiffusionGemma_E2B exactly the way diffusion/_models.py defines the 26B
class (Gemma4_E2B backbone + DiffusionMixin + self_conditioner), so the smoke
measures the real thing before we commit the class into the framework.
"""

import sys

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/gemma")

# pylint: disable=g-import-not-at-top
from gemma.diffusion import _transformer as _diffusion_transformer
from gemma.gm.nn.gemma4 import _gemma4

RESULTS_DIR = "/home/kaiwen/data/dgemma_e2b/feas"


class DiffusionGemma_E2B(  # pylint: disable=invalid-name
    _gemma4.Gemma4_E2B, _diffusion_transformer.DiffusionMixin
):
  """Gemma4 E2B + diffusion mixin — mirrors DiffusionGemma_26B_A4B (_models.py:21)."""

  self_conditioning_config: (
      _diffusion_transformer.SelfConditioningConfig | None
  ) = None

  # Same as the 26B class: keep the last prefill KV so indexes line up.
  keep_last_prefill_kv: bool = True

  def setup(self):
    super().setup()
    sc_config = self.self_conditioning_config
    if sc_config is None:
      sc_config = _diffusion_transformer.SelfConditioningConfig(
          features=self.config.embed_dim,
          hidden_dim=self.config.hidden_dim,
      )
    self.self_conditioner = sc_config.make()


def subtree_param_counts(params) -> dict[str, int]:
  """Per-top-level-subtree parameter counts from a (possibly abstract) tree."""
  import flax  # pylint: disable=g-import-not-at-top
  import numpy as np  # pylint: disable=g-import-not-at-top

  flat = flax.traverse_util.flatten_dict(params, sep="/")
  counts: dict[str, int] = {}
  for path, leaf in flat.items():
    top = path.split("/")[0]
    n = int(np.prod(leaf.shape)) if getattr(leaf, "shape", None) else 0
    counts[top] = counts.get(top, 0) + n
  counts["__total__"] = sum(v for k, v in counts.items() if k != "__total__")
  return counts


def gpu_memory_stats() -> dict:
  import jax  # pylint: disable=g-import-not-at-top

  dev = jax.devices()[0]
  stats = dev.memory_stats() or {}
  gb = 1024**3
  return {
      "peak_gb": round(stats.get("peak_bytes_in_use", 0) / gb, 2),
      "in_use_gb": round(stats.get("bytes_in_use", 0) / gb, 2),
      "limit_gb": round(stats.get("bytes_limit", 0) / gb, 2),
  }


def enable_block_remat():
  """Config-scoped gradient checkpointing — verbatim pattern from
  discreteGemma-modified/.../configs/sft_chartqa.py:64-91."""
  import functools  # pylint: disable=g-import-not-at-top
  import flax.linen as nn  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top
  from gemma.gm.nn.gemma4 import _modules  # pylint: disable=g-import-not-at-top

  orig_call = _modules.Block.__call__

  @functools.partial(
      nn.remat,
      policy=jax.checkpoint_policies.nothing_saveable,
      static_argnums=7,
  )
  def rematted_call_fn(
      self, x, segment_pos, cache, attn_mask, per_layer_input,
      kv_shared_cache, skip_sliding_mask,
  ):
    return orig_call(
        self, x, segment_pos, cache, attn_mask,
        per_layer_input=per_layer_input, kv_shared_cache=kv_shared_cache,
        skip_sliding_mask=skip_sliding_mask,
    )

  def new_call(
      self, x, segment_pos, cache, attn_mask, per_layer_input=None,
      kv_shared_cache=None, skip_sliding_mask=False,
  ):
    return rematted_call_fn(
        self, x, segment_pos, cache, attn_mask, per_layer_input,
        kv_shared_cache, skip_sliding_mask,
    )

  _modules.Block.__call__ = new_call
