# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma 4 models with diffusion capabilities."""

from gemma.diffusion import _transformer as _diffusion_transformer
from gemma.gm.nn.gemma4 import _gemma4


class DiffusionGemma_26B_A4B(  # pylint: disable=invalid-name
    _gemma4.Gemma4_26B_A4B, _diffusion_transformer.DiffusionMixin
):
  """DiffusionGemma 26B_A4B model."""

  self_conditioning_config: (
      _diffusion_transformer.SelfConditioningConfig | None
  ) = None

  # So the last prefill KV is kept. Otherwise, indexes will be off by 1.
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


class DiffusionGemma_E2B(  # pylint: disable=invalid-name
    _gemma4.Gemma4_E2B, _diffusion_transformer.DiffusionMixin
):
  """Gemma4 E2B + diffusion mixin (same structure as DiffusionGemma_26B_A4B).

  Initialized from the GEMMA4_E2B AR checkpoints; only `self_conditioner` is
  new (its FFW output projection is zero-initialized by the loader chain so the
  first forward is a no-op w.r.t. self-conditioning while staying trainable).
  """

  self_conditioning_config: (
      _diffusion_transformer.SelfConditioningConfig | None
  ) = None

  # So the last prefill KV is kept. Otherwise, indexes will be off by 1.
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
