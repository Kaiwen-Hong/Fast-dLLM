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

"""DiffusionGemma SFT config — ChartQA VQA (image input).

Multimodal analogue of ``sft_sudoku_full.py``: fine-tune the real 26B
DiffusionGemma (``text_only=False`` -> the release vision tower stays in the
param tree) on ChartQA, with FSDP sharding, gradient checkpointing, and the
release checkpoint loaded via ``GemmaDiffusionCheckpointLoader``.  Requires the
``gs://`` release checkpoint + a multi-accelerator host (see docstring of
``sft_sudoku_full.py``); it is the scale-up template.  For a single-GPU smoke
use ``sft_chartqa_tiny.py`` (which INHERITS this file's
``build_sft_chartqa_config`` with a shrunk random-init model).

The image path is threaded through the official SFT harness (see the ``# IMAGE``
edits in ``hd/hd_gemma_network.py``, ``hd/sft_model.py`` and
``hd/hd_gemma_ar_state_handler.py``): ``SFTDiffusion`` gains ``patches`` /
``positions_xy`` / ``n_soft``, and the vision soft tokens are merged at the
``-2`` placeholders during the encoder prefill (training) and the AR sampler
prefill (eval).  The ChartQA data pipeline (``chartqa_data``) emits the release
vision preprocessing (pooling=3, up to 280 soft tokens); a fixed square input
yields a constant ``n_soft`` so grain can batch (per-process ``batch_size=1``,
since the Gemma4 vision merge is single-sequence).

Run (multi-accelerator):
    python3 -m kauldron.main \
        --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_chartqa.py
"""

from kauldron import konfig

import functools
import jax
from flax import linen as nn
from gemma.gm.nn.gemma4 import _modules

_VOCAB = 262_144

# Release vision tower: pooling_kernel_size=3, output_length=280.  A fixed square
# image yields floor(sqrt(280))**2 = 256 soft tokens (deterministic; verified via
# _preprocessing.predict_soft_token_count(S, S, 16, 280, 3) == 256 for any S).
_IMAGE_SIZE = 448
_PATCH_SIZE = 16
_POOLING = 3
_MAX_SOFT_TOKENS = 280
_N_SOFT = 256
_PROMPT_LEN = 320   # >= 1(bos)+1(soi)+256(-2)+1(eoi)+question
_CANVAS = 64
_NUM_CANVASES = 1


def _enable_gradient_checkpointing():
  """Monkeypatch ``_modules.Block.__call__`` with ``nn.remat`` (26B fits memory).

  Identical to the module-level patch in ``sft_sudoku_full.py``, but wrapped in
  a function so that merely *importing* this config (e.g. from the tiny config)
  does NOT enable gradient checkpointing; only the full ``get_config`` does.
  """
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


# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma.diffusion import _models
  from gemma.diffusion import _paths  # pylint: disable=unused-import
  from gemma.diffusion.hackable_diffusion_adapter.data.chartqa import chartqa_data
  from gemma.diffusion.hackable_diffusion_adapter.hd import gemma_checkpointer
  from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
  from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model
  from hackable_diffusion import hd
  from hackable_diffusion.kdiff import core
  from hackable_diffusion.lib.training import discrete_loss
  from kauldron import kd
  import optax
  from gemma.diffusion.hackable_diffusion_adapter import safe_writer
# pylint: enable=g-import-not-at-top

CHECKPOINT_PATH = _paths.CheckpointPath.DIFFUSIONGEMMA_26B_A4B_IT


def build_sft_chartqa_config(
    *,
    gemma_model,
    n_soft,
    prompt_len,
    canvas_size,
    num_canvases,
    train_ds,
    eval_ds,
    num_train_steps,
    peak_lr,
    warmup_steps,
    checkpointer,
    sharding=None,
    init_transform=None,
    evals=None,
    eval_num_batches=None,
):
  """Assemble the multimodal SFT ``Trainer`` shared by full + tiny configs.

  Only the pieces that differ between the real-26B and the 5090-tiny setup are
  parameters; everything else (the diffusion corruption, the ``SFTDiffusion``
  wiring incl. the image keys, the two losses, the optimizer) is identical.
  """
  cfg = kd.train.Trainer()
  cfg.seed = 42
  cfg.aux = {}
  cfg.aux.vocab_size = _VOCAB

  cfg.aux.corruption_process = hd.corruption.CategoricalProcess.uniform_process(
      num_categories=cfg.ref.aux.vocab_size,
      schedule=hd.corruption.RFSchedule(),
  )
  cfg.aux.prompt_len = prompt_len
  cfg.aux.num_canvases = num_canvases
  cfg.aux.canvas_size = canvas_size
  cfg.aux.n_soft = n_soft
  cfg.aux.peak_lr = peak_lr
  cfg.aux.end_lr = cfg.ref.aux.peak_lr / 10
  cfg.aux.eval_num_batches = eval_num_batches
  cfg.aux.stop_gradient_from_denoiser_to_encoder = False
  cfg.aux.encoder_loss_weight = 1.0
  cfg.aux.decoder_loss_weight = 1.0

  if sharding is not None:
    cfg.sharding = sharding

  base_network = hd_gemma_network.WrappedDiffusionGemmaNetwork(
      gemma_model=gemma_model,
  )

  cfg.model = sft_model.SFTDiffusion(
      x0="batch.canvas",
      prompt="batch.prompt",
      canvas_id="batch.canvas_id",
      canvas_mask="batch.canvas_mask",
      encoder_target="batch.encoder_target",
      encoder_target_mask="batch.encoder_target_mask",
      # IMAGE: multimodal keys threaded through the harness.
      patches="batch.patches",
      positions_xy="batch.positions_xy",
      n_soft=n_soft,
      corruption_process=cfg.ref.aux.corruption_process,
      time_sampler=hd.training.time_sampling.UniformTimeSampler(
          span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4)
      ),
      gemma_network=base_network,
      prompt_len=cfg.ref.aux.prompt_len,
      canvas_size=cfg.ref.aux.canvas_size,
      num_canvases=cfg.ref.aux.num_canvases,
      stop_gradient_from_denoiser_to_encoder=cfg.ref.aux.stop_gradient_from_denoiser_to_encoder,
  )

  cfg.train_losses = {
      "diffusion_loss": core.KauldronLossWrapper(
          loss=discrete_loss.NoWeightDiscreteLoss(
              use_mask=True,
              mask_key="target_mask",
          ),
          weight=cfg.ref.aux.decoder_loss_weight,
      ),
      "encoder_loss": sft_model.EncoderARLoss(
          encoder_logits="preds.encoder_logits",
          encoder_target="preds.encoder_target",
          encoder_target_mask="preds.encoder_target_mask",
          weight=cfg.ref.aux.encoder_loss_weight,
      ),
  }

  cfg.num_train_steps = num_train_steps

  cfg.schedules = {
      "learning_rate": optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=cfg.ref.aux.peak_lr,
          end_value=cfg.ref.aux.end_lr,
          warmup_steps=warmup_steps,
          decay_steps=cfg.ref.num_train_steps,
      ),
  }

  cfg.optimizer = kd.optim.named_chain(**{
      "clip": optax.clip_by_global_norm(max_norm=1.0),
      "adafactor": optax.scale_by_factored_rms(),
      "decay": optax.add_decayed_weights(weight_decay=1e-4),
      "lr": optax.scale_by_learning_rate(cfg.ref.schedules["learning_rate"]),
  })

  cfg.checkpointer = checkpointer

  cfg._konfig_experimental_nofreeze = True  # pylint: disable=protected-access
  cfg.rng_streams = kd.train.RngStreams([
      kd.train.RngStream("default", train=True, eval=True),
      kd.train.RngStream("sampling", train=True, eval=True),
  ])

  if init_transform is not None:
    cfg.init_transform = init_transform

  cfg.train_ds = train_ds
  cfg.eval_ds = eval_ds
  cfg.evals = evals if evals is not None else {}

  cfg.writer = safe_writer.SafeMetricWriter()

  return cfg


def get_config():
  """Full multimodal SFT config for ChartQA on the real 26B (FSDP + TPU)."""
  _enable_gradient_checkpointing()

  # text_only=False keeps the release VisionEncoder (d_model=1152, 27 layers,
  # pooling=3, output_length=280) + use_bidirectional_attention='vision'.
  gemma_model = _models.DiffusionGemma_26B_A4B(text_only=False)

  sharding = kd.sharding.ShardingStrategy(
      params=kd.sharding.FSDPSharding(), opt_state=kd.sharding.FSDPSharding()
  )

  init_transform = gemma_checkpointer.GemmaDiffusionCheckpointLoader(
      path=CHECKPOINT_PATH,
  )

  train_ds = chartqa_data.make_chartqa_ds(
      training=True,
      batch_size=1,
      prompt_len=_PROMPT_LEN,
      num_canvases=_NUM_CANVASES,
      canvas_size=_CANVAS,
      n_soft=_N_SOFT,
      image_size=_IMAGE_SIZE,
      patch_size=_PATCH_SIZE,
      pooling_kernel_size=_POOLING,
      max_soft_tokens=_MAX_SOFT_TOKENS,
      mode="chartqa",
      split="train",
      query_max_len=48,
      num_workers=0,
  )
  eval_ds = chartqa_data.make_chartqa_ds(
      training=False,
      batch_size=1,
      prompt_len=_PROMPT_LEN,
      num_canvases=_NUM_CANVASES,
      canvas_size=_CANVAS,
      n_soft=_N_SOFT,
      image_size=_IMAGE_SIZE,
      patch_size=_PATCH_SIZE,
      pooling_kernel_size=_POOLING,
      max_soft_tokens=_MAX_SOFT_TOKENS,
      mode="chartqa",
      split="val",
      query_max_len=48,
      num_examples=256,
      num_workers=0,
  )

  checkpointer = kd.ckpts.Checkpointer(
      fast=False,
      save_interval_steps=1000,
      max_to_keep=5,
  )

  # Persist fused/original params via the release checkpoint formatter.
  evals = {
      "gemma_checkpointer": gemma_checkpointer.GemmaCheckpointFormatter(
          run=kd.evals.StandaloneEveryCheckpoint(),
          write_original_params=True,
          write_lora_params=False,
          write_fused_lora_params=False,
      ),
  }

  return build_sft_chartqa_config(
      gemma_model=gemma_model,
      n_soft=_N_SOFT,
      prompt_len=_PROMPT_LEN,
      canvas_size=_CANVAS,
      num_canvases=_NUM_CANVASES,
      train_ds=train_ds,
      eval_ds=eval_ds,
      num_train_steps=2_000,
      peak_lr=1.5e-4,
      warmup_steps=1_000,
      checkpointer=checkpointer,
      sharding=sharding,
      init_transform=init_transform,
      evals=evals,
      eval_num_batches=None,
  )
