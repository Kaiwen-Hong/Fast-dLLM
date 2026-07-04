# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""SHRUNK DiffusionGemma SFT config — ChartQA VQA (image input) — one RTX 5090.

INHERITS ``sft_chartqa.build_sft_chartqa_config`` (the shared assembler used by
the full ``sft_chartqa.py``) and only swaps in the pieces needed to run on a
single 5090:

  * a TINY, RANDOM-INIT DiffusionGemma carrying a TINY vision encoder
    (``text_only=False``) instead of the real 26B release tower;
  * fast pure-numpy fixed-square vision preprocessing (``pooling_kernel_size=1``,
    ``N_SOFT = (image_size // patch_size)**2 = 16``) instead of the release
    pooling=3 / 280-soft-token path;
  * NO ``init_transform`` (no ``gs://`` 26B weights), NO FSDP sharding;
  * 100 train steps, checkpoints at 10/25/50/100, a small ChartQA subset.

Everything structural (the real ``SFTDiffusion`` + ``WrappedDiffusionGemmaNetwork``
+ hackable_diffusion corruption/loss + ``EncoderARLoss``, the ``# IMAGE`` keys
that thread patches through the encoder prefill) is inherited unchanged from
``sft_chartqa.py``.

Success target = the official kauldron train + ``eval_main`` harness runs
end-to-end and train losses decrease.  Task accuracy is NOT expected to rise
(tiny random model, 100 steps).  Run:
    python3 -m kauldron.main \
        --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_chartqa_tiny.py
    python3 -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
        --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_chartqa_tiny.py \
        --step=100 --eval_names=sample_ar_steps32 \
        --cfg.workdir=<WORKDIR> --cfg.eval_ds.batch_size=1
"""

from gemma.diffusion.hackable_diffusion_adapter.configs import sft_chartqa
from kauldron import konfig

_VOCAB = 262_144
_IMAGE_SIZE = 64
_PATCH_SIZE = 16
_N_SOFT = (_IMAGE_SIZE // _PATCH_SIZE) ** 2  # 16 vision soft tokens (square img)
_PROMPT_LEN = 64   # >= 1(bos)+1(soi)+N_SOFT+1(eoi)+question
_CANVAS = 32
_NUM_CANVASES = 1
_SLIDING_WINDOW = 512  # >= prompt_len + canvas so context isn't truncated
_DATA_MODE = "chartqa"   # overridden to "color" by sft_chartqa_grounding_tiny

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma.diffusion import _models
  from gemma.gm.nn.gemma4 import _config as gemma_cfg
  from gemma.gm.nn.gemma4 import _modules as gemma_modules
  from gemma.gm.nn.gemma4.vision import _encoder as gemma_vision
  from gemma.diffusion.hackable_diffusion_adapter.data.chartqa import chartqa_data
  from kauldron import kd
# pylint: enable=g-import-not-at-top


def build_tiny_model():
  """Tiny random-init DiffusionGemma WITH a tiny vision encoder.

  output_length == N_SOFT and pooling_kernel_size == 1 so the encoder emits
  exactly N_SOFT soft tokens per (square) image, matching the N_SOFT ``-2``
  placeholders the data pipeline puts in the prompt.
  """
  tiny_vision = gemma_vision.VisionEncoder(
      d_model=64,
      num_layers=2,
      num_heads=2,
      ffw_hidden=128,
      patch_size=_PATCH_SIZE,
      output_length=_N_SOFT,
      pooling_kernel_size=1,
  )
  tiny_cfg = gemma_cfg.TransformerConfig(
      num_embed=_VOCAB,
      embed_dim=128,
      hidden_dim=256,
      num_heads=4,
      head_dim=32,
      num_kv_heads=1,
      final_logit_softcap=30.0,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      attention_types=gemma_cfg.make_attention_layers_types(
          (gemma_modules.AttentionType.LOCAL_SLIDING,), num_layers=4
      ),
      sliding_window_size=_SLIDING_WINDOW,
      qk_norm_with_scale=True,
      global_rope_proportion=0.25,
      local_rope_proportion=1.0,
      per_layer_input_dim=0,
      enable_moe=False,          # dense FFW: pure standard XLA ops
      vision_encoder=tiny_vision,
      audio_encoder=None,
      use_bidirectional_attention=None,  # sliding layers reuse the causal mask
  )
  # text_only=False keeps the vision encoder in the param tree.
  return _models.DiffusionGemma_26B_A4B(config=tiny_cfg, text_only=False)


def make_tiny_ds(training, split, num_examples):
  return chartqa_data.make_chartqa_ds(
      training=training,
      batch_size=1,
      prompt_len=_PROMPT_LEN,
      num_canvases=_NUM_CANVASES,
      canvas_size=_CANVAS,
      n_soft=_N_SOFT,
      image_size=_IMAGE_SIZE,
      patch_size=_PATCH_SIZE,
      pooling_kernel_size=1,   # fast pure-numpy fixed-square patchify
      mode=_DATA_MODE,
      split=split,
      num_examples=num_examples,
      query_max_len=24,
      num_workers=0,
  )


def get_config():
  """Shrunk multimodal SFT config for ChartQA on a single 5090."""
  checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=1000,
      save_on_steps=[10, 25, 50, 100],
      max_to_keep=10,
  )
  return sft_chartqa.build_sft_chartqa_config(
      gemma_model=build_tiny_model(),
      n_soft=_N_SOFT,
      prompt_len=_PROMPT_LEN,
      canvas_size=_CANVAS,
      num_canvases=_NUM_CANVASES,
      train_ds=make_tiny_ds(training=True, split="train", num_examples=256),
      eval_ds=make_tiny_ds(training=False, split="val", num_examples=64),
      num_train_steps=100,
      peak_lr=1e-3,
      warmup_steps=10,
      checkpointer=checkpointer,
      sharding=None,          # single GPU
      init_transform=None,    # random init (no gs:// 26B weights)
      evals={},               # offline eval_main is a separate step
      eval_num_batches=2,
  )
