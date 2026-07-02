# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""VISION-GROUNDING validation config (synthetic color task) — one RTX 5090.

Same official SFT harness + same tiny model as ``sft_chartqa_tiny.py``, but the
data is the SYNTHETIC color-grounding task (``mode='color'``): a solid-colour
image, the FIXED question ``"What is the color?"``, and the colour name as the
answer.  Because the question carries no information, the answer is determined
ONLY by the image -> if training/eval loss drops here, the vision encoder MUST
be doing the work.  This lets us PROVE, through the official harness, that the
vision path plays a causal role (real ChartQA cannot show this at tiny/random
scale -- that is a scale limit, not a broken pipeline).

Trains 300 steps, checkpoints at 10/25/50/100% (steps 30/75/150/300).  Pair with
``gpu_smoke/validate_vision_grounding.py`` for the counterfactual (correct vs
wrong image) gap + weight-scramble ablation on the harness checkpoints.

Run:
    python3 -m kauldron.main \
        --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_chartqa_grounding_tiny.py \
        --cfg.workdir=<WORKDIR>
"""

from gemma.diffusion.hackable_diffusion_adapter.configs import sft_chartqa
from gemma.diffusion.hackable_diffusion_adapter.configs import sft_chartqa_tiny
from kauldron import konfig

# Tight canvas so the (repeated-colour) answer FILLS the canvas -- the diffusion
# loss then trains on image-dependent tokens rather than EOS padding.
_CANVAS_G = 6

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma.diffusion.hackable_diffusion_adapter.data.chartqa import chartqa_data
  from kauldron import kd
# pylint: enable=g-import-not-at-top


def _make_color_ds(training, num_examples, color_seed):
  return chartqa_data.make_chartqa_ds(
      training=training,
      batch_size=1,
      prompt_len=sft_chartqa_tiny._PROMPT_LEN,
      num_canvases=sft_chartqa_tiny._NUM_CANVASES,
      canvas_size=_CANVAS_G,
      n_soft=sft_chartqa_tiny._N_SOFT,
      image_size=sft_chartqa_tiny._IMAGE_SIZE,
      patch_size=sft_chartqa_tiny._PATCH_SIZE,
      pooling_kernel_size=1,
      mode="color",
      num_examples=num_examples,
      color_seed=color_seed,
      query_max_len=8,
      num_workers=0,
  )


def get_config():
  """Vision-grounding SFT config (synthetic color) on a single 5090."""
  checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=10_000,
      save_on_steps=[40, 100, 200, 400],  # 10/25/50/100% of 400
      max_to_keep=10,
  )
  return sft_chartqa.build_sft_chartqa_config(
      gemma_model=sft_chartqa_tiny.build_tiny_model(),
      n_soft=sft_chartqa_tiny._N_SOFT,
      prompt_len=sft_chartqa_tiny._PROMPT_LEN,
      canvas_size=_CANVAS_G,
      num_canvases=sft_chartqa_tiny._NUM_CANVASES,
      train_ds=_make_color_ds(training=True, num_examples=512, color_seed=0),
      eval_ds=_make_color_ds(training=False, num_examples=128, color_seed=7),
      num_train_steps=400,
      peak_lr=1e-3,
      warmup_steps=20,
      checkpointer=checkpointer,
      sharding=None,
      init_transform=None,
      evals={},
      eval_num_batches=4,
  )
