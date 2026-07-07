# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""E2B multimodal SFT on ChartQA — AR-checkpoint init, single RTX 5090.

Design doc §5 B5 (0706-design-doc-prompts.md). Internal-parity geometry
(prompt 512 / canvas 2x256 / n_soft 256, budget 280) over local ArrayRecord
tf.Example shards; DiffusionGemma_E2B initialized from the GEMMA4_E2B AR
checkpoint (self_conditioner fresh, FFW output projection zero-init).

Runtime knobs (env vars, read at get_config time):
  DGEMMA_E2B_VARIANT  = pt | it            (default it)
  DGEMMA_E2B_BATCH    = per-process batch  (default 4; set from the T2 sweep)
  DGEMMA_E2B_STEPS    = train steps        (default 200)

Launch:
  DGEMMA_E2B_VARIANT=it python -m kauldron.main \
      --cfg=.../configs/sft_chartqa_e2b.py --cfg.workdir=<workdir>
"""

import glob as _glob
import os as _os

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma.diffusion import _models
  from gemma.diffusion.hackable_diffusion_adapter.data.chartqa import chartqa_data
  from gemma.diffusion.hackable_diffusion_adapter.hd import gemma_checkpointer
  from kauldron import kd
  import jax.numpy as jnp
# pylint: enable=g-import-not-at-top

from gemma.diffusion.hackable_diffusion_adapter.configs import sft_chartqa as _base

# ---- internal-parity geometry [internal·reply; schema.py] ----
_PROMPT_LEN = 512
_CANVAS = 256
_NUM_CANVASES = 2
_N_SOFT = 256          # actual soft tokens (fixed square, pooling 3, budget 280)
_ANSWER_LEN = 32

_DATA_ROOT = "/home/kaiwen/data/dgemma_e2b/chartqa"
_CKPT_ROOT = "/home/kaiwen/data/dgemma_e2b/ckpts"
_RESULTS = "/home/kaiwen/Desktop/research/Fast-dLLM/discreteGemma/results"


def get_config():
  """E2B ChartQA SFT (variant/batch/steps via env — see module docstring)."""
  variant = _os.environ.get("DGEMMA_E2B_VARIANT", "it")
  assert variant in ("pt", "it"), variant
  batch = int(_os.environ.get("DGEMMA_E2B_BATCH", "4"))
  steps = int(_os.environ.get("DGEMMA_E2B_STEPS", "200"))
  save_steps = sorted({max(1, steps // 4), max(1, steps // 2), steps})

  # Gradient checkpointing — required for the full-scale train step on 32GB
  # (T2 feasibility: b1 no-remat OOM; b1+remat peak 17.9GB).
  _base._enable_gradient_checkpointing()  # pylint: disable=protected-access

  train_paths = tuple(sorted(_glob.glob(f"{_DATA_ROOT}/human_train-*.arrayrecord")))
  val_paths = tuple(sorted(_glob.glob(f"{_DATA_ROOT}/human_val-*.arrayrecord")))
  assert train_paths and val_paths, f"run convert_chartqa.py first ({_DATA_ROOT})"

  gemma_model = _models.make_diffusion_e2b(
      text_only=False,  # vision stays; audio stripped inside the helper (D2)
      dtype=jnp.bfloat16,
  )

  init_transform = gemma_checkpointer.SequentialInitTransform(
      transforms=(
          gemma_checkpointer.GemmaDiffusionCheckpointLoader(
              path=f"{_CKPT_ROOT}/gemma4-e2b-{variant}",
              expected_missing=("self_conditioner",),
              coverage_json_path=f"{_RESULTS}/{variant}/load_coverage_train.json",
          ),
          # sc no-op-but-trainable init: zero ONLY the FFW output projection
          # (all-zeros would be a permanent zero-gradient saddle — doc §5 B2).
          gemma_checkpointer.ZeroInitLeaves(
              substrings=("self_conditioner/ffw/linear",),
          ),
      ),
  )

  train_ds = chartqa_data.make_chartqa_records_ds(
      training=True,
      batch_size=batch,
      paths=train_paths,
      fmt="arrayrecord",
      prompt_len=_PROMPT_LEN,
      num_canvases=_NUM_CANVASES,
      canvas_size=_CANVAS,
      n_soft=_N_SOFT,
      num_workers=2,
  )
  eval_ds = chartqa_data.make_chartqa_records_ds(
      training=False,
      batch_size=batch,
      paths=val_paths,
      fmt="arrayrecord",
      prompt_len=_PROMPT_LEN,
      num_canvases=_NUM_CANVASES,
      canvas_size=_CANVAS,
      n_soft=_N_SOFT,
      answer_len=_ANSWER_LEN,
      num_workers=2,
  )

  cfg = _base.build_sft_chartqa_config(
      gemma_model=gemma_model,
      n_soft=_N_SOFT,
      prompt_len=_PROMPT_LEN,
      canvas_size=_CANVAS,
      num_canvases=_NUM_CANVASES,
      train_ds=train_ds,
      eval_ds=eval_ds,
      num_train_steps=steps,
      # Pretrained init: 1e-3 (random-init tiny recipe) would damage the
      # backbone — design doc B5 default 3e-5 -> 3e-6 cosine, warmup 20.
      peak_lr=3e-5,
      warmup_steps=20,
      checkpointer=kd.ckpts.Checkpointer(
          save_interval_steps=10_000,  # effectively disabled
          save_on_steps=save_steps,
          max_to_keep=10,
      ),
      # FSDP strategy — no-op on one GPU, keeps the config internal-portable.
      sharding=kd.sharding.ShardingStrategy(
          params=kd.sharding.FSDPSharding(),
          opt_state=kd.sharding.FSDPSharding(),
      ),
      init_transform=init_transform,
      evals={},  # offline eval via eval_main --task=chartqa
      eval_num_batches=4,
  )
  return cfg
