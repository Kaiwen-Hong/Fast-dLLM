# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""SHRUNK DiffusionGemma SFT config — Sudoku — fits a single RTX 5090 (32GB).

Derived from sft_sudoku_full.py, with the only changes needed to run on one 5090:
  * a TINY, RANDOM-INIT DiffusionGemma (real code + real vocab, but embed_dim=128 /
    4 dense layers instead of the 26B MoE) so it fits in 32GB with no checkpoint load;
  * NO init_transform (no gs:// 26B weights);
  * 100 train steps, checkpoint every 5 steps (captures steps 10/25/50/100);
  * real Kaggle sudoku data (bagz on local disk); small batch / few workers for 30GB RAM.

Success target = the official SFT+eval harness runs end-to-end and train losses decrease.
Task accuracy is NOT expected to rise (tiny random model, 100 steps).
"""

from kauldron import konfig

_VOCAB = 262_144
_PROMPT_LEN = 256
_CANVAS = 256
_BAGZ_DIR = "/home/kaiwen/data/overnight_dgemma/data/sudoku_bagz"

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from gemma.diffusion import _models
  from gemma.gm.nn.gemma4 import _config as gemma_cfg
  from gemma.gm.nn.gemma4 import _modules as gemma_modules
  from gemma.diffusion.hackable_diffusion_adapter.data.sudoku import sudoku_data
  from gemma.diffusion.hackable_diffusion_adapter.hd import hd_gemma_network
  from gemma.diffusion.hackable_diffusion_adapter.hd import sft_model
  from hackable_diffusion import hd
  from hackable_diffusion.kdiff import core
  from hackable_diffusion.lib.training import discrete_loss
  from kauldron import kd
  import optax
  from gemma.diffusion.hackable_diffusion_adapter import safe_writer
# pylint: enable=g-import-not-at-top


def get_config():
  """Shrunk SFT config for Sudoku on a single 5090."""
  cfg = kd.train.Trainer()
  cfg.seed = 42
  cfg.aux = {}
  cfg.aux.vocab_size = _VOCAB

  cfg.aux.corruption_process = hd.corruption.CategoricalProcess.uniform_process(
      num_categories=cfg.ref.aux.vocab_size,
      schedule=hd.corruption.RFSchedule(),
  )
  cfg.aux.prompt_len = _PROMPT_LEN
  cfg.aux.num_canvases = 1
  cfg.aux.canvas_size = _CANVAS
  cfg.aux.peak_lr = 1e-3
  cfg.aux.end_lr = cfg.ref.aux.peak_lr / 10
  cfg.aux.eval_num_batches = 2  # used by ar_eval.make_ar_evals for offline eval_main
  cfg.aux.stop_gradient_from_denoiser_to_encoder = False
  cfg.aux.encoder_loss_weight = 1.0
  cfg.aux.decoder_loss_weight = 1.0

  cfg.aux.sudoku_prompt = (
      "<|turn>system Solve the following Sudoku puzzle. Empty cells are"
      " represented by 0. Output ONLY the solved puzzle immediately as"
      " a 9x9 grid of numbers separated by spaces. Do not include ####,"
      " explanations, or any other text.<turn|>\n<|turn>user"
      " {text}<turn|>\n<|turn>model\n"
  )

  # Tiny random-init backbone, built lazily so konfig keeps it configurable.
  tiny_cfg = gemma_cfg.TransformerConfig(
      num_embed=_VOCAB,          # must match the Gemma4 tokenizer vocab
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
      sliding_window_size=512,   # >= prompt_len + canvas_size so full context is visible
      qk_norm_with_scale=True,
      global_rope_proportion=0.25,
      local_rope_proportion=1.0,
      per_layer_input_dim=0,
      enable_moe=False,          # dense FFW: pure standard XLA ops, no ragged_dot
      vision_encoder=None,
      audio_encoder=None,
  )
  base_network = hd_gemma_network.WrappedDiffusionGemmaNetwork(
      gemma_model=_models.DiffusionGemma_26B_A4B(config=tiny_cfg),
  )
  gemma_network = base_network  # full-weight (no LoRA)

  cfg.model = sft_model.SFTDiffusion(
      x0="batch.canvas",
      prompt="batch.prompt",
      canvas_id="batch.canvas_id",
      canvas_mask="batch.canvas_mask",
      encoder_target="batch.encoder_target",
      encoder_target_mask="batch.encoder_target_mask",
      corruption_process=cfg.ref.aux.corruption_process,
      time_sampler=hd.training.time_sampling.UniformTimeSampler(
          span=hd.jax_helpers.SafeSpan(safety_epsilon=1e-4)
      ),
      gemma_network=gemma_network,
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

  cfg.num_train_steps = 100

  cfg.schedules = {
      "learning_rate": optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=cfg.ref.aux.peak_lr,
          end_value=cfg.ref.aux.end_lr,
          warmup_steps=10,
          decay_steps=cfg.ref.num_train_steps,
      ),
  }

  cfg.optimizer = kd.optim.named_chain(**{
      "clip": optax.clip_by_global_norm(max_norm=1.0),
      "adafactor": optax.scale_by_factored_rms(),
      "decay": optax.add_decayed_weights(weight_decay=1e-4),
      "lr": optax.scale_by_learning_rate(cfg.ref.schedules["learning_rate"]),
  })

  # Save ONLY at the requested eval points. save_on_steps spaces the async orbax
  # saves >=1s apart (unlike save_interval_steps=5, which races on this fast tiny run
  # -> "cannot schedule new futures after shutdown").
  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=1000,          # effectively disabled (> num_train_steps)
      save_on_steps=[10, 25, 50, 100],   # 10/25/50/100% of 100 steps
      max_to_keep=10,
  )

  cfg._konfig_experimental_nofreeze = True  # pylint: disable=protected-access
  cfg.rng_streams = kd.train.RngStreams([
      kd.train.RngStream("default", train=True, eval=True),
      kd.train.RngStream("sampling", train=True, eval=True),
  ])

  # NO init_transform -> random init (real 26B weights don't fit on a 5090).

  cfg.train_ds = sudoku_data.make_sudoku_ds(
      bagz_path=f"{_BAGZ_DIR}/sudoku_train.bagz",
      training=True,
      batch_size=4,
      prompt_len=cfg.ref.aux.prompt_len,
      num_canvases=cfg.ref.aux.num_canvases,
      canvas_size=cfg.ref.aux.canvas_size,
      prompt_template=cfg.ref.aux.sudoku_prompt,
      num_workers=2,
  )
  cfg.eval_ds = sudoku_data.make_sudoku_ds(
      bagz_path=f"{_BAGZ_DIR}/sudoku_eval.bagz",
      training=False,
      batch_size=4,
      prompt_len=cfg.ref.aux.prompt_len,
      num_canvases=cfg.ref.aux.num_canvases,
      canvas_size=cfg.ref.aux.canvas_size,
      slice_stop=200,
      prompt_template=cfg.ref.aux.sudoku_prompt,
      num_workers=2,
  )

  # Inline evals disabled (offline eval is a separate step). Keep training lean.
  cfg.evals = {}

  cfg.writer = safe_writer.SafeMetricWriter()

  return cfg
