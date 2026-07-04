# ChartQA VQA on the OFFICIAL DiffusionGemma SFT harness (kauldron)

Replaces the self-contained toy trainer (`vqa_train.py` / `vqa_multimodal_smoke.py`)
with the real kauldron `SFTDiffusion` / `eval_main` harness — the one that has
FSDP sharding and scales to multi-TPU 26B training. The image is threaded through
the harness end-to-end (training encoder prefill **and** the AR sampler prefill).

## What was added

**New files**
- `configs/sft_chartqa.py` — full 26B multimodal config (analogue of
  `sft_sudoku_full.py`): real `DiffusionGemma_26B_A4B(text_only=False)` (release
  vision tower), FSDP sharding, `nn.remat` gradient checkpointing, release
  checkpoint via `GemmaDiffusionCheckpointLoader`, release vision preprocessing
  (pooling=3, 256 soft tokens/image). Exposes `build_sft_chartqa_config(...)`.
- `configs/sft_chartqa_tiny.py` — **inherits** `sft_chartqa.build_sft_chartqa_config`
  with a tiny random-init model + tiny vision encoder (`text_only=False`), fast
  pure-numpy fixed-square preprocessing (16 soft tokens), no checkpoint/sharding,
  100 steps, ckpts @10/25/50/100. Fits one RTX 5090.
- `configs/sft_chartqa_grounding_tiny.py` — same harness/model, but the synthetic
  color-grounding data (answer determined ONLY by the image). 400 steps.
- `data/chartqa/chartqa_data.py` — grain pipeline (analogue of `sudoku_data.py`):
  HF `ahmed-masry/ChartQA` source + synthetic color source; emits fixed-shape
  `patches`/`positions_xy` + prompt (with `-2` placeholders) + canvas + encoder
  targets. Pure-numpy patchify (verified bit-exact vs `_images.patchify`).
- `gpu_smoke/validate_vision_grounding.py` — the grounding validation.

**Image threading edits (marked `# IMAGE` in the vendored gemma)**
- `hd/hd_gemma_network.py`: `encoder_call` forwards `images` to `gemma_model`;
  `prefill_kv_cache_with_encoder` accepts `images` and routes it via the
  conditioning dict.
- `hd/sft_model.py`: `sft_encode(images=...)`; `SFTDiffusion` gains `patches` /
  `positions_xy` kontext keys + `n_soft`, builds a `PreprocessedVisionInput`
  (`_build_vision_input`) and merges it into the encoder prefill only (the
  denoiser/canvas stays text-only); `GemmaSamplingEvaluator` packs images into
  the sampler conditioning.
- `hd/hd_gemma_ar_state_handler.py`: `init_ar_state` reads `conditioning['images']`
  → image-conditioned AR sampler prefill.
- `eval/text_metric.py`: strip negative (`-2`) placeholders before detokenizing
  the sampling summary (otherwise the tokenizer crashes on multimodal prompts).

All edits default to `images=None` → text-only sudoku/pubmedqa is unchanged.
Regression: `sft_model_test` (12), `hd_gemma_ar_state_handler_test` (23),
`mask_helpers_test` (21), `data_test` (8), `text_metric_test` (2) all pass.

**Vision batch constraint.** The Gemma4 vision merge (`merge_flat_embeddings`
vmaps a size-1 vision batch against the text batch) is single-sequence, so the
multimodal configs use per-process `batch_size=1`. FSDP still shards params for
multi-accelerator training; data-parallelism gives global batch = #devices.

## Run (canonical commands)

```bash
conda activate dgemma-jax
cd discreteGemma/gemma
export PYTHONPATH=$PWD HF_HOME=/home/kaiwen/data/huggingface
export XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_disable_hlo_passes=constant_folding"

# train (tiny, one 5090)
python -m kauldron.main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_chartqa_tiny.py \
  --cfg.workdir=$WORK

# eval (canonical eval_main; AR diffusion sampling, image-conditioned)
python -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
  --cfg=gemma/diffusion/hackable_diffusion_adapter/configs/sft_chartqa_tiny.py \
  --step=100 --eval_names=sample_ar_steps32 \
  --cfg.workdir=$WORK --cfg.eval_ds.batch_size=1
```

## Results (RTX 5090)

**Training (real ChartQA, `sft_chartqa_tiny`)** — kauldron.main, exit 0, ckpts
@10/25/50/100. Train loss total 24.56 → 4.64 (diffusion 12.1 → 1.09). Vision
encoder present in the param tree and receiving gradients.

**Eval (`eval_main`)** — exit 0, image-conditioned AR sampling ran end-to-end
(`sample_ar_steps32`, denoising metrics emitted, prompt+samples detokenized).

**Held-out diffusion loss @ checkpoints (harness `SFTDiffusion`)**
- real ChartQA: step 10/25/50/100 = 10.49 / 6.81 / 2.27 / 0.97 (monotone ↓)
- color grounding: step 40/100/200 = 5.59 / 1.67 / 0.70 (monotone ↓)

**Vision grounding — DECISIVE, through the harness** (`validate_vision_grounding.py`,
color task, probe the answer from a ~fully-corrupted canvas so it can only come
from image+prompt):

| metric @ step 200 | correct image | wrong image |
|---|---|---|
| color-restricted answer acc | **75.0%** (chance 25%) | **0.0%** |
| answer top-1 acc | 62.5% | 0.0% |
| answer CE | 1.34 | 3.64 (gap **+2.30**) |

Weight-scramble ablation: re-randomising ONLY the trained vision-encoder weights
collapses correct-image color acc **75.0% → 0.0%**. ⇒ the SPECIFIC harness-trained
vision weights causally drive the answer — **VISION-WEIGHTS-CAUSAL: True**.

**Real ChartQA at this scale**: correct ≈ wrong (color acc 0/0, CE gap +0.007) —
the tiny random model does NOT read real charts. This is a SCALE limit (real
chart reading needs the 26B + release weights + much more training), NOT a broken
pipeline — the color task proves the same harness IS causally grounded when the
task is learnable at 5090 scale.

## Evaluation: inference + ChartQA metric

Inference is the official AR diffusion sampler in `eval_main` (generates the
answer canvas `samples`, conditioned on image+question). The ChartQA accuracy
metric is now wired in:

- `eval/chartqa_eval.py` — `ChartQARelaxedAccuracy` (ChartQA relaxed correctness:
  numeric answers within 5% relative tolerance, else normalized exact match) and
  `ChartQAExactMatch`. Both are `BaseSimpleTextMetric`s that detokenize with the
  LOCAL `Gemma4Tokenizer` (offline, no GCS).
- `data/chartqa/chartqa_data.py` — eval split emits fixed-length ground-truth
  `answer_tokens` (like sudoku's `solution_tokens`).
- `eval_main.py` — `--task=chartqa` added; reports
  `chartqa_relaxed_accuracy` + `chartqa_exact_match` next to the denoising
  metrics and detokenized samples.

Run:
```bash
python -m gemma.diffusion.hackable_diffusion_adapter.eval_main \
  --cfg=.../configs/sft_chartqa_tiny.py \
  --task=chartqa --step=100 --eval_names=sample_ar_steps32 \
  --cfg.workdir=$WORK --cfg.eval_ds.batch_size=1 --cfg.aux.eval_num_batches=8
```

**The metric discriminates** (5090, `eval_main --task=chartqa`, AR sampling):
- tiny random model on real ChartQA @step100: relaxed_accuracy = **0.0**,
  exact_match = 0.0 (expected — like sudoku Phase 1).
- grounded color model @step200: relaxed_accuracy = **0.5**, exact_match = 0.5;
  samples like `"What is the color? yellow yellow yellow yellow"` (correct),
  `"blue blue blue blue"` (correct). ⇒ the sampler generates the image-derived
  answer and the metric scores it correctly (0.5 >> chance and >> the 0.0
  baseline). Evaluation loop = real and working.

## Known host gotcha

On this host, orbax's async checkpoint save intermittently deadlocks (`_SignalingThread.join()
blocking the main thread`) on fast tiny runs, stalling ~step 190-200 after a save.
The grounding run was validated at ckpt_200 (well-converged). If it recurs, reduce
`save_on_steps` frequency or set `kd.ckpts.Checkpointer(fast=False)`.
