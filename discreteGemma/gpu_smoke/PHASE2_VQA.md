# Phase 2 — VQA (ChartQA) on DiffusionGemma, vision input

Goal: set up DiffusionGemma to train a VQA task (ChartQA) with **image input**, on one RTX 5090.
The diffusion SFT adapter has **no image path at all**, so this is new wiring.

## What works (verified on GPU)
`vqa_multimodal_smoke.py` — a real ChartQA example (image + question + answer) is:
1. **loaded** via `datasets.load_dataset("ahmed-masry/ChartQA")` (cols: `image` PNG bytes, `query`, `label`, `type`).
2. **preprocessed** into a `PreprocessedVisionInput` via
   `gemma.gm.nn.gemma4.vision._preprocessing.preprocess_and_patchify([img], patch_size, max_soft_tokens, pooling_kernel_size)`
   → `(patches, positions_xy, soft_token_counts)`.
3. fed through a **tiny random-init DiffusionGemma WITH a tiny vision encoder** (`text_only=False`,
   `config.vision_encoder=VisionEncoder(d_model=64,num_layers=2,...)`) via the diffusion multimodal
   forward `call_with_self_conditioning(tokens, images=PreprocessedVisionInput, ...)`.
4. The forward **runs end-to-end on the 5090** and returns `logits (1, L, 262144)` — i.e. the image
   flows image→patchify→vision encoder→merge→diffusion transformer→logits. Model = 18.3M params,
   vision encoder confirmed present in the param tree.

## Non-obvious gotchas discovered (all handled in the smoke)
- The diffusion `call_with_self_conditioning` has a **WRONG `images` type annotation**
  (`UInt8['*B N H W C']`) but the code actually consumes a `PreprocessedVisionInput`. Wrap calls in
  `with kauldron.ktyping.config.Config(typechecking_enabled=False):` to bypass the bad check.
- Soft-token positions are marked with `vision._encoder.TOKEN_PLACEHOLDER` (== -2). The number of -2
  tokens in the sequence must equal `sum(soft_token_counts)` (variable per image aspect ratio).
- `add_extra_tokens_for_images` keys off **Gemma3** `IMAGE_PLACEHOLDER` (255999) which collides with
  Gemma4 `START_OF_IMAGE` (255999); Gemma4 `IMAGE_PLACEHOLDER` is 258880. Build the `[BOS, <soi>,
  -2*N, <eoi>, ...text...]` sequence manually instead of relying on that helper.

## The NaN — root-caused and FIXED
The combined multimodal forward produced a deterministic NaN. Root cause (found via JAX_DEBUG_NANS
after neutralising an intentional-but-masked `jnp.nan` in `factorized_posemb`):
**`_token_utils.remove_mm_logits` — the Gemma4 multimodal path hard-codes `Gemma3Tokenizer`
special-token IDs** (there's even a `# TODO: This value should be propagated from the model`), so on
a Gemma4 model its `jnp.take_along_axis` gathers with out-of-range indices → NaN, which then
propagates. The vision encoder, merge, and transformer blocks are all fine.

Two small upstream patches (in the vendored `discreteGemma/gemma`, both marked `# PATCH:`):
1. `gm/nn/gemma4/vision/_images.py` `factorized_posemb`: `jnp.where(nan, jnp.nan, ...)` → `0`
   (invalid/padding positions are masked anyway; this only removed a debug_nans false-positive).
2. `gm/vision/_token_utils.py` `remove_mm_logits`: make it **identity** (don't drop image logits;
   image positions are masked out in the loss instead). This removes the bad gather → finite logits.

## VQA training — DONE (train 100 steps + eval @10/25/50/100%, metrics improving) ✅
`vqa_train.py` — self-contained: tiny DiffusionGemma (128-d, 4 layers) + tiny vision encoder =
35.4M params, real ChartQA (image→square→PreprocessedVisionInput, question→prompt with 16 vision
soft-tokens, answer→diffusion canvas), hackable_diffusion uniform corruption + discrete diffusion
loss on the answer canvas. Trains 100 steps (batch=1, cycling 100 examples), saves params at
10/25/50/100, evaluates held-out diffusion loss at each. **Result (loss decreases = metric improving):**
```
step  10 (10%): eval_loss = 9.37
step  25 (25%): eval_loss = 4.31
step  50 (50%): eval_loss = 3.19
step 100 (100%): eval_loss = 2.72     improving (monotone down): True
```
Checkpoints: `/home/kaiwen/data/overnight_dgemma/xp_chartqa/ckpt_{10,25,50,100}.pkl`.

## Run
```bash
conda activate dgemma-jax
python vqa_multimodal_smoke.py   # forward-only sanity (shapes)
python vqa_train.py              # full: 100-step VQA diffusion training + eval @10/25/50/100%
```

## Note on the official SFT harness path (not used here)
This is a self-contained trainer (like `sft_smoke.py`), not the kauldron `SFTDiffusion`/`eval_main`
harness. To fold VQA into that harness you'd still thread `images` through
`WrappedDiffusionGemmaNetwork.encoder_call` + `sft_encode` + an `images` kontext key on
`SFTDiffusion`, and build a ChartQA kauldron data pipeline — but the modeling blocker (the NaN) is
solved and the training objective + metric-improvement are demonstrated here.

## ⚠️ Verification: does the vision encoder actually contribute? (ablations)
A decreasing loss does NOT prove the image is used — the model could just overfit text/answer-prior.
Controlled ablations on ckpt_100 (`vqa_ablation.py`, `vqa_textonly.py`):

| condition (held-out diffusion loss @ step100) | loss |
|---|---|
| VQA, **correct** image | 2.94 |
| VQA, **wrong** image (swap another chart) | **2.94** (≈identical) |
| VQA, **zero** image (degenerate) | 3.30 |
| **text-only** baseline (no image) | 3.01 |
| gradient norm: vision params 1.4e-2 vs text params 1.1e+1 (≈800× smaller) | |

**Verdict:** the vision pipeline is **functional & differentiable** (finite forward, non-zero vision
grads, real-vs-zero image changes the output logits, RMS 0.93) — the NaN fix and wiring are real.
BUT the model **does NOT use image content**: swapping the correct chart for a wrong one gives an
identical loss (2.9378 vs 2.9379; correct<wrong on only 12/20), and a text-only model reaches nearly
the same loss (3.01). So the loss drop is ~95% text/answer-prior, exactly the overfitting concern.
This is expected for a 35M random-init model trained 100 steps on 100 examples (same reason sudoku
accuracy is 0%) — real visual grounding needs pretrained weights + far more training (TPU).
**Correct metric for "vision works":** the counterfactual gap `loss(wrong image) - loss(correct image)`
(currently ≈0; should be ≫0 once the model reads charts), NOT raw loss decrease.
