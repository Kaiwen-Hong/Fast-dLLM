# ALIGNMENT_NOTES — E2B external twin vs the internal SFT stack

Ledger required by the design doc (§4.2 outcomes, every pristine-file edit,
deltas vs internal, open items). Overnight run 2026-07-07; branch
`dgemma-e2b-test`. Provenance: `response-from-jestki.md` ([internal·doc] /
[internal·reply]).

## §4.2 contract outcomes

| # | mechanism | outcome |
|---|---|---|
| 1 | record schema `image/encoded`/`question`/`answer`; ChartQA=ArrayRecord, toy=Bagz | **ADOPTED** — `data/chartqa/schema.py` (one-file constants), `convert_chartqa.py` (2000 train / 256 val / 10 toy produced), `ParseChartQARecord` verbatim incl. `response_text="The answer is: {answer}"` |
| 2 | kauldron Trainer + FSDP ShardingStrategy | **ADOPTED** — `configs/sft_chartqa_e2b.py` (FSDP no-op on 1 GPU, config internal-portable) |
| 3 | dual loss (NoWeightDiscreteLoss + EncoderARLoss) | **ADOPTED** (builder `sft_chartqa.build_sft_chartqa_config`, weights 1.0/1.0; EncoderARLoss keeps the 2.35B PLE table alive — §3.4 of the doc) |
| 4 | variable image expansion `[\n\n(108), SOI, n×(-2), EOI, \n\n(108)]` | **ADOPTED** — pristine `add_variable_extra_tokens_for_images` (`_token_utils.py:375`) IS the internal fn; E2B tower: patch 16 / pooling 3 / budget 280 → **actual n_soft = 256** (fixed square 768px), 216 padded patch rows per example (C6c precondition, verified in the pipeline smoke) |
| 5 | strip+gather encoder-target alignment | **ADOPTED** — `get_no_mm_indices` pasted verbatim into `_token_utils.py`; `remove_mm_logits` REWRITTEN (Gemma4 ids, dynamic offsets, dual-flow: pre-expanded adapter flow + gm-native legacy flow); `shift_encoder_targets_for_multimodal` pasted into `hd/sft_model.py` + wired when `images is not None`. Property test `tests/e2b_mm_alignment_test.py` PASS (image@start/middle/end + no-image row) |
| 6 | batch>1 vision merge reshape | **ADOPTED inline** in `_merge_mm_embeddings` (`gm/nn/gemma4/_transformer.py`): packed `[1, B·n, D] → [B, n, D]` when B>1; pipeline smoke ran batch=2 |
| 7 | `__getattr__` vision resolution patch | **AVOIDED** — `make_diffusion_e2b(text_only=False)` registers the tower properly; init's dummy-vision call (`_transformer.py:498-500`) creates tower params without images |
| 8 | hardcoded 26B vision tower config | **AVOIDED** — E2B tower comes from `Gemma4_E2B.config.vision_encoder` (d_model 768/16L, clipped_linears=True — differs from internal 26B's 1152/27L/False as expected) |
| 9 | tree-map shielding | **AVOIDED** — `images` annotation on `call_with_self_conditioning` widened to include `PreprocessedVisionInput` |
| 10 | sliding-mask length band-aid | **REPLACED by construction** — the adapter feeds pre-expanded prompts, so all masks are built over the expanded length; no length mismatch arises. [open: add an explicit assert-equal-length unit test] |
| 11 | nn.remat monkey-patches | **ADAPTED** — config-scoped `_enable_gradient_checkpointing()` (pattern from modified `sft_chartqa.py:64-91`), called in `sft_chartqa_e2b.get_config`. REQUIRED on 32GB: T2 b1 no-remat OOM; b1+remat peak 17.9GB, 0.3s/step |
| 12 | TensorBoardOnlyWriter under XM_XID | **ADAPTED** — `cfg.writer = SafeMetricWriter()` (builder); internal swaps theirs at the same seam |
| 13 | trajectory twin naming | **MIRRORED** — `schema.py` carries the trajectory prompt/response templates + geometry (512/128/3) verbatim for the future `trajectory_data.py` twin |

## Pristine-file edits (complete list)

- `diffusion/_models.py` — ADD `DiffusionGemma_E2B` + `make_diffusion_e2b` (audio-strip helper, D2).
- `diffusion/_transformer.py` — `images` annotation widened (PVI union); no behavior change.
- `hd/gemma_checkpointer.py` — `expected_missing` allowlist + coverage report (mirrors the LoRA carve-out; still raises on unexpected model-only keys); NEW `ZeroInitLeaves` + `SequentialInitTransform`.
- `gm/vision/_token_utils.py` — NEW `get_no_mm_indices` (internal verbatim); `remove_mm_logits` rewritten (was `Gemma3Tokenizer` hardcode at `:357` — the NaN-gather root cause; now Gemma4 + dynamic offsets, dual-flow).
- `gm/nn/gemma4/_transformer.py` — batch>1 reshape in `_merge_mm_embeddings`.
- `gm/nn/gemma4/vision/_images.py` — `factorized_posemb` NaN→zeros (ported one-line fix; internal's 4D→3D reshape NOT needed in this flow — posemb here is already `[S,2,dim]`-shaped [open: revisit if the internal 4D case ever appears]).
- Ported wholesale from `discreteGemma-modified` (validated there): `hd/hd_gemma_network.py`, `hd/sft_model.py`, `hd/hd_gemma_ar_state_handler.py`, `eval/text_metric.py`, `eval_main.py` (+chartqa task), `eval/chartqa_eval.py` (scoring swapped to internal-verbatim `score_chartqa`, 5.1%), `data/chartqa/chartqa_data.py` (+V2 internal-parity sources/transforms), `configs/sft_chartqa.py` (builder).

## Deliberate deltas vs internal (documented, accepted)

1. **`\n\n` = 108 confirmed by the actual tokenizer encode** (T0) — the internal
   reply's "tokenizer encodes 203" note is wrong for this vocab; no quirk to
   replicate.
2. **Tokenizer ids: NO drift** (T0 asserted all ids vs the internal table).
3. **l_no_mm uses `n_soft`(=256 actual), not the 280 budget**, in the adapter's
   target shift; the model's `remove_mm_logits` derives l_no_mm from the value
   passed by the gemma forward (`num_mm_tokens_per_image=280`) — both sides are
   self-consistent because logits length defines the gather target; the last 24
   tail positions (canvas PAD tail) drop out of the encoder loss. Negligible;
   matches internal's own budget-vs-actual gap.
4. **Encoder targets use the PLAIN shift** (no image-id invalidation): the
   strip+gather mapping collapses the image block and lands each kept position
   on the correct next-text-token target (property-tested). The
   modified-branch's `invalid_ids` masking is retained in the file for the V1
   pipeline but unused by the V2 internal-parity path.
5. **Eval wall-time reduction** (documented per D5): gate on steps=32 across
   saved ckpts; 64/96 on the best ckpt only; eval batch sized to the 5090.
6. **Fixed-square preprocessing** (768px) — internal preprocessing may keep
   aspect ratio; fixed square guarantees a constant n_soft=256 so grain
   batching and the static `n_soft` config agree. [open: confirm internal's
   resize policy when the internal code lands.]

## Feasibility (T-suite, 2026-07-07)

- T0: ckpts reachable; **5.12B params each; vision + audio towers present; no
  self_conditioner** (AR); trees pt≡it structurally; tokenizer ids match.
- T1: full E2B (text 4.66B) bf16 build + L=1024 diffusion forward on the 5090:
  init 85s, fwd 0.039s steady, peak 11.4GB.
- T2: SFT train step (internal geometry): b1 no-remat OOM (11GB single alloc);
  **b1+remat PASS — 0.3s/step steady, peak 17.9GB**; b2+remat [see
  results/feas/]. Loss at init ≈ 2×ln(262144) as expected.

## Open items for the (internal) coding agent

- Add the assert-equal-length sliding-mask unit test (row 10).
- Confirm internal image resize policy (delta 6) and, if aspect-preserving,
  switch `BuildChartQAInputsV2` to it and re-derive n_soft handling.
- C5 harness: prompt+canvas beyond the 512 sliding window diverges BY DESIGN
  between training (per-position sliding) and the sampler's block-local mask —
  whitelist, don't "fix".
- The `\n\n`(108) vs `203` discrepancy in the internal reply §5 — worth
  flagging back to the internal owner (their note, not their pipeline, is off).
