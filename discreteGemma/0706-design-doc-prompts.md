# 0706 Design Doc & Agent Task Book — DiffusionGemma-E2B: AR-init + Vision + Trainable
### (external twin of the internal SFT stack)

**rev 0707-final · status: ACTIVE — ALL decisions D1–D5 answered; Phase-A pre-flight (T0) run 2026-07-07 with all findings positive (§10); OVERNIGHT AUTONOMOUS MODE AUTHORIZED by Kaiwen (2026-07-07): STOP gates become log-and-proceed; remaining open issues are highlighted for the coding agent to resolve during development · executor: Claude Code agent on this PC (RTX 5090) · owner: Kaiwen**

> **Self-contained task book.** Execute against THIS file only. Every fact you need is inlined
> with a provenance tag; do not fetch project strategy docs or the web for project claims.
> If on-disk code contradicts an inlined fact, that is a BLOCKER (log + surface), not a license
> to improvise.
>
> Provenance tags: **[verified·code]** = read at file:line in this workspace (2026-07-06/07).
> **[internal·doc]** = the internal comparison doc; **[internal·reply]** = the internal agent's
> Q-D3/Q-D5 answers — BOTH archived verbatim in `discreteGemma/response-from-jestki.md` (read it
> for the full pasted sources: parse fns, `get_no_mm_indices`,
> `shift_encoder_targets_for_multimodal`, `score_chartqa`, trajectory metrics, prompt templates).
> **[derived]** = arithmetic from verified config values. **[decision Dn]** = human decision,
> §10. **[verify-PhaseX]** = confirm during that phase before relying on it.

---

## 0. Never-forget invariants (re-read after every context compaction)

1. **The WOD hypothesis is NOT tested here.** Context hypothesis: "with most weights initialized
   from the 2B AR checkpoint, a 2B diffusion model can perform comparably on WOD after finite
   adaptation." This milestone builds and validates the *mechanics* on ChartQA. Never claim the
   hypothesis is supported/refuted from anything produced here.
2. **Run-then-report.** Every number in `reports/REPORT.md` must be emitted by executed code into
   `results/*.json` and templated in. No invented or remembered numbers.
3. **No monkey-patching.** The internal stack ships 6 monkey-patches (§4.2). This build implements
   the same *semantics* as inline, committed code edits. Sole sanctioned exception: the
   config-scoped remat wrapper (§5 B5), which follows the existing pattern at
   `discreteGemma-modified/.../configs/sft_chartqa.py:64-91` and is applied inside a config, not at
   import time.
4. **`discreteGemma-modified/` is READ-ONLY reference.** All edits happen in `discreteGemma/`
   (the clean tree) and only in files authorized in §6. Any other file → BLOCKER first.
5. **Human-owned decisions** (§10: PT-vs-IT, audio scope, eval config, schema field names) —
   never change silently. Pre-set defaults apply only where §10 marks them as defaults.
6. **Stop-and-log, don't improvise.** Spec ambiguity → entry in `reports/BLOCKERS.md`, skip, keep
   working the queue. In interactive sessions, surface structural ambiguity to Kaiwen directly.
7. After compaction: re-read this file top-to-bottom, then `reports/QUEUE.md`, then resume.

---

## 1. Goal, hypothesis, staging

**Goal (one sentence):** Build `DiffusionGemma_E2B` (= `Gemma4_E2B` + `DiffusionMixin`), initialize
every backbone + vision weight from the `GEMMA4_E2B` AR checkpoint (self-conditioning fresh, with
zero-init output projection), thread visual inputs through the hackable-diffusion SFT harness for
**training and sampling**, and pass the success criteria in §7 on ChartQA — with the code
structured as an **external twin of the internal SFT stack** (§4) so the follow-up internal WOD
trajectory run is a data/config swap, not a rewrite.

**Why ChartQA:** internal already has `trajectory_data.py` / `sft_trajectory.py` analogues for WOD
[internal·doc]; ChartQA is the open-data mechanics proxy. Mirror internal file/class naming so the
trajectory swap later is mechanical (`chartqa_data.py` ↔ internal file of the same name).

**Two codebases in this repo — know which is which:**
- `discreteGemma/` — pristine upstream framework + our clean additions. **This is the edit target.**
- `discreteGemma-modified/` — earlier VQA/ChartQA experiment with validated vision threading and
  known bug fixes. **Port source, read-only.**

---

## 2. Environment & repo facts

- Machine: RTX 5090 (32 GB VRAM, sm_120), host RAM ~30 GB usable. **[verified·code]**
- Conda env **`dgemma-jax`** (python 3.12, jax 0.6.2 cuda12). Activation hook
  `zz_nvidia_libs.sh` sets `LD_LIBRARY_PATH` — without it JAX **silently falls back to CPU**.
  Verify with `discreteGemma/gpu_smoke/gpu_check.py` (must print `backend gpu`).
- Run pattern (copy from `discreteGemma/gpu_smoke/run_sudoku_train.sh`):
  `PYTHONPATH=<repo>/discreteGemma/gemma`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`, `TF_CPP_MIN_LOG_LEVEL=2`.
- Big artifacts (datasets, ckpts, logs, results) → `/home/kaiwen/data/dgemma_e2b/` — NOT in the repo.
- Git: work on branch **`dgemma-e2b-test`** (created and checked out by Kaiwen, 2026-07-07 —
  decision D4). One commit per phase gate (§8). Never commit datasets/ckpts.
- Host gotcha [verified in prior runs]: orbax **async checkpoint save can deadlock** on fast tiny
  runs → use sparse `save_on_steps` (already the pattern in `configs/sft_sudoku_tiny.py:154-158`)
  or `kd.ckpts.Checkpointer(fast=False)` if a run stalls at ~a save step with GPU at 0%.
- Checkpoints registry **[verified·code `gm/ckpts/_paths.py:68-74`]**:
  `GEMMA4_E2B_PT = gs://gemma-data/checkpoints/gemma4-e2b-pt`, `GEMMA4_E2B_IT = ...-it`.
  **[decision D1: BOTH variants in scope]** — fetch + inspect + load-test + train/eval both;
  everything parameterized by `ckpt_variant ∈ {pt, it}`; budget ~2×10 GB disk under
  `/home/kaiwen/data/dgemma_e2b/ckpts/`.
  **gs:// reachability from this PC is UNVERIFIED** → Phase A step 1. No access = BLOCKER; fallback:
  develop all blocks against a random-init `DiffusionGemma_E2B` and mark the real-weight acceptance
  tests `pending-ckpt` (do NOT fake them).
- Sizing **[derived from `gm/nn/gemma4/_gemma4.py:122-156`]**: text params ≈ 0.40 B embed
  (262144×1536, tied decode) + ≈1.80 B layers (15 layers FFW hidden 6144 + 20 KV-shared layers FFW
  hidden 12288 + attention) + ≈2.35 B PLE table (262144 × 35 layers × 256) + ~0.02 B PLE projection
  ≈ **4.6 B text**, + vision (+audio) towers. bf16 text ≈ 9.2 GB. With adafactor-style factored
  stats + remat + small per-call batch this fits 32 GB. Loader stages through CPU
  (`cheaply_load_params`) — watch the 30 GB host RAM; keep the reference-compare test streaming/CPU.

---

## 3. Verified code facts — do NOT re-derive; if contradicted, BLOCKER

### 3.1 Model & class
- Only `DiffusionGemma_26B_A4B` exists; it is `Gemma4_26B_A4B + DiffusionMixin` plus one new
  module `self_conditioner`, and sets `keep_last_prefill_kv=True`
  **[verified·code `diffusion/_models.py:21-42`]**. `DiffusionGemma_E2B` does not exist → B0.
- `SelfConditioning` = `pre_norm(RMSNorm) → FeedForward(features, hidden) → add to canvas
  embeddings → post_norm(RMSNorm, no scale)` **[verified·code `diffusion/_transformer.py:53-78`]**.
  Note: even a no-op FFW does not make the diffusion forward identical to the AR forward — the
  post_norm renormalizes canvas embeddings (`:76-78`). Expected; not a bug.
- Flax gotcha: `self_conditioner` params are only created when init goes through
  `call_with_self_conditioning` **[verified in gpu_smoke runs]** — init via that method.
- Plain `__call__` never touches `self_conditioner`; running it with the full (sc-containing)
  param tree works **[verified: `gpu_smoke/tiny_dgemma_smoke.py` base-AR forward]** → enables the
  AR-equivalence acceptance test (§7 C1).

### 3.2 Gemma4_E2B config **[verified·code `gm/nn/gemma4/_gemma4.py:36-156`]**
- `text_only: bool = True` **default strips BOTH vision_encoder and audio_encoder** (`:44,51-58`).
  Multimodal work must instantiate `text_only=False`; audio must then be removed explicitly
  (config `dataclasses.replace(..., audio_encoder=None)`) per [decision D2: audio OUT — confirmed 2026-07-07].
- E2B: embed_dim 1536, 35 layers, hidden 6144, heads 8 / kv 1 / head_dim 256, vocab 262144,
  final_logit_softcap 30.0, attention pattern (4×LOCAL_SLIDING, 1×GLOBAL), **sliding_window_size
  512**, `per_layer_input_dim=256` (PLE), KV-cache sharing `frac_shared_layers=20/35` with
  **doubled FFW hidden (12288) on shared layers**, `vision_encoder=VisionEncoder(use_clipped_linears=True)`,
  `audio_encoder=ConformerConfig()`, `use_bidirectional_attention=None` (26B uses `'vision'`).
- **First-ever pairings** in this build (integration risks, not bugs): DiffusionSampler cache
  surgery × KV-cache sharing (exercised by C4 eval and C5 inference — if it fails, diagnose cache
  layout, do NOT misread as a load bug); DiffusionMixin × PLE-256 (see 3.4).

### 3.3 Checkpoint loader **[verified·code `hd/gemma_checkpointer.py`]**
- `GemmaDiffusionCheckpointLoader` (a `kd.ckpts.InitTransform`,
  `gemma_param_path='gemma_network.gemma_model'`) exists and is used by
  `configs/sft_sudoku_full.py:219`. Core: `_remap_and_match_params` — flatten-and-match with
  dynamic `'/w'` strip; ckpt-only keys → warning + discard; **non-LoRA model-only keys → raises
  KeyError**. Loading an AR ckpt (no `self_conditioner`) into the diffusion model **hard-fails by
  design today** → B1 adds an `expected_missing` allowlist mirroring the existing `/lora/`
  carve-out, plus a machine-readable coverage report.
- `cheaply_load_params` stages the ckpt through CPU (`ocp.PyTreeCheckpointer` metadata → empty CPU
  arrays → restore → dtype/sharding put). Reuse; do not rewrite.

### 3.4 Diffusion forward & PLE bypass **[verified·code]**
- `call_with_self_conditioning` passes `ignore_ple_tokens=True` (`diffusion/_transformer.py:145`);
  in `Embedder.encode_per_layer_input` that branch returns the **projection only — the
  per-layer-input embedding TABLE lookup is skipped**, and the `(x+y)*rsqrt(2)` combine is skipped
  too (`gm/nn/gemma4/_modules.py:169-200`). The PLE table ≈ **2.35 B params — the single largest
  weight block in E2B — is unused by the denoiser pass**. The **encoder AR pass** (plain
  `__call__`, default `ignore_ple_tokens=False`) DOES use it.
  ⇒ **EncoderARLoss must stay ON** (framework default, weight 1.0) or the PLE table is dead weight;
  never disable it without [decision].
- Softcap 30.0 applied inside `call_with_self_conditioning` (`diffusion/_transformer.py:182-184`) —
  same in train and inference paths; not a consistency confound.
- Stale annotation: `images: UInt8[...]` on `call_with_self_conditioning`
  (`diffusion/_transformer.py:91`) rejects `PreprocessedVisionInput` under ktyping → B3 fixes the
  annotation inline (do NOT copy the old `typechecking_enabled=False` wrap, and do NOT copy the
  internal tree-map shielding patch).

### 3.5 Sampler semantics (needed for C5 and inference) **[verified·code `diffusion/_sampler.py`]**
- Entropy-bound acceptance: sort by entropy asc, accept `(cumsum − sorted) ≤ entropy_bound`
  (`:152-172`); non-accepted positions **renoise to uniform-random tokens** (`:174-182`).
- **Temperature shaping happens before sampling AND before the logits are returned**:
  `SampleStepOutput.logits` are the *shaped* logits (`:575-599`). The C5 harness must compare
  **pre-shaping** logits — take them from the transformer output, not from `SampleStepOutput`.
- **The EVAL path does not use the annealing shaper at all** [internal·reply + verified·code
  `eval/ar_eval.py:31-90`]: `make_ar_evals` builds `hd.sampling.DiscreteDDIMStep` with **constant
  temperature 0.7, bf16 logits, no early stopping**, over `AR_DENOISING_STEPS = [32, 64, 96]` —
  identical constants internal and external, so **no sampler porting is needed**. C5's "inference
  mode" = THIS `GemmaSamplingEvaluator`/ar_eval path (compare pre-temperature logits); the gm
  `DiffusionSampler` + `AnnealingTemperatureShaper` (`_sampler.py`) is a different, non-eval path.
- Block-local sliding mask for LOCAL_SLIDING layers gives **all canvas tokens the same context
  window** (`:753-821`, esp. `:761-770`) — a known approximation vs training's per-position
  sliding. With E2B window 512: keep the C5 toy example `prompt+canvas < 512` so the two coincide;
  longer sequences will diverge **by design** (document, don't "fix").
- Per-block commit: `append_tokens_to_cache` with causal mask (`:604-648`);
  `keep_last_prefill_kv=True` matters for index alignment (`diffusion/_models.py:30-31`).
- Early stop: argmax-stability and/or mean-entropy stoppers (`diffusion/_early_stopping.py:64-121`).

### 3.6 MM utils — already-correct vs broken vs missing **[verified·code + internal·reply]**
- **Already Gemma4-correct in pristine:** `add_variable_extra_tokens_for_images`
  (`_token_utils.py:375`) uses `Gemma4Tokenizer.special_tokens` and `_DOUBLE_NEW_LINE_TOKEN = 108`
  (`:25`) — verbatim-matching the internal version. Use IT for expansion; the legacy fixed-count
  `add_extra_tokens_for_images` (`:47`, Gemma3-hardcoded at `:81`) stays unused.
- **Broken in pristine:** `remove_mm_logits` (`:348`) uses `Gemma3Tokenizer.special_tokens`
  (`:357`) — the NaN-gather root cause from the earlier VQA work. Fix inline to Gemma4 ids AND
  make its stripping index-consistent with `get_no_mm_indices` (same dynamic offsets).
- **Missing in pristine:** `get_no_mm_indices` and `shift_encoder_targets_for_multimodal` —
  paste-implement verbatim from [internal·reply] (`response-from-jestki.md` Doc 2 §6). Note the
  single-image assumption (first `START_OF_IMAGE` only) — assert one image per example.
- `gm/nn/gemma4/vision/_images.py` `factorized_posemb`: NaN issue [fixed in modified] + internal's
  4D→3D posemb reshape [internal·doc §1.B3] — implement both inline.
- ✅ **Tokenizer-id drift risk: RESOLVED (T0, 2026-07-07,
  `results→/home/kaiwen/data/dgemma_e2b/feas/t0_gs_probe.json`).** Local
  `Gemma4Tokenizer.special_tokens` == internal table exactly (PAD 0 / EOS 1 / BOS 2 / turn
  105/106 / image 258880/255999/258882), and literal encodes confirm `<|image|>`→[258880],
  `<image|>`→[258882], `<|turn>`→[105], `<turn|>`→[106], **`"\n\n"`→[108]** (the [internal·reply]
  note that the tokenizer "encodes \n\n to 203" is wrong for this vocab — 108 is both the
  pipeline constant AND the actual encode; no quirk to replicate). The suspicious
  `_tokenizer.py:124-130` block belongs to a different class.

### 3.7 MM batching constraint
- The gemma4 vision merge path is single-sequence: vision output is packed `[1, B·n_soft, D]`
  against the text batch; per-call multimodal batch was 1 in all prior local work
  **[verified in the modified-branch ChartQA harness]**. The internal stack fixed this with a
  reshape/broadcast in its custom merge [internal·doc §1.B2] → B3 adopts that semantics inline;
  until it lands, global batch 8 = gradient accumulation 8×1.

### 3.8 Data precedents **[verified·code]**
- bagz of tf.Example is already the shipped pattern:
  `data/sudoku/convert_sudoku.py:56-80` builds `tf.train.Example` (bytes features) and writes with
  `bagz.Writer`; `data/sudoku/sudoku_data.py:33-63` provides a picklable `bagz.Reader` grain
  source (`PicklableBagzReader` / `Bagz` kd source). Reuse both patterns.
- ChartQA assets to port from `discreteGemma-modified` (read-only source), all under
  `gemma/gemma/diffusion/hackable_diffusion_adapter/`: `data/chartqa/chartqa_data.py` (incl. the
  pure-numpy patchify verified bit-exact vs `_images.patchify`), `eval/chartqa_eval.py`
  (`ChartQARelaxedAccuracy` / `ChartQAExactMatch`, local-tokenizer decode), `eval_main.py`
  `--task=chartqa`, and the vision threading in `hd/hd_gemma_network.py`, `hd/sft_model.py`,
  `hd/hd_gemma_ar_state_handler.py`, `eval/text_metric.py` (strips `-2` placeholders before
  detokenize).
- `kd.sharding.ShardingStrategy` + FSDP exists in this kauldron
  **[verified·code `discreteGemma-modified/.../configs/sft_chartqa.py:254`]**; the config-scoped
  remat wrapper pattern is at `sft_chartqa.py:64-91`.

---

## 4. Internal-stack alignment contract [internal·doc]

### 4.1 What the internal stack is (facts)
- `PrimeDiffusionGemmaModel` extends `DiffusionGemma_26B_A4B`; **hardcodes** the 26B vision tower
  (SigLIP-ish: d_model 1152, 27 layers, 16 heads, ffw 4304, `output_length=280`,
  `use_clipped_linears=False`, `standardize_embeddings=True`).
- Trainer = **kauldron `kd.train.Trainer`** with FSDP `ShardingStrategy` for params + opt_state —
  same framework as the external hd adapter. Dual loss = `NoWeightDiscreteLoss` +
  `EncoderARLoss` — same as external.
- Logging: swaps in `TensorBoardOnlyWriter` when Borg env `XM_XID` is set (suppresses heavy
  Datatable writes).
- Data: Grain loaders; `TrajectoryBagzDataSource` (bagz) and `ChartQAArrayRecordDataSource`
  (ArrayRecord) reading records of **`raw_image` (encoded bytes) + `question` + `answer`**;
  CPU-side PIL+numpy `preprocess_and_patchify` ("bypasses JAX multiprocessing locks");
  `ExpandImagePlaceholder` expands one `<|image|>` token to **280 soft tokens + 3
  boundary/newline tokens** (fixed length); `CanvasChunker` → `canvas`/`canvas_id`/`canvas_mask`.
- Encoder-target alignment: model returns `encoder_logits` of shape `[B, L_no_mm, V]` (MM logits
  stripped); `shift_encoder_targets_for_multimodal` gathers targets at
  `_token_utils.get_no_mm_indices(...)` via `take_along_axis` so AR CE is computed on text
  positions only.
- Six monkey-patches: (1) `Transformer.__getattr__` to resolve `vision_encoder`;
  (2) `_merge_mm_embeddings` replacement incl. **batch>1 reshape/broadcast of packed soft
  embeddings**; (3) `safe_factorized_posemb` (4D→3D reshape + coordinate matmul);
  (4) `tree_map_with_path` shielding of non-numeric kwargs; (5) `_encode_and_get_inputs` override
  that substitutes the attention mask when sliding-mask lengths mismatch; (6) `nn.remat` patches on
  gemma + vision Blocks.

### 4.2 Adopt / adapt / avoid — the contract this build implements

| # | internal mechanism | this build | why |
|---|---|---|---|
| 1 | record schema: tf.Example features **`image/encoded`** (JPEG/PNG bytes), **`question`**, **`answer`** (utf-8); ChartQA files = **ArrayRecord**, trajectory files = **Bagz** [internal·reply] | **ADOPT** — converters write exactly this; parse fns mirrored verbatim incl. `response_text = "The answer is: {answer}"` | answered — no longer an assumption |
| 2 | kauldron Trainer + FSDP `ShardingStrategy` | **ADOPT** — include `cfg.sharding` (no-op on 1 GPU) | config is internal-portable verbatim |
| 3 | dual loss (`NoWeightDiscreteLoss` + `EncoderARLoss`) | **ADOPT** (mandatory — PLE liveness, §3.4) | identical loss structure |
| 4 | image expansion = `add_variable_extra_tokens_for_images`: `[\n\n(108), START_OF_IMAGE, n×SOFT_TOKEN_PLACEHOLDER(-2), END_OF_IMAGE, \n\n(108)]`, net offset n+3 vs the replaced `<|image|>` [internal·reply] | **ADOPT** — the pristine `:375` function IS the internal one; n = tower output (constant per config: 280 for 26B; E2B's own value read at Phase A) — never hardcode 280 | machinery already shipped; C6(c) padding lives in the prompt PAD tail + padded patch rows, NOT unused soft slots |
| 5 | strip+gather encoder-target alignment (`remove_mm_logits` → `[B,L_no_mm,V]`; `get_no_mm_indices` + `take_along_axis` on targets) | **ADOPT — paste verbatim from [internal·reply]** (`response-from-jestki.md` Doc 2 §6): `get_no_mm_indices` → `_token_utils.py`, `shift_encoder_targets_for_multimodal` → `hd/sft_model.py`; fix `remove_mm_logits` to Gemma4 ids + the same dynamic offsets (§3.6). Fallback stays: modified-branch identity+loss-mask — record the choice in ALIGNMENT_NOTES.md | exact internal source in hand; no reverse-engineering |
| 6 | batch>1 vision reshape in merge | **ADOPT as inline code** in the merge path (packed `[1, B·n_soft, D] → [B, n_soft, D]`), with a unit test vs a loop-of-batch-1 reference | unblocks native bs=8 (C3); internal-proven semantics |
| 7 | `__getattr__` vision resolution patch | **AVOID** — instantiate `text_only=False` so the tower is a real child module | hack exists only because their model was built without the tower registered |
| 8 | hardcoded 26B vision tower config | **AVOID** — E2B's tower comes from `Gemma4_E2B.config.vision_encoder` (note: E2B is `use_clipped_linears=True`; internal 26B is `False` — do not copy their numbers) | tower config must match the ckpt weights |
| 9 | tree-map shielding of kwargs | **AVOID** — fix the `images` annotation properly (§3.4) | shielding hides type errors instead of fixing them |
| 10 | sliding-mask length band-aid (substitute attention mask on mismatch) | **REPLACE** — build the sliding mask over the MM-expanded length correctly; add an assert-equal-length unit test with a 283-token-expanded prompt | E2B has real sliding layers (window 512); the band-aid silently changes attention semantics |
| 11 | `nn.remat` monkey-patches on Blocks | **ADAPT** — config-scoped remat wrapper (pattern: modified `sft_chartqa.py:64-91`), applied inside the config | needed for memory at bs>1; keep it out of import-time global state |
| 12 | `TensorBoardOnlyWriter` under `XM_XID` | **ADAPT** — keep `cfg.writer` as an override seam; PC uses `SafeMetricWriter`; add a comment marking the internal swap point | internal drops their writer in without code change |
| 13 | trajectory twin (`sft_trajectory.py`, `TrajectoryBagzDataSource`) | **MIRROR NAMES** — structure `chartqa_data.py`/`sft_chartqa_e2b.py` so a `trajectory_data.py`/`sft_trajectory_e2b.py` pair can be added by copying the shape | the whole point of the ChartQA milestone |

---

## 5. Building blocks

### B0 — `DiffusionGemma_E2B` class
- Edit `discreteGemma/gemma/gemma/diffusion/_models.py`: add
  `class DiffusionGemma_E2B(_gemma4.Gemma4_E2B, _diffusion_transformer.DiffusionMixin)` with
  `keep_last_prefill_kv=True` and the same sc-config defaulting as the 26B class (sc features =
  embed_dim 1536, hidden = hidden_dim 6144).
- Add a construction helper (in the new config file, §B5) `make_e2b_diffusion_model(text_only,
  with_audio=False)` that handles `text_only=False` + `audio_encoder=None` config surgery so the
  audio tower is never silently resurrected [decision D2 default].
- **Acceptance:** init via `call_with_self_conditioning` (sc params created); one forward on GPU
  with PLE active; param count and per-subtree counts printed to `results/e2b_model_tree.json`.

### B1 — Loader extension + load acceptance
- Edit `hd/gemma_checkpointer.py`: add `expected_missing: tuple[str, ...] = ()` to
  `GemmaDiffusionCheckpointLoader` and thread it into `_remap_and_match_params` — model-only keys
  matching any substring keep their init values (exactly like the existing `/lora/` carve-out);
  all other model-only keys still raise. Emit `results/load_coverage.json`:
  `{n_model, n_loaded, n_expected_missing, expected_missing_subtrees[], n_ckpt_only,
  ckpt_only_sample[]}` and assert `n_loaded + n_expected_missing == n_model`.
- **Acceptance tests** (`tests/test_e2b_load.py`, needs real ckpt — else mark `pending-ckpt`):
  1. **Bit-equality:** reuse `_remap_and_match_params`'s matching to build the model-path ↔
     ckpt-path map; assert every matched leaf is bitwise equal to the raw ckpt leaf
     (`gm.ckpts.load_params` or `ocp` restore; compare streaming on CPU — do not materialize two
     full trees on device). Assert the unmatched-model set == the sc subtree exactly.
  2. **AR-forward equivalence:** same tokens → `Gemma4_E2B` (weights via `gm.ckpts.load_params`,
     `gm/ckpts/_checkpoint.py:202`) vs `DiffusionGemma_E2B` plain `__call__` → logits allclose
     (compare in fp32, rtol/atol ~1e-4). This is the decisive "loaded correctly AND config
     matches" test (§3.1 proves extra sc params are ignored by plain `__call__`).
  3. Coverage-json assertions.

### B2 — Self-conditioning init (NOT all-zeros)
- **All-zeros is a permanent saddle** [derived, one line]: FFW is
  `out = [gelu(xW₁) ⊙ (xW₂)]·W₃` with no biases; `W₁=W₂=W₃=0` ⇒ forward ≡ 0 AND
  `∂L/∂W₁ = ∂L/∂W₂ = ∂L/∂W₃ = 0` for all data, permanently (weight decay of 0 is 0).
- **Do instead:** after the loader merge, zero ONLY the FFW **output projection** (`linear`) leaf
  of `self_conditioner`; keep gating einsum + norms at default init. t=0 forward is still an exact
  no-op; `∂L/∂W₃ = hᵀδ ≠ 0` so the module trains.
- **Acceptance** (`tests/test_sc_init.py`): (a) at init, logits with `sc_embeddings=0` ==
  logits with `sc_embeddings=random` (bitwise — W₃=0 kills the signal); (b) after the C3 training
  run, `‖W₃‖ > 0` → `results/sc_liveness.json`.

### B3 — Vision threading (port + fix + align)
Port from `discreteGemma-modified` (§3.8 file list), then apply the alignment contract:
1. **Inline bug fixes** in the pristine files (§3.6): `remove_mm_logits` → Gemma4 ids + dynamic
   offsets consistent with `get_no_mm_indices` (the legacy `:47` fixed-count expander stays
   unused — the pipeline uses the already-correct `:375` variable expander);
   `factorized_posemb` NaN + 4D→3D handling; `images` annotation widened to accept
   `PreprocessedVisionInput`.
2. **Strip+gather alignment** (§4.2 row 5): paste-implement `get_no_mm_indices` +
   `shift_encoder_targets_for_multimodal` from `response-from-jestki.md` Doc 2 §6; property unit
   test (`tests/test_mm_alignment.py`): on a synthetic prompt containing an image expansion, every
   text position's encoder target == the next *text* token; MM positions excluded; the no-mm view
   keeps exactly one `\n\n` where the image block was; single-image assert; image at
   start/middle/end of prompt.
3. **Batch>1 merge** (§4.2 row 6): inline reshape; unit test `tests/test_batch_mm.py`:
   batch-4 forward logits == stacked loop-of-batch-1 logits (allclose fp32).
4. **Expansion** (§4.2 row 4): `ExpandImagePlaceholder` transform (mirror internal, source in
   `response-from-jestki.md` Doc 2 §4a) calling the pristine
   `add_variable_extra_tokens_for_images` (`_token_utils.py:375`); soft count = the E2B tower's
   constant output [verify-PhaseA: read from config, record in ALIGNMENT_NOTES.md]; then
   `gm.data.Pad(key='prompt', max_length=512, truncate=True)` [internal·reply geometry].
5. **Sliding mask over expanded length** (§4.2 row 10) + assert-equal-length test.
6. Sampler-side conditioning (images → encoder prefill → `init_ar_state`) ported as-is from
   modified; smoke a 1-sample image-conditioned generation (this also first-exercises
   KV-sharing × sampler — §3.2 risk note).

### B4 — Data: converters + sources + transforms (internal twin)
- **Schema [internal·reply — ANSWERED]**: `tf.train.Example` features — **`image/encoded`**
  (bytes: original encoded JPEG/PNG — NOT precomputed patches/embeds), **`question`**,
  **`answer`** (utf-8 bytes). Parse-dict keys + templates mirrored verbatim from
  `ParseChartQARecord`: `raw_image/prompt/prompt_text/short_answer/short_answer_text` +
  `response_text = "The answer is: {answer}"` — **the metric's extraction regex keys on this
  prefix, so it is load-bearing**. Prompt = the internal ChartQA user-turn template
  (`response-from-jestki.md` Doc 2 §3), turn tokens `<|turn>`/`<turn|>`.
  **All schema constants live in ONE module `data/chartqa/schema.py`.**
- `data/chartqa/convert_chartqa.py`: HF `ahmed-masry/ChartQA` → three artifacts under
  `/home/kaiwen/data/dgemma_e2b/` (formats mirror internal [internal·reply]):
  - `chartqa/human_train-*.arrayrecord` — ≥2,000 examples (**ArrayRecord**, like internal
    `human_train@90`; covers 200 steps × bs 8 with margin);
  - `chartqa/human_val-*.arrayrecord` — 256 examples (**ArrayRecord**; matches `slice_stop=256`);
  - `chartqa_toy.bagz` — 10 examples (**Bagz** — satisfies the original bagz requirement AND
    exercises the trajectory-side reader), **at least one image whose true patch count is below
    the patchify max so padded patch rows genuinely exist** (record which; the prompt PAD tail
    always exists at prompt_len=512 — C6c uses both).
  Writers: ArrayRecord + `bagz.Writer` (precedent §3.8). Emit `results/dataset_manifest.json`
  (counts, digests, the padding-bearing toy index).
- **Sources:** `ChartQAArrayRecordDataSource` (name-mirrors internal) + the generic `Bagz` grain
  source (`sudoku_data.py:33-63`) for the toy set.
- **Transforms**, mirroring internal names/order [internal·reply]: `ParseChartQARecord` →
  `PreprocessChartQAVision` (CPU PIL+numpy patchify — reuse modified's bit-exact patchify) →
  tokenize prompt (template) → `ExpandImagePlaceholder` (§B3.4) → `gm.data.Pad` to 512 →
  tokenize/`CanvasChunker` the `response_text` (`canvas`/`canvas_id`/`canvas_mask`, PAD-fill
  masked by `canvas_mask`) → target shift (+ `shift_encoder_targets_for_multimodal` at the loss
  boundary).

### B5 — Config `configs/sft_chartqa_e2b.py`
- model = `make_e2b_diffusion_model(text_only=False)`; the config takes **`ckpt_variant ∈
  {pt, it}`** (D1 = BOTH: every train/eval phase runs once per variant; workdirs and
  `results/{pt,it}/` namespaced accordingly);
  `init_transform = GemmaDiffusionCheckpointLoader(path=<variant ckpt>,
  expected_missing=('self_conditioner',))` followed by the sc-W₃ zeroing (B2) — implement the
  zeroing as a second init-transform step so it is ordered, explicit, and testable.
- losses: `NoWeightDiscreteLoss` + `EncoderARLoss`, weights 1.0 / 1.0 (mandatory, §3.4).
- optimizer: clip 1.0 → `scale_by_factored_rms` → weight decay 1e-4 → LR schedule
  **warmup 20 → peak 3e-5 → cosine to 3e-6** [default; rationale: pretrained init — the 1e-3 in
  the sudoku-tiny config was tuned for random-init tiny models and would damage a pretrained 2B].
- batch: **global 8** — native bs=8 if B3.3 lands and memory allows; else
  `optax.MultiSteps` grad-accum 8×1. Record which in `results/train_config_actual.json`.
- remat: config-scoped wrapper (§4.2 row 11). sharding: FSDP `ShardingStrategy` (§4.2 row 2).
  writer: `SafeMetricWriter` + internal-swap seam comment (§4.2 row 12).
- steps 200; `save_on_steps=[50,100,200]`; eval_ds from `chartqa/human_val-*.arrayrecord`.
- geometry **[internal·reply]**: `prompt_len=512`, `canvas_size=256`, `num_canvases=2`; prompt
  right-pad/truncate at 512 with PAD=0; canvas PAD-fill masked via `canvas_mask`. Deviations (if
  VRAM forces smaller) → ALIGNMENT_NOTES.
- eval **[internal·reply]**: `GemmaSamplingEvaluator` via the pristine `make_ar_evals`
  (`eval/ar_eval.py`: DiscreteDDIMStep, temp 0.7, bf16 logits, no early stop — §3.5); 256-example
  val slice; eval batch 16 if the B3.3 batch-MM path holds for generation, else 1; **local
  wall-time reduction: gate on steps=32 across ckpts {50,100,200}, run 64/96 on the best ckpt
  only** (full [32,64,96]×all-ckpts is the internal protocol — note the reduction in
  ALIGNMENT_NOTES); metric = internal `score_chartqa` **verbatim** (5.1% relaxed tolerance,
  `extract_chartqa_answer` keyed on "The answer is:"), replacing the modified-branch scoring
  internals; exact-match reported alongside.

### B6 — Test & validation suite
`tests/`: `test_e2b_load.py`, `test_sc_init.py`, `test_mm_alignment.py`, `test_batch_mm.py`,
`test_padding_invariance.py`, `test_sliding_mask_mm.py` + `gpu_smoke/consistency_harness_e2b.py`
(C5) + `gpu_smoke/run_chartqa_e2b_{train,eval}.sh`.
**C5 harness protocol (pinned comparison):** fix (prompt, noisy canvas, noise level t, sc). Run
(a) the training-path denoiser forward and (b) a forward built exactly the way the sampler builds
it (positions from cache end_index, masks from `_make_*` helpers). Assert: positions equal; mask
**visibility pattern** equivalent (not tensor-identical — shapes differ by design, §3.5); 
**pre-shaping** logits allclose (fp32, rtol/atol 1e-4). Repeat for the second pass (sc ≠ 0). Keep
total length < 512 (sliding-window coincidence, §3.5). Also compare the encoder/prefill pass
(training encoder forward vs sampler prefill) — that is where cache/position off-by-ones live.
**Criterion-3/6 note:** the SFT loss samples random time + corruption via the `sampling` rng
stream — all perturbation comparisons (image perturb, padding perturb) MUST pin the rng streams,
or the deltas are sampling noise.

---

## 6. File authorization list

**New files (create):**
`diffusion/_models.py` (add class — edit, see below), `configs/sft_chartqa_e2b.py`,
`data/chartqa/{schema.py, chartqa_data.py, convert_chartqa.py}` (ported+adapted), `eval/chartqa_eval.py`
(ported), `tests/*` (new), `gpu_smoke/{consistency_harness_e2b.py, run_chartqa_e2b_train.sh,
run_chartqa_e2b_eval.sh}`, `reports/*`, plus `hd/sft_model.py` additions.

**Pristine files authorized for inline edit (log each edit in ALIGNMENT_NOTES.md):**
`diffusion/_models.py`, `diffusion/_transformer.py` (annotation only),
`gm/vision/_token_utils.py`, `gm/nn/gemma4/vision/_images.py`,
`hd/{gemma_checkpointer.py, hd_gemma_network.py, sft_model.py, hd_gemma_ar_state_handler.py}`,
`eval/{text_metric.py}`, `eval_main.py`.

**Anything else → BLOCKER first. `discreteGemma-modified/` and `configs/sft_sudoku*.py` are
read-only.**

---

## 7. Success criteria (FINAL — supersedes all earlier drafts)

| # | criterion | gate |
|---|---|---|
| **C1 load** | for **BOTH** `pt` and `it`: `load_coverage.json` (`n_loaded + n_expected_missing == n_model`, `expected_missing_subtrees == {self_conditioner}`); bit-equality vs raw ckpt on every matched leaf; AR-forward logits allclose (fp32 1e-4) | HARD |
| **C2 sc-init** | at init: sc-input perturbation is a bitwise no-op on logits; after C3: `‖W₃‖ > 0` (liveness) — per variant | HARD |
| **C3 train** | for **BOTH** variants: 200 steps, global batch 8 (native or accum — recorded), LR per §B5; total loss decreases without NaN/divergence; ckpt save → restore → continue works | HARD |
| **C4 accuracy** | internal-verbatim `score_chartqa` (5.1% relaxed) on the 256-example val slice, sampler = DiscreteDDIMStep steps=32 / temp 0.7 [internal·reply]; per-ckpt {50,100,200}, best-ckpt reported (internal selection rule); **> 0 for at least one of {pt, it}** (expected: `it`); both variants + exact-match reported; steps 64/96 on the best ckpt | HARD; **≥ 0.3 is a stretch goal — report, do not gate** |
| **C5 train-infer consistency** | pinned-protocol (§B6) against the **ar_eval/GemmaSamplingEvaluator inference path** (DiscreteDDIMStep — §3.5): positions equal; mask visibility equivalent; **pre-temperature** logits allclose (fp32 1e-4); both sc passes; encoder/prefill pass included; toy length < 512 (code-path property — run once, with `it` weights) | HARD |
| **C6 vision** | (a) pinned-rng image perturbation changes the loss (plumbing; once); (b) post-C3 **counterfactual gap** on BOTH trained variants: wrong-image loss/acc worse than correct-image — direction gates on at least one variant, both magnitudes reported; (c) perturbing padded patch rows AND prompt-PAD-tail token ids (masks held fixed) leaves the loss **bit-identical** — patch padding proven present in the toy set; the PAD tail always exists at prompt_len=512 (once) | HARD |

Notes: C4/C5 first-exercise KV-sharing × sampler (§3.2) — a failure there is a cache-layout bug,
not a load bug. D1 = BOTH: the pt-vs-it contrast is itself a useful reported signal (how much
instruction-tuning matters under diffusion adaptation) — compare, don't gate on it. All numbers
land in `results/{pt,it}/*.json`, watermark-free but clearly labeled
`PC / ChartQA mechanics — not WOD evidence`.

---

## 8. Phase plan & commit gates (branch `dgemma-e2b-test`)

- **Phase A — pre-flight: ✅ DONE 2026-07-07** (T0 `gpu_smoke/e2b_feas_t0_gs_probe.py`, results
  in `/home/kaiwen/data/dgemma_e2b/feas/t0_gs_probe.json`; T1/T2 feasibility in the same dir):
  both ckpts reachable (OCDBT); **5.12B params each; vision tower PRESENT, audio tower present
  (discard per D2), self_conditioner absent** (as expected for AR ckpts); trees structurally
  identical pt vs it; **tokenizer ids match the internal table exactly** (§3.6 — drift resolved).
  Both ckpts downloading to `/home/kaiwen/data/dgemma_e2b/ckpts/`. Remaining Phase-A item folded
  into B0: read the E2B tower's constant soft-token count from config, record in ALIGNMENT_NOTES.
  **Overnight mode: Kaiwen reviewed the pre-flight plan and authorized autonomous execution —
  do not stop between phases; log everything; BLOCKERS.md for anything unresolvable.**
- **Phase B** — B0 + B1 + B2 + their tests → C1, C2(init-half) green → commit
  `e2b: assemble + AR-init load + sc zero-init`.
- **Phase C** — B3 + alignment/unit tests + C5 harness + C6(a/c) at fixture level → commit
  `e2b: vision threading + train-infer consistency harness`.
- **Phase D** — B4 + B5 → 200-step train → eval, **once per variant (`pt`, `it`)** → C3, C4,
  C6(b), C2(liveness) numbers for both → commit `e2b: chartqa 200-step train + eval (pt+it)`.
- **Phase E** — `reports/REPORT.md` (all numbers templated from `results/*.json`) +
  `reports/ALIGNMENT_NOTES.md` (adopt/adapt/avoid outcomes; every pristine-file edit; deltas vs
  internal doc; open items for the internal twin) → final commit.
- Unattended mode: never stop to ask — BLOCKERS.md, skip, continue. Interactive: ask.

---

## 9. Reporting

`reports/QUEUE.md` (checkboxes, work top-down), `reports/BLOCKERS.md`, `reports/REPORT.md`
(schema-templated numbers only; each table footer: ckpt id, git hash, config hash),
`reports/ALIGNMENT_NOTES.md` (the internal-twin ledger — §4.2 outcomes). Machine-emitted JSON only
in `results/` (checked in; small; variant-dependent artifacts under `results/{pt,it}/`). Big
artifacts stay in `/home/kaiwen/data/dgemma_e2b/`.

---

## 10. Decisions — ALL ANSWERED (D3/D5 sources archived in `response-from-jestki.md`)

- **D1 — checkpoint: ANSWERED (Kaiwen, 2026-07-07): BOTH.** Fetch, inspect, load-test, train and
  evaluate BOTH `GEMMA4_E2B_PT` and `GEMMA4_E2B_IT`. Everything is parameterized by
  `ckpt_variant ∈ {pt, it}` (workdirs `/home/kaiwen/data/dgemma_e2b/{pt,it}/`,
  results `results/{pt,it}/`). Per-criterion single/dual-run policy: §7.
- **D2 — audio: ANSWERED (Kaiwen, 2026-07-07): OUT.** `audio_encoder=None` explicitly everywhere;
  the loader treats audio subtrees in the ckpt (if any) as ckpt-only discards (warned + counted in
  the coverage report, §B1).
- **D4 — branch: ANSWERED (Kaiwen, 2026-07-07):** `dgemma-e2b-test` — already created and checked
  out by Kaiwen. Kaiwen merges.
- **D3 — internal data schema: ANSWERED (internal agent, 2026-07-07 — full reply archived in
  `response-from-jestki.md` Doc 2 Part 1; that file is the authoritative source, `schema.py`
  encodes it).** Key constants:
  - tf.Example features: **`image/encoded`** (JPEG/PNG bytes), **`question`**, **`answer`** (utf-8).
  - Parse-dict: `raw_image/prompt/prompt_text/short_answer/short_answer_text` +
    `response_text = "The answer is: {answer}"` (ChartQA) / `"Predicted Waypoints: {answer}"`
    (trajectory).
  - Files: ChartQA = **ArrayRecord** (`human_train@90` / `human_val@10`); trajectory = **Bagz**
    (`wod_e2e_train@100.data.bagz`); no sidecars.
  - Prompt templates: verbatim in the archive (ChartQA user-turn template; trajectory 20-waypoint
    @4Hz system prompt — keep for the WOD twin).
  - Expansion: `add_variable_extra_tokens_for_images` — `[\n\n(108), SOI, n×(-2), EOI, \n\n(108)]`;
    26B tower n=280; **E2B's n read at Phase A**.
  - Tokenizer: `Gemma4Tokenizer`, vocab `gemma4_cleaned_262144.model`; ids PAD 0 / EOS 1 / BOS 2 /
    `<|turn>` 105 / `<turn|>` 106 / `<|image|>` 258880 / `<|image` 255999 / `<image|>` 258882;
    pipeline `\n\n` = **108** (the tokenizer itself would encode 203 — replicate the pipeline's
    108). **Local-id equality asserted at Phase A (§3.6 drift risk).**
  - Alignment fns: `get_no_mm_indices` + `shift_encoder_targets_for_multimodal` — paste verbatim
    from the archive.
  - Geometry: ChartQA 512/256/2; trajectory 512/128/3; PAD-0 truncate/fill rules as archived.

- **D5 — eval/sampling config: ANSWERED (internal agent, 2026-07-07 — archived ibid., Doc 2
  Part 2).** `GemmaSamplingEvaluator` on a 256-example val slice (`slice_stop=256`), ChartQA eval
  batch 16; sampler = `make_ar_evals` / `hd.sampling.DiscreteDDIMStep`, **constant temp 0.7, bf16
  logits, no early stop, `AR_DENOISING_STEPS=[32,64,96]`** — identical to the pristine
  `eval/ar_eval.py` defaults **[verified·code]**, so no sampler porting is needed. Ckpt selection:
  evaluate every saved ckpt, report best-val. Metric: `score_chartqa` verbatim (5.1% relaxed,
  "The answer is:" extraction). Local wall-time reduction (→ ALIGNMENT_NOTES): gate on steps=32
  across ckpts {50,100,200}; 64/96 on the best ckpt only.

**All decisions answered + Phase-A findings (T0, 2026-07-07):**
- `pt` and `it` ckpt trees: **5.12B params, structurally identical**; subtrees = embedder,
  final_norm, layer_0..34, **vision_encoder (present)**, **audio_encoder (present — loader
  discards per D2)**; **no self_conditioner** (expected).
- Tokenizer: local ids == internal table; `"\n\n"`→108 confirmed by actual encode (internal
  "203" note wrong). **No vocab drift.**
- Both ckpts + tokenizer fetchable from this PC; local copies under
  `/home/kaiwen/data/dgemma_e2b/ckpts/`.
- T1/T2 (5090 build + train-step feasibility at the internal geometry) results in
  `/home/kaiwen/data/dgemma_e2b/feas/` — consult before picking batch/remat settings in B5.

---

## 11. Hard prohibitions (recap)

No monkey-patching (§0.3, one scoped exception). No edits outside §6. No fabricated numbers; no
copying numbers from docs into REPORT.md. No claiming WOD-hypothesis evidence. No disabling
EncoderARLoss or PLE without a [decision]. No all-zeros sc init (§B2 — it is a provable dead
saddle). No touching `discreteGemma-modified/`. No new training framework. No editing the §7
gates. No skipping the Phase-A stop.
