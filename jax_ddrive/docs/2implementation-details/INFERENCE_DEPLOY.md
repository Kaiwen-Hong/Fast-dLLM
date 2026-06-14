# INFERENCE_DEPLOY.md — internal-TPU SASD inference (B1 export + B2 self-contained sampler)

Stable reference for **how the trained SASD model is taken from MaxText training to generation
on the internal TPU**, where nothing can leave the server except scalar validation-log values.
This closes blockers **B1** (weight-format bridge) and **B2** (self-contained sampler), and wires
**B3** (torch-free inputs) + the **embedding-parity** check (user request) + **B4** (fp32/bf16 TPU
numerics). Companion: `../5blockers/0612-blocker-v0.md` (the blocker analysis this implements),
`DATASET_V2.md` (data), `LABELING.md` (labels).

> **Status legend:** ✅ implemented + tested · 🧪 implemented, GPU/TPU test pending · 📦 offline tool.

---

## 0. The end-to-end path (internal TPU)

```
                    [ all of this runs INSIDE the internal TPU env; only vlog scalars come out ]
MaxText SASD train ──▶ Orbax param ckpt ──(B1)──▶ bf16 HF safetensors snapshot
   (production, FSDP)        │  maxtext_to_hf_export.py        │
                            ckpt                        eval_sasd.hf_to_jax (bf16-hardened NNX loaders)
                                                                 │
   offline (local/free-TPU host, torch OK) ─────────▶ npz ──▶ eval_sasd.sampler_sasd  (block-diffusion loop)
   prep_jax_eval_inputs.py: scaffold + rope +              │       (validated 0.01m vs PyTorch)
   pixels + bf16 image_embeds + GT target_ids              ▼
                                                  decode (tokenizer) ──▶ 4-section JSON
                                                                 │
                                            compare vs target ──▶ validation_log.jsonl  (scalars only)
```

Two separations make this safe and portable:
1. **Inputs go IN, only scalars come OUT.** Inference inputs (scaffold + image embeds) are
   precomputed **offline** and shipped in via GCS; the only thing emitted is
   `validation_log.jsonl` (booleans / counts / max-deltas — no weights, images, or raw text).
2. **The inference package is self-contained.** `eval_sasd/` runs with `PYTHONPATH=fork/src`
   only — **no `ddrive_jax`** on the internal side (guarded by a test). Same vendoring pattern as
   `input_pipeline/sasd_data/`.

---

## 1. B1 — MaxText → HF weight bridge ✅

`maxtext-dlm-fork/scripts/maxtext_to_hf_export.py`. The inverse of
`save_fast_ddrive_params_ckpt.py`: restores the Orbax param ckpt, inverse-maps the **434 text
leaves** to HF layout via `QWEN_MAXTEXT_TO_HF_PARAM_*` hooks (`saving_to_hf=True`), copies the
**390 `visual.*`** tensors verbatim from a reference snapshot (the ViT is frozen during SASD
training), omits `lm_head` (tied), and writes a **bf16** HF snapshot (+ config/tokenizer copied
from the reference).

- Output is **bf16** (matches the bf16 param ckpt — lossless, half the host RAM of f32, which
  OOM-killed the 30 GB box at f32).
- **Round-trip identity gate** (`verify_against=`, BASE ckpt only): base → MaxText params → HF export must equal
  the base snapshot **bitwise**. ✅ **Verified 824/824 (434 text + 390 vision, 0 differing).**

```bash
# (CPU) round-trip the BASE param ckpt -> bf16 HF -> expect B1_ROUNDTRIP_PASS
# $DDRIVE = jax_ddrive root, $FORK = the fork (both from ~/.fastddrive_env); run from $FORK.
PYTHONPATH=$DDRIVE:$FORK/src JAX_PLATFORMS=cpu python scripts/maxtext_to_hf_export.py \
  src/maxtext/configs/sasd_waymo.yml model_name=qwen2.5-3b \
  param_ckpt_dir=<param ckpt> ref_snapshot=<base HF snapshot> \
  out_dir=<out> verify_against=<base HF snapshot>
```

---

## 2. B2 — self-contained inference (`src/maxtext/diffusion/eval_sasd/`)

**Vendored** from `Fast-dLLM jax_ddrive @ 4b0f4f2` (logic unchanged; only import lines + bf16):

| file | from | role |
|---|---|---|
| `models/rope.py` | `ddrive_jax/models/rope.py` | RoPE |
| `models/qwen2_5_text.py` | `ddrive_jax/models/qwen2_5_text.py` | NNX text decoder (`embed_tokens`/`hidden_forward_mrope_cs`/`attend`/`mrope_cos_sin`) |
| `models/vision_qwen25vl.py` | `ddrive_jax/models/vision_qwen25vl.py` | frozen ViT |
| `hf_to_jax.py` | `ddrive_jax/convert/hf_to_jax.py` | NNX loaders, **bf16-hardened** |
| `masks_eval.py` | `ddrive_jax/diffusion/masks.py` | inference attention mask (`eval_hybrid_block_causal_mask_dense`) |
| `sampler_sasd.py` | `ddrive_jax/eval/mm_sampler.py` | **the block-diffusion sampling loop** (`mm_section_diffusion_sample`, `decode_generation`) |

**Fork-authored (not vendored):**
- `driver.py` — `run_eval(npz, snapshot, dtype=…)`: load → sample → decode → parse JSON →
  T2 metrics → append `validation_log.jsonl`. Two image paths: **precomputed `image_embeds`**
  (preferred; TPU skips the ViT, via an `_EmbedShim`) or **pixel_values** (ViT on-device).
- `embedding_parity.py` — `run_parity(npz, snapshot)`: ViT embeds fp32 vs bf16 vs reference.
- `tests/eval_sasd_import_test.py` — asserts importing `eval_sasd` pulls in **zero ddrive_jax**.

**`eval_harness.py`** now re-exports `run_eval` / `run_parity`.

### Why bf16 hardening (the one correctness trap)
The B1 snapshot is **bf16**; safetensors' `framework="numpy"` **cannot decode bf16**
(`TypeError: data type 'bfloat16' not understood`). `hf_to_jax.py` therefore reads tensors via a
raw-byte `ml_dtypes` reader (`_st_tensor_f32`) — bit-identical to the old path for F32/F16, and
correct for BF16. (The same fix was applied to the two training-side loaders.) Downstream `_set`
casts to the model leaf dtype, so reading fp32 then casting is correct for any cfg dtype.

### Self-containment ✅ (tested)
```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python -m maxtext.diffusion.tests.eval_sasd_import_test
# -> EVAL_SASD_SELFCONTAINED_PASS  (zero ddrive_jax, flax.nnx present, public API resolves)
```
**Requirement for the internal env:** `flax.nnx` (flax ≥ 0.8) must be installed — the sampler is
NNX. (Local: flax 0.12.7. Pin it in the internal requirements.)

### Generation (GPU/TPU) 🧪 (post-A5 double-check armed)
```bash
PYTHONPATH=src python -m maxtext.diffusion.eval_sasd.driver \
  --npz <inputs.npz> --snapshot <B1 bf16 snapshot> --dtype fp32   # then --dtype bf16
# -> SASD_EVAL_PASS  (valid JSON + 5-waypoint trajectory; + traj_exact / traj_max_abs_delta / co_match / fmb_match vs target)
```

---

## 3. B3 — torch-free inputs (offline prep) 📦

`maxtext-dlm-fork/scripts/prep_jax_eval_inputs.py` (runs **offline**, imports ddrive_jax). For each
eval sample it builds the npz the driver consumes:
`x_t0, rbi, position_ids, orig_len, pixel_values, image_grid_thw, target_ids[, image_embeds]`,
using the parity-validated `build_scaffold` / `get_rope_index_numpy` + the HF processor. With
`--with_embeds` it runs the frozen ViT once (fp32→bf16) so the **internal TPU never loads the ViT**
— it only needs an `AutoTokenizer` to decode the output. **This precomputed-embeds path is the
CANONICAL one** (symmetric with training, which also feeds precomputed embeds → no ViT-recompute
drift, and it avoids the bf16-ViT matmul drift; see §4). ✅ 40-sample eval set built + shipped to
`gs://<bucket>/eval_inputs/` (20 train + 20 val); the 20 train npz carry the **exact training embeds**
(pulled from the v2 AR dataset by sample index, verified by prompt + grid match).

---

## 4. Embedding parity (user request) 🧪

`embedding_parity.py` loads the ViT in fp32 and bf16 and compares the merged embeds, plus vs the npz's
precomputed reference embeds. Since the **canonical path skips the ViT** (precomputed embeds), the
verdict **gates on `fp32_vs_reference`** — do the shipped embeds reproduce a fresh fp32 ViT? (cosine
≥ 0.999; measured **1.00000** on the training-exact npz). The `fp32_vs_bf16` / `bf16_vs_reference`
numbers are **DIAGNOSTICS** (the bf16-matmul drift you'd incur IF you ran the bf16 ViT — the BASE ViT
gives cosine ~**0.998**, larger than the release ViT's 0.99966; that is exactly why the canonical path
uses precomputed fp32→bf16 embeds and does NOT run the bf16 ViT). With no precomputed embeds (on-device
fallback) it falls back to gating on `fp32_vs_bf16` — prefer fp32 ViT. Both fp32+bf16 numbers reported.
```bash
PYTHONPATH=src python -m maxtext.diffusion.eval_sasd.embedding_parity \
  --npz <inputs_with_pixels.npz> --snapshot <B1 snapshot>   # -> EMBED_PARITY_PASS/FAIL
```

---

## 5. Internal-TPU runbook (deployment arc: local → free-1-TPU → internal-8-TPU)

**The actionable, step-by-step runbook for the internal coding agent is
`../6for_internal/0613-test_training.md`** (live GCS paths + exact commands). Summary of the arc:

1. **Build artifacts (local, this repo)** — base param ckpt (`save_fast_ddrive_params_ckpt.py`
   from the base snapshot), distilled-400 **base-ViT** v2 AR (`parquet_to_ar_with_embeds.py
   --snap <base>`), offline eval npz (`prep_jax_eval_inputs.py`). ✅ done 2026-06-13.
2. **Upload to GCS** — ✅ **done 2026-06-13**. Live objects in `gs://<project>-ddrive-sasd/`:
   `maxtext_sasd_params_base/fast_ddrive_qwen25_3b_BASE_params/` (base init weights),
   `wod_e2e_sasd_distilled_0613-400_baseViT_v2_ar/` (dataset, 7 shards), `code/fastddrive-<TS>.tgz`
   (+ `code/fastddrive-LATEST.txt` pointer; one bundle = fork + jax_ddrive, see `upload_code_to_gcs.sh`).
   Launch script: `launch_maxtext_sasd_tpu_frombase.sh`.
3. **Free single TPU smoke** — run `launch_maxtext_sasd_tpu_frombase.sh` (12-step train + resume
   validation), then `maxtext_to_hf_export.py` (expect 824/824), `run_eval` fp32 **then** bf16,
   `run_parity`. B4 numeric rehearsal + the reference the internal run reconciles against.
4. **Internal 8-TPU** — same launch script with `STEPS=30000` (FSDP shards optimizer → more memory
   headroom than 1 chip), export, `run_eval` (T2), `run_parity`, emit `validation_log.jsonl`.

Single-chip inference is fine (3B fits one chip); training uses FSDP across the pod.

---

## 6. What is tested vs pending (honest)

| item | status |
|---|---|
| B1 round-trip identity (824/824 bitwise) | ✅ tested |
| B1 export from a TRAIN-state ckpt (`items/`) | ✅ tested (restores 434 params directly; no extraction step) |
| B2 self-containment (zero ddrive_jax import) | ✅ tested (`EVAL_SASD_SELFCONTAINED_PASS`) |
| All B2 files `py_compile` + public API | ✅ tested |
| B2 fork-only generation (release F32 + oracle) | ✅ `SASD_EVAL_PASS` (valid JSON + 5-wp trajectory) |
| bf16 snapshot load via hardened loader | ✅ `B2_BF16_LOAD_PASS` (export base→bf16, 434 loaded, embed dtype BF16) |
| Embedding parity fp32/bf16 | ✅ cosine **0.99966** (max_rel 3.0e-2); thresholds recalibrated to cosine≥0.999 primary, max_rel≤5e-2 loose |
| Offline prep on the eval set | ✅ runs (`build_scaffold`). **GOTCHA: `--min_pixels`/`--max_pixels` MUST match the TRAINING resolution (784 / 784·64 → 56 merged tokens/img × 3 cams = 168), or token count / mRoPE won't line up.** |
| from-base overfit → T2 (real milestone) | 🔄 **pipeline works end-to-end** (overfit model → correct scaffold → coherent 4-section JSON; critical_objects matched), but **NOT yet verbatim**. 12k: traj Δ1.12m; 30k did **not** improve and **regressed** (traj Δ28.8m) → a training-recipe issue (LR schedule recomputed on resume), not the pipeline. NOTE: the old `token_agreement`-vs-target was a broken metric (denoised-scaffold vs flat-GT misalignment); replaced by `co_match`/`fmb_match` + trajectory Δ. |
| Free-1-TPU + internal-8-TPU smoke | ⏳ not started |

---

## 7. Maintenance

`eval_sasd/` is **vendored** — do **not** edit logic there; edit
`Fast-dLLM/jax_ddrive/ddrive_jax/...` and re-copy (only import lines differ; bf16 reader is the
one local addition, marked in `hf_to_jax.py`). Pinned source commit: `4b0f4f2`. The fork-side diff
is documented in `maxtext-dlm-fork/PATCHES.md`. Build/validation story for today's work:
`../4collect/07_from_base_b1_b2_progress.md`.
