I'll collate the doc-audit findings into a prioritized report. Let me analyze the 108 raw findings and auditor notes to produce the structured markdown report.

# Fast-dDrive Doc-Audit: Prioritized Collated Report

## 1. SUMMARY

**By verdict (108 total findings):**

| Verdict | Count |
|---|---|
| ok (no defect) | 76 |
| wrong | 8 |
| stale | 13 |
| misleading | 11 |
| **Total defects (wrong+stale+misleading)** | **32** |

**By severity (defects only, ok excluded):**

| Severity | wrong | stale | misleading | Total |
|---|---|---|---|---|
| high | 3 | 0 | 0 | 3 |
| medium | 3 | 5 | 3 | 11 |
| low | 2 | 8 | 8 | 18 |
| **Total** | **8** | **13** | **11** | **32** |

**Frozen vs living defects:** 11 frozen (under `1plans/` or `4collect/` → annotate inline), 21 living (→ edit directly).

---

## 2. HIGH-SEVERITY findings (fix first)

All three are **WRONG** and **living** (editable). Note all three are the same class of count-drift the audit was chartered to catch.

| Doc | Location | Claim | Code reality (file:line) | Fix | Frozen? |
|---|---|---|---|---|---|
| `2implementation-details/AUDIT.md` | "After fixes…" tail, lines 27-28 | `run_all_verification.sh` runs **9 gates** (lists 9 names) | Script runs **10**; summary loop iterates 10 keys incl. `cpu_eval_ports` at `run_all_verification.sh:51`; invoked at `:26` against `tests/test_eval_ports.py`. Ground truth §6 = 10. | Change "9 gates"→"10 gates"; add `cpu_eval_ports` to the enumerated list. | living — edit |
| `2implementation-details/DATASET_V2.md` | §1 "Embeds provenance", line 32 | ViT jit-vs-eager cross-check asserts `rel < 2e-4` | Two-tier guard: per-sample `assert d < 1e-3` (`parquet_to_ar_with_embeds.py:238`) AND rolling-median `assert med < 1e-4` over ≥8 shards (`:241`); runs on each shard's FIRST record only (`if n_shard == 0`, `:226`). Comment `:231` tolerates isolated ~2.5e-4 (a 2e-4 gate would falsely reject). | Replace "(assert rel < 2e-4; observed ≤ ~1.3e-5)" with "(two-tier guard on each shard's first record: per-sample max-rel < 1e-3 AND rolling median < 1e-4 over ≥8 shards; observed median ~1.3e-5)". | living — edit |
| `../to-host-chn.md` | §6.1 mesh (line 348) and §6.2 Option 1 (line 364) | Model is **3.75 B** params (mesh/FSDP sizing guidance) | Verified leaf-count is **3.086 B** (434/434 leaves) per `save_fast_ddrive_params_ckpt.py:72-73` and `docs/4collect/07_from_base_b1_b2_progress.md:33,86`. Doc itself uses 3.086B/3.09B at lines 31,53,78. | Replace both "3.75 B" → "3.09 B" (or 3.086 B). | living — edit |

---

## 3. MEDIUM-SEVERITY findings

### 3a. Living docs (edit directly)

| Doc | Location | Verdict | Claim | Code reality (file:line) | Fix |
|---|---|---|---|---|---|
| `2implementation-details/DATASET_V2.md` | §4 "Checkpoint/resume", line 98 | stale | `launch_maxtext_sasd_tpu_v2.sh` performs on-TPU resume validation | No such file exists in either repo (find returns nothing; name only in docs). Real SASD launcher is `maxtext-dlm-fork/scripts/train_sasd_waymo.sh`. | Point to `train_sasd_waymo.sh`, or label it an uncommitted internal-host launcher. |
| `2implementation-details/EVAL_PIPELINE.md` | §3b JAX eval command, lines 77-80 | misleading | Describes JAX eval prep without stating its resolution | `prep_jax_eval.py:24-25` defaults `--min_pixels=--max_pixels=200704` (paper-eval), differs from deploy/train 784/50176→168 tokens (`prep_jax_eval_inputs.py:53-54`, `prep_train_jax.py:56-57`). | Add a one-line note: this path uses paper-eval res 200704, distinct from training/deploy 784/50176. |
| `2implementation-details/ARCHITECTURE.md` | "Training data flow", line 60 | misleading | Vision protected via per-step `where(vision_mask, original_embeds, embed(noisy_ids))` | No training path reads `vision_mask`. Trainers re-scatter frozen embeds via `.at[img_pos].set(image_embeds)` (`train_overfit_mm.py:60,66`; `train_waymo_sasd_jax.py:95,113`); `make_batch` masks only response tokens (`& resp`, `noise.py:30`). `vision_mask` is a carried-but-unused schema field (Drift #11). | Replace line 60 with the `.at[img_pos].set` re-scatter mechanism; drop `vision_mask` from the flow. |
| `3summary/FEATURES.md` | "Inference / serving" row, line 44 | misleading | JAX eval = 52-frame partial 0.853/2.196/8.10 (vs PyTorch 0.888/2.250/7.913) | Final full-479 rated-val = JAX 0.839/2.072/7.929 vs PyTorch 0.814/1.990/7.914 (`REPORT.md:43`; `make_dataset_website.py:340`). 52-frame is a stale preliminary partial. | Update row to full-479 numbers, or explicitly label the 52-frame figure as preliminary partial and cross-ref `REPORT.md:43`. |
| `6for_internal/test_training.md` | §7 STAGE 3 status note (2026-06-14) | misleading | `token_agreement` metric removed entirely | Still present: `driver.py:17` (docstring), `driver.py:153` (ref_output branch computes it). Removed only from the T2/target_ids path (replaced by traj_exact/co_match/fmb_match, `driver.py:139-148`). | Scope the sentence to T2; note token_agreement still exists for the ref_output parity branch (`driver.py:153,17`). |

### 3b. Frozen docs (annotate inline — do NOT rewrite)

| Doc | Location | Verdict | Claim | Code reality (file:line) | Inline annotation |
|---|---|---|---|---|---|
| `1plans/02_tpu_plan.md` | ## Mesh bullet 1, line 10 | wrong | Model is 3.75B | 3.086B/434 leaves (`save_fast_ddrive_params_ckpt.py:72-73`; `07_from_base_b1_b2_progress.md:33`). 3.75B only in stale code comments `lora.py:5`, `launch_tpu.sh:11`. | `[CORRECTION 2026-06-16: 3.086B (3.09B rounded); "3.75B" is a stale over-estimate]` |
| `1plans/04_tpu_smallscale_validation.md` | ## 3 table row 1, line 75 | wrong | Real model is 3.75B | Same doc says 3.09B at line 28 (self-inconsistent); verified 3.086B (`save_fast_ddrive_params_ckpt.py:72-73`; `OVERNIGHT_TPU_PROGRESS.md:169`). | Change to 3.09B / `[CORRECTION: 3.086B, matching line 28]` |
| `4collect/HANDOFF.md` | Phase 3 bullet, line 24 | wrong | Full-FT of 3.75B | 3.086B/434 leaves (`sasd_waymo.yml:18`; `load_fast_ddrive_maxtext.py`; OVERNIGHT_TPU_PROGRESS). | `[CORRECTION: verified ~3.086B (434 text leaves); 3.75B is an early erroneous figure]` |
| `4collect/OVERNIGHT_PROGRESS.md` | lines 28, 89, 134, 135 | wrong | Real model is 3.75B (×4) | 3.086B; same doc says 3.09B at line 98 and "252/252 kernels" = 36×7 (self-contradictory). | Annotate all four: `[CORRECTION: ~3.086B; the 3.09B at line 98 is correct]` |
| `4collect/HANDOFF.md` | Phase 4b, line 28 | stale | MM overfit image downscaled to ~90 tokens | Committed `train_overfit_mm.py:42,48` uses 1365 tokens (×2=2730); no ~90-token path. | `[STALE: committed train_overfit_mm.py uses full 1365 img tokens (×2=2730); ~90-token downscale was an earlier run]` |
| `5blockers/0612-blocker-v0.md` | §3 B1 row, line 77 + §9/§10 | stale | Reverse bridge `maxtext_to_hf_export.py` does not exist yet | NOW EXISTS at `maxtext-dlm-fork/scripts/maxtext_to_hf_export.py` (824 keys, round-trip gate `:281`); created Jun 14, after this 2026-06-12 snapshot. | `[RESOLVED post-0612: scripts/maxtext_to_hf_export.py built and round-trip-verified — see 07_from_base_b1_b2_progress.md A6]` |
| `5blockers/0612-blocker-v0.md` | §10 appendix lines 217-218; §1.1/§8 | stale | distilled-400 v2 AR not built yet | Built since: `07_from_base_b1_b2_progress.md:30` ("400 / 7 shards / all verified"); used in `test_training.md:154` (`...distilled_0613-400_baseViT_v2_ar`). | `[RESOLVED post-0612: distilled-400 base-ViT v2 AR built (400/7 shards) — 07 A3]` |

---

## 4. LOW-SEVERITY findings

### 4a. Living docs (edit directly)

| Doc | Location | Verdict | Issue → Fix | Code ref |
|---|---|---|---|---|
| `quick-refresh.md` | Format legend, lines 8-9 | **wrong** | "13-array source schema" → "12-array". | `ARRAY_FIELDS` = 12 fields, `prep_to_parquet.py:26-32` |
| `../to-host-chn.md` | §6.2 sharding table, lines 352-359 | misleading | Implies `sharding.py` already maps 2D `('fsdp','tp')` specs; it emits pure-FSDP largest-axis only. Note the `'tp'` column is the Option-2 plan. | `sharding.py:22-28` (fsdp_pspec), `:31-48` (param_pspecs), `:17-19` (mesh) |
| `../to-host-chn.md` | §6.4 checkpointing, line 383 | misleading | `checkpoint.py` "multi-host safe" — it's the simple single-host StandardCheckpointer; multi-host path is `train/checkpoint_mgr.py`. | `checkpoint.py:1,15-29` |
| `../to-host-chn.md` | §3 tests listing, line 234 | stale | Add 4 missing tests (test_ar_pipeline, test_grain_pipeline, test_harness_fsdp, test_multihost_datafeed). | `jax_ddrive/tests/` |
| `../to-host-chn.md` | §3 scripts listing, lines 232-233 | stale | Partial glob; add dataset/AR scripts (parquet_to_ar_with_embeds.py, parquet_to_arrayrecord.py, check_real_fsdp_shard.py, build_full_dataset.sh, verify_ar_round2.*). | `jax_ddrive/scripts/` |
| `../to-host-chn.md` | §2.3/§5(b,d), lines 158,320-322 | misleading | Note eval prep pins 200704 (paper-eval) vs SASD-train prep 784/50176 — two intentional policies. | `prep_jax_eval.py:24-25`, `prep_train_jax.py:56-57` |
| `2implementation-details/ARCHITECTURE.md` | Package layout, line 9 | misleading | `rope.py` is SPLIT-form (equiv. to rotate-half under cat(c,c)), distinct from FULL-form `apply_rope_full`. Low pri (rope.py docstring itself says "rotate-half"). | `rope.py:27-32`; `qwen2_5_text.py:140-146` |
| `2implementation-details/DATASET.md` | §3 Stage 2, line 86 | misleading | Add parenthetical: grid (1,16,14)=56 merged tokens/img; the code's "~64" is a loose upper bound (actual 56). | `prep_train_jax.py:57`, `prep_jax_eval_inputs.py:52` |
| `2implementation-details/DATASET.md` | §1 header, line 19 | misleading | "13 fields" contradicts the doc's own "12 arrays". Reword to "sample_id + 12 SASD arrays + 2 scalars". | `prep_to_parquet.py:26-32` |
| `2implementation-details/LABELING.md` | §2 provenance table, line 37 | misleading | Add omitted "keep speed" longitudinal value: {speed up / slow down / come to stop / keep speed}. | `convert_wod_e2e.py:138-144` |
| `2implementation-details/INFERENCE_DEPLOY.md` | §5 step 2/3 | stale | `launch_maxtext_sasd_tpu_frombase.sh` is a deploy-host script, not in-repo; point to `../6for_internal/test_training.md`. | absent in both repos |
| `2implementation-details/INFERENCE_DEPLOY.md` | §5 step 2 | stale | Give absolute host path `/home/kaiwen/upload_code_to_gcs.sh` (not in-repo). | `test_training.md:81`, `transfer-codebase.md:59` |
| `2implementation-details/INFERENCE_DEPLOY.md` | §3 B3, line 114 | misleading | Add `input_ids` to listed npz keys (or annotate as driver-consumed subset). | `prep_jax_eval_inputs.py:88-93,105` |
| `2implementation-details/EVAL_PIPELINE.md` | §3b body line 83 vs Files table | misleading | Standardize on repo-root-relative paths: body `ddrive_jax/eval/mm_sampler.py` → `jax_ddrive/ddrive_jax/eval/mm_sampler.py`. | both paths resolve |
| `3summary/FEATURES.md` | "Inference/serving" evidence cell, line 44 | misleading | Disambiguate: inference drivers are top-level `jax_ddrive/eval/{prep_jax_eval,jax_batch_inference}.py` (not the `ddrive_jax/eval/` package). | `ls ddrive_jax/eval/` lacks them |
| `6for_internal/updates-latest-0614.md` | 2026-06-14 code-fix bullet | misleading | `token_agreement` only removed from T2 path; still in `driver.py:153` + docstring `:17`. | `driver.py:139-148,153,17` |

### 4b. Frozen docs (annotate inline)

| Doc | Location | Verdict | Issue → Inline note | Code ref |
|---|---|---|---|---|
| `1plans/03_scaleup_tpu_spec.md` | §3.1 item 1, line 76 | stale | masks/sasd_loss/noise not copied verbatim; re-derived into single `diffusion/sasd.py`. Already under "superseded" banner. | `sasd.py:1-3` |
| `1plans/02_tpu_plan.md` | ## Checkpointing, line 35 | misleading | Production multi-host path is `train/checkpoint_mgr.py` (composite params+opt+meta+grain); plain `checkpoint.py` saves model state only. | `checkpoint.py:15-29` |
| `1plans/00_PLAN.md` | §6 Proposed repo layout, lines 70-82 | stale | Aspirational; add banner mapping proposed→actual (no fusion.py/schedulers.py/waymo_json.py/ddrive_*.yml/top-level train.py; constants are inline dataclasses). | ground_truth §7.4 |
| `4collect/OVERNIGHT_PROGRESS.md` | lines 27,84 vs 99 | misleading | "14 kernels (2-layer FSDP proxy, `test_harness_fsdp.py:231`)" vs "252/252 kernels (real 36-layer, `check_real_fsdp_shard.py:42`)". | as cited |
| `4collect/OVERNIGHT_TPU_PROGRESS.md` | Key facts, line 91 | stale | `v5litepod-8|16` superseded same night → `v6e`/us-east5-a, canonical `ACCEL=v6e-16`. | same doc lines 118,128,132,144 |
| `4collect/OVERNIGHT_TPU_PROGRESS.md` | Cost log, lines 96-100 | stale | v5litepod cost rows are pre-switch placeholders; actual runs v6e-1/-8/-16. | same doc lines 111-115,130,136 |
| `4collect/OVERNIGHT_TPU_PROGRESS-chn.md` | Launch/cost/zone blocks | stale | Same v5litepod→v6e staleness as EN mirror; `[后改为 v6e/us-east5-a;规范 ACCEL=v6e-16]`. | -chn lines 116,132 |
| `4collect/HANDOFF.md` | Phase 2, line 18 | misleading | L=1120 is the one captured parity sample, not canonical L (prod 1184, distilled 1280). | `grain_pipeline.py:42,267`; `sasd_waymo.yml:25` |
| `4collect/05_maxtext_port_progress.md` | §7.4 lines 44,63 | misleading | "336 img-tok" = doubled (2×168); single-copy is 168 = 3 cams×56 at train-res. | `sasd_waymo.yml:26`; `parquet_to_ar_with_embeds.py:6` |
| `5blockers/0613-diffusiongemma-insights-v0.md` | §2/§7 C6, lines 66-69,149 | **wrong** | NULL_ID (151666) not in `sasd.py`; cite `load_fast_ddrive_maxtext.py:18`/`sampler_sasd.py:27`/`driver.py:42`. Drop `,84` (it's the `noisy[...]=MASK_ID` line). MASK_ID is at `sasd.py:23` (correct). | as cited |
| `5blockers/0613-diffusiongemma-insights-v0.md` | §2/§7 C6, lines 69,149 | stale | Unfilled-leaf guard is `load_fast_ddrive_maxtext.py:186-188` (not :162-164, which is the `_StreamingTextGetter` block); count printed at `:193`. | as cited |
| `5blockers/0613-diffusiongemma-insights-v0.md` | §2/§6 B1, lines 70-71 | misleading | `param_mapping.py:736` = def line; `:748-749` = docstring. Reverse-branch logic (`if saving_to_hf: ... x.T.reshape`) is in hook body (`:238-243` reshape_kernel pattern). | as cited |
| `5blockers/0612-blocker-v0.md` | §3 B1 row, line 77 | misleading | 434/434 leaf report comes from called loader `load_fast_ddrive_maxtext.py:193`, not `save_fast_ddrive_params_ckpt.py` (which prints ~3.086B / `SASD_PARAM_CKPT_SAVED` at `:72,80`). Count is correct. | as cited |
| `5blockers/0612-blocker-v0.md` | §4 D / §3 B2, lines 78,220-221 | stale | Sampler vendored to fork `src/maxtext/diffusion/eval_sasd/` (pin 4b0f4f2); cited NNX source paths remain accurate. | PATCHES.md, ground_truth §2 |
| `5blockers/0612-blocker-v0.md` | §3 B3, line 79 | misleading | "479-frame prep npz already exists" — on-disk `eval_inputs/` holds only the 20+20 T2/T2' demo subset; regenerate full-479 on demand (mechanism `prep_jax_eval.py` exists). | `/home/kaiwen/data/fast-ddrive/eval_inputs/` |

---

## 5. CROSS-DOC NUMBER ISSUES (feeds the canonical-numbers table)

### N1. Parameter count — 3.75B (WRONG) vs canonical 3.086B / 3.09B

The single most widespread drift: **9 wrong occurrences across 4 docs**.

- **Canonical value:** **3.086 B** (434/434 text leaves), rounded to **3.09 B** in loss-decrease records.
- **Source of truth:** `save_fast_ddrive_params_ckpt.py:72-73`; `07_from_base_b1_b2_progress.md:33,86`; `OVERNIGHT_TPU_PROGRESS.md:169`; config dims `qwen2_5_text.py:50-56` (36L/2048/16h/2kv/128hd/11008mlp/151936vocab, tied embeds).
- **Wrong occurrences:**
  - `../to-host-chn.md` §6.1 line 348, §6.2 line 364 — *living, edit* (HIGH: mesh-sizing guidance)
  - `1plans/02_tpu_plan.md` line 10 — *frozen, annotate*
  - `1plans/04_tpu_smallscale_validation.md` line 75 — *frozen, annotate* (self-inconsistent: line 28 says 3.09B)
  - `4collect/HANDOFF.md` line 24 — *frozen, annotate*
  - `4collect/OVERNIGHT_PROGRESS.md` lines 28, 89, 134, 135 — *frozen, annotate* (self-inconsistent: line 98 says 3.09B)
- **Residual stale code comments (out of doc scope, future cleanup):** `lora.py:5`, `launch_tpu.sh:11`.
- **Confirmed-correct docs (do NOT "fix" to 3.75B):** `REPORT.md:46`, `FEATURES.md:53`, `AUDIT.md:52-58`, `OVERNIGHT_TPU_PROGRESS.md`, `07_from_base_b1_b2_progress.md`, `00_PLAN.md`, `03_scaleup_tpu_spec.md:68`, `to-host-chn.md:31,53,78`.

### N2. Verification gate count — 9 (WRONG) vs canonical 10

- **Canonical value:** **10 gates** (`cpu_mask_loss, cpu_lora, cpu_noise, cpu_sharding, cpu_eval_ports, phase1_text, phase2_sasd, phase4_vit, phase4b_mm_fwd, phase3_lora_train`).
- **Source of truth:** `run_all_verification.sh:51` (10-key summary loop), `:26` (`cpu_eval_ports` invocation); ground truth §6.
- **Wrong occurrence:** `2implementation-details/AUDIT.md` lines 27-28 (omits `cpu_eval_ports`) — *living, edit* (HIGH).
- **Confirmed-correct docs:** `to-host-chn.md` (§2.5/§7.1), `03_scaleup_tpu_spec.md:83`, `06_dataset_v2_progress.md:76` ("10-gate re-run: 9/10 PASS"). The legacy "9 gates" drift appears in NO other audited doc.

### N3. Record-schema array count — 13 (WRONG) vs canonical 12 arrays

- **Canonical value:** **12 array fields** (`ARRAY_FIELDS`: input_ids, labels, rbi, turn, scaffold, weight_vec, block_alpha, block_beta, position_ids, vision_mask, pixel_values, image_grid_thw), plus the string `sample_id` and scalars L/n_blocks.
- **Source of truth:** `prep_to_parquet.py:26-32`; `ar_dataset.py:41-50`.
- **Wrong/misleading occurrences:**
  - `quick-refresh.md` lines 8-9 ("13-array source schema") — *living, edit* (WRONG)
  - `2implementation-details/DATASET.md` §1 header line 19 ("13 fields") — *living, edit* (misleading; contradicts doc's own "12 arrays")
- **Confirmed-correct:** `README.md` line 11, `DATASET_V2.md` §1, `DATASET.md` TL;DR.

### N4. Image-token count — context-dependent (NOT wrong; flag only when context omitted)

- **Legitimate variants:**
  - **168** = single-copy, train-res (3 cams × 56/img) — `parquet_to_ar_with_embeds.py:6`; `prep_jax_eval_inputs.py:52`; grid (1,16,14) at 784/50176.
  - **336** = doubled config count (2×168) — `sasd_waymo.yml:26` (`sasd_num_image_tokens`); doubled via `concat([ie,ie],0)` in `ar_dataset.py:7`.
  - **720** = paper-eval grid-derived count at 200704 res (diagnostic only, never asserted — `driver.py:128`).
- **Flagged only where context omitted (misleading, frozen → annotate):** `05_maxtext_port_progress.md:44,63` ("336 img-tok" without noting it's doubled).
- **Confirmed-correct-with-context:** `to-host-chn.md`, `DATASET_V2.md`, `06_dataset_v2_progress.md`, `07_from_base_b1_b2_progress.md:75-76`, `test_training.md`, `0612-blocker-v0.md`, `05_review_and_fixes`.

### N5. Sequence length L — context-dependent (NOT wrong; flag only when context omitted)

- **Legitimate variants:** **1184** (WOD-E2E pseudo, prod default — `sasd_waymo.yml:25`, `grain_pipeline.py:42,267`); **1280** (distilled-from-base override — `launch_maxtext_sasd_tpu_frombase.sh:74`, `max_target_length=2576`); **1120** (one-off captured parity sample only).
- **Flagged only where context omitted (misleading, frozen → annotate):** `HANDOFF.md:18` (L=1120 presented without "one-off parity sample" caveat).

### N6. Image resolution — context-dependent (NOT wrong; flag only when context omitted)

- **Legitimate variants:** **200704/200704** (paper-eval, `prep_jax_eval.py:24-25`); **784/50176** (train/deploy → 168 tokens, `prep_train_jax.py:56-57`, `prep_jax_eval_inputs.py:53-54`).
- **Flagged where context omitted (misleading, living → edit):** `EVAL_PIPELINE.md` §3b (never names 200704/paper-eval), `to-host-chn.md` §2.3/§5.

### N7. Sharded-kernel count — context-dependent (NOT wrong; flag only when context omitted)

- **Legitimate variants:** **14** (2-layer FSDP proxy = 2×7, `test_harness_fsdp.py:231` asserts `>=4`); **252** (real 36-layer = 36×7, `check_real_fsdp_shard.py:42`).
- **Flagged where unlabeled (misleading, frozen → annotate):** `OVERNIGHT_PROGRESS.md:27,84` (14) vs `:99` (252) without labels.

### N8. Launch/deploy-host scripts referenced but not in-repo (stale, scattered)

Same class of issue across docs — host-only scripts cited as if in the bundle:

- `launch_maxtext_sasd_tpu_v2.sh` — `DATASET_V2.md:98` (does not exist anywhere; real = `train_sasd_waymo.sh`).
- `launch_maxtext_sasd_tpu_frombase.sh` — `INFERENCE_DEPLOY.md` §5, `test_training.md` §5; owner-local `/home/kaiwen/launch_maxtext_sasd_tpu_frombase.sh` (`PATCHES.md:111`), NOT in bundle.
- `upload_code_to_gcs.sh` — `INFERENCE_DEPLOY.md` §5; host-local `/home/kaiwen/upload_code_to_gcs.sh` (`test_training.md:81`, `transfer-codebase.md:59`).

---

## 6. DEDUP NOTE

Where multiple auditor groups independently surfaced the same issue, findings are merged above:

- **3.75B param drift** — flagged by the `to-host-chn`, `1plans`, and `4collect` auditor groups; merged into **N1** (canonical-number) plus per-doc rows in §2/§3b/§4b. The reciprocal "3.09B is correct, do NOT change" confirmations (REPORT/FEATURES/AUDIT/07/03/00) are noted in N1 rather than listed as separate findings.
- **9-vs-10 gate count** — `AUDIT.md` is the only doc carrying the wrong value (§2); the to-host-chn / 03 / 06 auditors confirmed their docs are correct. Merged into **N2**.
- **`checkpoint.py` "multi-host safe"** — found by both the `to-host-chn` auditor (§6.4, living) and the `1plans` auditor (`02_tpu_plan.md:35`, frozen). Same root cause (production path is `train/checkpoint_mgr.py`); kept as two rows (different files/frozen-status) but cross-referenced — §3a/§4a (living) and §4b (frozen).
- **`token_agreement` "removed"** — surfaced in both `test_training.md` §7 (medium) and `updates-latest-0614.md` (low); same code reality (`driver.py:153,17`); kept as separate rows by doc, merged in reasoning.
- **launch-host-script-not-in-repo** — three distinct scripts flagged by the `DATASET_V2`, `INFERENCE_DEPLOY`, and `test_training` auditors; consolidated into **N8** while retaining per-doc rows in §3/§4.
- **eval-vs-train resolution (200704 vs 784/50176)** — flagged for `EVAL_PIPELINE.md` (medium) and `to-host-chn.md` (low); both legitimate policies, issue is omitted context; consolidated in **N6**.

**Frozen/living policy reminder:** all `1plans/*` and `4collect/*` findings (11 defects) are **frozen → inline bracketed annotations only**. All `2implementation-details/*`, `3summary/*`, `5blockers/*`, `6for_internal/*`, `README.md`, `quick-refresh.md`, and `to-host-chn.md` findings (21 defects) are **living → edit directly**.