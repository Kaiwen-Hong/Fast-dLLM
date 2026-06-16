# Doc-Fix Pass Report — Fast-dDrive Project

## 1. Totals

| Metric | Count |
|---|---|
| Docs touched | 24 |
| Edits applied (living, in-place) | 23 |
| Annotations added (frozen, bracketed) | 20 |
| Findings skipped | 1 |

Edits/annotations span 24 docs; 14 living docs received in-place corrections, 8 frozen docs received bracketed annotations (`[订正]`/`[STALE]`/`[SUPERSEDED]`/`[已解决]`), and 2 docs (`0612-blocker-v0.md`, mixed) carried both styles per their snapshot nature.

## 2. Applied Changes

### Living docs (in-place edits)

| Doc | Location | Before -> After |
|---|---|---|
| to-host-chn.md | §6.1 Mesh | `3.75 B 扩展得很好` -> `3.09 B 扩展得很好` |
| to-host-chn.md | §6.2 Option 1 | `跑完整 3.75 B` -> `跑完整 3.09 B` |
| to-host-chn.md | §6.2 Sharding table | +note: `sharding.py` only emits pure-FSDP; `'tp'` cols are Option-2 target |
| to-host-chn.md | §6.4 Checkpointing | `checkpoint.py` multi-host safe -> single-host StandardCheckpointer; real path = `checkpoint_mgr.py` |
| to-host-chn.md | §3 tests glob | 5 test files -> 9 (added ar/grain/harness_fsdp/multihost) |
| to-host-chn.md | §3 scripts glob | made illustrative + named Phase 6/8 AR scripts |
| to-host-chn.md | §2.3 eval prep | +note: eval 200704 vs train 784/50176 are intentional per-purpose policies |
| ARCHITECTURE.md | Training data flow §60 | `where(vision_mask, ...)` -> `.at[img_pos].set(image_embeds)`; vision_mask is carried-but-unused |
| AUDIT.md | Confirmed issues tail | `9 gates` -> `10 gates` (added `cpu_eval_ports`) |
| DATASET.md | §1 header | `13 fields` -> `sample_id + 12 SASD arrays + 2 scalars` |
| DATASET.md | §3 Stage 2 | `→ 16×14 grid` -> `grid (1,16,14) = 56 merged tokens/img` (corrects ~64 comment) |
| DATASET_V2.md | §1 Embeds provenance | `assert rel < 2e-4` -> two-tier guard (max-rel <1e-3 AND rolling median <1e-4 over ≥8 shards) |
| DATASET_V2.md | §4 Checkpoint/resume | `launch_maxtext_sasd_tpu_v2.sh` -> uncommitted internal launcher; real = `scripts/train_sasd_waymo.sh` |
| EVAL_PIPELINE.md | §3b line 83 | `ddrive_jax/eval/mm_sampler.py` -> `jax_ddrive/ddrive_jax/eval/mm_sampler.py` |
| EVAL_PIPELINE.md | §3b after mm_sampler | +resolution note: 200704 paper-eval vs 784/50176 deploy (168 image tokens) |
| INFERENCE_DEPLOY.md | §3 B3 npz keys | prepended `input_ids` to key list |
| INFERENCE_DEPLOY.md | §5 step 2 launch | marked `launch_maxtext_sasd_tpu_frombase.sh` deploy-host-only (`/home/kaiwen/...`) |
| INFERENCE_DEPLOY.md | §5 step 2 upload | marked `upload_code_to_gcs.sh` deploy-host-only |
| LABELING.md | §2 provenance table | longitudinal `(speed up / slow down / come to stop)` -> `+ keep speed` |
| FEATURES.md | Inference row line 44 | stale 52-frame metrics -> full-479 rated-val (0.839/2.072/7.929 vs PT 0.814/1.990/7.914) |
| FEATURES.md | Inference row evidence path | `eval/{...}` -> `jax_ddrive/eval/{...}` |
| test_training.md | §7 STAGE 3 note | `token_agreement ... removed` -> removed only from T2 path; persists in `ref_output` branch |
| test_training.md | §5 STAGE 1 smoke | launch wrapper marked owner-local `/home/kaiwen/...`, not in bundle |
| updates-latest-0614.md | 06-14 fix bullet | `删坏指标 token_agreement` -> only removed from T2/target_ids path; persists in ref_output branch |
| quick-refresh.md | Format legend | `13-array source schema` -> `12-array` |

### Frozen docs (bracketed annotations)

| Doc | Location | Annotation summary |
|---|---|---|
| 0612-blocker-v0.md | §1.1 parity micro-numbers | `[订正]` rel-max figures are source-comment claims, not re-measured; 479-frame ADE/RFS is corroborated |
| 0612-blocker-v0.md | §3 B1 434/434 leaves | `[订正]` leaf count printed by loader, not save script |
| 0612-blocker-v0.md | §3 B1 反向工具不存在 | `[已解决 post-0612]` `maxtext_to_hf_export.py` written, 824/824 round-trip |
| 0612-blocker-v0.md | §3 B2 vendor | `[已解决 post-0612]` vendored to `eval_sasd/` (pin 4b0f4f2) |
| 0612-blocker-v0.md | §3 B3 479-frame npz | `[订正]` mechanism ready; disk holds only 40 T2/T2' demo npz |
| 0612-blocker-v0.md | §10 v2 AR 未做 | `[已解决 post-0612]` distilled-400 base-ViT v2 AR built (400/7 shards) |
| 0613-diffusiongemma-insights-v0.md | §2 修正1 | MASK/NULL cites split (NULL=151666 not in sasd.py:84); guard at `:186-188`; reverse branch at `:766+` |
| 0613-diffusiongemma-insights-v0.md | §7 C6 row | same citation fixes (evidence cell) |
| 0613-diffusiongemma-insights-v0.md | Appendix file:line index | same citation fixes (index) |
| 00_PLAN.md | §6 Proposed repo layout | `[STALE]` implemented tree differs from proposed (no schedulers/waymo_json/fusion/configs/train.py) |
| 02_tpu_plan.md | §Mesh bullet 1 | `[订正]` 3.75B -> 3.086B/3.09B |
| 02_tpu_plan.md | §Checkpointing | `[订正]` checkpoint.py single-host; multi-host = `checkpoint_mgr.py` |
| 03_scaleup_tpu_spec.md | §3.1 graft item 1 | `[SUPERSEDED]` re-ported as consolidated `sasd.py` (bit-exact), not verbatim copy |
| 04_tpu_smallscale_validation.md | §3 table row 1 | `[订正]` 3.75B -> 3.086B/3.09B (matches doc line 28) |
| 05_maxtext_port_progress.md | 7.4 table + T5 log | `[订正]` 336 = doubled 2N; single = 168 (3 cam × 56) |
| HANDOFF.md | Phase 3 milestone | `[订正]` 3.75B -> ~3.086B/3.09B (434 text leaves) |
| HANDOFF.md | Phase 2 L=1120 | `[STALE]` one captured sample; canonical L=1184 (prod), 1280 (distilled) |
| HANDOFF.md | Phase 4b ~90 tokens | `[STALE]` committed code uses full 1365 img tokens (x2=2730) |
| OVERNIGHT_PROGRESS.md | 4× 3.75B sites | `[订正]` -> ~3.086B/3.09B (lines 28/89/134/135) |
| OVERNIGHT_PROGRESS.md | 3× kernel-count sites | `[订正]` 14 = 2-layer proxy harness; 252 = real 36-layer (lines 27/84/99) |
| OVERNIGHT_TPU_PROGRESS.md | Launch scripts + Cost log | `[SUPERSEDED]` v5litepod -> v6e/us-east5-a; canonical ACCEL=v6e-16 |
| OVERNIGHT_TPU_PROGRESS-chn.md | Launch/quota/cost-log (3 sites) | `[STALE]` v5litepod/v5e -> v6e @ us-east5-a (v5e PERMISSION_DENIED) |

## 3. Skipped Findings

| Doc | Location | Reason |
|---|---|---|
| ARCHITECTURE.md | Package layout line 9: `rope.py … rotate-half` | Doc is correct/defensible. `rope.py:27-32` uses split-form, but the source file's own docstring (`rope.py:3`) says "Standard HF rotate-half convention" — the doc faithfully echoes source naming. Split-form and rotate-half are numerically equivalent under the `cat(c,c)` cos/sin pattern. Finding self-rated low priority; skipped per "doc already correct -> SKIP" rule. |

## 4. Notes / Anomalies

- **No code_reality failures.** Every applied finding (42 across 23 docs) was independently re-verified against live code/filesystem before editing; all confirmed REAL. The only skip was a *defensible-as-written* case, not a false positive.
- **Recurring canonical-number drift — `3.75B` -> `3.086B/3.09B`.** The single most common error (8+ sites across 6 docs: to-host-chn, 02_tpu_plan, 04_tpu_smallscale, HANDOFF, OVERNIGHT_PROGRESS, OVERNIGHT_TPU). Canonical source: `save_fast_ddrive_params_ckpt.py:72-73` prints `{n_el/1e9:.3f}B`; 434 leaves / 3.086B confirmed in progress doc 07. Several docs were *internally inconsistent* (e.g. 04_tpu_smallscale said 3.09B on line 28 but 3.75B on line 75; to-host-chn said 3.09B in 4 places but 3.75B in 2).
- **Recurring `token_agreement` mischaracterization.** Two docs (test_training, updates-latest-0614) claimed the metric was "removed/deleted"; it was only dropped from the T2/target_ids path and still lives in the `ref_output` parity branch (`driver.py:153` + docstring `driver.py:17`).
- **Schema count drift — "13 fields/arrays" -> 12.** Hit DATASET.md and quick-refresh.md; `prep_to_parquet.py:26-32` `ARRAY_DTYPES` has exactly 12 array fields (`sample_id` is a string, `L`/`n_blocks` are scalars).
- **Two ghost launch scripts** referenced as in-repo but exist only on the owner-local host (`/home/kaiwen/launch_maxtext_sasd_tpu_frombase.sh`, `/home/kaiwen/upload_code_to_gcs.sh`) and one fully nonexistent (`launch_maxtext_sasd_tpu_v2.sh`, found nowhere — only in doc prose). Corrected in INFERENCE_DEPLOY, DATASET_V2, test_training.
- **Doubled-count ambiguity (336 vs 168).** `336 img-tok` is legitimate (the `[2N,D]` doubled embed count per `sasd_waymo.yml:26` / `parquet_to_ar_with_embeds.py`), not an error — annotated for provenance (single 168 = 3 cam × 56 -> doubled 336) rather than corrected.
- **Frozen-doc discipline maintained.** Frozen snapshots (blockers, plans, overnight logs) received only additive bracketed annotations in the doc's own language (English/Chinese), preserving historical narrative; per-finding `suggested_fix` guidance ("do NOT rewrite") was followed even where `is_frozen=false` for resolved-since items (0612-blocker).
- **Out-of-scope mirror noted:** `OVERNIGHT_TPU_PROGRESS-chn.md` was confirmed a faithful mirror of the EN doc's staleness and was handled; no other -chn mirrors were in scope.