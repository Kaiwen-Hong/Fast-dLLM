# 05 — Code/Doc Review + Fixes (2026-06-14)

Frozen record of the 2026-06-14 two-reviewer review pass and the fixes/actions taken. (Per the
docs convention `1plans/` is write-once; this is a record of what happened, not a forward plan.)

## Review setup
- **codex** `gpt-5.5` (xhigh), read-only, whole repo + docs — deep code-correctness + doc-vs-code.
- **Claude** 10-agent workflow (6 doc-verify + 4 code-verify), each cross-checking every concrete
  doc claim against the actual source.
- The two were **complementary**: Claude found the HIGH bf16-npz crash; codex found the scalar-only
  text leak, the `verify_against`-on-trained spurious-fail, and the eval-input-artifact gap. **Every
  finding was adversarially re-verified against the source before any change** (e.g. the bf16-npz
  crash was reproduced in-env; the "168 image tokens" claim was confirmed correct = 3 cams × 56, not
  changed).

## Code fixes — committed to fork `master` `20d9cd1`
| file | fix |
|---|---|
| `eval_sasd/driver.py` | **(HIGH)** bf16 `image_embeds` `.view(ml_dtypes.bfloat16)` — `np.savez` turns bf16 into `\|V2` void bytes → the precomputed-embeds ("TPU skips ViT") path crashed. Removed the misaligned `token_agreement`-vs-target (a broken metric: denoised scaffold vs flat GT, never aligned) → added `co_match`/`fmb_match`; `--show_text` gates the generated-text print OFF (scalar-only export); added scalar diagnostics `L`/`n_image_tokens`/`n_mask_remaining`; vlog records `threshold`. |
| `eval_sasd/embedding_parity.py` | **(HIGH)** same bf16 reference `.view`; fold `bf16_vs_reference` cosine into the PASS/FAIL verdict. |
| `ddrive_jax/convert/hf_to_jax.py` | **(HIGH)** `_load_all_tensors` now bf16-safe (was the one un-hardened ViT loader; bit-identical for F32) → `prep --with_embeds` works on the bf16 base snapshot. *(in the Fast-dLLM repo working tree — uncommitted there.)* |
| `scripts/maxtext_to_hf_export.py` | docstring `float32`→`bf16`; `verify_against` documented as base-ckpt-only. |

**Verified:** `py_compile` (all) + functional repro (bf16 `.view` recovers values; `_load_all_tensors`
bit-identical for F32) + module import (`driver._sections`, `IMAGE_ID`, `run_eval.show_text`) +
`EVAL_SASD_SELFCONTAINED_PASS` (fork-only property preserved). NOT yet run on a real TPU.

## Doc / script fixes
- **INFERENCE_DEPLOY.md**: parity `5e-3`→`5e-2`; `code/maxtext_fork.tgz`→`fastddrive-<TS>.tgz`(+LATEST);
  `--verify_against`→`verify_against=`; `$DDRIVE`/`$FORK` defined; `168 = 3×56` clarified; T2 status made
  honest (30k regressed; `token_agreement` was a broken metric); metric list updated.
- **0613-test_training.md**: §6 trained export drops `verify_against` (else spurious `B1_ROUNDTRIP_FAIL`);
  §7a npz are **OWNER-built + shipped to `eval_inputs/`** (the TPU has no torch) — skip prep on the TPU;
  §7b metric list + T2 status note.
- **PATCHES.md**: dual vendoring pins (`b18e861` + `4b0f4f2`); validation repointed to
  `launch_maxtext_sasd_tpu_frombase.sh` — fixes codex's `v6e-1` / `ckpt→resume` over-claim / stale-launcher
  in one move (the frombase script genuinely does ckpt@6→resume and defaults to v6e-1).
- **README.md** "two living docs"→"the living docs"; **to-host-chn.md** removed the nonexistent
  `to-host.md` ref + added an uncommitted-B1/B2/B3 note; **transfer-codebase.md** `~38 MB`→"几十 MB
  (脚本打印实际大小)".
- **launch_maxtext_sasd_tpu_frombase.sh**: old `maxtext_fork.tgz` + `jax_ddrive.tgz` two-tarball scheme
  → `fastddrive-LATEST.txt` → `<TS>.tgz` + `~/fastddrive-current` symlink; PYTHONPATH / cd / pip paths
  fixed (the old scheme would 404 on the internal pull). `bash -n` OK.

## Actions executed (2026-06-14)
1. **Committed** fork B1/B2/B3 + fixes → `master` `20d9cd1` (fork has no git remote → local commit,
   consistent with the existing SASD history on master).
2. **Re-published** the code bundle → `gs://<bucket>/code/fastddrive-20260614_021422.tgz` (38M);
   `fastddrive-LATEST.txt` updated. The internal pull (`0613-transfer-codebase.md`) now gets the fixes.
   *(Note: the published tarball packs the WORKING TREE, so it includes the uncommitted ddrive_jax fix.)*
3. **Built + shipped the eval npz set** → `eval_inputs/` — see below.

## Eval npz set (action 3)
- Built **on-device** (pixel_values, no torch needed on the TPU side) in the ddrive env from
  `train_targets_distilled_400.json` + `val_rated_targets.json`, snapshot `_t2_overfit30k_export`
  (ViT = frozen base, copied verbatim by B1), resolution 784/50176 → **168 image tokens** confirmed.
  Builder: `/tmp/batch_prep_eval.py` (one processor load, loops 20 train + 20 val). `section_utils`
  must be on PYTHONPATH (it lives in the Fast-dDrive HF snapshot's remote-code dir).
- **40 npz** (`train_s00..19`, `val_s00..19`), each carrying `pixel_values` + scaffold + `target_ids`
  **and** `image_embeds[bf16 168×2048]` (added by `/tmp/augment_embeds.py`: the fork ViT run once in the
  jax env over each npz's pixel_values). One artifact serves BOTH the skip-ViT path and on-device + parity.
- **Validated on real weights** (jax / 5090): the driver on `train_s00` reports
  `img_source=precomputed_embeds`, `n_image_tokens=168`, `n_mask_remaining=0` — A1 read works end-to-end,
  scalar-only (no text printed), new `co_match`/`fmb_match` emit. embedding_parity: `fp32_vs_reference`
  cosine **0.9999990** (embeds path correct) BUT `fp32_vs_bf16` cosine **0.9984** → **EMBED_PARITY_FAIL**
  vs the 0.999 gate. That is a NUMERICAL finding, not a bug: the BASE ViT's bf16-matmul drift is larger
  than the release-ViT calibration (0.99966 / max_rel 3e-2). It argues FOR the precomputed fp32→bf16
  embeds (negligible loss) over running the bf16 ViT. Internal TPU: prefer precomputed embeds; recalibrate
  the cosine gate (~0.998) or accept the drift, and measure it on the TPU itself.
- Uploaded to `gs://<bucket>/eval_inputs/` (internal `fileutil cp` → `$DATA_ROOT/eval_inputs/`).

## Open follow-ups (NOT done — flagged)
- **T2 still not verbatim**: the overfit regressed 12k→30k (traj Δ 1.12 m → 28.8 m). This is a
  **training-recipe** issue (the cosine LR schedule is recomputed on resume, perturbing memorised
  digits), NOT a pipeline bug. Needs a constant-/non-jumping-LR re-train. The trajectory Δ is the real
  signal — the old `token_agreement` numbers were from the broken metric.
- **`--with_embeds` env**: no local env has torch + jax together, so the skip-ViT npz embeds were
  produced via a 2-step (ddrive builds pixel_values → jax runs the fork ViT). The A5 ddrive_jax-loader
  fix is unit-verified but was NOT exercised through `prep --with_embeds` end-to-end (no combined env).
- **Benign smells left** (no correctness impact, flagged in review): `hf_to_jax` tie-check log message;
  `qwen2_5_text` two opposite cos/sin tuple orderings (B2 uses only the M-RoPE path).

## Follow-up (same day): parity reframed to a diagnostic + training-exact embeds
Per the design discussion, two more changes landed:
- **(a) `embedding_parity` reframed.** The canonical inference path skips the ViT (precomputed embeds),
  so the verdict now GATES on `fp32_vs_reference` — do the shipped embeds reproduce a fresh fp32 ViT?
  (→ **cosine 1.00000, EMBED_PARITY_PASS**). `fp32_vs_bf16` / `bf16_vs_reference` are reported DIAGNOSTICS
  (the bf16-ViT drift the canonical path never incurs). No-reference (on-device) falls back to gating on
  `fp32_vs_bf16`. Docstring + INFERENCE_DEPLOY §3/§4 + test_training §8 updated to match. This retires the
  earlier "EMBED_PARITY_FAIL @ 0.9984" — it was gating on the wrong (bf16-ViT) number.
- **(b) Training-exact embeds for the 20 train npz.** Replaced the recomputed embeds with the EXACT ones
  the model trained on, pulled from the v2 AR dataset (`…distilled_0613-400_baseViT_v2_ar`) by sample index
  (verified: npz prompt == AR `input_ids` prefix + matching `image_grid_thw`). Also swapped in the AR's
  training pixels (f16) so a fresh-ViT recompute reproduces them (parity 1.00000). The recomputed-vs-training
  cosine was 0.99908–0.99999 → (b) removed a small but real T2 embed confound. Re-uploaded the 20 train npz.
  (Val npz keep recomputed embeds — T2' only needs valid JSON.) Scripts: `/tmp/swap_training_embeds.py`.
- Net: precomputed-embeds path validated end-to-end on real weights; parity is a clean PASS + bf16 diagnostic;
  T2 memorisation now has zero embed confound on the train set.
