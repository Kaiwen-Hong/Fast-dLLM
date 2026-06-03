# Fast-dDrive JAX port — audit & resolutions

A multi-agent adversarial audit (35 agents across 5 dimensions: loss/mask correctness,
backbone/ViT correctness, sampler/training correctness, verification coverage, doc accuracy;
each finding independently verified before acceptance) reviewed the port.

**Verdict: no correctness defect was demonstrated in shipping JAX code.** Of 45 raw findings,
6 were confirmed real — all **verification-coverage gaps or doc inconsistencies**. The audit
also *rejected* several overreaching raw findings (e.g. a claimed PyTorch causal-loss "bug" in
the reference, not the port; and a wrong "complementary mask = bitwise inverse" assertion —
`im_end` is force-masked in both halves).

## Confirmed issues and resolutions
| # | Issue | Resolution |
|---|---|---|
| 1 | Default training path (LoRA) had no gate (only full-FT was gated) | Added `phase3_lora_train` smoke gate to `run_all_verification.sh` (trains LoRA, asserts loss decreases, no NaN) |
| 2 | `noise.py` `make_batch` (stochastic Beta path) untested | Added `tests/test_noising.py` (scaffold freeze, im_end always masked in both halves, complementary inverse on free positions, Beta(1,1)≈0.5 rate, edge cases) |
| 3 | ViT `patch_embed` conv-as-linear reshape not directly gated (the `iso` metric bypassed the JAX conv) | `parity_vit.py` now gates `patch_embed` rel-max < 5e-3 (catches reshape/transpose bugs; tolerates the ~6.5e-4 cuDNN seed) |
| 4 | `lora.py` `LoRALinear` forward/freeze untested | Added `tests/test_lora.py` (zero-init pass-through, base+`(x@A@B)·scale`, no-bias branch, pure-LoRA param structure/count) |
| 5 | HANDOFF Phase-5 status self-contradiction ([x] vs [ ]) | Fixed: Orbax + FSDP spec marked done; only physical TP (Phase 5b) open |
| 6 | HANDOFF Phase-2 loss number `7.85e-8` vs REPORT `7.9e-8` | Aligned to `7.9e-8` (the `%.2e` gate output) |

## Backlog (flagged, not blocking — currently ride on end-to-end parity)
Direct element-wise weight-conversion check; `mrope_cos_sin` isolation test; Orbax round-trip
regression gate; `remat=True/False` equivalence test; the non-deep `compute_response_block_idx_simple`.

After fixes, `run_all_verification.sh` runs **9 gates** (cpu_mask_loss, cpu_lora, cpu_noise,
cpu_sharding, phase1_text, phase2_sasd, phase4_vit, phase4b_mm_fwd, phase3_lora_train) — all PASS.

## Second audit (2026-06-03) — eval + training code (11 agents, review→verify)
Reviewed `convert_wod_e2e.py`, `eval/{mm_sampler,scaffold,rope_index}.py`,
`eval/{prep_jax_eval,prep_train_jax,jax_batch_inference}.py`, `train_waymo_sasd_jax.py`.
**7 raw findings → 2 confirmed, 5 refuted. No correctness defect in the parity-validated path.**

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | major (latent) | `messages_from_prompt` duplicated leading images when a prompt has ≥2 `<image>` placeholders (vs the reference's running cursor). Never triggered — all real data has exactly 1 placeholder, so both impls agree byte-for-byte (why the 52/479-frame eval passed). | **Fixed** to mirror `batch_inference.generate` exactly (running `image_idx`); verified identical on 1-placeholder, correct (no dup) on 2–3. |
| 2 | minor (by design) | converter lateral pseudo-label ∈ {go straight, turn left, turn right}, never "lane follow" (raw WOD-E2E has no text labels). | **Documented** in `build_meta_behavior` (trajectory is the real signal; no canonical lateral parity assumed). |

Refuted (verified non-issues): a claimed FMB trailing-space divergence (the NULL-padded target is in fact *more* scaffold-correct than `sample.json`'s trailing-space form — confirmed via the reference `section_utils` scaffold detector); the multi-image bug re-raised as "dead branch"; `embed_tokens` called twice for dtype (pure gather, XLA CSE dedups, no grad effect); Orbax saving model-state-only (operational, not correctness; the reference ships no training code); "come to stop" longitudinal vocab (intended weak label).
