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
