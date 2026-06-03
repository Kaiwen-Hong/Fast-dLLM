# Fast-dDrive → JAX/MaxText (TPU-ready) — Plan & Feasibility

Owner: kaiwen · Started 2026-06-02 · Status: **Phase 0 (env + oracle) in progress**

## 0. TL;DR verdict

**FEASIBLE, MEDIUM difficulty for the chosen bar.** The bar is **"JAX trains with decreasing SASD
loss on the 5090, architecture TPU/MaxText-ready"** — *not* paper quality/speed. The diffusion loss,
block-causal masking, Qwen text backbone, weight-conversion recipe, and Optax training loop already
exist in your `jax-mdlm-handoff` project and transfer almost directly. A **text-only** path reaches the
loss-decrease milestone **without** the two hard parts (Qwen2.5-VL vision tower; reverse-engineered
sampler), because **M-RoPE is a no-op for text-only training**. Vision is added afterward to make it the
true model. Estimated: **loss-decrease milestone ~1 week; full multimodal loss-decrease +~1 week.**

## 1. What Fast-dDrive is (authoritative, from `config.json`)

Qwen2.5-VL-**3B** block-diffusion VLA for Waymo WOD-E2E, saved **fp32** (16.3 GB, 4 shards).
- Text decoder: **36 layers, hidden 2048, FFN 11008, 16 heads / 2 KV (GQA), tied embeds, vocab 151936,
  RoPE θ=1e6, M-RoPE `[16,24,24]`, RMSNorm eps 1e-6, SiLU**.
- Vision: ViT depth 32, hidden 1280, 16 heads, patch 14, spatial_merge 2, window 112, full-attn [7,15,23,31], out 2048.
- Output = a **JSON scaffold**, 4 causal sections (critical_objects → explanation → future_meta_behavior →
  trajectory) filled by **block-diffusion** (bidirectional within `bd_size=32`, causal across).
- Training = **SASD**: per-section loss weights {traj 3.0, fmb 2.0, crit 1.5, expl 1.0} + per-section
  Beta noise schedule + deep-JSON-scaffold freezing. Full algorithm: `docs/01_pytorch_reference_algorithm.md`.

## 2. Decisions (locked with user, 2026-06-02)

| # | Decision |
|---|----------|
| Bar | **Loss decreases locally**; TPU-ready design; don't chase RFS/ADE/speed. |
| Weights | **Both** — convert trained ckpt (parity oracle + fine-tune) AND support Qwen2.5-VL-3B base (SASD-from-base). |
| Compute | Validate on **RTX 5090** now (`jax[cuda12]`); design sharding/Orbax for TPU; defer multi-host. |
| Data | Waymo not needed for the milestone (2 `sample.json` examples suffice). Help user download in parallel. |

## 3. Environment status

- **`ddrive` (PyTorch oracle)** — BUILT & verified: torch 2.11.0+cu128, transformers 4.57.1 on sm_120.
  (Deviated from pinned `torch==2.6.0+cu124`, which lacks Blackwell kernels.)
- **Checkpoint** — DOWNLOADED: `Efficient-Large-Model/Fast-dDrive` (16.3 GB) in HF cache on the data drive.
- **`ddrive_jax` (JAX)** — TODO: conda env, `jax[cuda12]` (driver 570 ⇒ cuda12, not cuda13), flax(nnx),
  optax, omegaconf, orbax, safetensors. Use an `activate.sh` that `unset LD_LIBRARY_PATH` (reference pitfall).
- Big artifacts: `/home/kaiwen/data/fast-ddrive/{ckpt,converted_jax,ref_logits,logs}`.

## 4. Feasibility per component (for the chosen bar)

| Component | Rating | Note |
|---|---|---|
| Text-decoder forward | **EASY** | `jax-mdlm-handoff/code/models/qwen3.py` is the same Qwen family; scale config, add sharding. |
| Weight conversion | **EASY–MEDIUM** | `a2d.py` recipe + standard Qwen2.5-VL key names; tied embeds; mean-init not needed (mask row exists). |
| SASD loss + masks | **MEDIUM** | Reuse `mdlm.py`/`bd3lm.py`; add doubled-seq, hybrid mask, section weights, per-section Beta, complementary loss. |
| Scaffold/section map | **MEDIUM** | Port `section_utils.py` boundary logic to build `response_block_idx`, `block_to_section`, `scaffold_mask`. |
| Loss-decrease validation | **EASY** (once above) | Overfit 2 samples; mirrors handoff Phase 7. |
| Vision tower + M-RoPE | **MEDIUM–HARD** (Phase 4) | New Qwen2.5-VL ViT (dyn-res, window attn, spatial_merge, M-RoPE 3D); ref ViT is square/224-only. |
| Sampler / KV-cache | **HARD** (deferred) | Only in remote `generation_utils.py`; not in the bar. |

**Single hardest in-scope risk:** faithfully reproducing the **doubled `[noisy|clean]` hybrid block-causal
mask + complementary loss** so the JAX loss matches a NumPy recomputation (Phase 2 hard gate). Everything
upstream (backbone, weights) is de-risked by the reference project.

## 5. Reuse map (jax-mdlm-handoff → here)

REUSE ~as-is: `code/diffusion/{mdlm.py,bd3lm.py,sampler.py,schedulers.py}`, `code/models/qwen3.py`
(+ sharding pattern from `code/models/llada.py:40-72`), `code/a2d.py`, `code/train.py`,
`code/utils/rope.py`, `code-fork/README.md` (MaxText loss_fn/attention-mode diff), `references/*NOTES.md`
(numerical-parity test scaffolding). BUILD new: Qwen2.5-VL ViT+fusion+M-RoPE, deep-scaffold/section map,
SASD loss extensions, Waymo JSON data loader, Orbax+sharding wiring.

## 6. Proposed repo layout (`Fast-dLLM/jax_ddrive/`)

```
jax_ddrive/
├── docs/            00_PLAN.md, 01_pytorch_reference_algorithm.md, HANDOFF.md (living)
├── ddrive_jax/
│   ├── models/      qwen2_5_text.py (← qwen3.py), vision_qwen25vl.py (Phase 4), fusion.py
│   ├── diffusion/   sasd_loss.py, masks.py (hybrid block-causal), scaffold.py (← section_utils), schedulers.py
│   ├── data/        waymo_json.py (JSON→arrays collator, jax/numpy)
│   ├── convert/     hf_to_jax.py (← a2d.py)
│   ├── configs/     ddrive_text_tiny.yml, ddrive_text_full.yml, ddrive_vl_full.yml
│   └── train.py
├── tests/           test_loss_parity.py, test_mask.py, test_convert.py, test_scaffold.py
└── scripts/         capture_oracle_logits.py, overfit_smoke.sh, make_jax_env.sh
```
Code adapted from the reference is copied+modified (provenance noted), not cross-repo imported.

## 7. Phased plan (exit criterion + parity check each)

- **Phase 0 — Oracle + archaeology** *(in progress)*. Env built; ckpt downloaded; remote `modeling.py`/
  `section_utils.py`/`generation_utils.py` read & documented; run `run_chatbot.py`.
  *Exit:* original model emits valid JSON trajectory on the example. *Artifact:* capture `ref_logits.npy`
  for one fixed (text-only masked) forward as the parity oracle.

- **Phase 1 — JAX text backbone + weight load.** Scale `qwen3.py`→Fast-dDrive text config; port LM weights
  with the `a2d`-derived loader (text-only; no vision/M-RoPE-3D).
  *Exit:* forward runs, no NaN, param count matches HF. *Parity:* JAX vs oracle logits, **rel-diff < 1e-3**.

- **Phase 2 — SASD loss + masks + scaffold (HARD GATE).** Doubled `[noisy|clean]` forward, hybrid
  block-causal mask, scaffold freeze, section weights, per-section Beta, complementary loss.
  *Exit:* one backward + AdamW step on `sample.json` (text-only). *Parity (gate):* JAX SASD loss vs a
  from-scratch NumPy recomputation on the same batch, **rel-diff < 1e-3** (algorithmic-correctness proof,
  independent of convergence — mirrors handoff Phase 6).

- **Phase 3 — Loss decreases (THE MILESTONE).** Overfit the 2 samples ~200 steps, both from converted ckpt
  (starts low) and from Qwen2.5-VL-3B base (drops).
  *Exit:* loss **monotonically decreases**, no NaN, near-memorization. Save `logs/loss_curve.csv`.

- **Phase 4 — Vision tower + fusion + M-RoPE.** Build Qwen2.5-VL ViT (dyn-res, window attn, spatial_merge,
  3D M-RoPE) + image-token splice; convert `visual.*` weights.
  *Exit:* full multimodal forward; **loss decreases on the real image-containing sample**.
  *Parity:* JAX image embeds vs oracle, **rel-diff < 1e-2**; end-to-end logits < 1e-2.

- **Phase 5 — TPU-readiness.** Add `PartitionSpec` sharding (from `llada.py`), Orbax checkpoint, mesh
  abstraction; validate single-device (mesh=1) on the 5090 is identical.
  *Exit:* sharded train step runs unchanged at mesh=1; a written TPU mesh/sharding plan.

Deferred (out of bar): decoder parity (SD/SS/multi-traj), real Waymo training, ADE/RFS metrics.

## 8. Waymo download (parallel track — your action needed)

Not required for the milestone, but to enable it later: (1) register at https://waymo.com/open with a
Gmail and accept the WOD-E2E license; (2) install/auth gcloud (`gcloud auth login` with that account).
Then I gsutil-pull the WOD-E2E train/val tfrecords into `/home/kaiwen/data/fast-ddrive/waymo/`, write the
tfrecord→Fast-dDrive-JSON converter (repo's is "coming soon"), and stand up the separate `autovla`
metrics env (tensorflow 2.12 + waymo-open-dataset-tf-2-12-0) for ADE/RFS.

## 9. Open risks

- Hybrid mask + complementary loss exact reproduction (Phase 2 gate) — primary risk.
- ViT dynamic-resolution + M-RoPE fidelity (Phase 4) — secondary.
- TPU mesh/sharding for a tied-embedding 3B + ViT — design-time, low (pattern exists in `llada.py`).
