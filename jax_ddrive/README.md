# jax_ddrive — Fast-dDrive in JAX/Flax-NNX (MaxText-style)

JAX port of Fast-dDrive (Qwen2.5-VL-3B block-diffusion VLA for Waymo driving), built to
run on the RTX 5090 and scale to Waymo TPU. Adapted from `DLM-policy4AV/jax-mdlm-handoff`.

**Start here:** [`docs/3summary/REPORT.md`](docs/3summary/REPORT.md) — what's done, results, how to reproduce.

Docs live in four folders under `docs/` (`3summary` + `2implementation-details` are kept current;
`1plans` + `4collect` are frozen historical record):

**`3summary/`** — current state, read first
| Doc | Purpose |
|---|---|
| `3summary/REPORT.md` | Build summary + results + reproduce commands (read first) |
| `3summary/FEATURES.md` | Capability matrix (done/verified/remaining) |

**`2implementation-details/`** — how it works + how it was verified
| Doc | Purpose |
|---|---|
| `2implementation-details/ARCHITECTURE.md` | Module layout + training data-flow + parity methodology |
| `2implementation-details/DATASET.md` | **Training dataset SSOT** — record schema, artifacts on disk/GCS, build chain, how a train step consumes it, round-2 verification, v2 roadmap |
| `2implementation-details/EVAL_PIPELINE.md` | WOD-E2E eval (PyTorch + JAX) + JAX real-data training — converter, ADE/RFS, commands |
| `2implementation-details/AUDIT.md` | Adversarial audit findings + resolutions |
| `2implementation-details/01_pytorch_reference_algorithm.md` | Exact PyTorch loss/mask/scaffold/M-RoPE/param spec (the porting bible) |

**`1plans/`** — plans & specs (frozen)
| Doc | Purpose |
|---|---|
| `1plans/00_PLAN.md` | Feasibility + phased plan |
| `1plans/02_tpu_plan.md` | TPU mesh/sharding scale-out plan (Phase 5) |
| `1plans/03_scaleup_tpu_spec.md` | Multi-node TPU scale-up spec (Phase 6) |
| `1plans/04_tpu_smallscale_validation.md` | Small-scale 2-host TPU validation runbook |

**`4collect/`** — dated build logs (frozen)
| Doc | Purpose |
|---|---|
| `4collect/OVERNIGHT_TPU_PROGRESS.md` | **Latest (2026-06-07):** MaxText SASD on real TPU (v6e) |
| `4collect/05_maxtext_port_progress.md` | Phase 7 MaxText port build log |
| `4collect/OVERNIGHT_PROGRESS.md` | Phase 6 scale-up build log |
| `4collect/HANDOFF.md` | Phases 0–5 living status log |

**Interactive dataset review site (v2):** [`visualizations/index.html`](visualizations/index.html) —
what a v2 record looks like (13 fields + precomputed `image_embeds`), the raw→AR-v2 build chain
with per-stage verification, the v2 train feed path, with **12 verified examples (10 train + 2
val)** incl. embeds-PCA triptychs (`bash visualizations/serve.sh`, then
`ssh -L 8890:localhost:8890`). Generator: `scripts/make_dataset_website.py`; verifier:
`scripts/verify_ar_round2.sh` (status SSOT: `docs/2implementation-details/DATASET_V2.md` §6).

**Status (2026-06-12).** The full model is ported, parity-verified, and trains end-to-end:
- **Model parity** vs PyTorch: text **3.2e-5** · SASD loss **7.9e-8** · ViT **4.0e-5** · multimodal forward **7.7e-5** · attention mask **bit-identical**. `run_all_verification.sh` → **10/10 gates PASS**; adversarial audit **0 code defects**.
- **Trains (loss ↓):** text **3.84→1.47** · multimodal **0.999→0.701** · real Waymo **0.682→0.600**.
- **Eval (both stacks, full 479 rated val):** PyTorch `scaffold_spec` **ADE3s 0.814 / ADE5s 1.990 / RFS 7.914** · JAX `section_diffusion` **0.839 / 2.072 / 7.929** (on par; trajectory **0.01 m**).
- **Phase 6 (scale-up):** TPU-ready dataset **50,331 frames / 787 Parquet shards** (private HF) · self-contained FSDP harness (`ddrive_jax/train/`, grain multi-host) — FSDP-vs-1device **9.5e-7**, ckpt-resume **0.0** · real **3.09B** model trains on GPU **0.985→0.598**.
- **Phase 7 (MaxText, production path):** SASD grafted into a MaxText fork (loss/weight/VLA parity **bit-exact** to NNX) · **trains on real TPU (v6e-1) with real weights, loss 0.31/0.56**.
- **Phase 8 (dataset v2, production data path, 2026-06-12):** **v2 ArrayRecord** = 13 fields + precomputed frozen-ViT **`image_embeds` bf16** · four splits (full **415,663**/50k/400/**val 479**) built + audited + on GCS · AR reader auto-detect + iterator-state-in-ckpt (resume continues the stream) · **83.4 TFLOP/s/device on v6e-1 (+28%)**, `V2_TPU_VALIDATION_PASS` · round-2 from-raw re-verification **12/12** incl. embeds recompute — see `docs/2implementation-details/DATASET_V2.md`.
- **Open:** the literal **≥8-chip multi-node TPU run** (blocked only by GCP trial capacity, external/transient); whole-model TP swap; JAX KV-cache fast decode; in-loop eval wiring (`eval_interval: 0`, val v2 AR ready).

Envs (always `unset LD_LIBRARY_PATH` first): PyTorch oracle =
`/home/kaiwen/miniconda3/envs/ddrive/bin/python`; JAX =
`/home/kaiwen/jax-dlm-baseline/.venv/bin/python` with `PYTHONPATH` set to this dir.
