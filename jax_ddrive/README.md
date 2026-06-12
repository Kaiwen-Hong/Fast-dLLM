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

**Interactive dataset review site:** [`visualizations/index.html`](visualizations/index.html) —
what a record looks like, the raw→ArrayRecord build chain, how a train step consumes it, with 10
bit-exact-verified examples (`bash visualizations/serve.sh`, then `ssh -L 8890:localhost:8890`).
Generator: `scripts/make_dataset_website.py`; verifier: `scripts/verify_ar_round2.sh`.

**Status (2026-06-12).** The full model is ported, parity-verified, and trains end-to-end:
- **Model parity** vs PyTorch: text **3.2e-5** · SASD loss **7.9e-8** · ViT **4.0e-5** · multimodal forward **7.7e-5** · attention mask **bit-identical**. `run_all_verification.sh` → **10/10 gates PASS**; adversarial audit **0 code defects**.
- **Trains (loss ↓):** text **3.84→1.47** · multimodal **0.999→0.701** · real Waymo **0.682→0.600**.
- **Eval (both stacks, full 479 rated val):** PyTorch `scaffold_spec` **ADE3s 0.814 / ADE5s 1.990 / RFS 7.914** · JAX `section_diffusion` **0.839 / 2.072 / 7.929** (on par; trajectory **0.01 m**).
- **Phase 6 (scale-up):** TPU-ready dataset **50,331 frames / 787 Parquet shards** (private HF) · self-contained FSDP harness (`ddrive_jax/train/`, grain multi-host) — FSDP-vs-1device **9.5e-7**, ckpt-resume **0.0** · real **3.09B** model trains on GPU **0.985→0.598**.
- **Phase 7 (MaxText, production path):** SASD grafted into a MaxText fork (loss/weight/VLA parity **bit-exact** to NNX) · **trains on real TPU (v6e-1) with real weights, loss 0.31/0.56**.
- **Dataset (2026-06-12):** full **415,663 frames → 130 ArrayRecord shards** (158 G local + GCS mirror) · round-2 semantic verification **10/10 bit-exact** vs a from-raw re-run · review website under `visualizations/` · v2 spec locked (add bf16 `image_embeds`) — see `docs/2implementation-details/DATASET.md`.
- **Open:** the literal **≥8-chip multi-node TPU run** (blocked only by GCP trial capacity, external/transient); whole-model TP swap; JAX KV-cache fast decode; AR reader + MaxText iterator-ckpt (dataset v2 plan).

Envs (always `unset LD_LIBRARY_PATH` first): PyTorch oracle =
`/home/kaiwen/miniconda3/envs/ddrive/bin/python`; JAX =
`/home/kaiwen/jax-dlm-baseline/.venv/bin/python` with `PYTHONPATH` set to this dir.
