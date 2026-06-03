# jax_ddrive — Fast-dDrive in JAX/Flax-NNX (MaxText-style)

JAX port of Fast-dDrive (Qwen2.5-VL-3B block-diffusion VLA for Waymo driving), built to
run on the RTX 5090 and scale to Waymo TPU. Adapted from `DLM-policy4AV/jax-mdlm-handoff`.

**Start here:** [`docs/REPORT.md`](docs/REPORT.md) — what's done, results, how to reproduce.

| Doc | Purpose |
|---|---|
| `docs/REPORT.md` | Overnight build summary + reproduce commands (read first) |
| `docs/EVAL_PIPELINE.md` | **WOD-E2E eval (PyTorch + JAX) + JAX real-data training** — converter, ADE/RFS, commands |
| `docs/HANDOFF.md` | Living status log + exact run commands/env |
| `docs/FEATURES.md` | Capability matrix (done/verified/remaining) |
| `docs/ARCHITECTURE.md` | Module layout + training data-flow + parity methodology |
| `docs/AUDIT.md` | Adversarial audit findings + resolutions |
| `docs/00_PLAN.md` | Feasibility + phased plan |
| `docs/01_pytorch_reference_algorithm.md` | Exact PyTorch loss/mask/scaffold/M-RoPE/param spec |
| `docs/02_tpu_plan.md` | TPU mesh/sharding scale-out plan |

**Status (all verified):** Phase 1 text parity 3e-5 · Phase 2 SASD loss parity 8e-8 · **Phase 3 text loss
decreases (base 3.84→1.47)** · Phase 4 ViT parity 4e-5 · Phase 4b multimodal forward 7.7e-5 + **multimodal
training 0.999→0.701** · JAX **section-diffusion sampler** (generates valid JSON trajectory) · Phase 5
Orbax + FSDP specs + **TP sharding primitives** (mesh=1). The full model is ported, parity-verified, trains
end-to-end (text+multimodal), and generates. `run_all_verification.sh` → 9/9 gates PASS. Audit: 0 code
defects (`docs/AUDIT.md`).

**Eval + training pipeline (2026-06-03, `docs/EVAL_PIPELINE.md`):** WOD-E2E `tfrecord→JSON`
converter (prompt byte-for-byte; 479 rated val frames) · official **ADE/RFS** metric on **both**
stacks (one backend) · PyTorch `scaffold_spec` **ADE3s 0.888 / ADE5s 2.250 / RFS 7.913** ·
JAX multimodal `section_diffusion` **ADE3s 0.853 / ADE5s 2.196 / RFS 8.100** (trajectory parity
0.01 m vs PyTorch) · JAX SASD training on real Waymo data (`train_waymo_sasd_jax.py`). Open:
whole-model TP swap, JAX KV-cache fast decode.

Envs (always `unset LD_LIBRARY_PATH` first): PyTorch oracle =
`/home/kaiwen/miniconda3/envs/ddrive/bin/python`; JAX =
`/home/kaiwen/jax-dlm-baseline/.venv/bin/python` with `PYTHONPATH` set to this dir.
