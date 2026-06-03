# jax_ddrive — Fast-dDrive in JAX/Flax-NNX (MaxText-style)

JAX port of Fast-dDrive (Qwen2.5-VL-3B block-diffusion VLA for Waymo driving), built to
run on the RTX 5090 and scale to Waymo TPU. Adapted from `DLM-policy4AV/jax-mdlm-handoff`.

**Start here:** [`docs/REPORT.md`](docs/REPORT.md) — what's done, results, how to reproduce.

| Doc | Purpose |
|---|---|
| `docs/REPORT.md` | Overnight build summary + reproduce commands (read first) |
| `docs/HANDOFF.md` | Living status log + exact run commands/env |
| `docs/00_PLAN.md` | Feasibility + phased plan |
| `docs/01_pytorch_reference_algorithm.md` | Exact PyTorch loss/mask/scaffold/M-RoPE/param spec |
| `docs/02_tpu_plan.md` | TPU mesh/sharding scale-out plan |

**Status:** Phase 1 (text parity 3e-5) · Phase 2 (SASD loss parity 8e-8) · **Phase 3 (text loss
decreases, base 3.84→1.47)** · Phase 4 (ViT parity 4e-5) · Phase 4b (multimodal forward 7.7e-5,
**multimodal training loss 0.999→0.701**) · Phase 5 (Orbax+sharding) — all verified. The full model
is ported + parity-verified + trains end-to-end. Open (not part of the rewrite): JAX generation/sampling,
physical TP sharding, real Waymo-data training.

Envs (always `unset LD_LIBRARY_PATH` first): PyTorch oracle =
`/home/kaiwen/miniconda3/envs/ddrive/bin/python`; JAX =
`/home/kaiwen/jax-dlm-baseline/.venv/bin/python` with `PYTHONPATH` set to this dir.
