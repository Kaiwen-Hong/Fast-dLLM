# Fast-dDrive JAX — feature matrix

Status of every Fast-dDrive capability in the JAX/Flax-NNX port. "Verified" = a numeric
parity gate or a loss-decrease run exists (see `REPORT.md` for commands/numbers).

> 📌 **STATUS UPDATE (2026-06-08):** The "TPU / scale-out" rows below predate Phase 6/7. Since then the
> FSDP harness is verified and the **MaxText port is done and trains on real TPU** (v6e-1, real weights,
> loss 0.31/0.56). The "TPU / scale-out" table below has been refreshed accordingly — current truth is
> `docs/4collect/OVERNIGHT_TPU_PROGRESS.md`.

## Model components
| Feature | Status | Evidence |
|---|---|---|
| Qwen2.5 text decoder (36L/2048d, GQA, qkv bias, tied embeds, RMSNorm, SiLU) | ✅ verified | logits 3.2e-5 vs PyTorch |
| RoPE (θ=1e6, rotate-half) | ✅ verified | part of text parity |
| 3D **M-RoPE** (mrope_section [16,24,24]) | ✅ verified | multimodal forward 7.7e-5 |
| HF safetensors → NNX weight conversion (text + ViT) | ✅ verified | 0 missing keys; parity |
| Qwen2.5-VL **vision tower** (Conv3D patch, 2D RoPE, window attn [7,15,23,31], spatial-merge, RMSNorm+SwiGLU, merger) | ✅ verified | arch parity 4.0e-5 |
| Vision↔text **fusion** (image-token scatter) | ✅ verified | multimodal forward parity |
| Tied LM head | ✅ verified | tie-check 0; parity |

## Diffusion / training
| Feature | Status | Evidence |
|---|---|---|
| Masked-diffusion (MDM) doubled `[noisy\|clean]` forward | ✅ verified | SASD loss 7.9e-8 |
| Hybrid block-causal attention mask (multi-turn) | ✅ verified | bit-identical to PyTorch |
| Deep-JSON scaffold + section detection (4 sections) | ✅ verified (via torch prep) | sections match; loss parity |
| **Section-weighted CE loss** (per-section weights) | ✅ verified | 1.7e-7 |
| **Complementary-mask causal loss** | ✅ verified | 4.2e-7 |
| Per-section **Beta noise schedule** | ✅ implemented | `diffusion/noise.py` |
| Scaffold-token freeze + always-mask-im_end | ✅ verified | part of loss parity |
| Text SASD training (loss decreases) | ✅ verified | base 3.84→1.47 |
| **Multimodal** SASD training (loss decreases) | ✅ verified | 0.999→0.701 |
| Optimizer (Optax Adafactor / AdamW), warmup-cosine, grad-clip | ✅ | `train_overfit*.py` |
| Memory: bf16 + gradient checkpointing (remat) + LoRA | ✅ | fits shared 21GB 5090 |

## Inference / serving
| Feature | Status | Notes |
|---|---|---|
| PyTorch model inference (`run_chatbot.py`) | ✅ works | Phase 0; valid JSON trajectory |
| JAX **forward** (full model) | ✅ verified | the basis for all generation |
| JAX **section-diffusion sampler** (text) | ✅ generates valid JSON | `diffusion/sample_sd.py` |
| JAX **multimodal** section-diffusion sampler (ViT fuse + denoise) | ✅ **verified** | `eval/mm_sampler.py`; trajectory matches PyTorch to **0.01 m** (`scripts/verify_sd_mm.py`) |
| JAX **eval pipeline** → official ADE/RFS | ✅ **verified** | top-level `jax_ddrive/eval/{prep_jax_eval,jax_batch_inference}.py`; full-479 rated-val ADE3s 0.839 / ADE5s 2.072 / RFS 7.929 (vs PyTorch 0.814/1.990/7.914) — see `REPORT.md`. (52-frame ADE3s 0.853 / ADE5s 2.196 / RFS 8.10 was the earlier partial preview.) |
| JAX scaffold-spec / multi-traj decoders | ⬜ not ported | speed features (need KV-cache); section_diffusion is the working JAX decoder |
| KV-cache (block-wise + fork) | ⬜ not ported | would speed JAX decode (~17 s/sample now) |

## TPU / scale-out
| Feature | Status | Notes |
|---|---|---|
| Orbax checkpoint save/restore | ✅ verified | `checkpoint.py` + harness `checkpoint_mgr.py`; round-trip, ckpt-resume diff 0.0 |
| FSDP PartitionSpec mapping + mesh | ✅ verified | `sharding.py`; 252/252 real-model kernels sharded (abstract trace) |
| **Self-contained FSDP harness** (shard_map+psum, AdamW, grain multi-host) | ✅ verified | `ddrive_jax/train/`; FSDP-vs-1device 9.5e-7; real 3.09B GPU 0.985→0.598 (CPU-8 emu + 1 GPU) |
| **TPU-ready dataset** (Parquet, private HF) | ✅ done | 50,331 frames / 787 shards / 22 GB; bit-exact decode; MaxText `hf`-path compatible |
| Physical TP sharding primitives (ShardedLinear/Embedding, `out_sharding=`) | ✅ verified mesh=1 | `models/sharded.py`, `tests/test_sharding.py` |
| Whole-model TP swap (Linear→ShardedLinear across the decoder) | ⬜ mechanical | ~80 LOC; primitives ready, see `docs/1plans/02_tpu_plan.md` |
| **MaxText fork integration** (SASD graft) | ✅ done & TPU-proven | loss/weight/VLA parity bit-exact to NNX; **trains on real v6e-1, real weights, loss 0.31/0.56** |
| **Multi-node (≥8-chip) TPU run** | ⬜ blocked (capacity) | built + armed; GCP trial gave no ≥8-chip capacity (external/transient) — `docs/4collect/OVERNIGHT_TPU_PROGRESS.md` |

## Data / eval
| Feature | Status | Notes |
|---|---|---|
| Example-sample data prep (text + multimodal) | ✅ | `scripts/prep_overfit_data*.py` |
| WOD-E2E download | ✅ val (226GB) + train (877GB) | `/home/kaiwen/data/fast-ddrive/waymo/` |
| **tfrecord→JSON converter** | ✅ **done** | `fast_ddrive/data/convert_wod_e2e.py`; prompt byte-for-byte; 479 rated val + train targets |
| **Official ADE/RFS metrics** | ✅ **done** | `autovla` env (compiled E2E proto); shared by both stacks |
| **JAX real-data SASD training** | ✅ done | `train_waymo_sasd_jax.py`; 400 real samples prepped; loss-decrease run in overnight batch |

Legend: ✅ done/verified · 🚧 in progress · ⬜ not started (scoped).
