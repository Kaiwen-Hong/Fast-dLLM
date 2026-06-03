# Fast-dDrive JAX — feature matrix

Status of every Fast-dDrive capability in the JAX/Flax-NNX port. "Verified" = a numeric
parity gate or a loss-decrease run exists (see `docs/REPORT.md` for commands/numbers).

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
| JAX **section-diffusion sampler** (`mdm_sample_deep_scaffold`) | ✅ generates valid JSON | `diffusion/sample_sd.py`; emits a parseable trajectory; forward parity-verified; exact-token match limited by bf16 + MDM sampling sensitivity |
| JAX scaffold-spec / multi-traj decoders | ⬜ not ported | inference-speed features (need KV-cache) |
| KV-cache (block-wise + fork) | ⬜ not ported | needed for fast decode |

## TPU / scale-out
| Feature | Status | Notes |
|---|---|---|
| Orbax checkpoint save/restore | ✅ verified | `checkpoint.py`, round-trip |
| FSDP PartitionSpec mapping + mesh | ✅ | `sharding.py` (spec artifact) |
| Physical TP sharding primitives (ShardedLinear/Embedding, `out_sharding=`) | ✅ verified mesh=1 | `models/sharded.py`, `tests/test_sharding.py` |
| Whole-model TP swap (Linear→ShardedLinear across the decoder) | ⬜ mechanical | ~80 LOC; primitives ready, see `docs/02_tpu_plan.md` |
| MaxText fork integration (5-line loss_fn diff) | ⬜ | `docs/02_tpu_plan.md` |

## Data / eval
| Feature | Status | Notes |
|---|---|---|
| Example-sample data prep (text + multimodal) | ✅ | `scripts/prep_overfit_data*.py` |
| WOD-E2E download | ✅ val (225GB); train downloading | `/home/kaiwen/data/fast-ddrive/waymo/` |
| tfrecord→JSON converter | ⬜ | repo's is "coming soon"; needed for real training |
| Official ADE/RFS metrics | ⬜ | separate `autovla` TF env (`fast_ddrive/eval/`) |

Legend: ✅ done/verified · 🚧 in progress · ⬜ not started (scoped).
