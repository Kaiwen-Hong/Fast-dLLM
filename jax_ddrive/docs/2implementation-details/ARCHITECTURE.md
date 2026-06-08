# Fast-dDrive JAX — architecture

How the JAX port is structured and how data flows. Mirrors the PyTorch model
(`modeling.py` + `section_utils.py` + `generation_utils.py`) but in Flax NNX.

## Package layout (`jax_ddrive/ddrive_jax/`)
```
models/
  rope.py             plain RoPE (rotate-half) + RoPE module
  qwen2_5_text.py     Qwen2.5 text decoder (NNX): RMSNorm, GQA attention (qkv bias),
                      SwiGLU MLP, tied embeds; plain-RoPE + 3D M-RoPE paths;
                      hidden_forward(remat) / hidden_forward_mrope_cs(remat) / attend
  vision_qwen25vl.py  Qwen2.5-VL ViT (NNX): Conv3D-as-linear patch embed, 2D RoPE,
                      window/full attention via per-segment masks, RMSNorm+SwiGLU,
                      PatchMerger; host-side window/position index ports (numpy)
diffusion/
  masks.py            hybrid_block_causal_mask_dense (training, doubled 2L) + eval mask
  sasd_loss.py        section_weighted_ce + causal_ce + sasd_total_loss
  noise.py            per-step stochastic noising (Beta per-section, scaffold freeze,
                      doubled [noisy|clean] + complementary)
convert/hf_to_jax.py  load_fast_ddrive_text / load_fast_ddrive_vit (safetensors→NNX)
lora.py               LoRALinear + apply_lora + freeze_non_lora_params
sharding.py           mesh + FSDP PartitionSpec mapping (TPU)
checkpoint.py         Orbax save/restore of NNX state
train_overfit.py      text SASD overfit (loss-decrease)
train_overfit_mm.py   multimodal SASD overfit (loss-decrease)
train_waymo_sasd_jax.py  real-Waymo multi-sample SASD training + Orbax ckpt
eval/                 JAX inference building blocks:
  rope_index.py         numpy get_rope_index (3D M-RoPE), validated vs PyTorch
  scaffold.py           deep-JSON scaffold + response_block_idx (generation_utils replica)
  mm_sampler.py         multimodal section-diffusion sampler (ViT fuse → block denoise)
convert/prep_to_parquet.py  npz → sharded Parquet (TPU-ready dataset writer)
data/                 (Phase 6) TPU-ready input pipeline:
  parquet_dataset.py    framework-free numpy decode contract (the data SSOT)
  grain_pipeline.py     grain MapDataset: per-host shard, online SASD noise, resumable
train/                (Phase 6) self-contained multi-host FSDP harness:
  dist.py               jax.distributed bootstrap + mesh context
  train_tpu.py          shard_map+psum FSDP train step (proxy on CPU-8 / real on GPU/TPU)
  checkpoint_mgr.py     Orbax CheckpointManager (params+opt+step+grain_state)
  launch_tpu.sh         TPU queued-resource launch template
```

**Two scale-out paths** (Phase 6/7): **Path A** = the self-contained NNX FSDP harness above
(`train/train_tpu.py`), kept as the algorithm source of truth. **Path B (production)** = the SASD
algorithm grafted into a **MaxText fork** (`/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/`), which
reuses this package's `data/` loader + `diffusion/` + ViT and **trains on real TPU** (v6e, real
weights). See `docs/4collect/OVERNIGHT_TPU_PROGRESS.md`.

## Training data flow (multimodal SASD)
```
sample.json + image
   │  (torch prep, one-time: tokenizer + section_utils + processor)
   ▼
input_ids, labels, pixel_values, image_grid_thw,
response_block_idx, turn_idx, scaffold_mask, weight_vec, block α/β, 3D position_ids
   │
   ▼  (JAX, per step)
image_embeds = ViT(pixel_values, grid)            # frozen, computed once
original_embeds = scatter(image_embeds, embed_tokens(input_ids))   # fusion
noisy_embeds   = where(vision_mask, original_embeds, embed(noisy_ids))  # vision protected
doubled = stack([ [noisy|clean], [comp_noisy|clean] ])             # [2, 2L, D]
cos,sin = mrope_cos_sin(tiled 3D positions)                        # [2L, 128]
mask4d  = hybrid_block_causal_mask_dense(rbi, turn, L)             # [2L, 2L]
hidden  = text.hidden_forward_mrope_cs(doubled, cos, sin, mask4d, remat=True)
loss    = section_weighted_ce(attend(hidden[:, :L]), labels, weights)     # noisy half
        + causal_ce(attend(hidden[:1, L:]), original_labels)              # clean half
   │  (Optax Adafactor + grad clip; bf16 + remat)
   ▼
param update  → loss decreases
```
Text-only training is identical minus the ViT/fusion and with plain RoPE (positions = arange).

## Parity methodology
Each component has a PyTorch "oracle" capture (`scripts/capture_oracle_*.py`, run in the
`ddrive` env with **eager** attention + explicit masks/positions to remove ambiguity) and a
JAX gate (`scripts/parity_*.py`, run with `jax_default_matmul_precision=highest` to disable
TF32). Gates assert rel-max < 1e-3 (looser for ViT due to the cuDNN-Conv3d seed; we isolate
the architecture by feeding PyTorch's patch-embed in). `run_all_verification.sh` runs them all.

## Key numerics/gotchas (also in `docs/4collect/HANDOFF.md`)
- Parity needs **fp32 `highest`** matmul precision; training uses default **TF32** + fp32 loss.
- Doubled `[noisy|clean]` 2L sequence; positions **tiled** `[0..L-1, 0..L-1]` (M-RoPE: `[pos3d|pos3d]`).
  PyTorch achieves this by splitting q/k into L-halves; JAX runs full 2L with tiled positions (equivalent).
- ViT patch-embed: Conv3D == a matmul; PyTorch's cuDNN conv adds a benign ~6.5e-4 seed.
- Memory on the shared 5090 (~21GB free): bf16 + Adafactor + per-layer `nnx.remat` (+ optional LoRA).
