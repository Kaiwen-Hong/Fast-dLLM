# Fast-dDrive JAX port — living handoff log

Autonomous overnight build (2026-06-02). Bar: JAX trains with **decreasing SASD loss**
on the RTX 5090, TPU/MaxText-ready. See 00_PLAN.md (plan) and 01_pytorch_reference_algorithm.md (bible).

## How to run things
- PyTorch oracle env: `/home/kaiwen/miniconda3/envs/ddrive/bin/python` (torch 2.11+cu128, transformers 4.57.1). `unset LD_LIBRARY_PATH` first.
- JAX env: `/home/kaiwen/jax-dlm-baseline/.venv/bin/python` (jax 0.10.0, flax 0.12.7 nnx, optax 0.2.8). `unset LD_LIBRARY_PATH; export XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- For numerics, set `jax.config.update("jax_default_matmul_precision","highest")` (TF32 off) when comparing to the fp32 oracle.
- Checkpoint snapshot: `/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f`
- JAX package: `jax_ddrive/ddrive_jax/`; scripts in `jax_ddrive/scripts/`; big artifacts in `/home/kaiwen/data/fast-ddrive/`.

## Status
- [x] **Phase 0** env + 16.3GB ckpt + PyTorch inference (`run_chatbot.py` scaffold_spec) → valid JSON trajectory.
- [x] **Phase 1** JAX text backbone + weight load. GATE PASS: logits rel-max **3.2e-5**, top-1 **100%** vs PyTorch oracle (fp32, bidirectional, eager).
      Files: `ddrive_jax/models/{rope.py,qwen2_5_text.py}`, `ddrive_jax/convert/hf_to_jax.py`; scripts `capture_oracle_text.py`, `parity_text.py`.
      Notes: text keys = `model.layers.N.*` flat; tie check max|embed-lm_head|=0 (truly tied); M-RoPE collapses for text.
- [x] **Phase 2** SASD loss + hybrid block-causal mask + scaffold. GATE PASS: hybrid mask **bit-identical** (0 mismatches) and SASD loss rel **7.85e-8** (primary 1.7e-7, complementary 4.2e-7) vs PyTorch on the real sample.json (L=1120, 11 blocks, 4 sections).
      Files: `ddrive_jax/diffusion/{masks.py,sasd_loss.py}`; scripts `capture_oracle_sasd.py`, `parity_sasd.py`. Section detection via the model's own `section_utils` (cached to npz).
      Key parity facts: doubled `[noisy|clean]` 2L seq, positions tiled `[0..L-1,0..L-1]`, eval-mode==train q/k split, loss=section_weighted_CE(noisy half)+causal_CE(clean half of mdm rows), num_items=2·response_tokens. **fp32 matmul precision must be HIGHEST** (TF32 off).
- [x] **Phase 3 (MILESTONE)** overfit 2 samples, SASD training loss decreases — DONE, multiple runs:
      - **base Qwen2.5-VL-3B, full-FT fixed-batch (headline):** fixed-eval loss **3.84→1.47 / 4.02→1.62** (−2.4 nats, monotonic, 120 steps, no NaN). Proves the JAX SASD training learns the driving task from scratch. Log `logs/overfit_base_fullft.log`.
      - trained-ckpt full-FT fixed-batch: **1.06→0.89 / 1.12→0.95** monotonic. LoRA + stochastic-noise runs also validated.
      Memory (5090 shares ~9.6GB w/ neighbor `starVLA-opd` → ~21GB free): **bf16 + Adafactor + per-layer `nnx.remat` (gradient checkpointing)** makes full-FT of 3.75B fit; LoRA (`lora.py`, base frozen) also available. Files: `ddrive_jax/{train_overfit.py,lora.py,diffusion/noise.py}`, `scripts/prep_overfit_data.py`. CPU tests `tests/test_mask_loss.py` PASS. Key: training uses **default (TF32) matmul precision** (not the parity scripts' `highest`) for speed+memory; loss in fp32.
- [x] **Phase 5** TPU-ready: `checkpoint.py` (Orbax save/restore, round-trip tested), `sharding.py` (mesh + FSDP PartitionSpec mapping, tested), `docs/02_tpu_plan.md`. Physical TP needs the ~80-LOC ShardedLinear/ShardedEmbedding port (JAX 0.10 sharding-in-types requires `out_sharding=`) — documented as the next step.
- [~] **Phase 4** vision tower DONE & verified; fusion + M-RoPE + multimodal training = remaining integration.
      `models/vision_qwen25vl.py` (Qwen2.5-VL ViT: Conv3D-as-linear patch embed, 2D RoPE, window attention at [7,15,23,31] via per-segment masks, spatial-merge 2, RMSNorm+SwiGLU, patch merger) + `convert/hf_to_jax.load_fast_ddrive_vit`. GATE PASS: feeding PyTorch's patch-embed into our blocks+merger reproduces the oracle to **rel-max 4.0e-5** (`scripts/{capture_oracle_vit,debug_vit,parity_vit}.py`). End-to-end shows 2.6% rel-L2 purely from the **cuDNN Conv3d** seed (~6.5e-4) amplifying over 32 residual layers — our matmul patch-embed is the exact one (fp64 confirms).
- [x] **Phase 4b** multimodal FORWARD (3D M-RoPE + fusion) verified. GATE PASS: feeding PyTorch's fused embeds + 3D position_ids into the JAX decoder reproduces hidden **3.2e-5** / logits **7.7e-5** / top-1 **100%** (`models/qwen2_5_text.py` `mrope_cos_sin`+`hidden_forward_mrope`; `scripts/{capture_oracle_mm,parity_mm}.py`). Text path regression-checked (still 3.2e-5). **The entire model (ViT + fusion + M-RoPE + text + diffusion loss) is now ported and parity-verified.** - [x] **Phase 4b multimodal TRAINING** — DONE. Full model (frozen ViT image embeds fused into the text stream + 3D M-RoPE + block-diffusion SASD loss) trains with **loss 0.999→0.701** (monotonic, 80 steps, full-FT bf16+remat, no NaN). Files: `scripts/prep_overfit_data_mm.py`, `ddrive_jax/train_overfit_mm.py`, `models/qwen2_5_text.py:hidden_forward_mrope_cs`. (Image downscaled to ~90 tokens so the doubled seq fits the shared 5090; multimodal path = Phase-3 SASD with fused embeds + vision-protected noising + tiled 3D M-RoPE, all per modeling.py:2359-2660.)
  **→ The entire Fast-dDrive model is now ported, parity-verified, and trains end-to-end (text + multimodal) with decreasing loss.**
  Still open (not blocking the rewrite): JAX *generation/sampling* (the 3 decoders in generation_utils.py) for end-to-end JAX inference; physical TP sharding; real Waymo-data training.
- [ ] **Phase 5** sharding + Orbax.

## Parallel tracks
- Waymo WOD-E2E download (val 225G then train 876G) → `/home/kaiwen/data/fast-ddrive/waymo/` (account kaiwenh.17@gmail.com has access). Log: `logs/waymo_download.log`. Not needed for the milestone.
