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
- [x] **Phase 2** SASD loss + hybrid block-causal mask + scaffold. GATE PASS: hybrid mask **bit-identical** (0 mismatches) and SASD loss rel **7.9e-8** (primary 1.7e-7, complementary 4.2e-7) vs PyTorch on the real sample.json (L=1120 [STALE: this is the one captured parity sample, not the canonical L; production uniform L=1184 (sasd_waymo.yml:25), distilled-from-base L=1280], 11 blocks, 4 sections).
      Files: `ddrive_jax/diffusion/{masks.py,sasd_loss.py}`; scripts `capture_oracle_sasd.py`, `parity_sasd.py`. Section detection via the model's own `section_utils` (cached to npz).
      Key parity facts: doubled `[noisy|clean]` 2L seq, positions tiled `[0..L-1,0..L-1]`, eval-mode==train q/k split, loss=section_weighted_CE(noisy half)+causal_CE(clean half of mdm rows), num_items=2·response_tokens. **fp32 matmul precision must be HIGHEST** (TF32 off).
- [x] **Phase 3 (MILESTONE)** overfit 2 samples, SASD training loss decreases — DONE, multiple runs:
      - **base Qwen2.5-VL-3B, full-FT fixed-batch (headline):** fixed-eval loss **3.84→1.47 / 4.02→1.62** (−2.4 nats, monotonic, 120 steps, no NaN). Proves the JAX SASD training learns the driving task from scratch. Log `logs/overfit_base_fullft.log`.
      - trained-ckpt full-FT fixed-batch: **1.06→0.89 / 1.12→0.95** monotonic. LoRA path (default) also runs: trained-ckpt LoRA fixed-eval **1.063→1.057**, base LoRA **3.838→3.817** (small, as expected — LoRA on a near-optimal/frozen-embed start; full-FT gives the headline drop). Gated by `phase3_lora_train` in run_all_verification.sh; LoRA forward + freeze unit-tested (`tests/test_lora.py`).
      Memory (5090 shares ~9.6GB w/ neighbor `starVLA-opd` → ~21GB free): **bf16 + Adafactor + per-layer `nnx.remat` (gradient checkpointing)** makes full-FT of 3.75B [订正 2026-06-16: ~3.086B (3.09B rounded), 434 text leaves; 3.75B is an early erroneous figure — 见 docs/0overview/02_gotchas.md#规范数字框-canonical-numbers] fit; LoRA (`lora.py`, base frozen) also available. Files: `ddrive_jax/{train_overfit.py,lora.py,diffusion/noise.py}`, `scripts/prep_overfit_data.py`. CPU tests `tests/test_mask_loss.py` PASS. Key: training uses **default (TF32) matmul precision** (not the parity scripts' `highest`) for speed+memory; loss in fp32.
- [x] **Phase 5** TPU-ready: `checkpoint.py` (Orbax save/restore, round-trip tested), `sharding.py` (mesh + FSDP PartitionSpec mapping, tested), `docs/02_tpu_plan.md`. Physical TP needs the ~80-LOC ShardedLinear/ShardedEmbedding port (JAX 0.10 sharding-in-types requires `out_sharding=`) — documented as the next step.
- [~] **Phase 4** vision tower DONE & verified; fusion + M-RoPE + multimodal training = remaining integration.
      `models/vision_qwen25vl.py` (Qwen2.5-VL ViT: Conv3D-as-linear patch embed, 2D RoPE, window attention at [7,15,23,31] via per-segment masks, spatial-merge 2, RMSNorm+SwiGLU, patch merger) + `convert/hf_to_jax.load_fast_ddrive_vit`. GATE PASS: feeding PyTorch's patch-embed into our blocks+merger reproduces the oracle to **rel-max 4.0e-5** (`scripts/{capture_oracle_vit,debug_vit,parity_vit}.py`). End-to-end shows 2.6% rel-L2 purely from the **cuDNN Conv3d** seed (~6.5e-4) amplifying over 32 residual layers — our matmul patch-embed is the exact one (fp64 confirms).
- [x] **Phase 4b** multimodal FORWARD (3D M-RoPE + fusion) verified. GATE PASS: feeding PyTorch's fused embeds + 3D position_ids into the JAX decoder reproduces hidden **3.2e-5** / logits **7.7e-5** / top-1 **100%** (`models/qwen2_5_text.py` `mrope_cos_sin`+`hidden_forward_mrope`; `scripts/{capture_oracle_mm,parity_mm}.py`). Text path regression-checked (still 3.2e-5). **The entire model (ViT + fusion + M-RoPE + text + diffusion loss) is now ported and parity-verified.** - [x] **Phase 4b multimodal TRAINING** — DONE. Full model (frozen ViT image embeds fused into the text stream + 3D M-RoPE + block-diffusion SASD loss) trains with **loss 0.999→0.701** (monotonic, 80 steps, full-FT bf16+remat, no NaN). Files: `scripts/prep_overfit_data_mm.py`, `ddrive_jax/train_overfit_mm.py`, `models/qwen2_5_text.py:hidden_forward_mrope_cs`. (Image downscaled to ~90 tokens [STALE: the committed train_overfit_mm.py uses the full 1365 image tokens/sample (x2=2730 doubled); the ~90-token downscale was an earlier run] so the doubled seq fits the shared 5090; multimodal path = Phase-3 SASD with fused embeds + vision-protected noising + tiled 3D M-RoPE, all per modeling.py:2359-2660.)
  **→ The entire Fast-dDrive model is now ported, parity-verified, and trains end-to-end (text + multimodal) with decreasing loss.**
  Still open (not blocking the rewrite): JAX *generation/sampling* (the 3 decoders in generation_utils.py) for end-to-end JAX inference; physical TP sharding; real Waymo-data training.
- [x] **Phase 5** Orbax checkpoint + FSDP PartitionSpec mapping done (see above). [ ] **Phase 5b** physical TP sharding (ShardedLinear/ShardedEmbedding port, ~80 LOC).

## Parallel tracks
- Waymo WOD-E2E download (val 225G then train 876G) → `/home/kaiwen/data/fast-ddrive/waymo/` (account kaiwenh.17@gmail.com has access). Log: `logs/waymo_download.log`. Not needed for the milestone.

## Overnight 2026-06-03 — eval pipeline (both stacks) + JAX real-data training
**See `docs/EVAL_PIPELINE.md` for the full pipeline + reproduce commands.** GPU note: the
user authorized freeing the whole 5090, so the neighboring `starVLA-opd` server (PID 1972816)
was killed → 32 GB free. Restart cmd: `/home/kaiwen/data/fast-ddrive/RESTART_starVLA_server.txt`.

- [x] **`autovla` env** (`/home/kaiwen/miniconda3/envs/autovla`): TF + waymo-open-dataset + the
  **compiled** `end_to_end_driving_data_pb2` (pip wheels lack it; compiled from GitHub `.proto`
  against installed descriptors). Needed by the converter + the official metric.
- [x] **Converter** `fast_ddrive/data/convert_wod_e2e.py`: the repo's missing preprocessor.
  Prompt reproduces `data/example/sample.json` **byte-for-byte** (1896 chars, verified). Produced
  `val_rated.json` = **479** rater-scored frames (the official RFS subset) + 1437 front-cam JPEGs,
  and `train_targets.json` (800 frames, `--with_target`: GT trajectory + derived meta + pseudo text).
- [x] **PyTorch eval (5090, multimodal)** — `batch_inference.py scaffold_spec` → `predictions.json`
  → `evaluate_waymo_metrics.py`. **Validated on 52 rated frames: ADE_3s 0.888, ADE_5s 2.250, RFS 7.913**
  (100% trajectory parse). Full-479 in the overnight run (`eval/pt_val_full_ss/`).
- [x] **JAX multimodal section-diffusion sampler** (`ddrive_jax/eval/mm_sampler.py`): ViT once →
  scatter image embeds → block-by-block denoise of the deep-JSON scaffold. **GATE PASS** vs PyTorch
  `mdm_sample_deep_scaffold`: trajectory matches to **0.01 m** (`scripts/verify_sd_mm.py`, bf16).
  Prep ports validated against PyTorch internals: x_t0 ✅, response_block_idx ✅ (exact generation
  replica), numpy `get_rope_index` ✅ (`scripts/capture_oracle_sd_mm.py`).
- [x] **JAX eval pipeline** (`eval/prep_jax_eval.py` ddrive-env CPU prep → `eval/jax_batch_inference.py`
  jax-env compute → same metric). **Full 479 rated frames: ADE_3s 0.839, ADE_5s 2.072, RFS 7.929**
  (100% parse) — on par with PyTorch (0.814/1.990/7.914). Results in `eval/jax_val_full_sd/`.
- [x] **JAX real-data SASD training** (`ddrive_jax/train_waymo_sasd_jax.py`): multi-sample, stochastic
  per-section Beta noise, Section-Importance-Weighted + complementary-mask loss, frozen ViT embeds,
  bf16+remat+Adafactor, Orbax ckpt. 400 real samples prepped (`eval/prep_train_jax.py`; all L=1184 /
  7 blocks → single compile). Loss-decrease run in the overnight batch (`logs/train_jax.log`).
- Runner: `scripts/run_overnight.sh` (PyTorch-479 eval+metric → JAX training → JAX-479 eval+metric).

## Overnight 2026-06-07 — MaxText SASD port runs on TPU (single-chip proof)
**Full session log: `OVERNIGHT_TPU_PROGRESS.md` (the TPU SSOT).** Headline: the MaxText SASD port
(`maxtext-dlm-fork/`, discovered already well underway) **trains on real TPU (v6e-1) with real
Fast-dDrive weights** — loss **0.308 / 0.556** at steps 10/11 (matching the GPU smoke ~0.6),
3.086 B params, frozen ViT (390 tensors) loaded, 65 TFLOP/s/device, EXIT=0. The whole stack proven
on TPU: provision + uv/py3.11 install + Waymo SASD grain pipeline + GCS param restore + section-weighted
SASD train step on TPU XLA. The literal **multi-node (≥2-host)** run was NOT reached that night —
GCP gave this trial account no ≥8-chip capacity (external/transient). [完成 2026-06-19: multi-host
is now done — see the trainable-ViT entry below.]

## Overnight 2026-06-19 — trainable in-graph ViT on multi-host TPU (MILESTONE)
**Rolling plan + full progress log: `../1plans/06_trainable_vit_plan.md` §9; frozen milestone log:
`08_trainable_vit_progress.md`.** Today the SASD ViT becomes **trainable & in-graph**: with
`sasd_vit_trainable=true` the Qwen2.5-VL ViT runs in-graph on `pixel_values` every step
(`flax.nnx.bridge.ToLinen` wraps the validated NNX ViT body), its params live in the MaxText train
state (sharded/checkpointed/gradient-receiving). The frozen/pre-baked path is KEPT behind the toggle
(default `false`, byte-unchanged).
- [x] **GPU end-to-end train PASS** (RTX 5090, real MaxText loop, `wod_e2e_sasd_v2_ar`): 3 real train
  steps, ViT params get gradients, frozen path regression PASS. Module numerics: in-graph ViT vs
  pre-baked embeds **cosine 0.99925** (release-weights-loaded).
- [x] **FULL MULTI-HOST TPU PASS** (run `be47jjta8`, v5e-16, **4 hosts / 16 chips**, jax 0.10.2):
  3 real train steps, **loss strictly DECREASING 5.199 → 3.921 → 3.042** (perplexity 181 → 50 → 21)
  ⇒ gradients flow through the in-graph trainable ViT and the optimizer updates it; checkpoint saved
  to GCS, **EXIT 0**, pod torn down. (First step 1455 s = one-time first-execution XLA compile; later
  steps 30 s / 1.8 s.) **→ The trainable in-graph ViT trains end-to-end on multi-host TPU.**
  New file `maxtext-dlm-fork/src/maxtext/diffusion/sasd_vit_ingraph.py`; ViT split in
  `jax_ddrive/ddrive_jax/models/vision_qwen25vl.py`; wired via `layers/decoders.py`,
  `input_pipeline/waymo_sasd_data_processing.py`, `utils/{maxtext_utils,sharding}.py`,
  `trainers/pre_train/train.py`, `configs/{types.py,sasd_waymo.yml}`. Bundle `fastddrive-20260619_112401`.
- **TPU operational lessons (hard-won, this run; full list → `0overview/02_gotchas.md` TPU 运维坑):**
  (a) GCS Regional Access Boundary (RAB) is REGION-scoped and walls the TPU compute SA → restore from a
  LOCAL copy of the ckpt (gsutil w/ USER creds), and SAVE to a SAME-REGION bucket (us-south1) with the
  TPU SA granted `roles/storage.admin`; (b) multi-host Orbax checkpointing needs a SHARED filesystem
  (per-host local disk fails layer by layer — grain-iter dir, per-process creation, then `array_metadatas`)
  → same-region GCS is the fix; (c) `enable_checkpointing=false` is rejected when `load_parameters_path`
  is set, and step 0 always saves (`0 % period == 0`); (d) fresh-pod SSH `Permission denied (publickey)`
  = key still propagating → warm-up retry; (e) a multi-host `process_state.cc Raising signal 6` /
  Shutdown-barrier abort is a SYMPTOM (one worker died) → two-phase SSH + full per-host logs to disk,
  read from a fresh session; (f) ViT params currently REPLICATED (no logical-axis sharding) — TODO for
  real multinode.
- [~] **DEFERRED:** ckpt-side ViT snapshot-init (init the in-graph ViT from base/release for real
  training, vs. today's random-init from the BASE text ckpt). The unconditional-ViT build approach
  leaked abstract params into the train state and was reverted; the right next approach is to pass a
  concrete pixel batch into the param-ckpt build, or `state.replace` the ViT subtree post-restore.
