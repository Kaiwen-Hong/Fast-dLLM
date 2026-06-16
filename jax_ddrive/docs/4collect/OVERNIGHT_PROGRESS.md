# Overnight autonomous job — Fast-dDrive JAX scale-up to multi-node TPU

**Started:** 2026-06-05 (user asleep ~24h). **Driver:** Claude (autonomous, ultracode).
**Goal:** (1) convert the dataset to a TPU-ready format + upload to HF; (2) build a
production multi-node-TPU training harness that consumes it; (3) seamless deploy to Waymo
TPU infra. **Honesty rule (user):** never fake; label TPU-verified (none here) vs
emulation/GPU-verified. **Engine:** chained background Workflows (each re-invokes me) +
adversarial subagent verification; this doc is updated every turn.

Locked decisions (see `docs/03_scaleup_tpu_spec.md` §0): Path A self-contained NNX harness;
Parquet format; **private** HF (not public — WOD license, see spec §7); subset now +
one-command full scale; verify on CPU 8-device emulation + single GPU.

## ✅ FINAL SUMMARY (what works, what's verified, what's next)

The single-5090 reference loop is now a **multi-host-ready training system**: a TPU-ready dataset
(400 + 50k frames, private HF) + a self-contained Flax-NNX FSDP harness that consumes it, every
layer verified on CPU 8-device emulation + the GPU.

| Claim | How verified | Result |
|---|---|---|
| Dataset format lossless & TPU-loadable | HF roundtrip decode vs source npz | **4800/4800 fields bit-exact** |
| 50k real training set built | parallel convert + decode sanity | **50,331 frames / 787 shards / 22 GB, 0 errors** |
| grain multi-host input pipeline | 3 adversarial agents | determinism+resume, disjoint shards, noising bit-exact |
| FSDP == single-device (math correct) | mesh (2,1) vs (1,1), 1 step | **\|diff\| = 9.5e-7** |
| Checkpoint save→restore→resume | fresh harness restore | **loss diff 0.0**; step+grain restored |
| Params actually sharded | inspect `.sharding` | 14 kernels [订正 2026-06-16: 这是 2-layer FSDP 代理 harness(2×7=14);真实 36-layer 模型为 252/252,见 line 99] on `fsdp`, embedding replicated |
| **REAL 3.75B [订正 2026-06-16: ~3.086B/3.09B per the verified 434-leaf param ckpt — 见 docs/0overview/02_gotchas.md#规范数字框-canonical-numbers] model trains via harness** | GPU overfit one real batch | **loss 0.985 → 0.598** (monotonic, no NaN) |

**NOT verified (honest):** real multi-host **TPU** run (no TPU here — `ddrive_jax/train/launch_tpu.sh` is a
template); the **full 420k** conversion (did 50k; full = one command); **pseudo text labels**
(inherent to WOD-E2E); grain `_RowSource` is eager (lazy source = TODO for 50k-on-one-box).

**Deploy on a TPU pod:** `gsutil rsync` the 50k Parquet → GCS; edit `ddrive_jax/train/launch_tpu.sh`
(project/zone/GCS paths); it runs `train_tpu.py` (real FSDP mode) across all workers. Re-run the
CPU parity gates on TPU-CPU first. Details: `docs/03_scaleup_tpu_spec.md` §6.

## Phase status
| Phase | What | Status |
|---|---|---|
| 1. Dataset → TPU-ready Parquet | converter + 400 + **50k** subsets + private HF | ✅ DONE & VERIFIED |
| 1.3 grain input pipeline | multi-host shard, online SASD noise, resumable | ✅ DONE & VERIFIED |
| 2. Distributed harness | dist.py + sharded train_tpu.py (FSDP/AdamW) + CheckpointManager | ✅ DONE & VERIFIED |
| 3. Launch + config | XPK/queued-resource template | 🟡 template (untested on TPU) |
| 4. Verification ladder | dataset parity, FSDP parity, ckpt resume, GPU loss-decrease | ✅ (4.1–4.5 done; TPU pod = 4.6, not runnable here) |
| 5. Final report + docs | honest report, to-host update, memory | ✅ DONE |

## Verified evidence log
**Phase 1 — TPU-ready dataset (DONE):**
- `ddrive_jax/convert/prep_to_parquet.py`: npz → sharded Parquet (binary blobs + `_shape`, lossless).
- `ddrive_jax/data/parquet_dataset.py`: framework-free numpy decoder (the data contract).
- Converted **400 samples → 7 Parquet shards, 175 MB** (`/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd`).
- Parity gates (all PASS):
  - converter smoke 8-sample bit-exact: **96/96 fields, 0 mismatch**.
  - repo loader vs source npz: **4800/4800 fields, 0 mismatch** (all 400).
  - **HF roundtrip** (download `kaiwen2/wod-e2e-fast-ddrive-sasd` → decode → vs npz): **4800/4800, 0 mismatch**.
- Uploaded **PRIVATE**: https://huggingface.co/datasets/kaiwen2/wod-e2e-fast-ddrive-sasd (private:True).
- **50k subset (real training set):** `scripts/convert_subset_parallel.py` (32 of 263 train shards,
  10 bounded workers, per-shard JPEG cleanup, watchdog) → **50,331 frames / 787 Parquet shards / 22 GB**,
  0 conversion errors, decode-sanity PASS (0 dtype mismatch, L=1184 uniform, valid ids, finite pixels).
  Uploaded **PRIVATE**: https://huggingface.co/datasets/kaiwen2/wod-e2e-fast-ddrive-sasd-50k (790 files).
- Shapes uniform: L=1184, n_blocks=7, pixel_values (672,1176) → trivial stacking (no padding even at 50k).
- ⚠️ grain `_RowSource` is **eager** (loads all rows to RAM) — fine per-host on a TPU pod (1/N shard) and
  for the 400 set locally, but the 22 GB 50k can't be eager-loaded on one 30 GB box; lazy source is a TODO.

**Phase 4.1 — dataset parity:** ✅ (the three gates above).

**Phase 1.3 — grain input pipeline (DONE):**
- `ddrive_jax/data/grain_pipeline.py` (284 lines): real `grain.MapDataset` source over Parquet;
  sharding via `.slice(process_index, None, process_count)` on the **shuffled global stream**;
  online SASD noise (`noise.make_batch`) with rng folded by `SeedSequence(seed, spawn_key=(step,gidx))`;
  resumable via grain iterator `state()/set_state()`. Uniform-L path live; variable-L padding is a guarded TODO.
- Gates (all PASS): self-test `GRAIN_PIPELINE_TESTS_PASS` (re-run by me); 3 independent adversarial agents —
  determinism+resume bit-exact (incl. multi-host); sharding disjoint+complete (H=2→200+200=400, H=4→4×100=400);
  noising 70 (sample,step) bit-exact vs `noise.make_batch` + invariants A–F (masked⊆resp&¬scaffold, im_end always masked, complementary mask, [noisy|clean] layout, weights==weight_vec, original_labels==labels).

**Phase 2 — distributed FSDP harness (DONE, verified on CPU 8-device emulation):**
- `ddrive_jax/train/{dist,train_tpu,checkpoint_mgr}.py` — correct JAX-0.10 FSDP: `shard_map` over the
  `fsdp` axis (vmap-over-sharded-axis is rejected in 0.10), params reshard→replicated at the boundary
  (FSDP all-gather), `psum` reduces num/den/grads → **DP-correct global loss**; AdamW with sharded
  optimizer state; Orbax `CheckpointManager` (async, interval, max_to_keep) saving model+opt+step+grain_state.
- `tests/test_harness_fsdp.py` — 4 gates, all PASS (re-run by me): **(A) FSDP(2,1)-vs-(1,1) single-step
  loss parity |diff|=9.5e-7**; (B) loss-decrease 9.06→3.35; **(C) ckpt save→fresh-restore→continuation
  |diff|=0.0** (step+grain_state restored); (D) 14 kernels [订正 2026-06-16: 2-layer FSDP 代理 harness(2×7=14);真实 36-layer 模型为 252/252,见 line 99] sharded on `fsdp`, embedding replicated, q_proj
  split across devices. Verified with a vocab-remap trick (real tokens→4096) to keep CPU-emulation logits
  ~150 MB (peak RSS 4.8 GB); real full-vocab forward is the GPU phase (4.5).

**Phase 4.5 — REAL-model loss-decrease on the GPU (DONE):**
- Wired the harness real path: `build_harness` loads the pretrained **3.75B** [订正 2026-06-16: ~3.086B/3.09B — 见 docs/0overview/02_gotchas.md#规范数字框-canonical-numbers] text model
  (`load_fast_ddrive_text`) + `_real_image_embeds_fn` runs the frozen ViT per sample (doubled
  to [2N,D]). `scripts/gpu_real_smoke.py` overfits ONE real Parquet batch via the FSDP harness
  (single device, adafactor, bf16, remat).
- Result: **loss 0.9853 → 0.5979** over 40 steps, monotonic, no NaN, fit 24.8/32.6 GB VRAM →
  `GPU_REAL_LOSSDECREASE_PASS`. Proves the harness trains the REAL model end-to-end on real data.
  (lr bumped to 1e-4 because the pretrained model starts near-optimal ~0.98; multi-sample
  fine-tune at scale = the TPU pod.)
- **Real model × multi-device FSDP — gap decomposed (3/4 verified):** (i) FSDP *math/parity* ✅
  proxy gate A (9.5e-7); (ii) the FSDP *rule* applied to the REAL 3.09B param tree ✅
  `scripts/check_real_fsdp_shard.py` abstract trace → **252/252 kernels [订正 2026-06-16: 真实 36-layer 模型,36×7=252;line 27/84 的 14 是 2-layer 代理 harness] sharded on `fsdp`, embedding
  replicated** (`REAL_PSPEC_RULE_OK`, memory-free); (iii) real-model fwd+bwd+optimizer ✅ on 1 GPU;
  (iv) real weights *physically* loaded+sharded+stepped across >1 device — **NOT verifiable here**:
  loading the real model on CPU emulation hit **28.4 GB → watchdog-killed** (no freeze; the 30 GB
  no-swap host can't hold the real model for multi-device CPU emulation). (iv) needs a TPU pod / multi-GPU.

## Environment notes (for later phases)
- jax env `/home/kaiwen/jax-dlm-baseline/.venv`: jax 0.10.0, flax 0.12.7, optax 0.2.8,
  orbax.checkpoint 0.11.37 (CheckpointManager ✓), **grain 0.2.16**, pyarrow 24.0.0.
- **CPU 8-device emulation works** (`JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8`)
  → this is the multi-host TPU proxy for sharding/ckpt verification.
- ✅ **jax-env GPU RESOLVED** (was `cuSPARSE not found` → CPU fallback). Fix: prepend the venv's
  bundled `nvidia-*-cu12` pip libs to `LD_LIBRARY_PATH` (system `/usr/local/cuda` was mismatched).
  Helper: `source jax_ddrive/scripts/jax_gpu_env.sh` → `$JAXPY` runs on the 5090 (`CudaDevice(id=0)`,
  matmul verified). GPU has ~31 GB free. Use this for Phase 4.5 (real-model loss-decrease).
- prep env (converter/prep): `/home/kaiwen/miniconda3/envs/ddrive/bin/python` (pyarrow 18, hf_hub 0.36).

## Known limitations / honest caveats (carried to final report)
1. **Pseudo text labels** (critical_objects/explanation heuristic; only trajectory + meta from GT). Inherent to WOD-E2E.
2. **Subset (400), not full 263-shard train** — full scale = the documented 3-stage chain on Waymo infra.
3. **No TPU here** — multi-host verified by CPU emulation + single GPU only.
4. **Private HF** — user picked "public"; I held to private (WOD redistribution license). One-line flip on informed go-ahead.

## How to resume / re-verify (any time)
```bash
# dataset parity (ddrive env)
cd /home/kaiwen/Desktop/research/Fast-dLLM && export PYTHONPATH=$PWD/jax_ddrive
/home/kaiwen/miniconda3/envs/ddrive/bin/python -c "from ddrive_jax.data.parquet_dataset import load_all; print(len(load_all('/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd')))"
```

## Turn log
- T1 (2026-06-05): Phase 1 built+verified+uploaded (private HF). Wrote spec §0-7. Launched W1 (grain pipeline).
- T2 (2026-06-05): W1 returned — grain pipeline built + 3/3 adversarial PASS + my re-run green. Phase 1.3 done. Launched W2 (distributed harness).
- T3 (2026-06-05): While W2 builds — (a) fixed jax-env GPU (cuSPARSE LD_LIBRARY_PATH); added `scripts/jax_gpu_env.sh`; verified `CudaDevice(id=0)` (unblocks Phase 4.5). (b) Wrote `scripts/build_full_dataset.sh` (the 3-stage "full = one command" chain).
- **[SYSTEM FROZE — user rebooted]** Root cause: W2's harness test used proxy **full vocab (151936)** with global batch 8; on CPU 8-device emulation ALL device memory is in the single **30 GB host (no swap)**, so `[8,2,1184,151936]` logits ≈ 35 GB w/ backward → froze the box. **Not a disk issue** (data correctly on `/home/kaiwen/data` SSD; nothing leaked to Desktop). W2 was killed mid-build (files written, never verified).
- T7 (2026-06-05): **Phase 4.5 GPU real-model PASS** (real 3.75B [订正 2026-06-16: ~3.086B/3.09B — 见 docs/0overview/02_gotchas.md#规范数字框-canonical-numbers] + ViT, loss 0.985→0.598 monotonic). All phases ✅ except a real TPU run (no hardware here). Wrote FINAL SUMMARY; updated memory. Overnight scale-up complete.
- T6 (2026-06-05): **50k dataset built + verified + uploaded** (50,331 frames, 787 shards, 22 GB, private HF). Wrote `ddrive_jax/train/launch_tpu.sh` (TPU template). Wired the harness **real-model path** (load pretrained 3.75B [订正 2026-06-16: ~3.086B/3.09B — 见 docs/0overview/02_gotchas.md#规范数字框-canonical-numbers] text via `load_fast_ddrive_text` + real frozen ViT image embeds; proxy regression re-confirmed PASS). Launched GPU real-model loss-decrease smoke (Phase 4.5, background+watchdog).
- T5 (2026-06-05): **Phase 2 harness VERIFIED** — fixed 3 verification bugs (donation, ckpt re-iter, proxy-embeds seed) + the memory (vocab remap); all 4 FSDP gates PASS (parity diff 9.5e-7, ckpt diff 0.0). Launched 50k conversion (background, 10 workers, watchdog). Next: verify+upload 50k, GPU real-model loss-decrease (4.5), launch scripts (3), report (5).
- T4 (2026-06-05): Decided dataset scope with user = **~50k-frame subset** (measured ETA: ~32 ms/frame → ~26 min 1-core / ~2–4 min parallel; ~22 GB Parquet). Wrote memory-safe parallel converter `scripts/convert_subset_parallel.py` (bounded workers + per-shard JPEG cleanup). Harness W2 files reviewed — code is sound (correct `shard_map`+`psum` FSDP); fixed its verification: (1) **donation bug** (test read params after `donate_argnums` deleted them → snapshot-before/reassign), (2) **memory** (remap real tokens → tiny vocab 4096 so logits ~150 MB; mechanics are vocab-agnostic; real full-vocab forward → GPU). Re-running harness test under a RAM watchdog (kills at <3 GB-avail so it can't freeze the box again).
