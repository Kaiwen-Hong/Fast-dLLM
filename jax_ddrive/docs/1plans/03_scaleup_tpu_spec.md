# Fast-dDrive JAX — scaled-up multi-node TPU training spec (Phase 6)

Status: **Phase 6 complete on local hardware (2026-06-05/06). Next: MaxText port (Phase 7).**
This is the SSOT for the scale-up. Decisions are locked and reflected below.

> 📌 **STATUS UPDATE (2026-06-08):** Phase 7 (MaxText port) is no longer "next/TODO" — it is **done and
> proven on real TPU** (MaxText SASD on v6e-1, real weights, loss 0.31/0.56). The "MaxText port (Phase 7
> — TODO)" section, the "MaxText integration ❌ Not yet started" and "TPU pod end-to-end ❌" rows below
> are **superseded**. Current truth: **`docs/4collect/OVERNIGHT_TPU_PROGRESS.md`**. Remaining open item: the
> ≥8-chip multi-node run, blocked only by GCP trial capacity (external/transient, not a code issue).

## 0. Locked decisions

| # | Decision | Choice | Status |
|---|---|---|---|
| 1 | Training framework | **MaxText fork (Path B)** — user confirmed 2026-06-06 | ⏳ next phase |
| 2 | Dataset format | **Apache Parquet shards** — HF-native, grain-readable, MaxText `hf`-path compatible | ✅ built |
| 3 | Dataset hosting | **PRIVATE** HF (WOD license prohibits redistribution; functionally identical for TPU) | ✅ uploaded |
| 4 | Conversion scope | **50k-frame subset** (50,331 frames) now; full 263-shard = one command | ✅ done |
| 5 | Verification bar | CPU 8-device emulation + single GPU (no TPU on this box) | ✅ all local gates pass |

**Decision 1 detail — why MaxText over Path A:**
Path A (self-contained NNX harness) was built and verified — it works. But MaxText is the
right production choice for Waymo infra: battle-tested multi-host, preemption recovery,
goodput tracking, XPK/GKE launch, and the `jax-mdlm-handoff` repo (LLaDA in MaxText) is the
exact template for grafting a masked-diffusion model. The Parquet dataset is already
MaxText `hf`-data-path compatible. The Path A harness code remains the reference
implementation and source of algorithm truth.

## 1. Dataset (DONE)

### 1.1 What exists on disk + HF
- **`kaiwen2/wod-e2e-fast-ddrive-sasd`** (private HF): 400-frame validation subset, 7 Parquet shards, 175 MB. Bit-exact vs source npz (4800/4800).
- **`kaiwen2/wod-e2e-fast-ddrive-sasd-50k`** (private HF): **50,331 frames, 787 Parquet shards, 22 GB.** 0 conversion errors; 1024-sample decode sanity PASS; count matches manifest. **This is the training set.**
- Local raw data: `/home/kaiwen/data/fast-ddrive/waymo/train/` (263 shards, 877 GB, ~420k frames).

### 1.2 3-stage conversion pipeline
1. `fast_ddrive/data/convert_wod_e2e.py` (autovla env): tfrecord → Fast-dDrive JSON + JPEGs (`--with_target`).
2. `jax_ddrive/eval/prep_train_jax.py` (ddrive env): JSON+JPEGs → per-sample SASD npz.
3. `jax_ddrive/ddrive_jax/convert/prep_to_parquet.py` (pyarrow): npz → sharded Parquet.

Orchestration (parallel, per-shard JPEG cleanup, RAM-bounded): `jax_ddrive/scripts/convert_subset_parallel.py`.
Full 420k run: `jax_ddrive/scripts/build_full_dataset.sh --full` (~4 h single-core, ~45 min parallelised).

### 1.3 Parquet row schema
Scalars: `sample_id:str`, `L:int32`, `n_blocks:int32`.
Arrays (little-endian binary blob + `<name>_shape:list<int64>`):
`input_ids[L] i64`, `labels[L] i64`, `rbi[L] i32`, `turn[L] i32`, `scaffold[L] bool`,
`weight_vec[L] f32`, `block_alpha/beta[n_blocks] f32`, `position_ids[3,L] i32`,
`vision_mask[L] bool`, `pixel_values[N,1176] f16`, `image_grid_thw[n_img,3] i64`.
Decode contract: `ddrive_jax/data/parquet_dataset.py` (numpy, no torch/jax).
Current subset shapes: L=1184 uniform, n_blocks=7, pixel_values (672,1176).

### 1.4 Known limitations
- **Pseudo text labels**: `critical_objects`/`explanation` are heuristic weak labels; only `trajectory` + `future_meta_behavior` derive from real GT/intent. Inherent to WOD-E2E (no native text labels).
- `grain._RowSource` is **eager** (reads all rows to RAM on construction). Fine per-host on a TPU pod (1/N shard), but 50k cannot be eager-loaded on the single 30 GB dev box. Lazy source = TODO before local large-scale runs.

## 2. Reference harness (Path A — VERIFIED, kept as algorithm source)

Built and verified. **Not the production training path** (MaxText is), but the implementation
truth for the SASD algorithm under JAX — consult this when grafting into MaxText.

Files: `ddrive_jax/train/{dist.py, train_tpu.py, checkpoint_mgr.py, launch_tpu.sh}`

Verified gates (all re-run 2026-06-06):
- `GRAIN_PIPELINE_TESTS_PASS`: determinism, disjoint sharding, noising bit-exact (3 adversarial agents).
- `HARNESS_FSDP_TESTS_PASS`: FSDP-vs-single-device parity **|diff|=9.5e-7**; ckpt resume **diff=0.0**; 14 kernels sharded; embedding replicated.
- `GPU_REAL_LOSSDECREASE_PASS`: real pretrained 3.09B model + frozen ViT, loss **0.985→0.598**, 40 steps, no NaN.
- `REAL_PSPEC_RULE_OK`: 252/252 real-model kernels get correct FSDP pspec (abstract trace, memory-free).

## 3. MaxText port (Phase 7 — TODO)

**Template**: `jax-mdlm-handoff` repo — LLaDA grafted into MaxText. The same 5 steps apply.

### 3.1 What to graft
1. **`diffusion/` module** into MaxText: `masks.py`, `sasd_loss.py`, `noise.py` — copy verbatim. [SUPERSEDED 2026-06-16: the SASD training math was re-ported as a single consolidated `src/maxtext/diffusion/sasd.py` (BIT-EXACT re-derivation, not a verbatim copy of masks/sasd_loss/noise); only the `eval_sasd/` inference stack is vendored verbatim.]
2. **`loss_fn` diff** (~5 lines): replace MaxText's standard CE with `sasd_total_loss` (section-weighted CE on noisy half + causal CE on clean half; global-token denom via `psum`).
3. **Bidirectional attention patch**: MaxText uses causal mask by default; patch to use `hybrid_block_causal_mask_dense(rbi, turn, L)` for the SASD doubled-sequence forward.
4. **Waymo grain data source**: wire `ddrive_jax/data/grain_pipeline.make_sasd_loader` as MaxText's data input (the Parquet format is already MaxText `hf`-path compatible).
5. **Model**: Qwen2.5-VL text decoder is plain Flax NNX. MaxText's `llada.py:40-72` is the reference sharding pattern. Port `Qwen25TextModel` into MaxText's model registry OR load it as an external NNX model (simpler, avoids re-parity).

### 3.2 Re-validation after graft
Re-run `jax_ddrive/scripts/run_all_verification.sh` on TPU-CPU (10 gates) before any large run.
Acceptance: parity gates < 1e-3, FSDP-vs-1device < 1e-4, loss decreases, ADE@3s < 1.0 m.

### 3.3 Data on GCS
```bash
gsutil -m rsync -r /home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd_50k gs://YOUR_BUCKET/wod_e2e_sasd_50k
# or download from private HF:
# huggingface-cli download kaiwen2/wod-e2e-fast-ddrive-sasd-50k --repo-type dataset --local-dir ...
```

## 4. Verification ladder (honest labels)

| Gate | Status | Evidence |
|---|---|---|
| Dataset parity (400) | ✅ | bit-exact 4800/4800 fields |
| Dataset integrity (50k) | ✅ | 787 shards = 50,331 rows; 1024-sample decode sanity |
| grain multi-host | ✅ | CPU emulation, 3 adversarial agents |
| FSDP-vs-1device parity | ✅ | \|diff\|=9.5e-7 |
| Checkpoint resume | ✅ | diff=0.0, step+grain restored |
| Real model trains via harness | ✅ | loss 0.985→0.598, GPU |
| Real model pspec rule | ✅ | 252/252 kernels, abstract trace |
| Real model × multi-device physical | ❌ | Needs multi-GPU or TPU (30GB box OOMs) |
| MaxText integration | ❌ | Not yet started (Phase 7) |
| TPU pod end-to-end | ❌ | No TPU on dev box |

## 5. Honest limitations (to carry into any paper/report)
- Pseudo text labels (inherent to WOD-E2E).
- 50k-frame subset; full 420k = one command.
- No real multi-host physical verification on this box.
- Private HF (WOD license). Flip to public only with explicit redistribution rights.
- MaxText port not yet done.
