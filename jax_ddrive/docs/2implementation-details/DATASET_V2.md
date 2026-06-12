# Dataset v2 — ArrayRecord with precomputed ViT embeds (the production training format)

*Living doc. Owner: data pipeline. Created 2026-06-12. Update when the schema, builder, or
reader changes.*

Dataset v2 is what TPU training consumes: **ArrayRecord shards of `tf.train.Example`
records carrying the 12 SASD arrays (including `pixel_values`, kept for verifiability) plus
`image_embeds` — the frozen-ViT output precomputed offline** so the training data loop is
pure IO (no ViT on the pod, no transformers/safetensors deps in the hot path).

## 1. Record schema

Scalars: `sample_id` (bytes), `L` (int64), `n_blocks` (int64).
Arrays (each a `tf.io.serialize_tensor` bytes feature; decode with
`tf.io.parse_tensor(bytes, dtype)`):

| field | dtype | shape | note |
|---|---|---|---|
| input_ids / labels | int64 | [L=1184] | |
| rbi / turn | int32 | [L] | |
| scaffold / vision_mask | bool | [L] | |
| weight_vec | float32 | [L] | |
| block_alpha / block_beta | float32 | [n_blocks] | |
| position_ids | int32 | [3, L] | 3D M-RoPE |
| pixel_values | float16 | [672, 1176] | processor output, kept as the verification source |
| image_grid_thw | int64 | [3, 3] | 3 imgs × (t,h,w) |
| **image_embeds** | **bfloat16** | **[168, 2048]** | **single copy — see doubling rule** |

**Embeds provenance:** frozen Fast-dDrive ViT (release snapshot), fp32 forward with
`jax_default_matmul_precision=highest` (TF32 off), per-sample K=1 calls, cast to bf16.
The jitted forward is a mirror of `VisionTransformer.__call__` cross-checked against the
validated eager path on every shard's first record (assert rel < 2e-4; observed ≤ ~1.3e-5).

**Doubling rule:** the model consumes the doubled `[noisy | clean]` sequence, so the
consumer must `concat([ie, ie], axis=0)` (per sample) → `[336, 2048]` — exactly what
`compute_fast_ddrive_image_embeds` produced online. `sasd_num_image_tokens: 336` in the
MaxText config is this doubled count.

**Verification chain** (all run during the build): (1) jitted-vs-eager ViT cross-check per
shard; (2) bitwise recompute self-check + bf16 serialize/parse roundtrip every
`--verify_every` records; (3) per-shard `written == source_rows` assertion; (4) independent
read-back: AR record fields are byte-identical to the source parquet row (covered
systematically by `tests/test_ar_pipeline.py`). Anyone can re-verify embeds later from the
stored `pixel_values`.

## 2. Builder

```
raw WOD-E2E tfrecords
  └─ convert_wod_e2e.py (--with_target)      autovla env   → JSON + JPEGs
      └─ prep_train_jax.py                    ddrive env    → per-sample npz
          └─ prep_to_parquet.py               pyarrow       → parquet (bit-exact blobs)
              └─ scripts/parquet_to_ar_with_embeds.py   jax venv, GPU
                                                            → AR v2 (this format)
```

The last stage is `jax_ddrive/scripts/parquet_to_ar_with_embeds.py` (resumable: atomic
`.tmp → os.replace` per shard, skips existing; ~19 samples/s on the 5090). 1:1 source
parquet file → AR shard. Writes `dataset_info_<split>.json` with `num_samples`,
`array_dtypes`, embeds provenance and the doubling rule.

## 3. Built datasets (local `…/hf/` + GCS `gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd/`)

| dir | rows | shards | role |
|---|---|---|---|
| `wod_e2e_sasd_v2_ar` | 400 | 7 | smoke/tests (mirrors the 400-sample parquet) |
| `wod_e2e_sasd_val_v2_ar` | 479 | 8 | **val split** (rated frames, `--with_target` rebuild; split=`val`) |
| `wod_e2e_sasd_50k_v2_ar` | 50,331 | 787 | shakeout training |
| `wod_e2e_sasd_full_v2_ar` | 415,663 | 130 | production training |

(Sources kept: `wod_e2e_sasd_full_packed` parquet = the bit-exact source of truth;
the v1 AR `wod_e2e_sasd_full_tfexample_ar` is superseded by v2 and kept only on GCS.)
All sets verified 100% uniform: L=1184, pixel (672,1176), 3 images → the grain uniform
fast path applies everywhere; the `_pad_sample` TODO is never hit.

## 4. Reader (training side)

`make_sasd_loader(data_dir, split, per_host_batch, seed, process_index, process_count)`
**auto-detects** `*.arrayrecord` (lazy `ArRecordSource`, full-scale) vs `*.parquet`
(eager `_RowSource`, small sets) — same public API, so callers (FSDP harness, MaxText
`WaymoSasdDataIterator`) need no change. With v2 data the batch dict additionally carries
`image_embeds (B, 168, 2048) bf16` and the loader sets `loader.has_embeds=True`:

* canonical copy: `jax_ddrive/ddrive_jax/data/{grain_pipeline,ar_dataset,parquet_dataset}.py`
* vendored copy (self-contained, no ddrive_jax import): MaxText fork
  `src/maxtext/input_pipeline/sasd_data/` — **edit the canonical copy and re-copy**; the
  vendor headers carry the pinned source commit.

MaxText consumption (`waymo_sasd_data_processing.py`): when `loader.has_embeds`, the ViT
is **never loaded** and embeds are concat-doubled from the batch; v1 pixels-only data falls
back to the on-host frozen-ViT path (lazy ddrive_jax import).

**Checkpoint/resume:** `dataset_type="waymo_sasd"` joined MaxText's grain checkpoint
family — the loader's grain `DatasetIterator` state (`{"next_index": N}`) is saved in every
checkpoint under `iter/process_<i>-of-<n>.json` and restored on resume, so the data stream
*continues* instead of replaying. Validated by a `GrainCheckpointHandler` save/restore
round-trip (bit-identical continuation) and by the on-TPU resume validation
(`launch_maxtext_sasd_tpu_v2.sh`).

## 5. Tests

* `jax_ddrive/tests/test_ar_pipeline.py` — AR-vs-parquet bit-exact (source + 3 loader
  batches), embeds doubling contract, AR resume. CPU, seconds.
* `jax_ddrive/tests/test_grain_pipeline.py` — the original 7 guarantees, regression-clean
  on the parquet path.
* Fork self-containment: import + loader + handler round-trip with **no** `ddrive_jax` on
  `PYTHONPATH` (see PATCHES.md §How to validate).

## 6. Round-2 from-raw re-verification + review website

Independent of the build-time checks (§1), `jax_ddrive/scripts/verify_ar_round2.sh`
re-verifies sampled v2 records **from the raw tfrecords**: it re-runs the whole chain
(stage 1 converter → stage 2 prep), compares the 12 source arrays bit-exact against the
AR rows, and re-derives `image_embeds` from the stored `pixel_values` (eager frozen ViT,
fp32 highest → bf16) under a two-tier criterion — bitwise fraction ≥ 98.5% (observed
~99.8%) AND every mismatched element ≤ 1.5× its own bf16 ulp or |Δ| ≤ 2e-3 (the bf16-level
global rel is reported, not gated: a benign 1-ulp flip at a near-max element already
exceeds 1e-3; the builder's 1e-3 cap applies to fp32 pre-quantization values). Plus
structure self-checks, answer-text round-trip, trajectory == GT@1 s, pixel reconstruction,
and an embeds-PCA triptych for eyeballing.

* **Status: 12/12 PASS (2026-06-12)** — 10 train (`wod_e2e_sasd_full_v2_ar` shard 0
  rows 0–9) + 2 val (`wod_e2e_sasd_val_v2_ar` rows 0–1). Artifacts:
  `/home/kaiwen/data/fast-ddrive/verify_round2_v2/review/report.md`.
* Re-run: `NT=10 NV=2 bash jax_ddrive/scripts/verify_ar_round2.sh` (~5 min; needs the
  autovla/ddrive/jax envs + the 5090 for the embeds recompute).
* **Review website** (what a v2 record is, the build chain, the v2 feed path, the 12
  verified examples with BEV/layout/M-RoPE/mask/embeds figures):
  `jax_ddrive/visualizations/index.html` — `bash jax_ddrive/visualizations/serve.sh`,
  then `ssh -L 8890:localhost:8890 <desktop>` → http://localhost:8890. Regenerate after
  a new verify run: `PYTHONPATH=jax_ddrive <ddrive-python>
  jax_ddrive/scripts/make_dataset_website.py`.
