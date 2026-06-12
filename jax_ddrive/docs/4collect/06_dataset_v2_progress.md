# 06 — Dataset v2 (pixels+embeds AR) + full-scale reader + TPU re-validation (2026-06-12 overnight)

*Frozen log of the overnight build. Truth lives in `../2implementation-details/DATASET_V2.md`
and `../../to-host.md`; this file records what happened and in what order.*

## Decisions (user-locked)

1. Training data source = **ArrayRecord** (grain-native) — not lazy parquet.
2. Dataset v2 keeps **both** `pixel_values` (resized images) **and** new precomputed
   `image_embeds` (frozen-ViT, fp32-computed/bf16-stored, single copy [168,2048]) — pixels
   retained so embeds are independently re-verifiable.
3. Build **both** 50k and full-415k. bf16 embeds (not fp32).
4. Tier-A disk cleanup approved & done first (~262 GB freed: 6,499-file small parquet dup,
   maxtext smoke run dirs, phase-5 ckpts, local params v1; free 1.2 → 1.5 TB).

## Timeline

* **Converter** `scripts/parquet_to_ar_with_embeds.py`: per-sample K=1 jitted ViT mirror
  (host-constant grid → window/rotary/masks precomputed once; the eager `__call__` is not
  jittable as-is), `jax_default_matmul_precision=highest` (first run with TF32 default
  diverged from eager by 7.5e-4 → with highest, 5.2e-6), per-shard eager cross-check
  (threshold 2e-4 after a 1.3e-5 fp32-noise false-trip at 1e-5), bitwise recompute +
  bf16 roundtrip self-checks, atomic resume. Throughput ~19 samples/s on the 5090.
* **400-sample smoke** → 7 shards, emb (168,2048), grids=1, all checks green; independent
  read-back: 12 fields byte-identical to parquet.
* **50k + full conversions** chained in background (50k ~45 min; full ~6 h).
* **val split (479 rated)**: rebuilt in TRAINING format (`--with_target --rated_only` from
  raw val tfrecords — the existing eval prep was inference-format, no labels) → parquet
  (8 shards) → AR v2. `AR_WITH_EMBEDS_DONE total=479 verified=8 max_eager_rel_diff=3.9e-06`.
* **AR reader** (`ddrive_jax/data/ar_dataset.py` + `grain_pipeline.py` auto-detect):
  `tests/test_ar_pipeline.py` 4/4 (source + loader bit-exact vs parquet, doubling, resume)
  and `test_grain_pipeline.py` 7/7 regression. All other CPU suites re-run green
  (noising, mask_loss, sharding, eval_ports, multihost_datafeed, harness_fsdp).
* **MaxText fork** (`maxtext-dlm-fork @ a645b25`): vendored self-contained
  `input_pipeline/sasd_data/` (no ddrive_jax on the pod for v2 data), iterator consumes
  precomputed embeds (ViT never loaded), `waymo_sasd` joined the **grain checkpoint
  family** (iterator state saved/restored → resume continues the stream; proven by a
  real-`GrainCheckpointHandler` round-trip, bit-identical). Working tree committed
  (was previously untracked tarball-only) + `PATCHES.md` documents the whole diff.
* **GCS**: refreshed `code/{maxtext_fork,jax_ddrive}.tgz`; uploaded
  `wod_e2e_sasd_v2_ar` (363 MiB) + `wod_e2e_sasd_val_v2_ar` (438 MiB).
* **TPU re-validation** `launch_maxtext_sasd_tpu_v2.sh` (v6e-1: 12 steps on v2 AR,
  ckpt@6 incl. iter state, relaunch must resume from 7): first attempt
  `PROVISION_FAILED — no v6e-1 capacity @ us-east5-a` (same transient GCP capacity issue
  as 06-08). Bounded catcher armed (≤10 rounds, alternating us-east5-a/b, one paid run).

## Results (filled as they land)

* 50k v2: ✅ **50,331 records / 787 shards / 45 GB**, eager-diff max 2.53e-4 (one tail
  sample; led to the two-tier check redesign) / **median 1.74e-6**; uploaded to GCS
  (44.72 GiB).
* val v2: ✅ 479 / 8 shards, eager-diff max 3.9e-6; uploaded.
* sampled audits (`scripts/verify_ar_v2_dataset.py`): ✅ 50k (98 records) and val
  (50 records) — counts 3-way consistent, all fields byte-identical to parquet.
* **TPU v6e-1 validation: ✅ V2_TPU_VALIDATION_PASS** (2nd attempt, `sasd-v2b`).
  - RUN 1: 12 steps on v2 AR, **precomputed embeds, ViT never loaded** — loss
    0.985 → 0.29-0.56 (same band as the 06-08 real-weights run);
    **83.4 TFLOP/s/device vs 65 on the old ViT-in-loop path (+28%)**.
    Checkpoint at step 6 + completion ckpt at 11, both with `iter/process_0-of-1.json`.
  - RUN 2 (steps=18): restored the step-11 state **+ grain iterator**
    (`CheckpointManager item_names=('items','iter')` with our GrainCheckpointHandler),
    trained exactly 12..17 (zero step-0 evidence → continued, not restarted),
    losses 0.35-0.59 continuous with RUN 1.
  - 1st attempt failed only on my test design: same `steps=12` in RUN 2 → MaxText
    restored the completion ckpt and exited with an empty step range (newsteps=0).
    Fixed by RUN 2 `steps=18` + segment-based criteria. ~$2 total across attempts;
    QRs auto-deleted.
  - Determinism note: both attempts produced identical per-step losses in RUN 1
    (e.g. step 5 = 0.398) — the deterministic grain stream at work.
* full v2: ✅ **415,663 records / 130 shards / 369 GB**, 347 min on the 5090,
  eager-diff max 6.55e-5 / median 1.60e-6, 406 in-run verifications.
* full v2 audit: ✅ counts 3-way consistent; 130 sampled records byte-identical to
  parquet; 6 embeds independently recomputed (eager fp32): 99.67% of elements
  bit-identical after bf16 quantization, max rel 2.58e-3 ≈ the bf16 resolution itself.
* upload: ✅ full v2 → GCS **368.68 GiB / 131 objects** (~55 min at ~114 MiB/s).
* 10-gate re-run: **9/10 PASS** — with a real find & fix on the way:
  - First runs OOM-killed the three big-model fp32 parity gates (kernel: anon 16.9G +
    shmem 8.2G ≈ 25 G peak vs 24 G available; the eager loader materialized the full
    6.2 GB tensor dict). **Fixed by making `load_fast_ddrive_text` streaming**
    (per-tensor, mirror of the ViT streaming loader; commit `7198a8a`) →
    phase1/2/4/4b all PASS again at the historical numbers (text logits rel-max
    3.23e-5, top-1 100%).
  - `phase3_lora_train`: the documented thin-margin flake
    (`run_all_verification.sh:43`, to-host §2.5). This round's regenerated 2-sample
    fixed batch drew a flat sample: 1.1182→1.1184 (+2e-4 vs the ~1e-3 signal; the other
    sample decreases). Deterministic for this draw (two identical standalone re-runs);
    PASSED earlier the same night under a different draw. No dependence on any v2
    change (weights load bit-identically; all parity gates green).
* MaxText GPU smoke on v2: **skipped deliberately** — its purpose was a fallback if TPU
  capacity never appeared; the real-TPU validation (`V2_TPU_VALIDATION_PASS`) covers the
  production path end-to-end, and the smoke's eager param-ckpt build is itself an OOM
  hazard on the 30 G box.
* local cleanup: ✅ superseded v1 AR (158 G) deleted (GCS copy verified, 157.78 GiB);
  final local layout = packed parquet source-of-truth (178 G) + v2 AR train
  full/50k/400 + val (+ small v1 sets); 1.2 T free.
* **round-2 + review website upgraded to v2 (06-12 afternoon): ✅ 12/12 PASS** (10 train
  + 2 val) — from-raw re-run of the whole chain, 12 columns bit-exact vs the v2 AR rows,
  plus `image_embeds` recomputed from stored pixels (eager fp32 highest → bf16): bitwise
  99.76–99.87%, every mismatch ≤1.5× its own bf16 ulp (criterion intentionally does NOT
  gate the bf16-level global rel — a benign 1-ulp flip at a near-max element exceeds
  1e-3; same lesson as the tail-sample two-tier redesign above). Website
  (`jax_ddrive/visualizations/`) regenerated for v2: schema/pipeline/feed-path rewritten,
  + per-sample embeds-PCA triptychs and recompute-diff heatmaps. Tooling:
  `scripts/verify_ar_round2.{py,sh}`, `scripts/make_dataset_website.py`; status SSOT:
  `DATASET_V2.md` §6.

## Where things landed (commits)

`Fast-dLLM @ jax-ddrive-port`: `b18e861` (converter+reader+tests) → `20014a7` (two-tier
check + audit tool) → `fff4a39` (docs) → `7198a8a` (streaming text loader) + to-host twins
& this log. `maxtext-dlm-fork @ master`: `a645b25` (vendored sasd_data, embeds
consumption, waymo_sasd ckpt family, PATCHES.md).
