# DATASET.md — the SASD training dataset (single source of truth)

> **Living doc.** Update this file whenever the dataset format, artifacts, build chain, or
> verification status changes. Last updated: **2026-06-12**.
>
> **中文 TL;DR:** 每条记录 = 一个驾驶瞬间:三张前视相机图(像素在 `pixel_values`,**不在**
> `input_ids` 里——那里只有 `<|image_pad|>` 占位符)+ 文本 prompt(指令、导航命令、3 秒自车历史)
> + JSON 答案(trajectory 是**真值**,critical_objects/explanation 是伪标签)+ SASD 训练结构
> (rbi/scaffold/权重/Beta 调度)。全量 **415,663 帧**已打包成 130 个 ArrayRecord shard,
> 10 个样本经 round-2 从 raw 重跑全链 **bit-exact 验证通过**。可视化复查网站:
> **`jax_ddrive/visualizations/index.html`**(`bash serve.sh` + `ssh -L 8890:localhost:8890`)。

---

## 1. Record schema (13 fields, every sample)

ArrayRecord records are serialized `tf.train.Example`; each array field is a
`tf.io.serialize_tensor` bytes feature (decode: `tf.io.parse_tensor(bytes, dtype)`).
The parquet rows are raw little-endian blobs + `<name>_shape` columns
(decode contract: `ddrive_jax/data/parquet_dataset.py`). Same logical schema in both.

| Field | Shape / dtype | Meaning | Used for |
|---|---|---|---|
| `sample_id` | str | WOD-E2E `frame.context.name` (joins back to GT) | provenance |
| `input_ids` | [1184] i64 | full chat-templated token sequence | model input (embed) |
| `labels` | [1184] i64 | answer-span token ids; `-100` elsewhere | loss targets |
| `rbi` | [1184] i32 | response block idx (−1 = prompt; ≤32 value-tokens per block, per section) | attention mask + section map |
| `turn` | [1184] i32 | +1 at each block boundary | training mask turn rule |
| `scaffold` | [1184] bool | frozen JSON skeleton (keys/quotes/separators) — never masked, never trained | noising freeze |
| `weight_vec` | [1184] f32 | per-token loss weight: CO 1.5 / explanation 1.0 / fmb 2.0 / trajectory 3.0 | section-weighted CE |
| `block_alpha`, `block_beta` | [n_blocks] f32 | per-block Beta(α,β) noise schedule: CO (1,2) · exp (1,1) · fmb (1,1.5) · traj (2,1) | online noising |
| `position_ids` | [3,1184] i32 | 3D M-RoPE (temporal/height/width) | RoPE cos/sin |
| `vision_mask` | [1184] bool | `ids ∈ {image_pad 151655, vision_start 151652, vision_pad 151654}` | aux/debug |
| `pixel_values` | [672,1176] f16 | 3 images × 224 patches; each row = 14×14×3ch×2 temporal frames, CLIP-normalized. **Losslessly invertible to the resized images** (verified) | frozen ViT input |
| `image_grid_thw` | [3,3] i64 | per-image (t,h,w) patch grid = (1,16,14) | ViT indexing + M-RoPE |
| `L`, `n_blocks` | scalars | sequence length / #blocks | static shapes |

**Key fact (common confusion):** image content is *not* in `input_ids`. The three images
appear there only as three runs of 56 identical `<|image_pad|>` placeholders
(`<|vision_start|> … <|vision_end|>`); pixels live in `pixel_values` and the frozen-ViT
embeds are scattered onto the placeholder positions at forward time.

**Uniformity:** all 415,663 rows verified uniform — L=1184 (multiple of bd_size 32),
`pixel_values` (672,1176), 3 images, 16×14 grid → 56 merged tokens/image (168 total;
336 image positions in the doubled 2L sequence). The grain fast path (no padding) is
valid everywhere; `_pad_sample` is a never-hit TODO.

**Answer provenance:** `trajectory` = real GT (5 wp @1 s, indices 3/7/11/15/19 of the 4 Hz
future, ±06.2f format); `future_meta_behavior` derived (longitudinal from waypoint speeds,
lateral from EgoIntent); `critical_objects`/`explanation` are **pseudo labels** (raw WOD-E2E
has no text labels). Upgrade path: teacher-distill via
`fast_ddrive/data/merge_distilled_labels.py` (hybrid fmb policy; validated on 10).

## 2. Artifacts on disk / GCS (truth as of 2026-06-12)

Local, under `/home/kaiwen/data/fast-ddrive/hf/`:

| Dir | Content | Size |
|---|---|---|
| `wod_e2e_sasd_full_packed/` | **full 415,663 rows**, 130 parquet — the sole local parquet source | 178 G |
| `wod_e2e_sasd_full_tfexample_ar/` | same rows, 130 ArrayRecord shards (1:1 with packed, same row order) | 158 G |
| `wod_e2e_sasd_50k/` | 50,331-frame subset, 787 parquet (also private HF `kaiwen2/wod-e2e-fast-ddrive-sasd-50k`) | 22 G |
| `wod_e2e_sasd/` + `wod_e2e_sasd_ar/` | 400-sample dev subset (parquet + AR) | 175 M + 156 M |

Deleted in the 2026-06-12 Tier-A cleanup: `wod_e2e_sasd_full/` (the 6,499 small parquet
files; the packed copy was integrity-gated first). GCS mirror at
`gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd/` holds full packed parquet (177.8 GiB),
full tfexample AR (157.8 GiB), 50k, 400-sample sets, and `maxtext_sasd_params{,_v2}`.
Raw WOD-E2E tfrecords kept locally (train 877 G / val 226 G) — only needed again if image
preprocessing changes (resolution/cameras/temporal); text-label redo does not need raw.
Datasets are **private** (WOD license forbids redistribution).

## 3. Build chain (raw → ArrayRecord)

```
Stage 0  raw WOD-E2E tfrecords (263 shards, 877 GB) — E2EDFrame protos
Stage 1  fast_ddrive/data/convert_wod_e2e.py --with_target          [autovla env]
         → 3 front cams as JPEG + canonical prompt (nav + 7-pt ego history,
           byte-verified vs sample.json) + target JSON (GT traj + derived fmb + pseudo text)
Stage 2  jax_ddrive/eval/prep_train_jax.py                          [ddrive env]
         → HF processor (chat template; min_pixels=784, max_pixels=784*64 → 16×14 grid),
           process_gpt normalization (NULL pads, ±06.2f traj), labels, deep-scaffold
           detection (rbi/turn/scaffold/b2s), weights, Beta(α,β), 3D M-RoPE position_ids,
           MASK-pad to %32 → one npz per frame
Stage 3  jax_ddrive/scripts/convert_full_chunked.py  (chunked resumable driver wrapping
         convert_subset_parallel.process_shard; npz → parquet via prep_to_parquet contract)
         → pack_parquet.py (pure concat, 6499→130 files, content unchanged)
         → full_to_tfexample_ar_driver.py / parquet_file_to_tfexample_ar.py → ArrayRecord
```

## 4. How a train step consumes a record

1. **Read** — `ddrive_jax/data/grain_pipeline.make_sasd_loader`: per-host slice of one
   global shuffled stream; deterministic, resumable (`state()` = `{"grain": {next_index}}`;
   noise re-derivable from `(seed, step, gidx)`).
2. **Online noising** — `diffusion/noise.make_batch`: per block `t ~ Beta(α,β)`,
   mask *value* tokens with p=(1−ε)t+ε (scaffold frozen, `im_end` always masked).
3. **Doubled sequence (2L=2368), two rows per sample** — row 0 `[x_t | x_0]` (labels =
   masked positions only), row 1 `[x̄_t | x_0]` (complementary positions).
4. **Images** — frozen ViT over `pixel_values` → [168, 2048] embeds, doubled
   (`concat([ie,ie])`), scattered at `<|image_pad|>` positions of both halves
   (`stop_gradient`; ViT never trained).
5. **Positions & attention** — position_ids tiled ×2 → contiguous-chunk M-RoPE cos/sin
   (`mrope_section (16,24,24)`, NOT MaxText's interleaved `use_mrope`); hybrid
   block-causal [2L,2L] mask from rbi/turn (`diffusion/masks.py`).
6. **Loss** — `Σ section_weighted_CE(noisy half, labels_final, w) +
   causal_CE(clean half row 0, original_labels)`, normalized by `2 × #response tokens`.

Consumers: the NNX FSDP harness (`ddrive_jax/train/train_tpu.py`) and the MaxText fork
(`objective="sasd"`, `dataset_type="waymo_sasd"`) — same math, bit-exact port
(`maxtext-dlm-fork/src/maxtext/diffusion/sasd.py`).

> **Reality check (verified in code 2026-06-12): nothing reads ArrayRecord yet.** All three
> training paths funnel into `make_sasd_loader`, whose `_RowSource` eagerly decodes
> *parquet* (fine at 400-sample scale only). MaxText also does not checkpoint the grain
> iterator for this dataset_type (resume replays data). Both gaps are scheduled in the v2
> plan (§6).

## 5. Verification status & how to re-verify

**Round-2 semantic verification (2026-06-12): 10/10 samples PASS** — for each sample the
full chain was re-run from the raw tfrecord and compared against the ArrayRecord row:
all 12 array columns bit-exact (incl. `pixel_values` fp16, max|Δ| = 0), 9 structure
self-checks, answer-text round-trip (`decode(labels span) == process_gpt(orig)`), encoded
trajectory == GT@1 s (±0.005 = 2-decimal rounding), 7 history points present, and pixel
reconstruction (inverse patchify + de-normalize) visually identical to the originals with
correct left/center/right camera order.

```bash
# re-run (N samples from train tfrecord shard 0; ~3 min):
N=10 bash jax_ddrive/scripts/verify_ar_round2.sh
# artifacts: /home/kaiwen/data/fast-ddrive/verify_round2/{review/report.md, review/NN_<sid>/}
```

End-to-end semantic evidence beyond round-2: the same prep pipeline feeds the 479-frame
rated-val eval where JAX/PyTorch official metrics are on par (ADE@3s 0.839/0.814,
RFS 7.93/7.91) — a mis-encoded dataset could not produce those numbers.

**Review website** (what the data looks like / how it's built / how it's fed, with the 10
verified examples): `jax_ddrive/visualizations/index.html`. Static, relative paths only.
View from a Mac: `bash jax_ddrive/visualizations/serve.sh` on the desktop, then
`ssh -L 8890:localhost:8890 <desktop>` → http://localhost:8890. Regenerate after a new
verify run: `PYTHONPATH=jax_ddrive <ddrive-python> jax_ddrive/scripts/make_dataset_website.py`.

## 6. Dataset v2 (planned — decisions locked 2026-06-12)

Spec: keep all 13 fields (incl. `pixel_values`, for easy verification) **+ new
`image_embeds`** = frozen-ViT output, fp32-computed → **bf16-stored**, single copy
(loader doubles), built for both 50k and 415k. Motivation: host-side per-step ViT is
~7% MFU on the v6e-1 smoke; precomputed embeds make the data loop pure IO and drop
ViT/transformers deps from the pod. Built-in verification: sample records → recompute
fp32 embeds from stored pixels → cast bf16 → bitwise compare.

Plan order: ① embeds+AR converter (packed parquet → new AR; atomic, resumable; smoke
400 → 50k → full) ② AR reader (grain `ArrayRecordDataSource` + tf.Example parse + embeds
tile) replacing `_RowSource`, with order-equality tests ③ wire `SasdLoader.grain_iterator`
into the MaxText checkpoint + kill/resume test ④ 479-frame val AR ⑤ MaxText fork commit +
PATCHES.md + vendor `ddrive_jax` modules ⑥ `gsutil rsync` (~720 GB) to the trial bucket
⑦ after validation, the old local AR may be deleted (GCS copy exists).

When v2 ships, update §1–§5 here (schema row for `image_embeds`, artifact table, reader
path in §4) and re-run round-2 + the website against the v2 AR.
