# quick-refresh.md — distilled-dataset quick tracker

> **Scope.** 本表只管 **teacher-distilled 标签数据集**。项目状态问题（如可训练 in-graph ViT 的 TPU 结果）在 [`0overview/00_START_HERE.md` §3 现状一览](0overview/00_START_HERE.md) + plan [`1plans/06_trainable_vit_plan.md` §9](1plans/06_trainable_vit_plan.md)；ViT 坑 → [`0overview/02_gotchas.md`](0overview/02_gotchas.md)。**别在这里记里程碑。**

One-glance tracker for the **teacher-distilled label datasets**: where they live
(local / GCS / CNS), when built, size, and format/AR status. Belongs to the dataset
docs. Concept + pipeline → [`2implementation-details/LABELING.md`](2implementation-details/LABELING.md);
format schemas → `DATASET.md` (v1 Parquet) / `DATASET_V2.md` (v2 AR + `image_embeds`).

> **Format legend.** `Parquet v1` = 12-array source schema incl. `pixel_values`, **no**
> `image_embeds` → **NOT ArrayRecord, NOT v2**. `v2 AR` = ArrayRecord + precomputed
> frozen-ViT `image_embeds` bf16 (the production loader format). Distilled sets are
> currently **Parquet v1 only, at uniform L=1280** (distilled answers are longer than
> the old L=1184 pseudo set; see LABELING.md §5). Converting to v2 AR (+ embeds) is the
> step needed to match the production training path. Dates local (−0500) unless UTC.

| dataset | rows | size | created | format / AR? | local | GCS | CNS | status |
|---|---|---|---|---|---|---|---|---|
| **distilled-400 (small)** | 400 | 175 MB / 7 shards (+1.7 MB JSON) | 2026-06-12 15:15 | **Parquet v1, L=1280** — pixel_values, **NO image_embeds → not AR / not v2** | `train/distill_400/{parquet_L1280, train_targets_distilled_400.json}` | `gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd/wod_e2e_sasd_distilled_0612-small_L1280/` (uploaded 2026-06-12 21:14 UTC) | `/cns/<cell>/home/<ldap>/fast_ddrive/wod_e2e_sasd_distilled_0612-small_L1280/` *(pending — run `fileutil cp` step 2, then fill cell/ldap)* | ✅ labels done (teacher Route A, hybrid fmb), loss-zero 400/400; trainable via Parquet loader. ⚠️ not yet v2-AR |
| **distilled-50k (middle)** | 50,331 | — | *stopped 2026-06-13* | (not built) | (cleaned) | — | — | ⏸ **stopped — not needed for this milestone** (see `5blockers/0612-blocker-v0.md`); stopped at 3/21 chunks, intermediates deleted; pipeline resumable (`distill_teacher_chunked.py` + `finalize_distill_50k.py`) if revisited |

## Notes / TODO
- **CNS copy (step 2)** is run on a Google-internal host (`fileutil` not on the 5090 desktop): `fileutil cp -R -f gs://…/wod_e2e_sasd_distilled_0612-small_L1280 /cns/<cell>/home/<ldap>/…`. Replace `<cell>`/`<ldap>` and paste the real path back into the CNS column above.
- **v2-AR redo (to reach the production loader):** the distilled Parquet must be run through the last builder stage `scripts/parquet_to_ar_with_embeds.py` → AR + precomputed `image_embeds` (uniform L=1280). `image_embeds` are unchanged by distillation (image-derived), so it's cheap/low-risk; the from-raw `verify_ar_round2` text-compare will MISMATCH for distilled sets (verify embeds/structure only). **Full context + exact command + verification caveat: [`LABELING.md` §7](2implementation-details/LABELING.md). Not done yet — owner will run it.**
- Date label in the GCS/CNS name is the **real build date 0612** (2026-06-12), not 0621.
