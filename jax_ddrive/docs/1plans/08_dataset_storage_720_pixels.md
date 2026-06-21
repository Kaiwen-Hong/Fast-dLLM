# 08 · 数据集存储瓶颈：720 pixels-only 全量 = 2.85TB（待深挖解决）

> **状态**：`DATASET_STORAGE_720: OPEN`（2026-06-21 记录，留待深挖；OOM 已解决后，这是全量训练数据的下一个真问题）。
> **一句话**：可训练 in-graph ViT 需要 **pixels-only 720** 数据；但现 pipeline 把图**预-patchify 成未压缩 fp16** 存盘，使 415k 全量膨胀到 **~2.85TB**（本地 free 仅 ~1TB，GCS 月费 ~$57）。**这不是固有的**——根因是"提前解压成 fp16 落盘"，可通过"存压缩 JPEG + loader 现 patchify"降到 **~394GB**。
> 背景见 [`../6for_internal/cc-delta-compared-with-831baf4.md`](../6for_internal/cc-delta-compared-with-831baf4.md) §4、[`../6for_internal/sanitycheck_dataprocessing.md`](../6for_internal/sanitycheck_dataprocessing.md)。

---

## 1. 问题

要做真正的 **415k 全量 720 trainable-ViT 训练**，需要 720 **pixels-only** AR 数据集（现 GCS 上只有 168/embeds：`wod_e2e_sasd_full_v2_ar`=369G、`50k_v2_ar`=45G；720 只有 toy）。按现 pipeline 重建 415k 会到 **~2.85TB**，本地装不下、GCS 又贵。

## 2. 根因：为什么"解析后"比 raw 还大（反直觉点）

| 阶段 | 量（/样本 → 415k 全量） | 说明 |
|---|---|---|
| 完整 raw WOD-E2E（`/home/kaiwen/data/fast-ddrive/waymo`） | — → **1.1 TB** | 多相机 + 全分辨率 + 元数据，**JPEG/tfrecord 压缩**，含我们不用的部分 |
| ① 抽 3 相机(FRONT/FL/FR) + 降到 200704px + JPEG | **0.95 MB** → **~394 GB** | 比 raw **小**（扔了多余相机/分辨率，仍压缩） |
| ② patchify + 转 **fp16 浮点** `[2880,1176]`（=`pixel_values`，现 pipeline 落盘的东西） | **6.77 MB** → **~2.85 TB** | 比来源 JPEG **大 ~7×**（浮点 patch 几乎不可压缩）→ 盖过①省下的 |

**结论**：2.85TB > 1.1TB **不是信息更多，而是把压缩 JPEG 提前解压成未压缩 fp16 落盘**。`pixel_values` 单样本 6.77MB（`[2880,1176] fp16`），是 prep 在 `prep_train_jax.py:106`（`inputs["pixel_values"].float().astype(np.float16)`）固化的。

实测单样本 720 pixels-only npz = **6.86 MB**（其中 `pixel_values` 6.77MB、`input_ids/labels/position_ids` 仅 KB）。

## 3. 为什么必须 pixels（不能用 embeds）

可训练 in-graph ViT（`sasd_vit_trainable=true`）**每步在 pixels 上现跑 ViT**（参数进 train state、可训）；预烤 embeds 是**冻结 ViT** 的老路，无法训练 ViT。所以 720 训练数据必须带 `pixel_values`，不能只存 embeds。（注：720 embeds `[720,2048]bf16`=2.95MB 反而比 pixels 6.77MB **小**，但 embeds 不能训 ViT——所以问题不在"embeds vs pixels"，在"压缩 vs 未压缩落盘"。）

## 4. 解决方案（三条，待深挖选定）

| 方案 | 415k 总量 | 本地(free~1T)? | GCS/mo | 代码改动 | loader 开销 |
|---|---|---|---|---|---|
| **X. 保持现 pipeline + 流式分块 build 到 GCS** | 2.85 TB | 只留单块、传完即删 | ~$57 | 无（只要 chunked build 脚本） | 低（直接读 fp16） |
| **Y. 存 JPEG bytes + grain loader 现 patchify**（推荐 415k） | **~394 GB** | ✅ | ~$8 | **要改 AR schema + loader**（真活） | 中（每步解码+patchify，可 prefetch 掩盖） |
| **Z. 只用 50k 子集（现 pipeline）** | ~343 GB | ✅ | ~$8 | 无 | 低 | （若 50k 是训练目标则无瓶颈） |

> raw uint8 图（未压缩、不 patchify）≈ 747GB 是中间态，一般不如 Y。

## 5. 推荐：Y（415k 必然选择）+ 实现草图

把"提前 patchify 落盘"改成"读取时 patchify"：

1. **AR schema**：每条记录存 **JPEG bytes（3 相机）** + `input_ids/labels/rbi/turn/scaffold/weight_vec/block_alpha/block_beta/position_ids/image_grid_thw`（这些小，仍预算好），**去掉 `pixel_values`**。
   - 改 `eval/prep_train_jax.py`（不再 `astype(fp16)` 存 pv，改存原图字节/路径）、`ddrive_jax/convert/prep_to_parquet.py`、`scripts/parquet_file_to_tfexample_ar.py`。
2. **grain map 现 patchify**：在 `input_pipeline/sasd_data/grain_pipeline.py` 的 map 里（noise 之后/之前）对 JPEG bytes 跑 **Qwen2.5-VL image processor**（min/max_pixels=200704）得到 `[2880,1176]` 喂给 ViT。
   - 关键：processor 是 host CPU、numpy，可在 grain `num_threads`/prefetch 里并行掩盖；确定性要保证（同一图 → 同一 patch）。
3. **校验**：`scripts/check_dataset_format.py` 加一条"JPEG-bytes schema"分支；对几条样本断言 loader 现 patchify 的 `pixel_values` 与现 pipeline 预烤的 **bit/数值一致**（parity gate）。
4. **shape**：`image_grid_thw` 必须仍恒定 `[1,32,30]×3`（in-graph ViT 编译期常量假设，见 `02_gotchas.md` ViT 段）。

## 6. 开放问题（深挖时先答）
1. **训练目标到底是 50k 还是 415k？** 50k → 用 Z，**今天就能建（343G 装得下）**、零代码、无瓶颈；只有 415k 才需要 Y。（原版 Fast-dDrive 常用 50k 蒸馏集。）
2. Y 的 **loader CPU 预算**：v5e host 每步 patchify 3 张 200704px 图是否拖慢 step？需 micro-bench（prefetch buffer 是否够）。
3. processor 在 grain map 里的**确定性 + 可序列化**（grain worker 进程）。
4. GCS 成本/生命周期：训练期 2.85TB(X) vs 394GB(Y) 的真实月费 + 训完是否删。

## 7. 现状锚点（开建前确认）
- 现 free：`/home/kaiwen/data` 1021G（df，2026-06-21）。raw WOD-E2E 源 1.1T 在盘。
- 现 GCS 168 数据未删、仍有效（一个 checker 两套都认）。
- 720 数据 pipeline（pixels-only）已验证（toy720 `CHECK_PASS`）：`convert_wod_e2e → prep_train_jax(720,label+2) → prep_to_parquet → parquet_file_to_tfexample_ar → check_dataset_format --expect pixels`。
