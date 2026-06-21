# CC-Delta · 当前形态 vs commit `831baf4` 的差异（给内部 TPU 团队）

> **用途**：内部团队在 commit **`831baf4afebef92982779b68545a0461d6404344`**（短 `831baf4`）上把训练跑通了。本文档总结**从 `831baf4` 到当前形态（HEAD `d627036` + 本会话未提交改动）**改了什么——代码 + 数据集——便于你把那套 setup 升级到现在。
> **living doc**：重新打包 code+dataset 时再更新本文（见末尾「重新打包时要复核的点」）。
>
> ⚠️ 文件名说明：用户原话写的是 `813baf4`，但该对象不存在；真实 commit 是 `831baf4`（`git cat-file -t 831baf4 == commit`，`813baf4` 无效）。本文按 **831baf4** 撰写。

---

## 0. 一句话

`831baf4` 是 **168-token 分辨率 + 朴素 CE + 冻结/预烤-embeds ViT** 的可跑训练。当前形态是 **720-token 分辨率 + vocab-tiled CE + in-graph 可训 ViT**，并把这套在**真 v5e-16 上的 OOM 解决了**（编译 + step0 + 存档都过，`TRAINABLE_EXIT=0`）。中间 9 个 commit + 本会话一组未提交修复。

## 1. TL;DR 差异表

| 维度 | `831baf4`（基线） | 当前形态 | 出处 |
|---|---|---|---|
| **图像分辨率** | 168 tokens（784/50176 px；`sasd_seq_len=1184`、`sasd_num_image_tokens=336`、`max_target_length=2376`） | **720 tokens**（200704 px；`1856`/`1440`/`3712`） | `86857b4` + 本会话；`sasd_waymo.yml`、`prep_train_jax.py` |
| **数据集形态** | pixels + **预烤 image_embeds**`[168,2048]bf16`（冻结 ViT 路径） | **pixels-only AR**（无 embeds；in-graph ViT 每步吃 pixels） | `prep_train_jax.py`、`check_dataset_format.py`、[`sanitycheck_dataprocessing.md`](sanitycheck_dataprocessing.md) |
| **ViT** | 冻结/预烤（`sasd_vit_ingraph.py` **不存在**） | **in-graph TRAINABLE**（`nnx.bridge.ToLinen` 包 ViT body，每步可训/可 shard/可 ckpt） | `3a375cb`；新增 `sasd_vit_ingraph.py`(+119)、`vision_qwen25vl.py`(+81)、`decoders.py`(+27) |
| **CE 实现** | 朴素 `jax.nn.log_softmax(logits.fp32)`（materialize 整 `[N,V]`） | **vocab-tiled online-logsumexp**（`_CE_VOCAB_TILES=8`，峰值 `[N,V_tile]`，无 lax.map/checkpoint） | 本会话；`sasd.py:_ce_per_token` |
| **response label** | 到 `<\|im_end\|>` 为止 | 含 `<\|im_end\|>` **和后随 `\n`**（对齐原版 `response_end+=2`） | 本会话；`prep_train_jax.py:117` |
| **v5e 装得下?** | （168 装得下） | **720 装得下**（vocab-tiled CE + bf16 logits；真 v5e 验证通过） | 本会话；见 [`8doc_updates/2026-06-21_sasd-tpu-oom-fix.md`](../8doc_updates/2026-06-21_sasd-tpu-oom-fix.md) |
| optimizer/lr | adafactor / 1e-4（smoke 默认） | 同 smoke；另有 canonical config（AdamW/1e-5，见下） | `sasd_waymo.yml` + 新增 `sasd_waymo_canonical.yml` |

## 2. 9 个 commit（`831baf4..HEAD`，时序）

```
db09f14 delete unnecessary files, update docs on how to pack up codebase
1817d23 update inference code
d299795 include data processing internal docs
6f5eeda make the doc more self-contained
d1fb763 fix bugs in inference
3a375cb implement training use unfrozen vision encoder, validate on multi tpu, also update doc
9d32bf8 update doc
86857b4 fix problems in dataset, previously token number is 168, now corrected to 720, also update doc
d627036 update doc   (= 当前 HEAD)
```
**外加本会话未提交改动**（工作区，尚未 commit）——见 §4。

## 3. 代码改动（按类别；`git diff --stat 831baf4 -- '*.py' '*.yml'` = 30 文件 +691/−2273）

### 3.1 分辨率 168→720（`86857b4` + 本会话）
- `configs/sasd_waymo.yml`：`sasd_seq_len 1184→1184`(smoke 仍 168) ——注意：**committed smoke 配置仍是 168**；720 走 launch override 或新的 canonical 配置。`02_gotchas.md` 规范数字：L=1856 / 2L=3712 / 2N=1440 @720。
- `eval/prep_train_jax.py`：默认 `min/max_pixels=200704`（原 784/50176）；图像 token 720。
- `utils/maxtext_utils.py`(+16)：720 shape 推导。

### 3.2 in-graph 可训 ViT（`3a375cb`）
- **新增 `diffusion/sasd_vit_ingraph.py`**(+119)：把 Qwen2.5-VL ViT 拆成 `body`(纯 jax 可 jit+autodiff) + `precompute_structural`(host-only 几何，编译期常量)；`nnx.bridge.ToLinen` 包成 Linen 子模块，进 MaxText train state。
- `models/vision_qwen25vl.py`(+81)：body/structural 拆分、window 重排、2D-RoPE、seg mask。
- `layers/decoders.py`(+27)：`sasd_vit_trainable=true` 时在图里跑 ViT，把 embeds scatter 到 IMAGE_TOK 位（双行）。
- `utils/sharding.py`(+6)：ViT param 的 logical-axis。
- 数据侧：`input_pipeline/waymo_sasd_data_processing.py`(+20) 携带 `sasd_pixel_values`（不再只喂预烤 embeds）。

### 3.3 真 v5e 720 OOM 修复（**本会话，未提交**）
> 详细推导见 [`8doc_updates/2026-06-21_sasd-tpu-oom-fix.md`](../8doc_updates/2026-06-21_sasd-tpu-oom-fix.md)。
- `diffusion/sasd.py:_ce_per_token` → **vocab-tiled online-logsumexp**（`_CE_VOCAB_TILES=8`）。背景：720 的 doubled `[2L=3712]` 让 loss 侧 fp32 全词表 CE 撑爆 v5e 单芯 15.75G；先试 chunked-CE(`lax.map`+checkpoint)在 TPU 上**反而 87G**（病态），改 fused logsumexp 回到 17.11G（仍超 1.36G），最终 **vocab-tiled（分 V 维，峰值 `[N,V_tile]`）+ bf16 logits** 装下。
- `trainers/pre_train/train.py`(+9)：SASD 时 `if not is_sasd: intermediate_outputs["logits"]=logits`（避免 grad-accum 下堆全量 logits）。
- `configs/types.py`(+24)：`sasd_vit_remat` 默认 `False→True`；修正 `sasd_num_image_tokens` 契约 docstring（是已 double 的 2N）。
- 新增 `configs/sasd_waymo_canonical.yml`：720 shapes + trainable ViT + **AdamW（`adam_b2=0.999`/`weight_decay=0`/`mu_dtype=float32` 对齐 HF）** + lr 1e-5 + constant_with_warmup + grad_accum 4。**这是大规模训练用的；`sasd_waymo.yml` 是单卡 smoke。**
- **bf16 logits**：真 v5e 跑时 launch 加 `logits_dot_in_fp32=false cast_logits_to_fp32=false`（让 `[2B,2L,V]` logits 为 bf16）。⚠️ 尚**未**写进 canonical 配置（见 §6 待办）。

### 3.4 数据 prep / 校验（本会话 + 之前）
- `eval/prep_train_jax.py`(+30)：720 默认 + **response label `+2`**（含 `\n`，对齐原版 collator）。
- `scripts/check_dataset_format.py`(+121)：改成**按 `image_grid_thw` 推断分辨率** + **image_embeds 可选**（一个 checker 同时认 720/pixels-only 与 legacy 168/embeds）；尾部检查兼容新旧 label 约定。
- 新增 `scripts/validate_prep_consistency.py`(+85)、`scripts/tpu_vit_body_smoke.py`(+114)。

### 3.5 多 host checkpoint（之前）
- `common/checkpointing.py`(+16)：多 host 本地盘的 process-dir/primary_host 修复（gated 到 `process_count>1 and not gs://`；真跑用 same-region GCS 桶）。

### 3.6 推理修复（`d1fb763`、`1817d23`）
- `eval/evaluate_waymo_metrics.py`(+19)、`diffusion/load_fast_ddrive_maxtext.py`(+38) 等。

### 3.7 清理（`db09f14`）—— 占了 −2273 删除的绝大部分
- 删掉 MaxText 上游无关物：`local_datasets/*`、`logits_generation/*`、`scripts/temp/parity_*`、golden-logits 导出脚本等。**不影响 SASD 训练/推理**。

## 4. 数据集变化（用户特别要求，重点）

| | `831baf4` | 当前 |
|---|---|---|
| 分辨率 | 784 / 50176 px → **168** image tokens | 200704 / 200704 px → **720** image tokens |
| AR 记录内容 | pixels + **`image_embeds [168,2048] bf16`**（预烤，冻结 ViT 用） | **pixels-only**（`pixel_values`；**无 embeds**——in-graph 可训 ViT 每步现算） |
| per-sample L / 2L / 2N | 1184 / 2368 / 336 | 1856 / 3712 / 1440 |
| response label 跨度 | rs..`<\|im_end\|>` | rs..`<\|im_end\|>`,`\n`（多 1 个 token） |
| GCS 上现有数据 | `wod_e2e_sasd*_v2_ar`（168/embeds，**未删**，仍有效） | `wod_e2e_sasd_toy720_v2_ar`（720 toy，已用于本次验证）；**415k 全量 720 重建 DEFERRED**（磁盘 ~2.8TB vs 现 free ~1TB） |

> 一个 checker（`check_dataset_format.py`）现在**两套都认**（分辨率推断 + embeds 可选），所以现有 168/embeds 数据没被破坏。pipeline：`convert_wod_e2e.py`→`prep_train_jax.py`(720,+2)→`prep_to_parquet.py`→`parquet_file_to_tfexample_ar.py`→`check_dataset_format.py --expect pixels`。详见 [`sanitycheck_dataprocessing.md`](sanitycheck_dataprocessing.md)。

## 5. 当前状态（截至 2026-06-21）

- ✅ **720 trainable step 在真 v5e-16 装得下**：run `vit720conf`（bundle `fastddrive-20260621_065405`=vocab-tiled CE + launch bf16 logits）**编译通过、step 0 执行 + 存档成功（GCS `checkpoints/0/commit_success.txt`）、`TRAINABLE_EXIT=0`（无 NaN）**。前三次（17.11G / 87.25G / 17.11G）都在编译就 OOM，这次过了。
- ✅ **loss 数学不变**：GPU lossdecrease 用 vocab-tiled CE = `0.981→0.636` PASS（与 fused/chunked 实质一致）。
- ⏳ **未捕获**：本次 v5e run 的 10 步 loss 轨迹数字（pod 删早了一步，SSH-B 没读到；run 本身成功）。若要这组数字需再跑一个 ~$5 cycle，或视 GPU PASS + 干净退出为足够证据。

## 6. 重新打包 / 升级时要复核的点（living）

1. **本会话改动尚未 commit**（工作区 dirty）——打包前先 commit（见下一步）。
2. **bf16 logits 还没进 canonical 配置**——目前靠 launch override；若大规模训练用 canonical，需把 `logits_dot_in_fp32=false cast_logits_to_fp32=false` 写进配置（注意：parity 测试仍需 fp32 logits → config 拆分 parity vs training）。
3. **415k 全量 720 数据集**未建（磁盘约束）；重建用修好的 `prep_train_jax.py`（720 + label `+2`）。
4. GCS 上 168/embeds 旧数据**暂留**；等 720 全量建好 + 验证后再单独决定是否删。
