# 内部推理文档 — 更新记录（changelog）

> `7for_internal_inference/` 的**滚动更新记录**,最新在最上。**agent 不必读这篇**;要完整上下文读
> [`00_run_inference.md`](00_run_inference.md)（本 track 的主 runbook）+ 设计 SSOT
> [`../2implementation-details/INFERENCE_DEPLOY.md`](../2implementation-details/INFERENCE_DEPLOY.md)。
> 当出现新批次的 `updates-latest-<新日期>.md` 时,把旧的这份移到 `history/`。

---

## 2026-06-18 — Track 2 改成 proto-free（owner 预烤 prep_val）+ 修 §T1.1 Mode R bug

**触发:** 内部 agent 跑 Mode R 的 §T1.3 parity 时 `fp32_vs_reference cosine=0.54 FAIL`。本地复现确认:GCS 上的
`eval_inputs`(20 val+train npz)是 **base-ViT** 烤的(给 Mode D),**Mode R 不能用**;旧 §T1.1 "Mode R 直接拉 GCS
eval_inputs" 是 bug（那批 npz 的 `image_embeds` 与 release ViT 不同源 → 0.54）。

**改动:**
- **`fast_ddrive/eval/evaluate_waymo_metrics.py`**:把 `import tensorflow` + waymo `end_to_end_driving_data_pb2`
  改成**惰性 import**（挪进 `load_waymo_e2e_data()`,仅 tfrecord-GT 分支用）。`--gt <pkl>` 路径现在**纯 numpy**,
  不碰 tf/proto。本地在无 tf/无 proto 的 env 端到端验证通过(479 帧出 ADE/RFS)。`waymo_rfs_utils` 本就纯 numpy。
- **owner 预烤 + 发布 `prep_val_full`**（479 帧,`pixel_values`+text,**与 ViT 无关**）→ `$SRC/eval/prep_val_full/`。
  内部 Track 2 直接拉它跑 `jax_batch_inference`（TPU 上自加载 `FASTDDRIVE_SNAP` 的 ViT）+ metric,**全程 ~/venv(jax)
  +numpy**:零 convert、零 proto、零 tf、零 torch、**不需要 `$PY`**。
- **`00_run_inference.md`**:§0 ③ / Track 2 / §T1.1 / track 表 / canonical 表全部按上面重写;§T1.1 标明 GCS
  `eval_inputs`=base-ViT(只配 Mode D),Mode R 复现 ADE/RFS 走 Track 2。
- **`6for_internal/00_owner_publish.md` §6**:加发布 `prep_val_full` 的步骤。

**Mode R 期望:** Track 2 bf16 ≈ golden **ADE@3s 0.839 / @5s 2.072 / RFS 7.929**（num_samples=479）。

---

## 2026-06-17 — 新建 `7for_internal_inference/`（内部 TPU 推理 / eval 的专属 track）

**为什么:** 推理内容原本散在三处（`INFERENCE_DEPLOY.md` 设计 / `test_training.md` STAGE 2-4 当训练尾巴 /
`6for_internal/03` 的 val parity）。给内部推理一个专属机械 runbook,和 `6for_internal/`（训练）平行——
**共享 STEP 0/1 bootstrap,是 STEP 2（训练）的姊妹 track。**

**`00_run_inference.md`（主 runbook,两个 track）:**
- **Track 1 — B2 自包含部署推理**（egress-safe）:`eval_sasd/driver.py`（168-res,`~/venv`,`PYTHONPATH=$FORK/src`）。
  §T1.0 自包含 sanity（`EVAL_SASD_SELFCONTAINED_PASS`）→ §T1.1 eval-inputs（Mode R 拉 GCS;Mode D 离线 base-ViT 重建）
  → §T1.2 (Mode D) B1 导出 → §T1.3 `run_parity`（`EMBED_PARITY_PASS`,gated fp32_vs_reference cosine≥0.999）
  → §T1.4 `run_eval` fp32+bf16（`SASD_EVAL_PASS` + 逐样本标量 vlog）。Mode D=from-base 导出 / Mode R=发布 ckpt。
- **Track 2 — 官方 ADE/RFS**（从 `6for_internal/03 §2B` **迁来**）:`jax_batch_inference`@200704 →
  `evaluate_waymo_metrics --gt rated_val_gt.pkl` → ADE/RFS。Mode R 复现 **0.839/2.072/7.929**;Mode D 报自己的分。

**钉死的坑（grounding 出来的,都写进 runbook 了）:**
- `PYTHONPATH=$FORK/src`——driver/parity 的 docstring 写的是旧 `jax-dlm-baseline` 路径,**别照抄**。
- 训练 ckpt 导出**绝不加** `verify_against`（逐位 round-trip 仅 BASE ckpt 有效,否则假 FAIL）。
- metric 用 `--gt`（接 pkl 或 tfrecord glob）,**不是** docstring 里过时的 `--gt_tfrecords`/`--gt_dict_pkl`。
- 分辨率:Track1 driver=**168**,Track2 metric=**200704**,别交叉（用错→静默乱码）。
- eval_inputs 的 `image_embeds` 必须与 `--snapshot` 的 ViT 同源（Mode D=base,Mode R=release);parity 门强制。
- egress:driver **禁用** `--show_text`;只出 `validation_log.jsonl` 标量。

**配套改动（同批）:**
- `6for_internal/03_data_and_inference_parity.md` → **瘦身 + 改名** `03_data_processing.md`:移除 §2B（val parity/metric →
  迁到本 track Track 2),只留数据处理（处理 sanity + 格式检查 + 新 split）。
- 前门 `0overview/00_START_HERE.md` 路由 + 导航补上"内部跑推理/eval → 7"和改名后的 03;`README.md` 把 7 列为 Living。
