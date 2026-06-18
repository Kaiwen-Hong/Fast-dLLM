# 内部推理文档 — 更新记录（changelog）

> `7for_internal_inference/` 的**滚动更新记录**,最新在最上。**agent 不必读这篇**;要完整上下文读
> [`00_run_inference.md`](00_run_inference.md)（本 track 的主 runbook）+ 设计 SSOT
> [`../2implementation-details/INFERENCE_DEPLOY.md`](../2implementation-details/INFERENCE_DEPLOY.md)。
> 当出现新批次的 `updates-latest-<新日期>.md` 时,把旧的这份移到 `history/`。

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
