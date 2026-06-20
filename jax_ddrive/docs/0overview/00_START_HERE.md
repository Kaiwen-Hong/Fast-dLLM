# 00 · START HERE —— Fast-dDrive JAX 移植的总入口

> 你（人或 coding agent）读项目的**第一个文件**。读完这一篇就知道：系统是什么、按身份该读哪、代码按什么顺序读、现状在哪查、哪些文件别读。
> 细节一律**链出**，不在这里复述（唯一例外：规范数字的家在 [`02_gotchas.md`](02_gotchas.md#规范数字框-canonical-numbers)）。

---

## 1. 这是什么（一段话）

**Fast-dDrive** 是 NVIDIA 把 **Qwen2.5-VL-3B** 改成的**块扩散 VLA**（vision-language-action）自动驾驶模型（HF: `Efficient-Large-Model/Fast-dDrive`，snapshot `0fda81009f4efa58a2debbb48c0c09818e45341f`），跑 Waymo WOD-E2E open-loop。它对一个**全 MASK 的深层 JSON 脚手架**去噪，输出 4 个因果有序段 `critical_objects → explanation → future_meta_behavior(fmb) → trajectory`（5 个 1s 路点）。算法叫 **SASD**（semi-autoregressive sub-block diffusion）：块内双向、块间因果，`bd_size=32`。

本项目把它 **PyTorch → JAX/Flax-NNX（算法真值参考）→ MaxText fork（生产/TPU 路径）**，目标是在 **Google 内部 TPU** 上 train + infer。三套实现由 **parity gate** 保持数值对齐。

代码库三方关系、子系统地图、阅读顺序 → [`01_codebase_map.md`](01_codebase_map.md)。

---

## 2. 受众路由表（你是谁 → 读哪）

| 你想做的事 | 先读 | 再读 |
|---|---|---|
| **读懂 / 修改代码** | [`01_codebase_map.md`](01_codebase_map.md)（地图+阅读顺序）+ [`02_gotchas.md`](02_gotchas.md)（易误读组件） | 算法真值 [`../2implementation-details/01_pytorch_reference_algorithm.md`](../2implementation-details/01_pytorch_reference_algorithm.md) |
| **部署到 TPU** | [`../../to-host-chn.md`](../../to-host-chn.md)（交接文档，含验收标准） | [`../2implementation-details/INFERENCE_DEPLOY.md`](../2implementation-details/INFERENCE_DEPLOY.md)（B1 导出 / B2 采样器） |
| **内部 Google TPU：训练 + 数据** | owner 先 [`../6for_internal/00_owner_publish.md`](../6for_internal/00_owner_publish.md)（STEP 0 发布到 GCS）→ 内部 [`../6for_internal/transfer-codebase.md`](../6for_internal/transfer-codebase.md)（STEP 1）→ [`../6for_internal/test_training.md`](../6for_internal/test_training.md)（STEP 2 训练）/ [`../6for_internal/03_data_processing.md`](../6for_internal/03_data_processing.md)（STEP 3 数据处理） | blocker/红线 [`../5blockers/0612-blocker-v0.md`](../5blockers/0612-blocker-v0.md) |
| **内部 Google TPU 跑推理 / eval** | （STEP 0/1 同上 →）[`../7for_internal_inference/00_run_inference.md`](../7for_internal_inference/00_run_inference.md)（STEP I：B2 自包含部署 + 官方 ADE/RFS） | 设计 [`../2implementation-details/INFERENCE_DEPLOY.md`](../2implementation-details/INFERENCE_DEPLOY.md) |
| **理解数据格式** | [`../2implementation-details/DATASET_V2.md`](../2implementation-details/DATASET_V2.md)（生产 v2 AR） | v1 源格式 [`../2implementation-details/DATASET.md`](../2implementation-details/DATASET.md)、标签 [`../2implementation-details/LABELING.md`](../2implementation-details/LABELING.md) |
| **跑评测 / 复现指标** | [`../2implementation-details/EVAL_PIPELINE.md`](../2implementation-details/EVAL_PIPELINE.md)（两套栈一个 metric） | — |
| **查某个数字到底是多少** | [`02_gotchas.md` §规范数字框](02_gotchas.md#规范数字框-canonical-numbers) | — |

---

## 3. 现状一览（截至 2026-06-20，**只链出证据，不复制数字**）

| 阶段 | 状态 | 证据出处 |
|---|---|---|
| 算法 port（loss/mask/M-RoPE/ViT/权重转换） | ✅ parity-gated | [`../3summary/REPORT.md`](../3summary/REPORT.md)、`run_all_verification.sh`（**10 个 gate**） |
| 数据 v2（AR + 预算 bf16 ViT embeds，415,663 帧） | ✅ byte-audit + TPU 验证 | [`../2implementation-details/DATASET_V2.md`](../2implementation-details/DATASET_V2.md) §6 |
| 训练（NNX 单机 + MaxText 真实 v6e-1 单芯） | ✅ 单芯真权重 loss 下降 | [`../4collect/OVERNIGHT_TPU_PROGRESS.md`](../4collect/OVERNIGHT_TPU_PROGRESS.md)（TPU SSOT） |
| 可训练 in-graph ViT（`sasd_vit_trainable=true`：每步吃 pixels，ViT 进 train state） | ✅ GPU 3-step PASS + 真实 v5e-16 多节点 3-step loss 下降、ckpt 落 GCS、EXIT 0；✅ **720 保真度修正** GPU toy PASS；✅ **720-OOM 已定位+修复**：瓶颈是 loss 里的 fp32 全词表 `log_softmax`（`sasd.py:_ce_per_token`），**不是** ViT / sharding（ViT-remat = 零内存变化）；改为 chunked-CE（按行分块 + remat，数值 bit-identical）后，720 train step 在 capped-GPU v5e proxy 下放得进、step 内无 OOM（残留 OOM 是 ckpt-save 路径）；⚠️ 真实 v5e 确认仍 pending（~$5）；168 已在 v5e PASS | [`../1plans/06_trainable_vit_plan.md`](../1plans/06_trainable_vit_plan.md)（滚动日志）、[`../1plans/07_fidelity_fixes_2026-06-20.md`](../1plans/07_fidelity_fixes_2026-06-20.md)（720 保真度 + EXP_BUDGET=192）、[`../4collect/08_trainable_vit_progress.md`](../4collect/08_trainable_vit_progress.md)、ViT 坑 [`02_gotchas.md`](02_gotchas.md) |
| 导出 B1（MaxText→HF bf16） | ✅ 本地 round-trip | [`../2implementation-details/INFERENCE_DEPLOY.md`](../2implementation-details/INFERENCE_DEPLOY.md) |
| 推理 B2（自包含 section-diffusion 采样器） | ✅ 本地 GPU；🧪 TPU 数值彩排未做 | 同上 |
| 评测（WOD-E2E ADE/RFS + 嵌入 parity） | ✅ 全 479 帧 on-par | [`../3summary/REPORT.md`](../3summary/REPORT.md) |
| from-base overfit 全在内部 TPU 完成 | ⏳ 当前活跃里程碑 | [`../6for_internal/test_training.md`](../6for_internal/test_training.md) |
| ≥8 芯多节点 TPU 跑批 | ✅ v5e-16 已端到端跑过 3 步真训（含可训练 ViT，ckpt 落 GCS）；⚠️ 剩 ViT 参数 logical-axis sharding（当前 REPLICATED）+ ckpt 侧 ViT snapshot-init 未做 | [`../1plans/06_trainable_vit_plan.md`](../1plans/06_trainable_vit_plan.md)、[`../3summary/FEATURES.md`](../3summary/FEATURES.md) |

> 实时 / 更细的状态以 [`../../to-host-chn.md`](../../to-host-chn.md) 顶部状态横幅与 [`../4collect/OVERNIGHT_TPU_PROGRESS.md`](../4collect/OVERNIGHT_TPU_PROGRESS.md) 为准。

---

## 4. 文档导航地图（哪个 doc 管什么）

> 维护规则（one-fact-one-home、里程碑落地流程、frozen 不改、big-artifacts 路径等）在 [`../README.md`](../README.md)。这里只管"去哪读"。

### Living —— 必须随现实更新（真相在这里）
| doc | 范围 |
|---|---|
| [`../../to-host-chn.md`](../../to-host-chn.md) | **交接文档**（中文）：有什么、验证了什么（数字）、怎么复现/部署、验收标准；顶部横幅带最新日期 |
| [`../2implementation-details/DATASET_V2.md`](../2implementation-details/DATASET_V2.md) | 生产数据格式：v2 ArrayRecord（12 array + 预算 `image_embeds`）、builder、verify 链、reader API、built-set 清单 |
| [`quick-refresh.md`](../quick-refresh.md) | distilled 数据集快速追踪表（位置/日期/大小/格式） |
| [`../5blockers/0612-blocker-v0.md`](../5blockers/0612-blocker-v0.md) | 内部 TPU 部署 blocker（B1–B5）+ 备选路线 + overfit 成功标准 + 验证日志规格 + 待拍板清单 |
| [`../5blockers/0613-diffusiongemma-insights-v0.md`](../5blockers/0613-diffusiongemma-insights-v0.md) | DiffusionGemma 代码精读的可迁移洞见（对抗校验过）+ 红线清单 |
| [`../6for_internal/00_owner_publish.md`](../6for_internal/00_owner_publish.md) | **STEP 0（owner）**：把代码 + 数据 + manifest 发布到 GCS（`upload_code_to_gcs.sh` + `data_manifest.py`）；内部 STEP 1 的前提 |
| [`../6for_internal/transfer-codebase.md`](../6for_internal/transfer-codebase.md) | **STEP 1**：内部 bootstrap（拉代码包进 google3、写 env） |
| [`../6for_internal/test_training.md`](../6for_internal/test_training.md) | **STEP 2**：内部 TPU 从-base 训练 → B1 导出 → B2 推理 → 嵌入彩排 → 验证日志 |
| [`../6for_internal/updates-latest-0616.md`](../6for_internal/updates-latest-0616.md) | 内部文档 changelog（滚动,最新在最上;agent 不必读；`6for_internal/history/` 是归档,含旧 0614） |
| [`../6for_internal/03_data_processing.md`](../6for_internal/03_data_processing.md) | **STEP 3**：内部数据处理 sanity + 格式检查 + 新 split 处理（数据 track,独立于训练/推理） |
| [`../7for_internal_inference/00_run_inference.md`](../7for_internal_inference/00_run_inference.md) | **STEP I**：内部 TPU 推理 / eval —— Track1 B2 自包含部署（`eval_sasd`,168-res,标量 vlog）+ Track2 官方 ADE/RFS；共享 STEP 0/1,是 STEP 2 的姊妹 track |
| [`../7for_internal_inference/updates-latest-0617.md`](../7for_internal_inference/updates-latest-0617.md) | 7 的 changelog（滚动；agent 不必读） |

MaxText fork 侧 file-by-file diff / vendor 规则 → fork 内 `maxtext-dlm-fork/PATCHES.md`。

### Stable references —— 组件变了才更新
| doc | 范围 |
|---|---|
| **[`00_START_HERE.md`](00_START_HERE.md) · [`01_codebase_map.md`](01_codebase_map.md) · [`02_gotchas.md`](02_gotchas.md)** | **本 0overview 三档：前门 / 代码地图 / 易误读组件 + 规范数字** |
| [`../2implementation-details/ARCHITECTURE.md`](../2implementation-details/ARCHITECTURE.md) | JAX/Flax-NNX 模型 port 结构 |
| [`../2implementation-details/01_pytorch_reference_algorithm.md`](../2implementation-details/01_pytorch_reference_algorithm.md) | 被 port 的 PyTorch SASD 算法 |
| [`../2implementation-details/EVAL_PIPELINE.md`](../2implementation-details/EVAL_PIPELINE.md) | WOD-E2E 评测（两套栈一个 metric） |
| [`../2implementation-details/INFERENCE_DEPLOY.md`](../2implementation-details/INFERENCE_DEPLOY.md) | 内部 TPU 推理：B1 导出 + B2 采样器 + bf16 + 嵌入 parity + 部署 runbook |
| [`../2implementation-details/DATASET.md`](../2implementation-details/DATASET.md) | v1 Parquet 源格式（训练已由 v2 取代，仍是 bit-exact 源） |
| [`../2implementation-details/LABELING.md`](../2implementation-details/LABELING.md) | 标签来源（pseudo vs real）、teacher-distill pipeline |
| [`../2implementation-details/AUDIT.md`](../2implementation-details/AUDIT.md) | 对抗式代码审计发现 |
| [`../3summary/REPORT.md`](../3summary/REPORT.md) · [`../3summary/FEATURES.md`](../3summary/FEATURES.md) | 阶段完成总结 / 能力矩阵（**状态+证据**型，非代码导读） |

### Frozen —— 历史，**永不编辑**（onboarding 时**不要读**，只在考古时看）
`1plans/{00_PLAN,02_tpu_plan,03_scaleup_tpu_spec,04_tpu_smallscale_validation,05_review_and_fixes_2026-06-14,07_fidelity_fixes_2026-06-20}.md`（+ `06_trainable_vit_plan.md` 是该里程碑的滚动 plan/日志，§9 滚动更新） · `4collect/{HANDOFF,OVERNIGHT_PROGRESS,OVERNIGHT_TPU_PROGRESS(,-chn),05_maxtext_port_progress,06_dataset_v2_progress,07_from_base_b1_b2_progress}.md`

### 文档维护记录（append-only，非项目里程碑）
[`8doc_updates/`](../8doc_updates/README.md) —— 对 docs 体系本身的大修/审计记录 + 可复核证据产物（如本 0overview 的建立过程）。

---

## 5. context 有限就按这个顺序读

1. **本文件**（系统是什么 + 去哪读）。
2. [`01_codebase_map.md`](01_codebase_map.md) §3 黄金阅读顺序（代码的 7 个起手文件）。
3. [`02_gotchas.md`](02_gotchas.md)（易误读组件 + 规范数字框 + 10 个 gate）。
4. 按任务深挖：算法 → `01_pytorch_reference_algorithm.md`；数据 → `DATASET_V2.md`；部署 → `to-host-chn.md` / `INFERENCE_DEPLOY.md`。

**绝不为了 onboarding 整读** `1plans/` 与 `4collect/`（frozen 历史，按需考古）。
