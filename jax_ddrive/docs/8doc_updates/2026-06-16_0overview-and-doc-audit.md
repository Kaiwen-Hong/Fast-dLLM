# 2026-06-16 · 建立 0overview 前门 + 全文档 doc↔code 审计/对账

> 一次性的文档体系大修记录。**触发**：文档已有 ~30 篇（~3600 行），细节足但 context 有限的 coding agent 无法整读；需要一个"如何读代码 + subtle 组件"的详细入口，并顺手保证现有文档与代码一致。
> 证据产物见同目录 `evidence_*` 与 `design_spec.md`。

---

## 1. 决策（拍板记录）

| 决策 | 选择 | 理由 |
|---|---|---|
| 受众边界 | **完整前门 (C)**：`0overview/` 成唯一入口，吸收 README 的导航地图；README 收缩为维护规则 | 主受众是"读懂/改代码的 coding agent"，区别于部署(to-host)、内部 TPU(6for_internal) |
| 分层粒度 | **3 档**：START_HERE / codebase_map / gotchas | 让 context 有限的 agent 只读需要的那层 |
| D1 语言 | 中文行文 + 英文术语/关键词 | 用户中文母语便于把握结构、英文流利保留术语最准 |
| D2 数字漂移 | **建规范数字框 + 回填修正源头**；合理变体记语境+出处 | one-fact-one-home；区分"错值"与"语境不同的合法值" |
| D3 doc↔code 审计 | **本次就做完整审计** | 系统性核对每条文档声明 vs 真实代码 |
| frozen 文档策略 | **不重写历史**，错处加内联订正标注；living 直接改 | 遵守 README 维护规则"frozen append-only" |

---

## 2. 交付物（5）

### 2.1 新建 `0overview/`（414 行）
- `00_START_HERE.md`（91）：系统一段话 · 受众路由表 · 现状一览（仅链出）· 文档导航地图 · 最小阅读顺序。
- `01_codebase_map.md`（127）：三代码库关系 · 黄金阅读顺序（7 起手文件）· 子系统地图 · 深挖索引。
- `02_gotchas.md`（196）：易误读组件目录（~40 条带 file:line）· 10 个 gate 可执行规格 · **规范数字框（数字唯一 SSOT）**。

全部由**实读代码的 ground truth** 撰写，非二手转述。

### 2.2 README 重构 + 指针
- `README.md` → 纯维护规则（导航表移入 START_HERE）；新增 rule #7（0overview 以代码为源 + 规范数字唯一家）；顶部指向 START_HERE。
- `to-host-chn.md` 顶部加代码入口指针。

### 2.3 全文档审计 + 数字对账 + 回填（见 §4）

### 2.4 5 处 stale 代码注释订正（`3.75B → 3.086B`）
`ddrive_jax/lora.py:5`、`ddrive_jax/train/launch_tpu.sh:11`、`scripts/gpu_real_smoke.py:2,30`、`scripts/check_real_fsdp_shard.py:1`。漂移根因（人写注释的过计数），动态打印 `save_fast_ddrive_params_ckpt.py:73` 一直是对的（`n_el/1e9`）。

---

## 3. 执行 pipeline（自主，多 agent 工作流）

| Phase | 做法 | 规模 |
|---|---|---|
| 0 文档精读 | 9 reader 读全部 30 文档 → 认知图谱 | 10 agent / 39 万 tok |
| 1 代码 ground truth | 8 reader 实读 `ddrive_jax/` + fork + configs + gates → 子系统地图 / gotchas / 带出处数字 | 9 agent / 58 万 tok |
| 2 doc↔code 审计 | 10 auditor 逐条核对 30 文档 vs ground truth + 抽查代码 → findings | 11 agent / 98 万 tok |
| 3 数字对账 + 校验 | 规范数字表 + 承重修正直接 grep 代码复核 | 直接核验 |
| 4 写前门 | 亲自撰写 3 档 + README 重构 + 指针 | 手写 |
| 5 回填 | 23 writer 逐文档，每条 apply 前再对代码复核 → living 改 / frozen 标注 | 24 agent / 110 万 tok |

**总成本 ≈ 305 万 subagent token。** 每步对抗式校验（用户不在，错的"修正"比原错更糟）。

---

## 4. 审计结果

- 代码 ground truth 抽出 **17 条 drift 疑点**；审计 30 文档得 **108 条 findings**（76 ok / 8 wrong / 13 stale / 11 misleading，去重后 **32 个真实缺陷**）。
- **回填：56 处应用 / 1 处跳过**（`ARCHITECTURE.md` 的 `rope.py` rotate-half——source docstring 自称如此，可辩护，正确跳过）。
- frozen 缺陷 11 个 → 内联订正标注；living 缺陷 21 个 → 直接改。

### 关键漂移（已修）
| 量 | 错值 | 规范值 | 出处 |
|---|---|---|---|
| 参数量 | 3.75B（9 处 + 5 处代码注释） | **3.086B**（434 text leaves；824=434+390） | `save_fast_ddrive_params_ckpt.py:72-73` |
| gate 数 | 9 | **10**（补 `cpu_eval_ports`） | `run_all_verification.sh:51` |
| array 字段 | 13 | **12** | `prep_to_parquet.py:26-32` |
| `vision_mask` | "训练时门控 vision" | **训练从不读**（carried-but-unused） | `grain_pipeline` 不读 |
| `checkpoint.py` | "multi-host safe" | 单机 StandardCheckpointer；多机走 `checkpoint_mgr.py`/MaxText | `checkpoint.py:15-29` |
| `sharding.py` | "已 map 2D (fsdp,tp)" | 仅 pure-FSDP 最大轴；'tp' 是 Option-2 目标 | `sharding.py:22-28` |

### 语境相关数字（补语境，非误改）
- image tokens：**168**(单份,3 cams×56,train-res) / **336**(doubled config) / 672(patch) / 720(paper-eval 200704 诊断)。
- 序列长 L：**1184**(pseudo 生产) / **1280**(distilled 覆盖,`max_target_length=2576`) / 1120(一次性 parity 样本)。
- 图像分辨率：训练/部署 **784/50176**(→168) vs paper-eval **200704**。
- mrope_section：真值 **(16,24,24)** vs proxy **(4,6,6)**。
- eval：全 **479 帧**(规范) vs 52 帧(早期 partial)。

完整 32 缺陷 + 56 修正逐条见 `evidence_doc_audit_report.md` / `evidence_fix_summary.md`；全 108 raw findings 见 `evidence_doc_audit_findings.json`。

---

## 5. 改动文件清单

- **新增**：`0overview/{00_START_HERE,01_codebase_map,02_gotchas}.md`、`8doc_updates/*`（本记录 + 证据）。
- **重写**：`README.md`。
- **living 文档修正（直接改）**：`to-host-chn.md` + `2implementation-details/{ARCHITECTURE,AUDIT,DATASET,DATASET_V2,EVAL_PIPELINE,INFERENCE_DEPLOY,LABELING}.md` + `3summary/FEATURES.md` + `5blockers/{0612,0613}*.md` + `6for_internal/{test_training,updates-latest-0614}.md` + `quick-refresh.md`。
- **frozen 文档标注（不重写）**：`1plans/{00,02,03,04}*.md` + `4collect/{05,HANDOFF,OVERNIGHT_PROGRESS,OVERNIGHT_TPU_PROGRESS(,-chn)}*.md`。
- **代码注释订正**：`lora.py`、`launch_tpu.sh`、`gpu_real_smoke.py`、`check_real_fsdp_shard.py`（仅注释/docstring/log）。

验证：`git status` 显示仅文档 + 4 个代码文件（注释）改动；frozen 原值保留且带 `[订正 2026-06-16: … 见 0overview/02_gotchas.md#规范数字框-canonical-numbers]`。

---

## 6. 维护提示

- 以后**代码结构/规范数字变化** → 更新 `0overview/`（重读代码，别从 prose 转抄）。
- `0overview/02_gotchas.md#规范数字框-canonical-numbers` 是**数字唯一家**，别处链接它、不重写值。
- 再有大的文档大修/审计 → 在本目录 `8doc_updates/` 追加 `YYYY-MM-DD_<topic>.md` 记录。
