# Design Spec — `jax_ddrive/docs/0overview/` 前门 + 全文档审计/对账

- **日期**: 2026-06-15
- **状态**: 已批准（用户口头批准全部决策，并要求自主执行到完成）
- **作者**: Claude (brainstorming skill)
- **触发**: 文档过多，context 有限的 coding agent 无法整读；需要一个"如何读代码 + subtle 组件"的详细入口，并保证现有文档质量。

## 1. 目标 (Goal)

为 Fast-dDrive JAX 移植项目建立一个**分层前门** `jax_ddrive/docs/0overview/`，让一个**新来的、context 有限的 coding agent** 只读它需要的那一层，就能：(a) 懂系统、(b) 知道按什么顺序读代码、(c) 拿到"易误读组件"目录、(d) 知道每块去哪深挖、(e) 拿到唯一权威的规范数字。同时**审计并修正**现有 ~30 个文档与真实代码之间的漂移。

## 2. 受众边界 (已决策)

- 主受众 = **理解并修改代码的 coding agent**（区别于：部署用 `to-host-chn.md`；内部 TPU 跑 from-base 用 `6for_internal/`）。
- 选定方案 = **完整前门 (C)**：`0overview/` 成为唯一入口，吸收 README 的文档导航地图；README 收缩为纯维护规则。

## 3. 语言 (D1, 已决策)

中文行文 + 英文术语/关键词（函数名、`flag`、SASD/M-RoPE/FSDP/ViT 等一律英文）。用户是中文母语，便于把握高层结构；英文流利，技术名词保留英文最准。

## 4. 交付物 (5)

### D-1 `0overview/` 三档文档
- `00_START_HERE.md`（~120–150 行，唯一要求读完）：一段话讲清系统 · 受众路由表 · 现状一览（仅链出证据，不复制数字）· 文档导航地图（从 README 吸收 living/stable/frozen 分类）· "context 有限就按这个顺序读"。
- `01_codebase_map.md`（~200 行）：三代码库关系图（`ddrive_jax/` NNX 真值 · `maxtext-dlm-fork/` 生产/TPU · PyTorch oracle）· 黄金阅读顺序 · 子系统地图（概念→文件→关键函数→一句话）· 深挖索引（链到 `2implementation-details/`）。
- `02_gotchas.md`（~150 行）：易误读组件目录（~12 条，每条 现象+为何易错+看哪个文件确认）· 可执行规格（gates + `*_PASS` marker）· **规范数字框**（数字的唯一家）。

### D-2 doc↔code 漂移审计 (D3=本次做)
逐文档把每条可核对声明（文件名/函数名/`flag`/路径/数字/行为描述）对照**代码 ground truth**，产出 findings：`{doc, section, 声明, 代码现实, 判定(stale/wrong/ok), 建议修法}`。

### D-3 规范数字对账表 (D2)
每个关键数字分类：规范值（代码为准）/ 合理变体（记语境+出处）/ 错误写法（记出处+待修）。已知样例：参数量 3.086B（`3.75B` 错）；image tokens 168 单序列 vs 336 doubled vs 720 错误 paper 分辨率；L=1184 pseudo vs 1280 distilled；eval 479 帧 vs 52 帧早期 partial。

### D-4 README 重构 + 指针接线
- README 移出 living/stable/frozen 导航表 → `00_START_HERE.md`；README 保留维护规则；顶部加 `> Start here → 0overview/00_START_HERE.md`。
- `to-host-chn.md` 顶部加一行：`> 想读懂代码而非部署？→ docs/0overview/00_START_HERE.md`（其余不动）。

### D-5 回填修正
把 D-2 + D-3 的 fix 落回源头文档。

## 5. 关键规则

- **只链不复制**：0overview 唯一例外是 `02` 的规范数字框（数字的唯一 SSOT，反向被其它文档链接）。
- **代码为准**：`01`/`02` 必须实读代码生成，不二手转述，避免继承 doc 漂移。
- **frozen 文档处理 (已决策)**：living/stable（README、`2implementation-details/*`、`to-host-chn.md`、`6for_internal/*`）→ 直接改；frozen 日志（`1plans/`、`4collect/`）→ **不重写历史**，在错处加内联订正标注，如 `3.75B ⟵ 订正: 3.086B（见 0overview/02_gotchas.md#规范数字）`，遵守 README 维护规则"frozen append-only"。
- **对抗式校验**：用户不在，任何"修正"在写入前必须对照代码二次校验——错的修正比原错更糟。

## 6. 执行 phase（自主）

1. **Phase 1 — 代码 ground truth**：多 agent 实读 `ddrive_jax/` + fork + configs + gates → 子系统地图 / subtle 行为 / 带出处的原始数字。
2. **Phase 2 — doc↔code 审计**：~30 文档逐条核对代码 → findings。
3. **Phase 3 — 数字对账 + 校验**：聚合所有数字 → 规范表（含合理变体语境 + 错误出处），逐条对代码校验。
4. **Phase 4 — 写前门 + README 重构 + 指针**。
5. **Phase 5 — 回填修正**（living 改 / frozen 标注）。

每个修正在应用前对抗式校验通过才落地。

## 7. 非目标 (Non-goals)

- 不改代码、不改训练/推理行为，只动文档。
- 不删除 frozen 历史记录的"发现过程"。
- 不把已经 frozen 的 plan/log 重写。
