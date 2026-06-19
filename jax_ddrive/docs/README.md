# Docs maintenance rules

> **Start / navigation → [`0overview/00_START_HERE.md`](0overview/00_START_HERE.md).**
> 那是文档的总入口：系统是什么、受众路由、living/stable/frozen 文档导航地图、代码阅读指南（[`0overview/01_codebase_map.md`](0overview/01_codebase_map.md)）、易误读组件 + 规范数字（[`0overview/02_gotchas.md`](0overview/02_gotchas.md)）。
> 本文件只保留**维护规则**（怎么让文档保持健康），不再承担导航。

## 文档分层（一行速记；完整带 scope 的导航表在 [`0overview/00_START_HERE.md`](0overview/00_START_HERE.md) §4）

- **Living（必须随现实更新）**：`../to-host-chn.md`、`2implementation-details/DATASET_V2.md`、`quick-refresh.md`、`5blockers/*`、`6for_internal/*`、`7for_internal_inference/*`。
- **Stable references（组件变了才更新）**：`0overview/*`、`2implementation-details/{ARCHITECTURE,01_pytorch_reference_algorithm,EVAL_PIPELINE,INFERENCE_DEPLOY,DATASET,LABELING,AUDIT}.md`、`3summary/*`。
- **Frozen（历史，永不编辑，append-only）**：`1plans/*`（如 `06_trainable_vit_plan.md` —— 可训练 in-graph ViT 的 plan + 滚动 §9 状态/日志）、`4collect/*`（含 `08_trainable_vit_progress.md` 里程碑卷）。
- **文档维护记录（append-only）**：`8doc_updates/` —— 对 docs 体系本身的大修/审计记录 + 证据产物（不是项目里程碑，那在 `4collect/`）。

MaxText fork 侧的逐文件 diff / vendor 规则在 fork 内 `maxtext-dlm-fork/PATCHES.md`。

## Maintenance rules

1. **One fact, one home.** Current state lives in the living docs; don't restate it
   elsewhere (link instead). Frozen logs keep the *discovery* story, not the truth.
2. When a milestone lands: update the living docs **in the same change**, and append a
   dated entry to a `4collect/` log (create `NN_<topic>_progress.md`, numbered).
3. The handoff doc is `to-host-chn.md` (中文 only; the English `to-host.md` twin is retired).
4. Plans in `1plans/` are written once and frozen; deviations are recorded in the living
   docs, not by editing the plan. **Correcting a wrong number in a frozen log = inline
   bracketed annotation** (e.g. `3.75B [订正: 3.086B, 见 0overview/02_gotchas.md#规范数字框-canonical-numbers]`),
   never a rewrite — preserve the original record.
5. Big artifacts (datasets, ckpts, oracles) live under `/home/kaiwen/data/fast-ddrive/`
   and on GCS — docs reference them by path; nothing heavy in git.
6. **`6for_internal/` + `7for_internal_inference/` runbooks** (`transfer-codebase.md`, `test_training.md`,
   `03_data_processing.md`, `7for_internal_inference/00_run_inference.md`) are **living, undated,
   edited in place** and shipped to the internal side in the code bundle. Each dir has its own rolling
   changelog `updates-latest-<date>.md` (newest on top); when a newer `updates-latest-*` supersedes it,
   move the old one into the dir's `history/` (which the internal agent does not read). Do NOT re-add a
   date prefix to the runbook filenames. STEP 0/1 (publish + transfer) are shared by both tracks;
   `6for_internal` = train + data, `7for_internal_inference` = inference/eval.
7. **`0overview/` is sourced from CODE ground-truth, not from other docs.** When code
   structure, subsystem layout, or a canonical number changes, update `0overview/` (re-read the
   code, don't transcribe from prose). `0overview/02_gotchas.md#规范数字框-canonical-numbers` is the
   **single home for key numbers** — other docs link to it rather than restating values.
