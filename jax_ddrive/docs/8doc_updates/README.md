# 8doc_updates —— 文档维护 / 审计记录（append-only）

本目录记录对 `docs/` 体系本身的**大修与审计**（不是项目里程碑——那在 `4collect/`）。每次大的文档重构/全量审计在这里追加一篇 `YYYY-MM-DD_<topic>.md`，并把可复核的证据产物一起留存。

## 记录

| 记录 | 内容 |
|---|---|
| [`2026-06-16_0overview-and-doc-audit.md`](2026-06-16_0overview-and-doc-audit.md) | 建立 `0overview/` 分层前门（START_HERE / codebase_map / gotchas）+ README 重构；从代码建立 ground truth → 审计 30 文档（108 findings / 32 缺陷）→ 规范数字对账 → 回填修正（56 处）+ 5 处代码注释订正。决策、pipeline、token 成本、改动清单全在内。 |
| [`2026-06-16_fork-consolidation-and-release-versioning.md`](2026-06-16_fork-consolidation-and-release-versioning.md) | 解决内部交接的版本控制问题：把独立、无 remote 的 `maxtext-dlm-fork` **并入 Fast-dLLM repo**（git bundle 备份历史）；发布脚本 `upload_code_to_gcs.sh` 进 `jax_ddrive/scripts/` 并改为单 repo 打包 + 写 `MANIFEST.json`（git commit SHA + dirty）；更新相关文档路径。含 owner 待办（commit + push）。 |

## 证据产物（供复核）

| 文件 | 内容 |
|---|---|
| `design_spec.md` | 经批准的设计 spec（受众边界、3 档结构、D1–D3 决策、frozen 策略、执行 phase） |
| `evidence_code_ground_truth.md` | 实读 `ddrive_jax/` + MaxText fork 的代码真值地图（子系统/gotchas/带 file:line 的数字/10 gate） |
| `evidence_doc_audit_report.md` | 去重后的审计报告（按严重度排序 + 跨文档数字问题 N1–N8） |
| `evidence_doc_audit_findings.json` | 全部 108 条 raw findings（结构化，含每条 file:line 证据 + 建议修法） |
| `evidence_fix_summary.md` | 回填 pass 的 applied/skipped 逐条记录 |
