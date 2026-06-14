# history/ — 归档(agent 无需阅读)

被取代的内部文档版本、以及旧的 `updates-latest-*.md` 更新记录归档在这里。

**内部 coding agent 不需要读本目录。** 上一层的 [`../transfer-codebase.md`](../transfer-codebase.md)(STEP 1)
+ [`../test_training.md`](../test_training.md)(STEP 2)是当前唯一权威;
[`../updates-latest-0614.md`](../updates-latest-0614.md) 是当前 changelog。

维护约定:
- `6for_internal/` 的两篇 runbook 是 **living**、**原地编辑**(`git log --follow` 保留全部历史)。
- 当一个新批次的 `updates-latest-<新日期>.md` 出现时,把旧的那份移到这里。
- 任何被**整体取代**(而非小改)的内部文档也归档到这里,文件名带日期。

(当前为空:0614 是首个 changelog;两篇 runbook 由 `git mv` 去前缀改名而来,旧 `0613-` 历史在 git 里。)
