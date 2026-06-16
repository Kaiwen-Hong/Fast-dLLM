# 2026-06-16 · maxtext-dlm-fork 并入 Fast-dLLM + 发布脚本版本化

> **触发**：内部交接的版本控制问题。代码包是"时间戳级、与 git 脱钩"的工作区 tarball；MaxText fork 是**独立 repo、零 remote**（只在 owner 笔记本上）；数据无版本。讨论后定方案、执行。

## 问题（详见讨论）
- `upload_code_to_gcs.sh` 打的是**工作区** + `--exclude='.git'` → 包身份**只有 `<TS>` 时间戳，无 git commit**，还可能含未提交改动。
- `maxtext-dlm-fork`（生产主代码，6 个 commit、手写 SASD 全套）是**独立 repo、无任何 remote** → 笔记本即单点。
- `jax_ddrive` 在 Fast-dLLM repo（有 GitHub `Kaiwen-Hong/Fast-dLLM`）；fork 不在。

## 决策
- **并入（简单派）**：把 fork 拷进 Fast-dLLM 作普通子目录 → 一个 repo、一个 git commit 描述全部、白蹭 GitHub remote 做异地备份。
- 不再追上游 MaxText（内部 rebase 到 google3 MaxText 走 `PATCHES.md`，与并入无关）→ 放弃"丝滑 rebase"几乎无成本。
- 历史用 `git bundle` 备份，不用 submodule/subtree。

## 做了什么
1. **备份 fork 历史**：`git bundle create /home/kaiwen/data/fast-ddrive/maxtext-dlm-fork-history-2026-06-16.bundle --all`（36MB，verify=complete history，HEAD `da8c92b`）。**原目录 `/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork` 保留不删**（活备份）。
2. **拷入** `Fast-dLLM/maxtext-dlm-fork/`（rsync 去 `.git`/缓存；1129 文件 / 93MB；保留 fork 自带 `.gitignore` + `LICENSE` + `PATCHES.md`；无 `.git`）。
3. **发布脚本版本化**：新建 `jax_ddrive/scripts/upload_code_to_gcs.sh`（repo 内）——
   - 从**单 repo** 打包 `maxtext-dlm-fork` + `jax_ddrive`（repo root 由脚本自身位置推导，便携）；
   - 记 `git rev-parse HEAD` + branch + remote + **dirty 检查**（packed 子目录有未提交改动则警告 + 包名加 `-dirty`）；
   - 写 **`MANIFEST.json`** 进包根 + 上传同名 `.MANIFEST.json` + `fastddrive-LATEST-MANIFEST.json`；
   - 包名 `fastddrive-<TS>-<sha7>[-dirty].tgz`，`LATEST.txt` 仍指向 .tgz（内部拉取流程不变）。
   - 旧 `/home/kaiwen/upload_code_to_gcs.sh` 改成**转发 stub**（手感不变、单一真相源）。
4. **文档更新**（路径/概念）：
   - 绝对路径修正：`0overview/01_codebase_map.md`（§1 表 + 仓库结构说明）、`2implementation-details/ARCHITECTURE.md`。
   - 发布脚本引用→新 in-repo 路径 + manifest/SHA：`INFERENCE_DEPLOY.md`、`6for_internal/{transfer-codebase,test_training}.md`、`to-host-chn.md` §6.6。
   - `maxtext-dlm-fork/PATCHES.md` provenance：注明已并入 Fast-dLLM（同 repo）+ 历史备份位置。
   - 相对名 `maxtext-dlm-fork/...`（目录名没变）**未动**，仍有效。

## 新发布流程（owner）
```bash
# 改完代码、commit（让 SHA 有意义）后：
bash jax_ddrive/scripts/upload_code_to_gcs.sh
# → gs://…/code/fastddrive-<TS>-<sha7>.tgz + .MANIFEST.json + 刷新 LATEST(.txt / -MANIFEST.json)
```
内部侧 `transfer-codebase.md` 不变（仍按 `LATEST.txt` 拉），但现在每包带 `MANIFEST.json` → **可反查/钉 git commit**。

## ⚠️ 还需 owner 手动做（本次未替你执行）
1. `git add maxtext-dlm-fork/ jax_ddrive/ docs/`（+ 已删 redirect 的旧脚本不在 repo）→ **commit**（fork ~93MB / 1129 文件首次入库）。
2. `git push origin jax-ddrive-port` → **fork 代码这才真正有了 GitHub 备份 + 可 resolve 的 SHA**（这是整件事的最终闭环）。
3. （可选）确认旧 `/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork` 不再编辑——**从此 canonical 是 `Fast-dLLM/maxtext-dlm-fork/`**。

## 数据版本化（2026-06-16 同日补做）
代码 provenance 闭环后，把数据侧"命名路径原地覆盖、无版本"的缺口也补上（**简单派、不改数据名**）：
- 新增 `jax_ddrive/scripts/data_manifest.py`：为任一 artifact（GCS prefix / 本地 dir）生成 `DATA_MANIFEST.json`——GCS 读 `crc32c`+`size`+`generation`（**不下载**，369GB 也秒级）、本地 sha256（>8GB cap 则只记 size + 提示）；一个汇总 `digest` 标识整份；`--upload` 写到 artifact 同目录。
- `parquet_to_ar_with_embeds.py` 加 best-effort hook：建完数据集自动写 `DATA_MANIFEST.json`（distilled-400 实测 sha256 / 8 文件 / digest 确定可复现）。
- **运行时闭环**：`validation_log` 的 `data_provenance`（S0 事件）记 `{code_git（取代码 MANIFEST.json）, data_digests:{dataset, base_params, snapshot, eval_inputs}}` → 一次 run = `{哪个 code commit} × {哪几份数据 digest}`，完全可反查/校验。
- **升级而非新增槽**：`0612 §7` 原有 `refs.dataset_sha`（只哈希 schema 文件 `dataset_info`、仅覆盖数据集）→ 升级为 `refs.data_digests`（**内容指纹**，覆盖全部消费 artifact）。
- 文档同步：`5blockers/0612-blocker-v0.md §7`、`2implementation-details/DATASET_V2.md`、`6for_internal/test_training.md §9`、`to-host-chn.md §6.6`。
- **未做（你同意先不做）**：eval_inputs 路径版本化（`eval_inputs/<sha7>/`）—— 现 digest 已能检测变更；路径版本化改 runbook、较重，缓。

### owner 待办（数据侧）
发布数据时对每个 artifact 跑 `python jax_ddrive/scripts/data_manifest.py gs://…/<artifact> --upload`（写 `DATA_MANIFEST.json` 到 GCS）。本次**未替你执行 GCS 上传**（gsutil 需 owner 认证）。`data_manifest.py` + builder hook 已就位、本地 smoke-test 通过。

## owner 发布 runbook（STEP 0，新增）
之前 owner 侧发布（commit → 发代码 → 建/传数据 → 写 manifest）散落在 `to-host §6.6` prose + 3 个脚本，且没有"贴给 Claude 就能发布"的单篇 doc（内部侧有 STEP 1/STEP 2，owner 侧空缺）。补 `6for_internal/00_owner_publish.md`：
- 一页分步:§1 commit → §2 `upload_code_to_gcs.sh`(代码) → §3 `build_full_dataset.sh --upload-gcs`(数据变了) → §4 `for A in …; data_manifest.py $SRC/$A --upload`(4 个 artifact 的 manifest) → §5 自检。
- 顶部标明"给 owner、不是给内部 agent";流程序列 **STEP 0(owner 发布) → STEP 1(transfer-codebase) → STEP 2(test_training)**。
- 已注册进 `0overview/00_START_HERE.md`(§2 路由 + §4 living 表) + `transfer-codebase.md` 顶部 PREREQ 指针。
