# 内部文档 — 更新记录(changelog)

> `6for_internal/` 的**滚动更新记录**,最新在最上。**agent 不必读这篇**;要完整上下文读
> [`00_owner_publish.md`](00_owner_publish.md)(STEP 0)+ [`transfer-codebase.md`](transfer-codebase.md)(STEP 1)
> + [`test_training.md`](test_training.md)(STEP 2)。
> 当出现新批次的 `updates-latest-<新日期>.md` 时,把旧的这份移到 [`history/`](history/)。
> 上一批 `updates-latest-0614.md` 已归档到 `history/`。owner 侧大型文档工程的叙事记录见 `../8doc_updates/`。

---

## 2026-06-16 — 代码/数据版本化 + owner 发布 runbook

**发布管线版本化(fork 并入 + git-SHA manifest)**
- `maxtext-dlm-fork` 并入 Fast-dLLM repo → 一个 git commit 描述整套代码。发布脚本进 repo:
  `jax_ddrive/scripts/upload_code_to_gcs.sh`(单 repo 打包 + 写 `MANIFEST.json`:git commit SHA + dirty;
  包名 `fastddrive-<TS>-<sha7>[-dirty].tgz`)。旧 `/home/kaiwen/upload_code_to_gcs.sh` 转成转发 stub。
- runbook 同步:`transfer-codebase.md` / `test_training.md` 里 upload 路径改为 in-repo + 提 MANIFEST/SHA。

**数据版本化(provenance)**
- 新增 `jax_ddrive/scripts/data_manifest.py`:为任一 artifact 写 `DATA_MANIFEST.json`(GCS crc32c / 本地
  sha256 → 汇总 `digest`,不下载)。`parquet_to_ar_with_embeds.py` 建数据集时自动写。
- `test_training.md §9` 加 `data_provenance` 事件(记 code git_sha × 各 artifact 的 `DATA_MANIFEST.digest`);
  日志 schema 见 `../5blockers/0612-blocker-v0.md §7`(原 `refs.dataset_sha` 升级为内容指纹 `refs.data_digests`)。

**新增 owner 发布 runbook**
- `00_owner_publish.md`(**STEP 0**):owner 把代码 + 数据 + manifest 发到 GCS 的分步 runbook
  (commit → `upload_code_to_gcs.sh` → 建/传数据 → `data_manifest.py … --upload` → 自检)。
  三步流程:**STEP 0(owner) → STEP 1(transfer-codebase) → STEP 2(test_training)**;`transfer-codebase.md`
  顶部加了 PREREQ 指针。

完整改动清单 / 决策见 `../8doc_updates/2026-06-16_fork-consolidation-and-release-versioning.md`。

**仍挂着的(owner 待办)**:`git add` + commit + `git push origin jax-ddrive-port`(fork 首次入库 + GitHub
备份);发布数据时对每个 artifact 跑 `data_manifest.py … --upload`。
