# 内部文档 — 更新记录(changelog)

> `6for_internal/` 的**滚动更新记录**,最新在最上。**agent 不必读这篇**;要完整上下文读
> [`00_owner_publish.md`](00_owner_publish.md)(STEP 0)+ [`transfer-codebase.md`](transfer-codebase.md)(STEP 1)
> + [`test_training.md`](test_training.md)(STEP 2)。
> 当出现新批次的 `updates-latest-<新日期>.md` 时,把旧的这份移到 [`history/`](history/)。
> 上一批 `updates-latest-0614.md` 已归档到 `history/`。owner 侧大型文档工程的叙事记录见 `../8doc_updates/`。

---

## 2026-06-16 (后续 2) — 新增 STEP 3:数据处理 sanity + 发布ckpt val parity + 新 split

内部要做 4 件事(对一个**同 WOD-E2E proto** 的新 split):① 数据处理链跑通 ② 数据格式检查 ③ 用我们的方法
处理新 split ④ 发布 NVIDIA ckpt 在 val 上推理复现我们的数字。**原则:内部 coding agent 较弱 → 凡需判断的活
在本地(5090)做完、测好,内部只机械 replay。**

**本地预制(已写 + 实测通过)**
- `jax_ddrive/scripts/probe_dataset.py`(只读探针:`--raw` 解 E2EDFrame / `--processed` 读 AR|parquet schema)。
  实测:v2 AR(479)+ parquet(400)+ 原始 val shard 全 `PROBE_DONE`。
- `jax_ddrive/scripts/check_dataset_format.py`(独立格式校验,复用 `verify_ar_v2` 断言、去掉源-parquet diff)。
  实测:val_v2_ar / parquet 均 `CHECK_PASS failures=0`(**修了一个 bf16 假阳**:TF `.numpy()` 的 bfloat16 与
  `ml_dtypes.bfloat16` 是两个不相等的 numpy 扩展 dtype → 改按 `.name` 比较)。
- `jax_ddrive/scripts/build_rated_val_gt.py`(479-rated GT → 0.55MB pkl,镜像 `evaluate_waymo_metrics` 的
  `--gt *.pkl` 结构)。实测:对现有 predictions.json 复现 ADE **bit 一致**、RFS 差 ~4e-9。
- `jax_ddrive/scripts/extract_rated_val_subset.py`(从 226GB val 抽 479 rated → 单 tfrecord **1.1GB**)。实测 kept=479。
- 4 个 prep/推理脚本硬编码路径参数化为 `FASTDDRIVE_SNAP`/`FASTDDRIVE_REPO`(env 默认,本地行为不变)。

**runbook + 发布**
- 新增 `03_data_and_inference_parity.md`(**STEP 3**,独立 track、用**发布 ckpt** 非 from-base):§0 env+从 GCS 拉
  artifact → §1 probe 新数据 → §2 val 环控(§2A item1 处理+格式 `CHECK_PASS`;§2B item4 推理+官方 metric 对
  golden **0.839/2.072/7.929**)→ §3 处理新 split。顶部 pin 死分辨率(eval 200704 vs AR 168)、精度(fp32 bar+bf16
  诊断)、golden。
- `upload_code_to_gcs.sh` 的 bundle 加入 **`fast_ddrive/`**(convert + 官方 metric 在那;原来只打包
  `maxtext-dlm-fork`+`jax_ddrive`),并排除 `.claude`。`00_owner_publish.md` 加 **§6** 发布 STEP 3 的 3 个 artifact
  (发布快照 / val 子集 / GT pkl)。

**仍挂着(owner 待办)**:commit + push;跑 §6 发布 3 个 artifact(并对快照写 `data_manifest.py … --upload`)。

---

## 2026-06-16 (后续) — test_training 全量生产训练变体

- `test_training.md` 加 **§12「全量生产训练(变体)」**:换 `sasd_data_dir=…/wod_e2e_sasd_full_v2_ar`
  (415,663 帧 / 130 shards / ~369G,pseudo 标签,**L=1184** → 用 `sasd_waymo.yml` 默认配置、**不加** distilled 的
  1280 覆盖);overfit 的 T1/T2 逐字标准不适用,改判固定噪声 eval loss 下降 + held-out(val 479)不发散。
  **代码/数据已就绪,无需改代码**;按 §4 先把全量 ingest 到 CNS。
- `00_owner_publish.md` §4/§5 的 manifest 循环加入 `wod_e2e_sasd_full_v2_ar`。

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
