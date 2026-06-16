# STEP 0 — owner 发布到 GCS（publish code + data + manifests）

*Last updated: 2026-06-16 — changelog: [`updates-latest-0616.md`](updates-latest-0616.md).*

> **本篇是给 owner(你)的**,不是给内部 agent。它是内部 `transfer-codebase.md`(STEP 1) 的**前提**:
> 把代码 + 数据 + provenance manifest 发到 GCS,内部侧 STEP 1 才能从 GCS 拉。
> 可以把这一整篇**贴给本地机器(5090)上的 Claude Code** 来执行。
>
> 全流程顺序:**STEP 0(本篇,owner 发布) → STEP 1(`transfer-codebase.md`,内部拉代码) → STEP 2(`test_training.md`,内部训练)**。

- bucket:`gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd`(下记 `$SRC`)
- repo root:`/home/kaiwen/Desktop/research/Fast-dLLM`(`maxtext-dlm-fork/` 与 `jax_ddrive/` 同 repo)

---

## 何时发布什么
- **只改了代码** → 跑 §1 + §2(代码 bundle)。
- **改/重建了数据**(dataset / base params / snapshot / eval_inputs)→ 再加 §3 + §4(传数据 + 写 manifest)。
- 每次发布后做 §5 自检。

## 1. 先 commit(让 code 版本身份有意义)
```bash
cd /home/kaiwen/Desktop/research/Fast-dLLM
git add -A && git commit -m "..."     # 干净工作树 → bundle 的 git SHA 才真正描述它
```
> 不 commit 也能发,但 bundle 会标 `-dirty`、SHA 不完整描述内容(`MANIFEST.json` 里 `dirty: true`)。

## 2. 发布代码(bundle + MANIFEST)
```bash
bash jax_ddrive/scripts/upload_code_to_gcs.sh
# → $SRC/code/fastddrive-<TS>-<sha7>.tgz + 同名 .MANIFEST.json
#   + 刷新 fastddrive-LATEST.txt(指向最新 .tgz) 与 fastddrive-LATEST-MANIFEST.json(最新 commit)
```
打包同一个 repo 的两个子目录 `maxtext-dlm-fork` + `jax_ddrive`;`MANIFEST.json` 记 git commit + dirty。

## 3.(数据变了才做)构建/上传数据
```bash
SRC=gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd
# 例:重建并推全量数据集
bash jax_ddrive/scripts/build_full_dataset.sh --full --upload-gcs $SRC/wod_e2e_sasd_full_v2_ar
# 或对已有 artifact 直接同步: gsutil -m rsync -r <local_dir> $SRC/<artifact>
```

## 4.(数据变了才做)给每个 artifact 写 DATA_MANIFEST
```bash
SRC=gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd
for A in maxtext_sasd_params_base \
         wod_e2e_sasd_distilled_0613-400_baseViT_v2_ar \
         base_qwen25vl_3b_snapshot \
         eval_inputs; do
  python jax_ddrive/scripts/data_manifest.py "$SRC/$A" --upload   # 读 crc32c(不下载) → 写 $SRC/$A/DATA_MANIFEST.json
done
```
内部侧 STEP 2 会把每个 artifact 的 `DATA_MANIFEST.json` 的 `digest` 记进 `validation_log` 的
`data_provenance` 事件 → 一次 run 可反查"哪个 code commit × 哪几份数据"(版本可追溯)。

## 5. 自检
```bash
SRC=gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd
gsutil cat $SRC/code/fastddrive-LATEST.txt                # 指向你刚发的 .tgz
gsutil cat $SRC/code/fastddrive-LATEST-MANIFEST.json      # git_commit 对得上你 commit 的 SHA、dirty=false
for A in maxtext_sasd_params_base wod_e2e_sasd_distilled_0613-400_baseViT_v2_ar base_qwen25vl_3b_snapshot eval_inputs; do
  gsutil stat $SRC/$A/DATA_MANIFEST.json >/dev/null 2>&1 && echo "OK       $A/DATA_MANIFEST.json" || echo "MISSING  $A"
done
```

---
**→ 接 STEP 1(内部侧):`transfer-codebase.md`**
