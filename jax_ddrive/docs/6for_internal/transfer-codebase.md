# STEP 1 — 把 Fast-dDrive 代码库搬进内部环境(transfer codebase)

*Last updated: 2026-06-16 — 更新记录见 [`updates-latest-0616.md`](updates-latest-0616.md);`history/` 是归档(含旧 0614),agent 无需读。*

> **PREREQ(owner 侧):** 本 STEP 1 假设代码/数据已在 GCS。owner 先按 [`00_owner_publish.md`](00_owner_publish.md)(**STEP 0**:发代码 + 数据 + manifest)发布,内部侧才有东西可拉。

> **怎么用:** 把这一整篇**直接粘贴给内部 coding agent**(在 Cloudtop 上、能访问 GCS)。它会从 GCS
> 拉带时间戳的代码包、解到 google3 源码树、并写一个稳定的 env 文件。**跑完这一步,agent 就有了完整
> 仓库(代码 + 全部文档)作为上下文**,再进 STEP 2(`test_training.md`,那时已在仓库里)。
>
> 这一步**只搬代码**(几十 MB,几秒;上传脚本会打印实际大小):先拿到 codebase 拿 full context。**不**拉 12 GB+ 的数据、
> **不**装 venv —— 那些在 STEP 2(数据只有真正训练时才用到)。

---

## 这一步做什么
1. 从 `gs://<bucket>/code/fastddrive-<TS>.tgz`(`fastddrive-LATEST.txt` 指向最新)拉代码包,解到
   google3 树:`.../third_party/fastddrive-<TS>/{maxtext-dlm-fork, jax_ddrive}`。
2. 写 `~/.fastddrive_env`(`FORK` / `DDRIVE` / `DATA_ROOT` / `SRC`)+ 稳定软链 `~/fastddrive-current`,
   这样 STEP 2(可能是另一个 shell session)`source` 一下就有全部路径,不用再去推时间戳。

## 跑(整段复制)
```bash
gcloud auth login kaiwenh@google.com
SRC=gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd
G3=/google/src/cloud/kaiwenh/fastdllm/google3/experimental/waymo/users/xqin/third_party
mkdir -p "$G3"

# 拉最新代码包(要钉某个版本就把下一行换成 TS=fastddrive-<具体时间戳>):
TS=$(gsutil cat "$SRC/code/fastddrive-LATEST.txt" | sed 's/\.tgz$//')
gcloud storage cp "$SRC/code/$TS.tgz" ~/ && tar xzf ~/"$TS.tgz" -C "$G3" && rm ~/"$TS.tgz"
ln -sfn "$G3/$TS" ~/fastddrive-current

# 写稳定 env 文件(STEP 2 起手 `source ~/.fastddrive_env`):
cat > ~/.fastddrive_env <<EOF
export SRC=$SRC
export FORK=$G3/$TS/maxtext-dlm-fork          # 训练 + B1 导出 + B2 推理(自包含)
export DDRIVE=$G3/$TS/jax_ddrive              # 仅离线输入 prep(STEP 2 §3a)用
export DATA_ROOT=/cns/is-d/home/chauffeur/perception_training/kaiwenh/data   # CNS 大数据池
EOF
echo "code -> $G3/$TS"
echo "env  -> ~/.fastddrive_env  (内容如下)"; cat ~/.fastddrive_env
```

## 现在你有了完整上下文 —— 跑任何东西前先读这几篇(都在刚解出来的仓库里)
```bash
source ~/.fastddrive_env
ls $DDRIVE/docs/6for_internal/test_training.md            # STEP 2:数据 + 训练 + 导出 + 推理(接着做这个)
ls $DDRIVE/docs/2implementation-details/INFERENCE_DEPLOY.md    # 设计权威:B1/B2、bf16、嵌入彩排、tested-vs-pending
ls $DDRIVE/docs/5blockers/0612-blocker-v0.md                  # 内部 TPU 部署的 blocker 分析(背景)
ls $FORK/PATCHES.md                                          # MaxText fork 逐文件 diff
```
- **`$FORK`**(`maxtext-dlm-fork`)= 生产代码:训练(`maxtext.trainers.pre_train.train`)、B1 导出
  (`scripts/maxtext_to_hf_export.py`)、B2 自包含推理(`src/maxtext/diffusion/eval_sasd/`)。
- **`$DDRIVE`**(`jax_ddrive`)= 文档 + 离线 prep 脚本所需的 `ddrive_jax` 包。
- 项目一句话:Qwen2.5-VL-3B 微调成的**掩码 block-diffusion** VLA(SASD),输出 4 段 JSON
  (critical_objects / explanation / future_meta_behavior / trajectory);本任务从**干净 base**
  过拟合一个小数据集,验证训练+推理管线**全程在内部 TPU 内**完成(只有标量验证日志带出)。

## 代码怎么更新(owner 侧)
owner 改了代码就 `bash jax_ddrive/scripts/upload_code_to_gcs.sh`(repo 内、版本化;旧 `/home/kaiwen/upload_code_to_gcs.sh` 已转成转发 stub,仍可用)重发一个新的 `fastddrive-<TS>-<sha7>.tgz` + 刷新
`fastddrive-LATEST.txt` + 写一份 `MANIFEST.json`(记 git commit SHA + dirty)。agent 重跑上面的"拉最新代码包"即可拿到新版本;
要确认/钉某个 commit,先看 `code/fastddrive-LATEST-MANIFEST.json`(给出 `git_commit` / `dirty`)。

---
**→ 接 STEP 2:`$DDRIVE/docs/6for_internal/test_training.md`**
