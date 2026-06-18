# STEP 3 — 数据处理 sanity + 格式检查 + 新 split 处理

*Last updated: 2026-06-17 — changelog: [`updates-latest-0616.md`](updates-latest-0616.md)。
推理 / 官方 ADE/RFS 已拆到 [`../7for_internal_inference/00_run_inference.md`](../7for_internal_inference/00_run_inference.md)。*

> **PREREQ:** STEP 0(owner 已按 [`00_owner_publish.md`](00_owner_publish.md) 发布代码 + 本 STEP 的
> artifacts 到 GCS)→ STEP 1(你已按 [`transfer-codebase.md`](transfer-codebase.md) 拉代码并
> `source ~/.fastddrive_env`)。
>
> **怎么用:** 把这一整篇**贴给内部 coding agent**。它是一条**独立 track**,目标是 3 件事:
> ① 证明我们的数据处理链能在内部跑通 ② 有脚本能检查数据集格式 ③ 用我们的方法处理一个新的 WOD-E2E split。
>
> **要在 val 上做推理 / 复现官方 ADE/RFS?** → 见 [`../7for_internal_inference/00_run_inference.md`](../7for_internal_inference/00_run_inference.md)（姊妹 track）。

---

## 📌 钉死的常量（CANONICAL — 不要改,改了就静默出错）

| 项 | 值 | 为什么 |
|---|---|---|
| **训练/AR 格式分辨率** | `prep_train_jax.py` 内部固定 `784 / 50176 → 168 image tokens`(pixel 672×1176) | v2 AR / 训练格式 |
| **MASK/markers** | MASK=151665,im_end=151645,image_pad=151655,assistant 三元 151644/77091/198 | checker 内部用,仅供排错参考 |
| **统一形状** | L=1184、pixel(672,1176)、image_grid_thw(3,3)、image_embeds(168,2048) bf16 | 同 proto 的子集应 100% uniform |

> 推理 / 官方 metric 的常量(200704 paper-res、fp32/bf16、golden 0.839、GT pkl)在
> [`../7for_internal_inference/00_run_inference.md`](../7for_internal_inference/00_run_inference.md) 顶部。

---

## §0 一次性:env + 确认 artifacts 在 CNS

```bash
source ~/.fastddrive_env          # 给出 FORK / DDRIVE / DATA_ROOT / SRC(STEP 1 写的)

# 本 STEP 用到的两个 env(脚本读它们定位 repo 和快照,STEP 1 的脚本已参数化、无需改代码):
export FASTDDRIVE_ROOT="$(dirname "$DDRIVE")"            # bundle 根:含 jax_ddrive / fast_ddrive / maxtext-dlm-fork
export FASTDDRIVE_REPO="$DDRIVE"                          # = .../jax_ddrive
export FASTDDRIVE_SNAP="$DATA_ROOT/release_fast_ddrive_snapshot"   # 给 prep 的 processor + ViT(embeds)用
                                                          #   若产物要喂 from-base 训练,改成 base snapshot(base ViT)
export PY="<你的 python>"   # 必须能 import: torch, transformers, jax(+TPU), flax, array_record, ml_dtypes
                            #   (新 split 探测/转换;不需要 tf+waymo —— 那是 7 的 metric 才要)
export WORK="$DATA_ROOT/step3_work"   # 本 STEP 的中间产物落这(可 glob 的 CNS/本地 POSIX 路径)
mkdir -p "$WORK"

# 首次跑本 STEP:把 STEP 0 发布的快照 + val 子集从 GCS 拉到 CNS:
mkdir -p "$DATA_ROOT/eval"
gcloud storage rsync -r "$SRC/release_fast_ddrive_snapshot" "$FASTDDRIVE_SNAP"
gcloud storage cp "$SRC/eval/val_rated_479.tfrecord" "$DATA_ROOT/eval/val_rated_479.tfrecord"

# 确认都在:
ls -d "$FASTDDRIVE_SNAP"                                  # 快照(prep 的 processor/ViT)
ls -l "$DATA_ROOT/eval/val_rated_479.tfrecord"           # 479 rated val 子集(~1.1GB,item 1 的已知答案数据)
```

`fast_ddrive` 在 bundle 里(STEP 0 已含),所以 `convert_wod_e2e.py` 在 `$FASTDDRIVE_ROOT/fast_ddrive/`。

---

## §1 PROBE 新数据(item 2:先看清"这数据到底是什么",秒级、只读)

新 split 的原始 tfrecords 已在 CNS(同 WOD-E2E `E2EDFrame` proto)。先探:

```bash
export NEW_SPLIT='<新 split 的 CNS tfrecord glob,如 $DATA_ROOT/new_split/*.tfrecord*>'

$PY "$DDRIVE/scripts/probe_dataset.py" --raw "$NEW_SPLIT" -n 3
```
**期望**:打印 `total records`、每帧 `context.name present: True`、`#images`、`intent`、`len(past/future_states)`、
`rated`,结尾 `PROBE_DONE`。若字段/计数与预期不符(例如不是同一 proto),**停下来报告**,不要往下处理。

---

## §2 VAL 处理 sanity(item 1:处理链跑通 + 格式正确,已知答案的对照组)

把 479 rated val 转成 JSON+图片,再跑整条训练/AR 格式链(168 tokens),最后 checker 校验:

```bash
# convert(带 target,rated only)
$PY "$FASTDDRIVE_ROOT/fast_ddrive/data/convert_wod_e2e.py" \
   --tfrecords "$DATA_ROOT/eval/val_rated_479.tfrecord" \
   --out_json  "$WORK/val.json" --image_root "$WORK/val_images" --rated_only --with_target

# JSON+图片 → per-sample npz(torch+HF processor,内部固定 784/50176→168)
$PY "$DDRIVE/eval/prep_train_jax.py" \
   --train_json "$WORK/val.json" --image_root "$WORK/val_images" --out_dir "$WORK/val_npz"

# npz → parquet
$PY "$DDRIVE/ddrive_jax/convert/prep_to_parquet.py" \
   --npz_dir "$WORK/val_npz" --out_dir "$WORK/val_pq" --split val

# parquet → v2 ArrayRecord(+ 冻结 ViT embeds,纯 JAX,跑在 TPU/GPU 上)
$PY "$DDRIVE/scripts/parquet_to_ar_with_embeds.py" "$WORK/val_pq" "$WORK/val_ar" --split val

# 格式校验(item 2 的 checker)→ 期望最后一行 CHECK_PASS failures=0
$PY "$DDRIVE/scripts/check_dataset_format.py" --dir "$WORK/val_ar" --expect v2 --samples 16
```
**期望**:`CHECK_PASS failures=0`。这就证明了 **item 1**:整条 S1→S4 处理链在内部装得上、跑得通、且产出 schema 正确的数据。

> (在 val 上**推理**并复现官方 ADE/RFS 是另一条 track → `../7for_internal_inference/00_run_inference.md` Track 2。)

---

## §3 用我们的方法处理新 split(item 3)

新 split 是同 proto → S1 直接复用,只换输入路径(与 §2 同一条链,带 `--with_target`):

```bash
$PY "$FASTDDRIVE_ROOT/fast_ddrive/data/convert_wod_e2e.py" \
   --tfrecords "$NEW_SPLIT" --out_json "$WORK/new.json" \
   --image_root "$WORK/new_images" --with_target              # 有 GT 才训得了;纯推理集可去掉

$PY "$DDRIVE/eval/prep_train_jax.py" \
   --train_json "$WORK/new.json" --image_root "$WORK/new_images" --out_dir "$WORK/new_npz"
$PY "$DDRIVE/ddrive_jax/convert/prep_to_parquet.py" \
   --npz_dir "$WORK/new_npz" --out_dir "$WORK/new_pq" --split train
$PY "$DDRIVE/scripts/parquet_to_ar_with_embeds.py" "$WORK/new_pq" "$WORK/new_ar" --split train

# 校验产出格式(item 2 的 checker)
$PY "$DDRIVE/scripts/check_dataset_format.py" --dir "$WORK/new_ar" --expect v2 --samples 16
```
**期望**:`CHECK_PASS failures=0`,且 `probe_dataset.py --processed "$WORK/new_ar"` 显示 image_embeds 在、L=1184、
pixel(672,1176)、embeds(168,2048)。新 split(同 proto)应当 100% uniform;若 checker 报非均匀 L,说明该 split
形态与 WOD-E2E 子集不同 → 报告(可能要走变长 padding 路,目前未启用)。

---

## §4 收尾 / 回报

把这些标量回报(只出标量,符合内部数据外流策略):
- §1 probe 的 `total records` + 字段是否符合预期。
- §2 / §3 的 `CHECK_PASS failures=N`。

> 这几步全部在内部环境内完成(数据处理),除标量验证数字外无数据带出。
> 新 split 的 v2 AR(`$WORK/new_ar`)留在 CNS,可直接喂后续训练(`test_training.md` 的 grain 路径,
> `dataset_type=waymo_sasd`,指向该 AR 目录);要在 val 上推理见 `../7for_internal_inference/00_run_inference.md`。
