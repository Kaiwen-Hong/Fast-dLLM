# STEP 3 — 数据处理 sanity + 发布ckpt val 推理-parity + 新 split 处理

*Last updated: 2026-06-16 — changelog: [`updates-latest-0616.md`](updates-latest-0616.md).*

> **PREREQ:** STEP 0(owner 已按 [`00_owner_publish.md`](00_owner_publish.md) 发布代码 + 本 STEP 的
> artifacts 到 GCS)→ STEP 1(你已按 [`transfer-codebase.md`](transfer-codebase.md) 拉代码并
> `source ~/.fastddrive_env`)。
>
> **怎么用:** 把这一整篇**贴给内部 coding agent**。它是一条**独立 track**,目标是 4 件事:
> ① 证明我们的数据处理链能在内部跑通 ② 有脚本能检查数据集格式 ③ 用我们的方法处理一个新
> 的 WOD-E2E split ④ 用**发布的 NVIDIA ckpt** 在 val 上推理、复现我们已有的数字。

> ⚠️ **与 STEP 2 严格分开,别混 provenance。** STEP 2(`test_training.md`)是**从干净 base 过拟合训练**、
> 走 **B1 bf16 导出**。本 STEP 4 用的是**发布的 NVIDIA fp32 ckpt**(我们 0.839/2.072/7.929 这套 val
> 数字就是它产的)。如果你拿 STEP 2 的 from-base/导出模型来跑 §2B,**一定对不上**——那是设计使然,不是 bug。

---

## 📌 钉死的常量(CANONICAL — 不要改,改了就静默出错)

| 项 | 值 | 为什么 |
|---|---|---|
| **eval 推理分辨率** | `prep_jax_eval.py` 默认 `min/max_pixels=200704`(**别加 flag 覆盖**) | 这是复现我们 0.839 的 paper-eval 配置 |
| **训练/AR 格式分辨率** | `prep_train_jax.py` 内部固定 `784 / 50176 → 168 image tokens`(pixel 672×1176) | v2 AR / 训练格式;**与上面那条是两条独立的路,别交叉** |
| **fp32 信任 bar** | `jax_batch_inference.py --fp32`(脚本内自动 `matmul_precision=highest`) | fp32 跨 backend **位等**;bf16 是默认+诊断 |
| **bf16 诊断** | 不加 `--fp32`(默认) | section-diffusion 的 conf>0.9 unmask 在 bf16 会翻、早块级联(~1.5% token),**设计性分歧**,不是错 |
| **golden(479 rated val)** | JAX bf16: **ADE@3s 0.839 / @5s 2.072 / RFS 7.929**;PyTorch: 0.814/1.990/7.914 | §2B 对照基线 |
| **metric GT** | `rated_val_gt.pkl`(已 ship,0.55MB,复现官方 metric 到 ~1e-9) | 不需要 226GB 整 val |
| **MASK/markers** | MASK=151665,im_end=151645,image_pad=151655,assistant 三元 151644/77091/198 | checker 内部用,仅供排错参考 |

> **数字对不上时,99% 是这两个原因之一:** (1) 分辨率交叉(eval 误用了 168 或 AR 误用了 200704);
> (2) 精度没 pin(bf16 vs fp32 比)。先查这两个,再查别的。

---

## §0 一次性:env + 确认 artifacts 在 CNS

```bash
source ~/.fastddrive_env          # 给出 FORK / DDRIVE / DATA_ROOT / SRC(STEP 1 写的)

# 本 STEP 新增的两个 env(脚本读它们定位 repo 和发布快照,STEP 1 的脚本已参数化、无需改代码):
export FASTDDRIVE_ROOT="$(dirname "$DDRIVE")"            # bundle 根:含 jax_ddrive / fast_ddrive / maxtext-dlm-fork
export FASTDDRIVE_REPO="$DDRIVE"                          # = .../jax_ddrive
export FASTDDRIVE_SNAP="$DATA_ROOT/release_fast_ddrive_snapshot"   # 发布的 NVIDIA fp32 ckpt(STEP 0 ship)
export PY="<你的 python>"   # 必须能 import: torch, transformers, tensorflow, waymo_open_dataset(+编译好的
                            # end_to_end_driving_data_pb2), jax(+TPU), flax, array_record, ml_dtypes
export WORK="$DATA_ROOT/step3_work"   # 本 STEP 的中间产物落这(可 glob 的 CNS/本地 POSIX 路径)
mkdir -p "$WORK"

# 首次跑本 STEP:把 STEP 0 发布的 3 样 artifact 从 GCS 拉到 CNS(和 STEP 1 拉代码同样用 gcloud):
mkdir -p "$DATA_ROOT/eval"
gcloud storage rsync -r "$SRC/release_fast_ddrive_snapshot" "$FASTDDRIVE_SNAP"
gcloud storage cp "$SRC/eval/val_rated_479.tfrecord" "$DATA_ROOT/eval/val_rated_479.tfrecord"
gcloud storage cp "$SRC/eval/rated_val_gt.pkl"       "$DATA_ROOT/eval/rated_val_gt.pkl"

# 确认 3 样东西都在(缺了就回 STEP 0 让 owner 发布):
ls -d "$FASTDDRIVE_SNAP"                                  # 发布快照
ls -l "$DATA_ROOT/eval/val_rated_479.tfrecord"           # 479 rated val 子集(~1.1GB)
ls -l "$DATA_ROOT/eval/rated_val_gt.pkl"                  # GT pkl(~0.55MB)
```

`fast_ddrive` 在 bundle 里(STEP 0 已含),所以 `convert_wod_e2e.py` / `evaluate_waymo_metrics.py` 都在
`$FASTDDRIVE_ROOT/fast_ddrive/`。

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

## §2 VAL 环控 —— 一次跑通同时满足 item 1 和 item 4(已知答案的对照组)

先把 479 rated val 转成 JSON+图片(§2A、§2B 共用这一步,带 `--with_target`):

```bash
$PY "$FASTDDRIVE_ROOT/fast_ddrive/data/convert_wod_e2e.py" \
   --tfrecords "$DATA_ROOT/eval/val_rated_479.tfrecord" \
   --out_json  "$WORK/val.json" \
   --image_root "$WORK/val_images" \
   --rated_only --with_target
# 期望:写出 ~479 条样本 + 前视 JPEG。
```

### §2A — item 1:处理链跑通 + 格式正确(训练/AR 格式,168 tokens)

```bash
# JSON+图片 → per-sample npz(torch+HF processor,内部固定 784/50176→168）
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

### §2B — item 4:发布 ckpt 推理 + 官方 metric(eval 格式,200704)

```bash
# JSON+图片 → eval 输入 npz(默认 200704,别加 --min/max_pixels）
$PY "$DDRIVE/eval/prep_jax_eval.py" \
   --eval_json "$WORK/val.json" --image_root "$WORK/val_images" --out_dir "$WORK/prep_val"

# 推理:先 bf16(对我们 bf16 golden),再 fp32(信任 bar)。在 TPU 上跑。
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  $PY "$DDRIVE/eval/jax_batch_inference.py" --prep_dir "$WORK/prep_val" --out_dir "$WORK/jax_val_bf16"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  $PY "$DDRIVE/eval/jax_batch_inference.py" --prep_dir "$WORK/prep_val" --out_dir "$WORK/jax_val_fp32" --fp32

# 官方 metric(用 ship 进来的 pkl GT，不需要整 val):
for tag in bf16 fp32; do
  $PY "$FASTDDRIVE_ROOT/fast_ddrive/eval/evaluate_waymo_metrics.py" \
     --pred_json "$WORK/jax_val_${tag}/predictions.json" \
     --gt "$DATA_ROOT/eval/rated_val_gt.pkl" \
     --output_dir "$WORK/metric_${tag}"
  echo "== $tag =="; cat "$WORK/metric_${tag}/waymo_eval_results.json"
done
```
**期望(item 4 通过的判据):**
- **bf16** ≈ 我们的 golden **ADE@3s 0.839 / @5s 2.072 / RFS 7.929**(同精度比,应当很接近;num_samples=479)。
- **fp32** 也在同一 on-par 带(fp32 与 bf16 因 confidence-cascade 会差 ~3% ADE 量级,正常)。
- 100% parse(479/479)。
- 若 **bf16 都对不上 0.839** → 几乎一定是 §0 的分辨率/快照配错(回顶部常量框排查),**不要**当成"内部 TPU 不行"。

> **(更强但可选)真 TPU fp32 位等:** 用 fork 自包含部署采样器 `$FORK/src/maxtext/diffusion/eval_sasd/driver.py`
> (168 tokens,fp32)在几个 val 样本上跑,验证 TPU-fp32 的确定性(重跑逐 token 一致)。理论依据见
> `$DDRIVE/docs/2implementation-details/INFERENCE_DEPLOY.md` §B4(本地已证 GPU==CPU fp32 在 20 样本逐 token
> 一致 → TPU-fp32 预期 bit-for-bit 匹配)。**注**:本地那份 GPU-fp32 token golden 在 owner 的 `scripts/temp/`,
> **默认不随 bundle 发**;若要做真 TPU↔GPU 的逐 token 对照,需 owner 另发那个 golden。深度信任检查,item 4 不强制。

---

## §3 用我们的方法处理新 split(item 3)

新 split 是同 proto → S1 直接复用,只换输入路径(与 §2A 同一条链,带 `--with_target`):

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
- §2A / §3 的 `CHECK_PASS failures=N`。
- §2B 的 `waymo_eval_results.json`(bf16 + fp32 的 ADE_3s/ADE_5s/RFS/num_samples)对照 golden。
- (可选)§2B 注里的 eval_sasd 20 样本 token 位等结果。

> 这 4 步全部在内部环境内完成(数据处理 + 推理 + 官方 metric);除标量验证数字外无数据带出。
> 新 split 的 v2 AR(`$WORK/new_ar`)留在 CNS,可直接喂后续训练(`test_training.md` 的 grain 路径,
> `dataset_type=waymo_sasd`,指向该 AR 目录)。
