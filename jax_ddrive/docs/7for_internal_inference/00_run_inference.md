# STEP I — 内部 TPU 推理 / eval（B2 自包含部署 + 官方 ADE/RFS）

*Last updated: 2026-06-18 — changelog: [`updates-latest-0617.md`](updates-latest-0617.md).*

> **PREREQ（共享 6for_internal 的 bootstrap）:** 先 STEP 0（owner 按 [`../6for_internal/00_owner_publish.md`](../6for_internal/00_owner_publish.md)
> 发布到 GCS）→ STEP 1（内部按 [`../6for_internal/transfer-codebase.md`](../6for_internal/transfer-codebase.md) 拉代码、写
> `~/.fastddrive_env`）。**STEP 1 只写 env,不建 venv、不 ingest 数据**(那本来在 STEP 2);**没跑过 STEP 2 的纯推理路径**,本篇 §0 会幂等地现建 venv + 拉模型快照。**本篇是 STEP 2（训练）的姊妹 track**——拿一个**已有的 ckpt**（from-base 训练导出 或
> 发布 NVIDIA ckpt）在内部 TPU 上做推理 / eval。
>
> **怎么用:** 把这一整篇**贴给内部 coding agent**。
>
> **设计 SSOT（为什么/tested-vs-pending）:** [`../2implementation-details/INFERENCE_DEPLOY.md`](../2implementation-details/INFERENCE_DEPLOY.md)（B1 导出 / B2 采样器 / 嵌入 parity / B4 数值）。**本篇只讲怎么跑。**

---

## 两个 track（按目标选一个或都跑）

| | **Track 1 — B2 自包含部署推理** | **Track 2 — 官方 ADE/RFS** |
|---|---|---|
| 工具 | fork 的 `eval_sasd/driver.py`（自包含、零 ddrive_jax） | `jax_batch_inference.py` + `evaluate_waymo_metrics.py` |
| 分辨率 | **168 tokens**（784/50176，部署） | **200704**（paper-eval） |
| 出什么 | `SASD_EVAL_PASS` + 逐样本标量（valid_json/traj_exact/co_match/fmb_match）+ `EMBED_PARITY_PASS` | `ADE@3s / @5s / RFS` |
| egress | ✅ **只出标量 vlog,部署安全** | predictions.json 留内部,只出标量 metric |
| 跑哪个 env | `~/venv`（jax+flax.nnx,STEP 1 建的） | **也用 `~/venv`**（jax）+ numpy —— prep_val 由 owner 预烤,内部**零 convert/proto/tf/torch**,不需要 `$PY` |
| 何时用 | 部署 / 逐样本可信度 / egress-safe | 要**官方 ADE/RFS 数字**（复现 0.839 或报模型自己的分） |

两个 track 都支持 **Mode D**（评测 from-base 训练并导出的模型）和 **Mode R**（发布 NVIDIA ckpt）——只差 `--snapshot`。

---

## 📌 钉死的常量（CANONICAL — 不要改,改了就静默出错）

| 项 | 值 | 备注 |
|---|---|---|
| **分辨率** | Track1 driver = **168**（784/50176）；Track2 metric = **200704** | **两条路别交叉**；用错→mRoPE/结构错位→**静默乱码** |
| **精度** | **fp32 = 信任 bar**（跨 backend 位等）；**bf16 = 诊断**（conf-cascade 设计性分歧,~1.5% token） | 两个都跑、都记（内部 TPU 充裕） |
| **threshold** | `0.9`（driver/sampler 默认） | confidence-unmask 阈值 |
| **PYTHONPATH** | Track1 一律 `PYTHONPATH=$FORK/src` | ⚠️ driver/parity 的 **docstring 里写的是旧 `jax-dlm-baseline` 路径,别照抄**——用 `$FORK/src` |
| **egress** | 只出 `validation_log.jsonl` 标量/布尔/计数；**driver 禁用 `--show_text`** | `--show_text` 会打印原文,违反内部外流策略（本地 debug 才用） |
| **snapshot** | Mode D=`$DATA_ROOT/overfit400_base_hf`（from-base 导出）；Mode R=`$DATA_ROOT/release_fast_ddrive_snapshot` | |
| **eval_inputs ↔ ViT** | eval_inputs 的 `image_embeds` 必须与 `--snapshot` 的 **ViT 同源** | Mode D=**base** ViT；Mode R=**release** ViT。parity 的 `fp32_vs_reference` 门强制这点（不匹配→FAIL）。⚠️ **现 GCS `eval_inputs`=base-ViT(只配 Mode D)**;Mode R 跑 Track 1 parity 会 `cosine≈0.54 FAIL`（base/release ViT 不同,**预期,非 TPU 问题**）→ Mode R 复现 ADE/RFS 走 **Track 2**（prep_val 与 ViT 无关,见下） |
| **导出坑** | 训练 ckpt 导出**绝不加** `verify_against` | 它做逐位 round-trip,训练后 434 text 权重变了 → 会假 `B1_ROUNDTRIP_FAIL`（仅 BASE ckpt 才加） |
| **metric flag** | `--gt`（接 `.pkl` 或 tfrecord glob） | ⚠️ docstring 的 `--gt_tfrecords`/`--gt_dict_pkl` 是**过时的**,会报错 |

---

## §0 一次性 env

```bash
source ~/.fastddrive_env        # FORK / DDRIVE / DATA_ROOT / SRC（STEP 1 写的）
export FASTDDRIVE_ROOT="$(dirname "$DDRIVE")"   # bundle 根（含 fast_ddrive —— Track2 的 metric 在这）

# ① venv（jax + flax.nnx,采样器是 NNX,必需）。STEP 1 不建 venv —— 没跑过 STEP 2 就在这里幂等现建:
if [ ! -d ~/venv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
  uv venv -p 3.11 ~/venv
  uv pip install -q -r "$FORK/src/dependencies/requirements/generated_requirements/tpu-requirements.txt"
  uv pip install -q safetensors pyarrow transformers ml_dtypes flax     # flax/nnx 必需
fi
source ~/venv/bin/activate

# ② 模型快照 → CNS（幂等;Mode D 用 STEP 2 训练+导出的,本来就在 CNS,可跳过这步）:
[ -d "$DATA_ROOT/release_fast_ddrive_snapshot" ] || \
  gcloud storage rsync -r "$SRC/release_fast_ddrive_snapshot" "$DATA_ROOT/release_fast_ddrive_snapshot"   # Mode R 的模型
[ -d "$DATA_ROOT/base_qwen25vl_3b_snapshot" ] || \
  gcloud storage rsync -r "$SRC/base_qwen25vl_3b_snapshot" "$DATA_ROOT/base_qwen25vl_3b_snapshot"         # tokenizer/decode + Mode D 的 ref

# ③ Track 2（官方 ADE/RFS）—— prep_val 由 owner 预烤,内部**只用上面的 ~/venv（jax）+ numpy**:零 tf、零
#    waymo/proto、零 torch、零 convert,**没有 $PY 这回事了**（旧文档要的 tf+waymo+编译 proto 全是 owner
#    侧预烤时用的,内部不需要）。把预烤好的 prep_val_full(479 帧,与 ViT 无关,只含 pixel+text) + GT pkl 拉到 CNS:
mkdir -p "$DATA_ROOT/eval"
[ -d "$DATA_ROOT/eval/prep_val_full" ] || \
  gcloud storage rsync -r "$SRC/eval/prep_val_full" "$DATA_ROOT/eval/prep_val_full"     # 479 npz, ~3.1GB
[ -f "$DATA_ROOT/eval/rated_val_gt.pkl" ] || \
  gcloud storage cp "$SRC/eval/rated_val_gt.pkl" "$DATA_ROOT/eval/rated_val_gt.pkl"      # 0.55MB
```

---

# Track 1 — B2 自包含部署推理（egress-safe）

## §T1.0 自包含 sanity（跑 Track 1 前先确认 B2 零 ddrive_jax）

```bash
PYTHONPATH=$FORK/src JAX_PLATFORMS=cpu python -m maxtext.diffusion.tests.eval_sasd_import_test
# → EVAL_SASD_SELFCONTAINED_PASS
```

## §T1.1 准备 eval-inputs（Track 1 driver 用；owner 离线产、含 frozen-ViT image_embeds）

eval-inputs 是**离线**在 torch host 上产的 per-sample npz（含预烤 frozen-ViT `image_embeds`,168-res,内部 TPU 不跑 ViT）。**embeds 与 `--snapshot` 的 ViT 同源是硬约束**（§T1.3 parity 门强制；不同源→`cosine≈0.54 FAIL`）。

- **Mode D（from-base,base-ViT）**——GCS 上已发布的 `eval_inputs`(20 val + 20 train) **就是 base-ViT 烤的**,直接拉(配 from-base 导出快照):
  ```bash
  gcloud storage cp -r "$SRC/eval_inputs" "$DATA_ROOT/eval_inputs"   # base-ViT；20 val + 20 train
  ```
- **Mode R（发布 ckpt,release-ViT）**——⚠️ **release-ViT 的 eval_inputs 当前没发布**:
  - **别**把上面 base-ViT 的 `eval_inputs` 喂给 Mode R 跑 §T1.3 parity —— 会 `fp32_vs_reference cosine≈0.54 FAIL`（base/release ViT 不同,**预期,不是 TPU 问题**）。
  - **要在 Mode R 复现 ADE/RFS → 直接跳到 Track 2**（prep_val 与 ViT 无关,内部加载 release ViT,proto-free,见下）。Track 1 的 driver 只产逐样本标量,不产 ADE/RFS。
  - 若**确实**要 Mode R 的 Track-1 driver 标量,需 owner 用 **release** snapshot 离线重烤 eval_inputs 再 ship（找 owner;命令同下,`--snapshot` 换成 release）。

  <owner 离线重烤参考（torch host,不是 TPU;`--snapshot` 决定 ViT 同源；循环 idx 取多个样本）:>
  ```bash
  PYTHONPATH=$DDRIVE python $FORK/scripts/prep_jax_eval_inputs.py \
    --sample <WOD targets JSON,如 train_targets_distilled_400.json> \
    --img_dir <对应相机 JPEG 目录> \
    --snapshot <base 或 release snapshot —— 决定 ViT 同源> \
    --out $DATA_ROOT/eval_inputs/val_s0.npz --idx 0 --with_embeds
  # 默认 min/max_pixels=784/50176 → 168 image tokens（别覆盖!）;结尾 SASD_PREP_INPUTS_DONE
  ```

## §T1.2 （Mode D only）B1 导出 from-base ckpt → bf16 HF 快照

Mode R 跳过本步（直接用发布快照）。详细背景见 `../6for_internal/test_training.md` STAGE 2。

```bash
STEP=30000                       # 想导出的那个 checkpoint
cd $FORK
PYTHONPATH=$FORK/src JAX_PLATFORMS=cpu \
python scripts/maxtext_to_hf_export.py src/maxtext/configs/sasd_waymo.yml model_name=qwen2.5-3b \
  param_ckpt_dir=$DATA_ROOT/run_overfit400_base/overfit400-base/checkpoints/$STEP/items \
  ref_snapshot=$DATA_ROOT/base_qwen25vl_3b_snapshot \
  out_dir=$DATA_ROOT/overfit400_base_hf
# → MAXTEXT_TO_HF_EXPORT_DONE（824 keys = 434 text + 390 vision,bf16,lm_head omit）
# ⚠️ 训练 ckpt 绝不加 verify_against（见顶部"导出坑"）
```

## §T1.3 embedding-parity 门（确认 embeds 与 ViT 同源）

```bash
SNAP=$DATA_ROOT/overfit400_base_hf       # Mode R: SNAP=$DATA_ROOT/release_fast_ddrive_snapshot
PYTHONPATH=$FORK/src python -m maxtext.diffusion.eval_sasd.embedding_parity \
  --npz $DATA_ROOT/eval_inputs/val_s0.npz --snapshot $SNAP \
  --vlog validation_log.jsonl --run_id infer-r1
# → EMBED_PARITY_PASS（gated on fp32_vs_reference: cosine ≥ 0.999）;bf16-ViT 那组数是诊断
# 若 FAIL: 多半是 eval_inputs 的 ViT 与 --snapshot 不同源（Mode D 用了 release-ViT 的 npz 等）
```

## §T1.4 run_eval：fp32 + bf16（逐样本）

```bash
for s in val_s0 val_s1 val_s2; do          # 内部 TPU 充裕,样本随意扩
  for dt in fp32 bf16; do                   # fp32 = 信任 bar;bf16 = 诊断;两个都记
    PYTHONPATH=$FORK/src python -m maxtext.diffusion.eval_sasd.driver \
      --npz $DATA_ROOT/eval_inputs/$s.npz --snapshot $SNAP --dtype $dt \
      --tokenizer $DATA_ROOT/base_qwen25vl_3b_snapshot \
      --vlog validation_log.jsonl --run_id infer-r1 --sample $s
  done
done
# → 每次打印 SASD_EVAL_PASS (valid JSON + 5-waypoint trajectory)
# 核对 metrics 字典: n_image_tokens=168, n_mask_remaining=0;
#   npz 带 target_ids 时还有 traj_exact / traj_max_abs_delta / co_match / fmb_match
# ⚠️ 不要加 --show_text（会打印原文,违反外流策略）
```

**Track 1 通过判据:** 每样本 `SASD_EVAL_PASS`、`n_image_tokens=168`、`n_mask_remaining=0`;fp32 的 `traj_max_abs_delta`（vs target_ids）小;`EMBED_PARITY_PASS`。只有 `validation_log.jsonl` 的标量离开内部。

---

# Track 2 — 官方 ADE/RFS on val（metric 交叉核对）

> 要**官方 ADE/RFS 数字**才用这个（Track 1 不产 ADE/RFS）。**prep_val 由 owner 预烤,内部全程 `~/venv`（jax）+ numpy**:
> 零 convert、零 proto、零 tf、零 torch、**不需要 `$PY`**。前提:§0 ③ 已把 `prep_val_full` + `rated_val_gt.pkl` 拉到 CNS。
>
> 为什么不要 proto/tf:① `prep_val_full` 是 owner 预烤的 479 帧 `pixel_values`+text(**与 ViT 无关** —— `jax_batch_inference`
> 自己在 TPU 上加载 `FASTDDRIVE_SNAP` 的 ViT),所以**跳过了** convert(解析 tfrecord 才需 proto)+ prep;② metric 用 `--gt <pkl>`
> 走纯 numpy 路径(`evaluate_waymo_metrics` 已把 tf+proto 改成**惰性 import**,只有 tfrecord-GT 分支才触发,pkl 分支不碰)。

```bash
source ~/venv/bin/activate
WORK=$DATA_ROOT/infer_work; mkdir -p "$WORK"

# snapshot：Mode R(复现 0.839)用 release；Mode D 换成 from-base 导出快照。
export FASTDDRIVE_SNAP=$DATA_ROOT/release_fast_ddrive_snapshot   # Mode D: =$DATA_ROOT/overfit400_base_hf
PREP=$DATA_ROOT/eval/prep_val_full          # owner 预烤的 479 帧（pixel+text,ViT 无关）
GT=$DATA_ROOT/eval/rated_val_gt.pkl

# 1) 推理 bf16 + fp32（jax_batch_inference 自己加载 FASTDDRIVE_SNAP 的 text+ViT,在 TPU 上跑 ViT）
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python "$DDRIVE/eval/jax_batch_inference.py" --prep_dir "$PREP" --out_dir "$WORK/jax_val_bf16"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python "$DDRIVE/eval/jax_batch_inference.py" --prep_dir "$PREP" --out_dir "$WORK/jax_val_fp32" --fp32

# 2) 官方 metric（--gt 用 pkl → 纯 numpy,不 import tf/proto；**不是** --gt_tfrecords）
for tag in bf16 fp32; do
  python "$FASTDDRIVE_ROOT/fast_ddrive/eval/evaluate_waymo_metrics.py" \
     --pred_json "$WORK/jax_val_${tag}/predictions.json" \
     --gt "$GT" --output_dir "$WORK/metric_${tag}"
  echo "== $tag =="; cat "$WORK/metric_${tag}/waymo_eval_results.json"
done
```

**Track 2 通过判据:**
- **Mode R**：bf16 ≈ golden **ADE@3s 0.839 / @5s 2.072 / RFS 7.929**（同精度比,应当很接近;`num_samples=479`,100% parse）。
- **Mode D**：报该模型自己的 ADE/RFS（没有固定 golden;和你的训练目标比）。
- **对不上时 99% 是**：(1) 分辨率交叉（误用了 168 而非 200704）；(2) 精度没 pin（bf16 vs fp32 混比）；(3) snapshot 拿错。先查这三个,**不要**当成"内部 TPU 不行"。

---

## §I.report — 回报（只出标量,符合外流策略）

`validation_log.jsonl` 里只出标量/布尔/计数/hash,**无权重/图像/原文**。最少回报:
- `EVAL_SASD_SELFCONTAINED_PASS`（§T1.0）
- `EMBED_PARITY_PASS` + `fp32_vs_reference.cosine`（§T1.3）
- 每样本 `SASD_EVAL_PASS` + `traj_exact/co_match/fmb_match` + 汇总（如 `19/20`）（§T1.4）
- （跑了 Track 2）各精度的 `ADE_3s/ADE_5s/RFS/num_samples`

> 全程在内部 TPU 内完成;除 `validation_log.jsonl` 的标量外无数据带出。
> 相关:数据处理见 `../6for_internal/03_data_processing.md`;训练 + B1 导出闭环见 `../6for_internal/test_training.md`。
