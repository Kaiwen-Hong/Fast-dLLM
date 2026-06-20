# STEP 3 — 数据处理 sanity + 格式检查 + 新 split 处理（pixels-only / 720）

*Last updated: 2026-06-20 — changelog: [`updates-latest-0616.md`](updates-latest-0616.md)（history/ 是归档）。
推理 / 官方 ADE/RFS 在姊妹 track [`../7for_internal_inference/00_run_inference.md`](../7for_internal_inference/00_run_inference.md)。*

> **PREREQ:** STEP 0(owner 已按 [`00_owner_publish.md`](00_owner_publish.md) 发布代码 + 本 STEP 的
> artifacts 到 GCS)→ STEP 1(你已按 [`transfer-codebase.md`](transfer-codebase.md) 拉代码并
> `source ~/.fastddrive_env`)。
>
> **怎么用:** 把这一整篇**贴给内部 coding agent**。它是一条**独立 track**,目标是 4 件事:
> ① 证明我们的 pixels-only 数据处理链能在内部跑通 ② 有脚本能检查产出的 AR 格式 ③ 证明
> train-prep 与 eval-prep 在同分辨率下逐字节一致(训练/推理不漂移) ④ 用我们的方法处理一个新的
> WOD-E2E split。
>
> **本篇 100% 是 canonical 的 pixels-only / 720 路径。** 旧的 168-token / 内嵌 image_embeds 路径**已退役**,
> 本篇不再涉及——任何地方看到 168 / image_embeds 都是过时材料,别照抄。
>
> **要在 val 上做推理 / 复现官方 ADE/RFS?** → 见 [`../7for_internal_inference/00_run_inference.md`](../7for_internal_inference/00_run_inference.md)(姊妹 track)。

---

## 📌 钉死的常量(CANONICAL — pixels-only / 720;不要改,改了就静默出错)

| 项 | 值 | 为什么 |
|---|---|---|
| **分辨率 / pixels** | `prep_train_jax.py` 默认 `min_pixels=max_pixels=200704` → **720 image tokens** | 原始 Fast-dDrive paper-res;这就是 canonical 训练/AR 分辨率 |
| **统一形状** | `L=1856`、`pixel_values(2880,1176) float16`、`image_grid_thw(3,3) int64` = 三行 `[1,32,30]` | 同 proto 的子集应 100% uniform |
| **image tokens** | `N = sum(t*h*w) = 3*960 = 2880` = pixel_values 行数;`image tokens = N//4 = 720` | 每图 `32*30//4 = 240`,3 图 3 段共 720 个 `IMAGE_PAD` |
| **EXP_BUDGET** | `192` = `block_length×6`(6 blocks) | **explanation 段**固定 NULL-pad 预算:`prep_train_jax.process_gpt` 把 `obj['explanation']` pad 到 192 token,保证 train/infer scaffold 一致(**不是**轨迹/AR 解码预算) |
| **NO image_embeds** | pixels-only:AR 里**没有** image_embeds 字段 | 模型在 graph 内重算 ViT embeds(可训练 ViT 路径) |
| **base arrays** | 12 个数组 + 3 个 sidecar 标量键(`sample_id` 非空 str、`L`==1856、`n_blocks`) | 见下方 schema 框 |

> 推理 / 官方 metric 的常量(paper-res、fp32/bf16、golden、GT pkl)在
> [`../7for_internal_inference/00_run_inference.md`](../7for_internal_inference/00_run_inference.md) 顶部——**本篇不重复**。

### pixels-only / 720 AR schema(产出契约,checker 校验这个)

12 个 base 数组(NO image_embeds),`L=1856`:

```
input_ids       (L,)        int64     -> L=1856
labels          (L,)        int64
rbi             (L,)        int32
turn            (L,)        int32
scaffold        (L,)        bool
weight_vec      (L,)        float32
block_alpha     (n_blocks,) float32   -> n_blocks 标量,非固定形状(本样本 11)
block_beta      (n_blocks,) float32
position_ids    (3, L)      int32     -> (3,1856)
vision_mask     (L,)        bool
pixel_values    (N, 1176)   float16   -> (2880,1176);N = sum(t*h*w) over grid rows
image_grid_thw  (3, 3)      int64     -> 行 [[1,32,30],[1,32,30],[1,32,30]]
```

外加 `to_example()` 写的 3 个标量/字符串 sidecar 键:`sample_id`(非空 str)、`L`(==1856)、`n_blocks`(本样本 11,按内容变化)。
派生量(已核对):grid = 三张 `[1,32,30]`;`N = 2880` = pixel_values 行;`image tokens = N//4 = 720`;
每图 image_pad run 长 = `32*30//4 = 240`,3 段 240 求和 = 720;input_ids 里 IMAGE_PAD 总数 = 720;`L%32==0`。
dtype 与旧 schema 完全一致——**只有 SHAPE 和 token 数不同**,且**没有** image_embeds。

---

## §0 一次性:env + 拉 artifacts(只要 HF processor 快照,不要 ViT/embeds)

```bash
source ~/.fastddrive_env          # 给出 FORK / DDRIVE / DATA_ROOT / SRC(STEP 1 写的)

# 本 STEP 用到的 env(脚本读它们定位 repo 和快照,STEP 1 的脚本已参数化、无需改代码):
export FASTDDRIVE_ROOT="$(dirname "$DDRIVE")"            # bundle 根:含 jax_ddrive / fast_ddrive / maxtext-dlm-fork
export FASTDDRIVE_REPO="$DDRIVE"                          # = .../jax_ddrive(checker / prep 读它做 import 根)
export FASTDDRIVE_SNAP="$DATA_ROOT/release_fast_ddrive_snapshot"   # 给 prep 的 HF processor 用
                                                          #   pixels-only:只需 processor,不取 ViT、不存 embeds
export PY="<你的 python>"   # 必须能 import: torch, transformers, jax, flax, array_record, ml_dtypes, pyarrow
                            #   (新 split 探测/转换;不需要 tf+waymo——那是 7 的 metric 才要)
export WORK="$DATA_ROOT/step3_work"   # 本 STEP 的中间产物落这(可 glob 的 CNS/本地 POSIX 路径)
mkdir -p "$WORK"

# 首次跑本 STEP:把 STEP 0 发布的快照 + val 子集从 GCS 拉到 CNS:
mkdir -p "$DATA_ROOT/eval"
gcloud storage rsync -r "$SRC/release_fast_ddrive_snapshot" "$FASTDDRIVE_SNAP"
gcloud storage cp "$SRC/eval/val_rated_479.tfrecord" "$DATA_ROOT/eval/val_rated_479.tfrecord"

# 确认都在:
ls -d "$FASTDDRIVE_SNAP"                                  # 快照(prep 的 HF processor)
ls -l "$DATA_ROOT/eval/val_rated_479.tfrecord"           # rated val 子集(已知答案数据)
```

`fast_ddrive` 在 bundle 里(STEP 0 已含),所以 `convert_wod_e2e.py` 在 `$FASTDDRIVE_ROOT/fast_ddrive/`。
pixels-only 路径**不需要 ViT、不生成 image_embeds**——快照只用于 HF processor(tokenizer + image_processor)。

---

## §1 PROBE 原始数据(先看清"这数据到底是什么",秒级、只读)

新 / val 的原始 tfrecords 已在 CNS(同 WOD-E2E `E2EDFrame` proto)。`probe_dataset.py` 的
`--raw` / `--processed` 是**互斥必选**——这里用 `--raw`(autovla env 解析前 N 帧 proto):

```bash
export NEW_SPLIT='<新 split 的 CNS tfrecord glob,如 $DATA_ROOT/new_split/*.tfrecord*>'

$PY "$DDRIVE/scripts/probe_dataset.py" --raw "$NEW_SPLIT" -n 3
```
**期望**:打印 `total records`、每帧 `context.name present: True`、`#images`、`intent`、`len(past/future_states)`、
`rated`,结尾 `PROBE_DONE`。若字段/计数与预期不符(例如不是同一 proto),**停下来报告**,不要往下处理。

---

## §2 VAL 处理 sanity:pixels-only 全链跑通 + 格式正确(已知答案对照组,2 个 val 样本)

把 val 转成 JSON+图片(这里**只取 2 个样本**做 sanity:`--max_frames 2`),再跑整条 pixels-only AR 链
(prep 默认 200704 → 720 tokens),最后 checker 校验。AR 是**逐 parquet shard** 转的:
`parquet_file_to_tfexample_ar.py` 是**无 argparse、两个位置参数**的 per-file worker
(`argv[1]`=单个 parquet shard、`argv[2]`=输出 arrayrecord;原子写 `.tmp -> os.replace`,跑在 CPU)。

```bash
# ① convert(带 target,rated only;只取 2 帧做 sanity)
$PY "$FASTDDRIVE_ROOT/fast_ddrive/data/convert_wod_e2e.py" \
   --tfrecords "$DATA_ROOT/eval/val_rated_479.tfrecord" \
   --out_json  "$WORK/val.json" --image_root "$WORK/val_images" \
   --rated_only --with_target --max_frames 2

# ② JSON+图片 → per-sample npz(torch + HF processor;默认 min=max=200704 → 720 tokens)
$PY "$DDRIVE/eval/prep_train_jax.py" \
   --train_json "$WORK/val.json" --image_root "$WORK/val_images" --out_dir "$WORK/val_npz"

# ③ npz → parquet
$PY "$DDRIVE/ddrive_jax/convert/prep_to_parquet.py" \
   --npz_dir "$WORK/val_npz" --out_dir "$WORK/val_pq" --split val

# ④ 单个 parquet shard → 一个 ArrayRecord shard(pixels-only;NO embeds;CPU)
#    per-file worker:一个 parquet shard 进、一个 arrayrecord 出。2 样本一个 shard 即可:
CUDA_VISIBLE_DEVICES= $PY "$DDRIVE/scripts/parquet_file_to_tfexample_ar.py" \
   "$WORK/val_pq/val-00000-of-00001.parquet" "$WORK/val_ar/val-00000-of-00001.arrayrecord"

# ⑤ 格式校验(checker,pixels-only / 720 期望)→ 期望最后一行 CHECK_PASS failures=0
$PY "$DDRIVE/scripts/check_dataset_format.py" --dir "$WORK/val_ar" --expect pixels --split val --samples 2
```
**期望**:`CHECK_PASS failures=0`。checker 校验的是上面 pixels-only / 720 schema(`L=1856`、
`pixel_values(2880,1176)`、`image_grid_thw` 三行 `[1,32,30]`、image_pad 3 段各 240 共 720、`n_blocks==block_alpha/beta`
长度、`L%32==0`、所有 `pixel_values` finite、`sample_id` 非空;**且不含 image_embeds**)。
这就证明了:整条 convert→prep→parquet→AR 处理链在内部装得上、跑得通、产出 schema 正确的 pixels-only 数据。

> ④ 一整个目录(多 shard)怎么转?用并行 driver `full_to_tfexample_ar_driver.py`(无 argparse、位置参数:
> `argv[1]`=源 `train-*.parquet` 目录、`argv[2]`=目标 AR 目录、可选 `argv[3]`=workers 默认 7)——它 glob 每个
> parquet shard、各起一个上面的 worker 子进程映射成 `train-{i:05d}-of-{N:05d}.arrayrecord`,**可断点续传**(已存在的输出跳过),
> 并写 `dataset_info_train.json` sidecar。2 样本的 sanity 用单 worker 直接调即可;§4 处理大 split 才用 driver。
>
> (在 val 上**推理**并复现官方 ADE/RFS 是另一条 track → `../7for_internal_inference/00_run_inference.md` Track 2。)

---

## §3 prep-consistency:train-prep vs eval-prep 在同分辨率下逐字节一致(CPU-only)

证明**训练侧的 prep**(`prep_train_jax.py`)与**推理侧的 prep**(`prep_jax_eval.py`)在**同一分辨率**下产出
**逐字节相同**的视觉输入与 prompt 前缀——这是"训练/推理不漂移"的硬保证。`validate_prep_consistency.py`
会**同时跑两个 prep**(同 toy json、同 `--pixels` = min = max)并断言 `pixel_values` 逐字节一致、
`image_grid_thw` 一致、prompt-token 前缀一致。**纯 CPU,不需要模型/TPU。**

```bash
# 用 §2 已生成的 val.json 当 toy(或 fast_ddrive 自带 example);pixels=200704 = canonical 720-tok 分辨率
/home/kaiwen/miniconda3/envs/ddrive/bin/python "$DDRIVE/scripts/validate_prep_consistency.py" \
   --json "$WORK/val.json" --image_root "$WORK/val_images" --pixels 200704 --n 2
```
**期望**:逐样本打印 `pixels_equal=True`、`grid_equal=True`、`prompt_lcp_tokens=...`,结尾 `PREP_CONSISTENCY_PASS`。
`--pixels 200704` 把 min=max 都设成 canonical paper-res(720 tokens),两个 prep 用同一 HF processor + 同图,
所以视觉输入应逐字节相等。若打印 `PREP_CONSISTENCY_FAIL`,说明两条 prep 路径已漂移 → **停下来报告**,别拿这数据去训。

> 推荐解释器 `/home/kaiwen/miniconda3/envs/ddrive/bin/python`(脚本会 spawn 两个 prep 子进程,需 torch+transformers)。

---

## §4 用我们的方法处理一个新 split(同 proto → 复用整条链,新输入路径,带 `--with_target`)

新 split 是同 proto → §2 整条 pixels-only 链直接复用,只换输入路径。这里转**整个 split**,所以 AR
用并行 driver(目录级):

```bash
# ① convert(整 split;有 GT 才训得了 → --with_target;纯推理集可去掉)
$PY "$FASTDDRIVE_ROOT/fast_ddrive/data/convert_wod_e2e.py" \
   --tfrecords "$NEW_SPLIT" --out_json "$WORK/new.json" \
   --image_root "$WORK/new_images" --with_target

# ② prep(默认 200704 → 720 tokens)
$PY "$DDRIVE/eval/prep_train_jax.py" \
   --train_json "$WORK/new.json" --image_root "$WORK/new_images" --out_dir "$WORK/new_npz"

# ③ npz → parquet(多 shard;--shard_size 默认 64)
$PY "$DDRIVE/ddrive_jax/convert/prep_to_parquet.py" \
   --npz_dir "$WORK/new_npz" --out_dir "$WORK/new_pq" --split train

# ④ 目录级并行转 AR(driver:源 parquet 目录 → 目标 AR 目录;7 workers;可续传;写 dataset_info_train.json)
$PY "$DDRIVE/scripts/full_to_tfexample_ar_driver.py" "$WORK/new_pq" "$WORK/new_ar" 7

# ⑤ 校验产出格式(pixels-only / 720)→ 期望 CHECK_PASS failures=0
$PY "$DDRIVE/scripts/check_dataset_format.py" --dir "$WORK/new_ar" --expect pixels --split train --samples 16
```
**期望**:`CHECK_PASS failures=0`。可选再 `probe_dataset.py --processed "$WORK/new_ar"`(自动识别 `.arrayrecord`)
dump 第一条记录,确认 `L=1856`、`pixel_values(2880,1176)`、`image_grid_thw` 三行 `[1,32,30]`、**无 image_embeds**。
新 split(同 proto)应当 100% uniform;若 checker 报非均匀 `L`,说明该 split 形态与 WOD-E2E 子集不同 → **报告**
(可能要走变长 padding 路,目前未启用)。`--expect pixels` 会硬断言**无 image_embeds**(pixels-only)。
> ⚠️ `image_grid_thw` 也必须 uniform `[1,32,30]`:可训练 in-graph ViT 用**编译期固定 grid**(`SASD_GRID_THW`)。WOD-E2E 是固定前视相机、200704 下恒为 `[1,32,30]`;若新 split 图像尺寸不同 → grid 不同 → 固定-grid ViT 不匹配,也要**报告**。

---

## §5 收尾 / 回报(只出标量,符合内部数据外流策略)

把这些**标量**回报:
- §1 probe 的 `total records` + 字段是否符合预期(布尔)。
- §2 的 `CHECK_PASS failures=N`(val sanity)。
- §3 的 `PREP_CONSISTENCY_PASS` / `FAIL` + 逐样本 `pixels_equal` / `grid_equal` / `prompt_lcp_tokens`。
- §4 的 `CHECK_PASS failures=N`(新 split)。

> 这几步全部在内部环境内完成(数据处理),除上述标量验证数字外**无数据带出**。
> 新 split 的 pixels-only AR(`$WORK/new_ar`)留在 CNS,可直接喂后续训练(`test_training.md` 的 grain 路径,
> 指向该 AR 目录);要在 val 上推理见 `../7for_internal_inference/00_run_inference.md`。
