# flume_pipeline — WOD-E2E → ArrayRecord, 2-job split (doc 09 实现)

本目录是 [`docs/1plans/09_internal_flume_data_pipeline.md`](../../docs/1plans/09_internal_flume_data_pipeline.md)
里 **option A(全-raw AR + tokenize-in-Grain)** 的可运行实现 + 本地 parity 锚点。
**全部为新增、独立脚本,未改动任何被跟踪代码**(`git status` 仅显示本目录 untracked)。
本 README 是这套 harness 的 **SSOT**;数字只在这里记一处,doc 09 只链接不复制。

## 一句话

把 legacy 的 4-stage(tfrecord→JSON→npz→parquet→tf.Example-AR、跨 3 env)重构成 2-job split:

- **Job ①(ETL)** 只产 raw-bytes 的 **msgpack ArrayRecord**(无 TF / 无 torch / 无大模型库)。
- **Job ②(训练期 Grain)** 把 tokenize + 全部 SASD 结构搬到训练 host 现算 —— 即
  `prep_train_jax.py` 重打包成一个 `grain.MapTransform`。

## 文件

| 文件 | 角色 | 本地可跑? |
|---|---|---|
| `etl_record.py` | **可移植记录核**:msgpack schema v1 + `assemble_record`/`pack`/`unpack` + `build_record_from_json`(本地 JSON 前端)。零重依赖。 | ✅ |
| `etl_local.py` | **Job ① 本地替身**:读 Fast-dDrive JSON → records → 写 `.array_record`。站位真 Flume job。 | ✅ |
| `etl_flume.py` | **Job ① 真 google3 Flume 骨架**:`build_record_from_proto`(proto 字段路径)+ Beam pipeline。可 import 作文档,**本地不可跑**(proto 是 `//third_party` 内部)。 | ⛔(import-only) |
| `tokenize_sasd.py` | **Job ② 核心**:`TokenizeSASD` = `prep_train_jax.py` 重打包成 `grain.MapTransform`。13 字段张量契约。 | ✅ |
| `grain_loader.py` | **Job ② loader 接线**:`ArrayRecordDataSource` + `IndexSampler` + `TokenizeSASD` → `grain.DataLoader`;另有 `read_raw_records`(纯 array_record 读)。 | ✅ |
| `parity_test.py` | **bit-exact gate**:ETL→AR→read-back→`TokenizeSASD` 的 13 字段 == 现跑 `prep_train_jax` npz oracle;+ ArrayRecord round-trip gate。 | ✅ |
| `run_parity.sh` | 一键跑 gate,输出 sentinel `FLUME_PIPELINE_PARITY_PASS`。 | ✅ |

## 怎么跑

环境 = **ddrive**(oracle 的同款 stack:torch 2.11 + transformers 4.57.1 + trust_remote_code Qwen2.5-VL +
`section_utils`)。本目录额外装了 `msgpack`(1.2.1)、`array_record`、`grain`(0.2.18)进 ddrive —— **additive**,
未拉入 jax、torch/transformers 不变。**务必 `unset LD_LIBRARY_PATH`**。

```bash
# 一键 parity gate(2 样本 toy)
bash run_parity.sh

# 单独跑各 Job
unset LD_LIBRARY_PATH; PY=/home/kaiwen/miniconda3/envs/ddrive/bin/python
$PY etl_local.py  --train_json .../sample.json --image_root .../fast_ddrive --out /home/kaiwen/data/flume_pipeline/x.array_record
$PY grain_loader.py --ar_path /home/kaiwen/data/flume_pipeline/x.array_record   # Job ② smoke
```

## Parity 结果(2026-06-23,本地 ddrive/CPU)

`sample.json` 2 条真 WOD-E2E 样本,canonical @720:`L=1856`、`n_blocks=11`、
`pixel_values=[2880,1176] float16`(2880 = 3 图 ×960 unmerged patch;序列内 image token=720 = merge 后)。

- **GATE 1 ArrayRecord round-trip**:bytes=OK,fields=OK(sample_id/images/prompt/target 无损)。
- **GATE 2 TokenizeSASD vs `prep_train_jax` npz**:**13/13 字段 bit-exact,两样本全 PASS**
  (含 float16 `pixel_values`、train-time 展开的 `weight_vec`/`block_alpha`/`block_beta`)。
- **sentinel**:`FLUME_PIPELINE_PARITY_PASS`。
- `grain.DataLoader` 路径 smoke:`GRAIN_LOADER_SMOKE_DONE`(两样本经真 grain 管线产 13 字段)。

> 这就是 doc 09 §8 要的"只新增一个 *Grain 输出 vs npz* 的 bit-exact gate"。其余链路(NNX/MaxText/真 TPU)
> 沿用现成 `scripts/mm_step_parity/`(memory `fast-ddrive-mm-step-parity-harness`)。

## 关键设计点(实现里落实的)

1. **config 不再烤进数据**(doc 09 §2.4 修的 wart):`SECTION_W`/`NOISE_SCHED` 成 `TokenizeSASD` 构造参数
   (默认 == oracle 常量),`weight_vec`/`block_alpha`/`block_beta` **训练期从 config 展开**。改权重/噪声
   **不必重跑 ETL**。
2. **parity-safe**:record 存**原始 prompt/target 字符串**(= `conversations[*].value`,pre-`process_gpt`)
   + 3 张原始 JPEG bytes;`TokenizeSASD` 自己跑 `process_gpt`/Qwen processor。msgpack 保字符串、JPEG 同字节
   → 与 oracle 逐字节一致。
3. **单一真值源**:`TokenizeSASD` **复用** oracle 的叶子函数(`process_gpt`、
   `compute_section_block_idx_deep_static`、`get_rope_index_numpy`、`messages_from_prompt`),只测"重打包"。
4. **镜像 oracle 的 processor 细节**(parity 必需):`return_tensors="pt"` 后 `.float().numpy().astype(float16)`
   (doc 09 §4.4 写的 `"np"` 是近似);`proc=AutoProcessor(use_fast=False)` + `proc.tokenizer=tok` +
   `min_pixels==max_pixels==200704`。

## §7 待核清单 —— 本地已核 / 仍 google3-only

| § | 项 | 状态 |
|---|---|---|
| 2 | `compute_section_block_idx_deep_static` 返回 `b2s`(block→section) | ✅ 已核:返回 **5** 元(`rbi,turn,n_blocks,scaff,b2s`),见 `prep_train_jax.py:125` |
| 8 | `min/max_pixels=200704`、`EXP_BUDGET=192`、`BD=32`、labels `+2`、MASK pad | ✅ 已核:parity bit-exact 即证同源 |
| 5 | msgpack `use_bin_type=True`/`raw=False`、bytes 往返无损、`schema_version=1` | ✅ 已核:GATE 1 OK |
| 1 | proto 字段路径 `frame.frame.images[i].image` 等 | ⛔ google3-only:写在 `etl_flume.py`,需真 proto 自核 |
| 6 | google3 ArrayRecord 写 sink + CNS 路径 | ⛔ google3-only:`etl_flume._WriteToArrayRecord` 占位 |
| 7 | 离线 Qwen `/cns/.../qwen_local` 全程不触网 | ⛔ google3-only:本地用 SNAP 离线快照已等价验证加载逻辑 |
| 3/4 | `render_prompt`/`render_target` = `convert_wod_e2e` 的 TF-free 纯 python 半身 | ⛔ google3-only:`etl_flume` 里 stub;实现时从 `convert_wod_e2e` 抽共享 lib |

## google3 落地映射

| 本地 | google3 |
|---|---|
| `etl_local.py`(JSON 前端) | `etl_flume.py` 的 `build_pipeline`(proto 前端 + Flume runner) |
| `array_record.ArrayRecordWriter` | 内部 ArrayRecord Beam sink |
| SNAP 离线快照 | `/cns/.../qwen_local` 离线目录 |
| `grain.ArrayRecordDataSource`(本地 CPU) | 同 API,源指向 CNS,`worker_count>0` prefetch 喂 TPU |

## Overnight 端到端验证(2026-06-23)—— 见 [`docs/1plans/10_flume_validation_overnight.md`](../../docs/1plans/10_flume_validation_overnight.md)

新增脚本把"raw WOD-E2E → AR → 训练 + 推理"全链路在 GPU + 真 **v6e-1 TPU** 上验证:
- `convert_raw_to_ar.py`(autovla,one-script raw→AR)+ `sasd_loader_msgpack.py`/`materialize_batches.py`
  (新 dataloader → SASD batch,**13/13 bit-exact vs `prep_train_jax` oracle**,两 track 全 PASS)。
- `train_jax_driver.py`:真 3B + frozen ViT 跑 **1-step(两样本 bs=1)+ 10-step**,GPU loss 0.647/0.672。
- 推理:`ar_to_eval_json.py`→`prep_jax_eval`→`jax_batch_inference`,轨迹复现 recorded GT(首点 0.02m)。
- `tpu_validate_flume.sh`:v6e-1 queued-resource + auto-reaper(money-safe)在 TPU 上跑训练+推理。
- 交付物全部上传 GCS:`gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd/flume/deliverable/`(data/scripts/results)。

## 死掉的 legacy(被本目录取代)

`prep_to_parquet.py`、`parquet_file_to_tfexample_ar.py`、`ar_dataset.py::decode_example`(tf.Example→msgpack)、
JSON 中间格式。`prep_train_jax.py` **保留为 parity oracle**(本目录的 gate 直接调它)。
