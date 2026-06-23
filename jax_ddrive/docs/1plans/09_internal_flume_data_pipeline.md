# 09 · 内部 Flume 数据流水线:WOD-E2E → ArrayRecord(option A 全-raw + tokenize-in-Grain)

> **状态**:设计已定稿(架构决策 = option A,2026-06-23);**本地实现 + parity 已 PASS(2026-06-23)** ——
> 见 `jax_ddrive/scripts/flume_pipeline/`(独立新增脚本,SSOT = 该目录 `README.md`)。Job ②(`TokenizeSASD`
> = 本文重打包)与 msgpack/ArrayRecord 存储层在本地 ddrive/CPU 上对 `sample.json` 2 样本 **13/13 字段
> bit-exact vs `prep_train_jax` npz oracle**,`grain.DataLoader` 路径亦 smoke PASS。**仍 google3-only**:
> proto decode(`etl_flume.py` 骨架)、Flume runner、内部 AR sink、CNS/离线 Qwen 路径(见 README §7 表)。
> 本文保留为实现蓝图 + 经验记录。
> **目标读者**:实现这条流水线的人(我们自己 / 内部 google3 agent)。
> **一句话**:把 legacy 的 4-stage(tfrecord→JSON→npz→parquet→ArrayRecord、跨 3 个 env)
> 重构成 google3 钦定的 **2-job split**——① Flume ETL 只产 raw bytes 的 ArrayRecord,
> ② tokenize + 全部 SASD 结构搬进训练期的 Grain `MapTransform`。
>
> **关联**:
> - [[08_dataset_storage_720_pixels]] —— 现行 720/200704 pixels-only 数据存储方案(本文取代其
>   parquet/tf.Example 链路,但沿用 720 分辨率与字段语义)。
> - [[03_scaleup_tpu_spec]] —— TPU scale-up 总规划(本文是其"数据侧"的内部落地版)。
> - [[07_fidelity_fixes_2026-06-20]] —— 720/EXP_BUDGET=192 等保真修正(本文 Grain 侧必须延续)。
> - memory `fast-ddrive-wode2e-to-ar-internal-flume` —— Waymo 内部 agent 的 Q&A 原始结论。
> - memory `fast-ddrive-mm-step-parity-harness` —— 现成的三方 parity harness,作回归锚点。

---

## 0. North star / 为什么做这件事

我们要把 WOD-E2E 原始 tfrecord 变成 TPU 训练可直接消费的数据。现行实现是历史演进出来的
**4 个阶段、3 个互相冲突的 Python 环境**,既难一键跑、又把"调参旋钮"烤死进数据 artifact。
北极星:**一条干净、内部可跑、改 tokenize/权重/噪声/(未来)分辨率都不必重跑全量的流水线**,
并且全程能被现有 parity harness 验证为数值正确。

---

## 1. 现状(legacy 4-stage)——要被取代的东西

| 步 | 脚本 | env | 产物 | 关键依赖 |
|---|---|---|---|---|
| 1 | `fast_ddrive/data/convert_wod_e2e.py` | autovla | Fast-dDrive JSON + JPEG | `tensorflow` + `waymo-open-dataset` + 编译版 `end_to_end_driving_data_pb2` |
| 2 | `jax_ddrive/eval/prep_train_jax.py` | ddrive | per-sample **npz**(13 字段) | `torch` + `transformers`(Qwen2.5-VL)+ `trust_remote_code` 自定义 modeling(`section_utils`) |
| 3 | `jax_ddrive/ddrive_jax/convert/prep_to_parquet.py` | 任意 | **parquet** 分片 | `numpy` + `pyarrow` |
| 4 | `jax_ddrive/scripts/parquet_file_to_tfexample_ar.py` | jax | **ArrayRecord**(`tf.train.Example` 编码) | `tensorflow`(仅编码)+ `array_record` |

训练侧 loader:`jax_ddrive/ddrive_jax/data/ar_dataset.py::decode_example`(用 `tf.train.Example.FromString`
+ `tf.io.parse_tensor` 逐字段反序列化)。

### npz 的 13 个字段(= 训练步真正消费的张量契约,务必保留语义)
来自 `prep_train_jax.py:138-141` 与 `prep_to_parquet.py:ARRAY_DTYPES`:

| 字段 | dtype | shape | 含义 |
|---|---|---|---|
| `input_ids` | int64 | [L] | 序列 token(已 pad 到 `BD=32` 的整数倍,尾部用 `MASK_ID=151665`) |
| `labels` | int64 | [L] | response span 才 != -100;span = `<im_start>assistant\n` 之后到 `<im_end>` **+2**(含 im_end 和后续 `\n`) |
| `rbi` | int32 | [L] | response block index(prompt 区 = -1) |
| `turn` | int32 | [L] | turn id |
| `scaffold` | bool | [L] | 结构骨架位(被冻结、不去噪) |
| `weight_vec` | float32 | [L] | 逐 token section 权重(**来自 config `SECTION_W`**) |
| `block_alpha` | float32 | [n_blocks] | 逐 block 噪声 α(**来自 config `NOISE_SCHED`**) |
| `block_beta` | float32 | [n_blocks] | 逐 block 噪声 β(**来自 config `NOISE_SCHED`**) |
| `n_blocks` | int64 | scalar | block 数 |
| `pixel_values` | float16 | [N, 1176] | ViT patch 输入(N = 3 图的 merged patch 总数) |
| `image_grid_thw` | int64 | [3, 3] | 3 张图各自 (t,h,w);canonical = `[[1,32,30]]×3` |
| `position_ids` | int32 | [3, L] | mrope 三维坐标(图像区铺成 2D) |
| `vision_mask` | bool | [L] | image/vision token 位 |

canonical 量(@720):`L=1856`,`N_img=720`,`n_blocks≈11`,`2N=1440`。见 [[07_fidelity_fixes_2026-06-20]]。

---

## 2. 调查与 blocker(**lessons** — 这一节是核心经验)

### 2.1 我最初的(错误)假设:"段一是 TF↔torch 硬墙"
我曾断言步骤 1(必须 `tf.data` + 整个 `waymo-open-dataset` + 编译 proto)与步骤 2(`torch`+`transformers`)
两套重型依赖在单解释器里共存会崩,因此 raw→张量"不可能一个脚本"。**这个判断保守且部分错误。**

### 2.2 Waymo 内部 agent 的回答(2026-06-23,通过复制粘贴 MC prompt 求证)
| 题 | 答 | 关键结论 |
|---|---|---|
| F1 读 tfrecord | 官方轻量 reader | proto target = `//third_party/waymo_open_dataset/protos:end_to_end_driving_data_py_pb2`;读取直接用 `apache_beam.io.ReadFromTFRecord`(纯 Python TFRecordIO)+ `E2EDFrame.FromString`,**零 TF** |
| F2 图像存储 | 内嵌 JPEG bytes | `frame.frame.images[i].image` 是 JPEG 字节流 → `PIL.Image.open(io.BytesIO(...))`,**零 TF、零 CNS 引用** |
| F3 AR 编码 | 换掉 tf.Example | 推荐 **msgpack / 极简自定义 proto**(`tf.Example` 会把 TF 拖回来);训练侧用纯 Python `grain.MapTransform` over `grain.ArrayRecordDataSource` 读回,替代 `tf.io.parse_tensor` |
| F4 Flume 里跑 HF tokenizer | **D(否)** | **Borg worker 默认无外网**:HF `trust_remote_code=True` 去 huggingface.co 拉代码/词表 → ConnectionRefused 硬崩;且 torch+多模态 processor 塞进 worker → 臃肿/OOM。**单 Flume job(decode→tokenize→AR)不是 Google-blessed 形状。** |

### 2.3 结论(lessons 提炼)
1. **段一的"硬墙"在内部路径下不成立**:decode 这步**既不需要 TF 也不需要 torch**(纯 protobuf + Pillow + apache_beam)。我之前预设"必须用 tf.data + 整个 SDK"是错的。
2. **但 proto 是 google3-internal**(F1 的 target 在 `//third_party/...`),所以被打通的是**内部 google3 路径**,OSS 本地"一个脚本"仍因拿不到 proto 而受限。
3. **正确形状不是"一个脚本",而是边界不同的 2-job split**:tokenization 从"离线烤进数据集"挪到"训练期 Grain 现算"。这也顺带修掉了一个真实设计 wart(见 2.4)。
4. **不要盲信外部反馈**:F2 的字段路径 `frame.frame.images[i].image`、ego/future 字段、AR msgpack schema,都要在实现时用真 proto 自己核一遍(见 §7 待核清单)。

### 2.4 顺带修掉的设计 wart:config 被烤进数据集(此前 (A)/(B) 讨论)
现行 npz 把 `weight_vec`(逐 token 段权重)、`block_alpha/block_beta`(逐 block 噪声调度)**从纯 config
常量(`SECTION_W`、`NOISE_SCHED`)展开后 bit-exact 烤进每一行**。后果:想重调段权重或噪声调度,
**必须重跑全量 ETL**。新架构里它们在 Grain 里 **train-time 从 config 展开**,旋钮回到 config。
> 区分清楚:`rbi/turn/scaffold/position_ids` 是**逐样本结构**(依赖该样本的 tokenization,放不进 config,
> 必须 per-sample 算)——这些**本就该**在数据/transform 里算;而段权重/噪声**值**是 config。

---

## 3. 架构决策:option A(全-raw AR + tokenize-in-Grain)

2026-06-23 经 AskUserQuestion 选定 **A**(对比 B=ETL 预算 pixel_values、C=维持全烤)。

**为什么 A**:最灵活(改 tokenize/权重/噪声/未来分辨率都不重跑 ETL)、ETL 最纯(连 `transformers` 都不引)、
AR 体积最小、最贴合 Google-blessed 形状。**唯一代价**:训练 host 每步做 Qwen 图像 resize+patchify+tokenize
(§6 给出可逆的性能对策)。

---

## 4. 目标架构:2-job split

```
                      ┌─────────────────────── Job ① Flume ETL (Borg, CPU) ───────────────────────┐
WOD-E2E tfrecord ──►  ReadFromTFRecord ─► E2EDFrame.FromString ─► 抽 raw 字段(+ join 文本 target) │
                      │                        ─► msgpack 编码 ─► 写 ArrayRecord(/cns/...)         │
                      └──────────────── 无 TF / 无 torch / 无大模型库 ─────────────────────────────┘
                                                    │
                                                    ▼  ArrayRecord(raw msgpack 记录)
                      ┌──────────────── Job ② JAX/TPU 训练输入管线 (TPU host) ─────────────────────┐
                      │ grain.ArrayRecordDataSource ─► DecodeMsgpack(MapTransform)                  │
                      │   ─► Pillow 读 3 图 ─► 离线 Qwen AutoProcessor/Tokenizer ─► SASD 结构 ─►    │
                      │   ─► 13 字段张量(= §1 契约)─► 喂 JAX 编译的 TPU 训练步                     │
                      └──────────────── torch/transformers 在 host,可随模型迭代 ───────────────────┘
```

### 4.1 ArrayRecord 记录 schema(msgpack,每条样本一条)

设计原则:**存能保 parity 的最小集**。`prompt_text`/`target_text` 存**已渲染字符串**(保证 Grain 的
tokenize 与现行 `prep_train_jax`(读 JSON `conversations[*].value`)**逐字节一致**,parity 才铁);
其余 raw 数值字段为**可选**(留作未来改 prompt 模板 / eval metrics)。

| key | 类型(msgpack) | 必选 | 来源 / 说明 |
|---|---|---|---|
| `sample_id` | str | ✅ | `frame.context.name`;join key + eval GT keying |
| `images` | list[bytes] (len 3) | ✅ | `frame.frame.images[i].image` 原始 JPEG,顺序 = (FRONT_LEFT, FRONT, FRONT_RIGHT) |
| `prompt_text` | str | ✅ | 已渲染的 human prompt(= `convert_wod_e2e.build_prompt(intent, past_states)`) |
| `target_text` | str | ✅ | 最终 gpt JSON(见 §4.2 provenance);其中 `trajectory` 是真 GT |
| `provenance` | str | ✅ | `"pseudo"`(convert 时生成)或 `"distilled"`(teacher 升级);影响可比性 |
| `schema_version` | int | ✅ | msgpack schema 版本,先用 `1` |
| `ego` | dict[str→bytes] | ⬜ | 原始 `past_states` 6 数组(pos_x/pos_y/accel_x/accel_y/vel_x/vel_y,各 float32[16],`np.tobytes()`)。留作未来重渲染 prompt |
| `intent` | int | ⬜ | 原始 `EgoIntent.Intent`;配合 `ego` 可在 Grain 重渲染 prompt(A2 变体) |
| `future_xy` | bytes | ⬜ | GT 未来轨迹 float32[20,2](`future_waypoints_20`),供 ADE/RFS metrics |
| `rated` | bool | ⬜ | 是否 rater-scored(eval 子集标记) |

> **parity-safe 取舍**:canonical 存 `prompt_text`+`target_text`(字符串),Grain 直接 tokenize。
> 想要"连 prompt 模板都能 train-time 改"的 A2 变体,则改存 `ego`+`intent`,在 Grain 里重跑
> `build_prompt`——但要自担"重渲染需与旧 JSON 逐字节匹配"的风险,**默认不选 A2**。

### 4.2 文本 target 的 provenance(实现必须决定的点)
原始 WOD-E2E **没有文本标签**。`convert_wod_e2e.build_target()` 用规则造出弱标签:
- `critical_objects`:全 `"no"`(无感知标签);
- `explanation`:模板句;
- `future_meta_behavior`:由 GT 轨迹的速度段 + intent 粗导出(`build_meta_behavior`,**弱标签**:
  lateral 只有 go straight/turn left/turn right,缺 canonical lane-follow/lane-change/yield 词表);
- `trajectory`:**真 GT**(5 wp @1s,`gt_5waypoints`)。

可选地用 teacher-distill(memory `fast-ddrive-teacher-distill`,`merge_distilled_labels.py`)**升级**文本部分,
按 `sample_id` 合并。**ETL 决策**:`target_text` 取 pseudo 还是 distilled,由一个开关决定;若 distilled,
ETL 需读 distilled JSON 并按 `sample_id` join。`provenance` 字段记录实际来源。

### 4.3 Job ① Flume ETL(无 TF / 无 torch)

依赖:`//third_party/py/apache_beam`、`//third_party/waymo_open_dataset/protos:end_to_end_driving_data_py_pb2`、
`//third_party/py/msgpack`、`//third_party/py/PIL`(仅用于校验,可选)、ArrayRecord sink。

骨架(伪代码,实现时按 google3 Beam API 落地):
```python
import apache_beam as beam, io, msgpack, numpy as np
from third_party.waymo_open_dataset.protos import end_to_end_driving_data_pb2 as e2e

FRONT_TRIPLET = [(2, "FRONT_LEFT"), (1, "FRONT"), (3, "FRONT_RIGHT")]  # CameraName ints
PROMPT_PREFIX = "...(从 convert_wod_e2e 原样搬)..."

def to_record(frame: e2e.E2EDFrame, distilled: dict | None):
    sid = frame.frame.context.name
    by_name = {im.name: im.image for im in frame.frame.images}
    if any(cam not in by_name for cam, _ in FRONT_TRIPLET):
        return None                                  # 缺前视 → 丢(对齐旧 save_front_images 行为)
    images = [by_name[cam] for cam, _ in FRONT_TRIPLET]
    prompt_text = build_prompt(frame.intent, frame.past_states)        # 复用 convert_wod_e2e 逻辑
    if distilled and sid in distilled:
        target_text, prov = distilled[sid], "distilled"
    else:
        target_text, prov = build_target(frame), "pseudo"             # 复用 convert_wod_e2e 逻辑
    rec = {"sample_id": sid, "images": images, "prompt_text": prompt_text,
           "target_text": target_text, "provenance": prov, "schema_version": 1,
           # 可选:
           "intent": int(frame.intent),
           "ego": {k: np.asarray(getattr(frame.past_states, k), np.float32).tobytes()
                   for k in ("pos_x","pos_y","accel_x","accel_y","vel_x","vel_y")},
           "future_xy": np.asarray([[p.x,p.y] for p in ...], np.float32).tobytes(),
           "rated": is_rated(frame)}
    return msgpack.packb(rec, use_bin_type=True)

with beam.Pipeline() as p:
    (p | beam.io.ReadFromTFRecord("/path/*.tfrecord")
       | beam.Map(e2e.E2EDFrame.FromString)
       | beam.Map(lambda fr: to_record(fr, DISTILLED))
       | beam.Filter(lambda x: x is not None)
       | WriteToArrayRecord("/cns/.../wod_e2e_sasd.array_record"))   # 按内部 sink API
```
注意:`FRONT_TRIPLET` 顺序、`build_prompt`/`build_target`/`is_rated` 与 `convert_wod_e2e.py` **逐字节对齐**
(prompt 模板、history 7 点 @0.5s、轨迹 GT 索引 `[3,7,11,15,19]`、meta 弱标签规则)。

### 4.4 Job ② Grain `MapTransform`(= `prep_train_jax.py` 重打包,train-time,TPU host)

这是把 `prep_train_jax.py:96-141` 的主体从"离线脚本"改成"train-time transform"。**逐步**(对每条样本):

1. **解码**:`msgpack.unpackb(record_bytes)` → `sample_id, images[3], prompt_text, target_text`。
2. **target 归一化**:`process_gpt(target_text, tok)`(`prep_train_jax.py:42-64`:去 mdm 标记、`clean_nulls`、
   explanation pad 到 `EXP_BUDGET=192`/6-block、fmb 字段 pad 到 3、trajectory 格式化)。
3. **构 messages**:`messages_from_prompt(prompt_text, imgs)`(`ddrive_jax.eval.scaffold`)→ user content(插 3 个
   `<image>` 占位)+ assistant content = 归一化后的 gpt;`proc.apply_chat_template(..., add_generation_prompt=False)`。
4. **离线 Qwen processor**:`proc(text=[text], images=imgs, return_tensors="np")` → `input_ids`、
   `pixel_values`(→float16)、`image_grid_thw`。**processor 必须离线加载**(见 §4.5)。
5. **pad**:`len(ids) % BD` → 尾部补 `MASK_ID=151665` 到 `BD=32` 整数倍;`L=len(ids)`。
6. **labels**:扫 `<im_start>assistant\n`(token `151644,77091,198`)定位 response start;到 `<im_end>=151645`,
   再 **`+2`**(含 im_end 与后续 `\n`,对齐 `multi_modal_dataset_fast_ddrive.py:408`)。
7. **SASD 结构**:`seqs = build_deep_scaffold_sequences(tok)`(可在 transform `__init__` 里**缓存一次**);
   `rbi, turn, n_blocks, scaffold, b2s = compute_section_block_idx_deep_static(labels, ids, seqs, BD)`。
8. **config 展开(train-time!)**:`weight_vec[i] = SECTION_W[b2s[rbi[i]]]`;
   `block_alpha[b], block_beta[b] = NOISE_SCHED[b2s[b]]`。`SECTION_W`/`NOISE_SCHED` 现在是 **Grain 侧 config**,
   不再来自数据。
9. **rope / vision**:`position_ids = get_rope_index_numpy(ids, image_grid_thw)`(`ddrive_jax.eval.rope_index`);
   `vision_mask = (ids∈{IMAGE_TOK 151655, VSTART 151652, 151654})`。
10. **输出** §1 的 13 字段张量 dict,喂 JAX 训练步。

`build_deep_scaffold_sequences` / `compute_section_block_idx_deep_static` 签名见
`section_utils.py:305 / 374`(返回 `response_block_idx, turn_idx, n_blocks, scaffold_mask`;本仓库另需
`b2s` block→section 映射——`prep_train_jax` 当前从该函数的扩展返回拿到,实现时确认返回元数)。

Grain 接线:
```python
import grain.python as grain, msgpack
class TokenizeSASD(grain.MapTransform):
    def __init__(self, model_dir, section_w, noise_sched):
        self.proc = AutoProcessor.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
        self.tok  = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.proc.image_processor.min_pixels = self.proc.image_processor.max_pixels = 200704  # 锁 720
        self.seqs = build_deep_scaffold_sequences(self.tok)        # 缓存
        self.SECTION_W, self.NOISE_SCHED = section_w, noise_sched
    def map(self, record_bytes):
        d = msgpack.unpackb(record_bytes, raw=False)
        # ...步骤 2-10...
        return {"input_ids": ..., "labels": ..., ...}              # 13 字段

loader = grain.DataLoader(
    source=grain.ArrayRecordDataSource("/cns/.../wod_e2e_sasd.array_record"),
    operations=[TokenizeSASD(MODEL_DIR, SECTION_W, NOISE_SCHED), Batch(...), Pad(...)],
    worker_count=N)                                                # 多 worker prefetch 喂 TPU
```

### 4.5 离线 Qwen 在 TPU host 的加载(F4 的 C-fix,即便在 Grain 里也适用)
预先把 Qwen2.5-VL 全部文件(`config`、自定义 `*.py`、tokenizer、processor 配置)放到 `/cns/.../qwen_local/`,
用 `AutoProcessor.from_pretrained("/cns/.../qwen_local", trust_remote_code=True)` **离线**加载,**绝不触网**。
BUILD 依赖声明 `//third_party/py/torch`、`//third_party/py/transformers`。
> 关键:Grain transform 跑在 **TPU host**(非 Borg ETL worker),网络/依赖更宽松,但**仍按离线加载处理**最稳。

---

## 5. 迁移地图:死 / 复用 / 新增

| 类别 | 内容 |
|---|---|
| **死掉** | JSON 中间格式;`prep_to_parquet.py`;`parquet_file_to_tfexample_ar.py`;`ar_dataset.py::decode_example`(tf.Example→msgpack) |
| **复用(搬进 transform/ETL)** | `convert_wod_e2e`:`PROMPT_PREFIX`/`build_prompt`/`build_history`/`build_target`/`is_rated`/`FRONT_TRIPLET`/轨迹 GT 索引 → ETL;`prep_train_jax`:`process_gpt`/labels/SASD 结构/rope → Grain transform;`section_utils` 两函数;`messages_from_prompt`;`get_rope_index_numpy` |
| **新增** | Flume ETL(proto→msgpack→AR);Grain `ArrayRecordDataSource` + `DecodeMsgpack/TokenizeSASD`;`SECTION_W`/`NOISE_SCHED` 提为 Grain config;离线 Qwen 目录 `/cns/.../qwen_local` |

---

## 6. 性能注脚 + 可逆对策

**风险**:option A 在训练 host 每步做 Qwen 图像 resize(200704px)+ patchify(→720×1176)+ tokenize。
720-token×3 图是真 CPU 活。Grain 多 worker prefetch 可与 TPU 计算 overlap,但需实测吞吐是否喂得饱 TPU。

**可逆对策(A→B,不重做其余)**:若 host 喂不饱,**只把"图像预处理那一段"从 Grain 挪进 ETL**——
即在 AR 里额外存预算好的 `pixel_values`(float16)+`image_grid_thw`,Grain 只读不算图。Qwen image processor
可纯 numpy 输出、**不需要 torch、非 trust_remote_code**,所以 ETL 加它代价可控。文本 tokenize 仍留 Grain。
这就是 option B,**架构其余部分一字不改**。先上 A、按实测决定是否退 B。

---

## 7. 实现期待核清单(不要盲信,自己验)

1. **proto 字段路径**:`frame.frame.context.name`、`frame.frame.images[i].name/.image`、`frame.intent`、
   `frame.past_states.{pos_x,...}`、`frame.future_states`、`frame.preference_trajectories` —— 用真 proto 核一遍
   (Waymo agent 给的 `frame.frame.images[i].image` 双层 `frame` 需确认)。
2. **`compute_section_block_idx_deep_static` 的 `b2s` 返回**:确认本仓库版本返回 block→section 映射
   (`prep_train_jax` 依赖它做 `weight_vec`/`alpha`/`beta`);若签名只回 4 元,定位 `b2s` 实际来源。
3. **ego subsample**:`HIST_IDX=[3,5,7,9,11,13,15]`(7 点 @0.5s)与 `HIST_LABELS` 对齐;prompt 渲染逐字节匹配。
4. **distilled join**:distilled labels 的 key 是否就是 `sample_id`(`frame.context.name`);缺失样本回退 pseudo。
5. **msgpack schema**:`use_bin_type=True`/`raw=False`;bytes 字段(images/ego/future)往返无损;定 `schema_version`。
6. **AR sink / Grain source**:google3 内部 ArrayRecord 写 API + `grain.ArrayRecordDataSource` 读路径(CNS)。
7. **离线 Qwen**:`/cns/.../qwen_local` 文件齐全(含自定义 modeling/`section_utils`),`from_pretrained` 全程不触网。
8. **数值同源**:`min_pixels==max_pixels==200704`、`EXP_BUDGET=192`、`BD=32`、labels `+2`、MASK pad —— 与
   [[07_fidelity_fixes_2026-06-20]] 完全一致。

---

## 8. Parity / 回归测试计划(白送的验证链)

`prep_train_jax.py` 直接当 **oracle**:对 `fast_ddrive/data/example/sample.json` 那 2 条样本,
断言 **Grain `TokenizeSASD.map` 输出 == `prep_train_jax` npz**,13 字段 **bit-exact**
(注意:`weight_vec/alpha/beta` 现在是 train-time 从同一 `SECTION_W`/`NOISE_SCHED` 展开,数值应完全相等)。

再接上现有三方 harness(memory `fast-ddrive-mm-step-parity-harness`,`jax_ddrive/scripts/mm_step_parity/`):
```
ETL→AR(msgpack)──decode──► Grain transform 输出 ──==──► prep_train_jax npz ──==──► PyTorch oracle / NNX / MaxText
```
即:**只需新增"Grain 输出 vs npz"一个 bit-exact gate**,其余沿用已 PASS 的 GPU/CPU + 真 v6e-1 TPU 链路。
另加一个 **ArrayRecord round-trip gate**(msgpack 写入→读回 bytes 无损),对齐现 `verify_dataset.py` 的角色。

---

## 9. Lessons learned(汇总,供下次别再踩)

1. **"环境依赖冲突"≠"逻辑不可能"**:段一看似 TF↔torch 硬墙,实则 decode 可纯 protobuf + Pillow,**零 TF/零 torch**。
   下次先问"这步真正不可约的依赖是什么",别默认"得用整个官方 SDK"。
2. **proto 是 google3-internal**:OSS 本地"一个脚本"被 proto 可得性卡住,**内部路径才是被打通的那条**——
   与现有 internal-TPU 计划([[03_scaleup_tpu_spec]]、memory `fast-ddrive-internal-tpu-step3`)一致。
3. **正确的拆分边界 = "raw bytes 进 AR(ETL)| tokenize 进 Grain(train)"**,不是"全塞一个脚本"。
   Borg 无外网 + worker 臃肿,使"ETL 内 tokenize"成为反模式。
4. **别把 config 烤进数据**:段权重/噪声调度是 config,不是数据;烤进去 = 调参必重跑全量。train-time 展开才对。
   但要分清**逐样本结构**(rbi/scaffold/position_ids,必须算)与 **config 值**(weight/α/β,该留 config)。
5. **外部 agent 答案要自核**:字段路径、schema、target 都按 §7 清单用真 proto 复验。
6. **保 parity 优先于"极致 raw"**:存**已渲染** prompt/target 字符串(而非全 raw 重渲染),让 Grain tokenize
   与现行 oracle 逐字节一致;raw 数值字段作可选项并存,兼顾灵活与可验证。

---

## 10. References

- 代码:`fast_ddrive/data/convert_wod_e2e.py`、`jax_ddrive/eval/prep_train_jax.py`、
  `jax_ddrive/ddrive_jax/convert/prep_to_parquet.py`、`jax_ddrive/scripts/parquet_file_to_tfexample_ar.py`、
  `jax_ddrive/ddrive_jax/data/ar_dataset.py`、`<SNAP>/section_utils.py`、
  `jax_ddrive/scripts/mm_step_parity/`(parity harness)。
- google3 target:`//third_party/waymo_open_dataset/protos:end_to_end_driving_data_py_pb2`、
  `//third_party/py/apache_beam`、`//third_party/py/{msgpack,torch,transformers}`。
- memory:`fast-ddrive-wode2e-to-ar-internal-flume`、`fast-ddrive-mm-step-parity-harness`、
  `fast-ddrive-teacher-distill`、`fast-ddrive-internal-tpu-step3`、`fast-ddrive-dataset-v2-spec`。
