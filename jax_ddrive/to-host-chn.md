# Fast-dDrive → JAX / TPU —— Hosting & Deployment 交接文档(`to-host-chn.md`)

给在 Waymo TPU 基础设施上跑这个项目的人的自包含交接文档。覆盖:做了什么、验证了什么(带数字)、
怎么复现、怎么在 TPU 上部署、以及验收标准。分支 `jax-ddrive-port`,未推到 GitHub。

> **最后更新:2026-06-22**(mm-SASD-步三方 parity:GPU/CPU + 真 TPU 全 PASS;此前 from-base overfit 管线 + B1 导出 + B2 自包含推理)。**想读懂代码而非部署？→ `docs/0overview/00_START_HERE.md`(文档总入口 + 代码阅读指南)**。文档维护规则见
> `docs/README.md`;数据格式权威规格见 `docs/2implementation-details/DATASET_V2.md`(v1 细节与
> Round-2 语义验证见 `DATASET.md`);**内部 TPU 推理部署权威规格见
> `docs/2implementation-details/INFERENCE_DEPLOY.md`**。
> 📌 **状态(2026-06-22,mm-SASD-步三方 parity,当前真相):** 多模态 SASD **训练步**在**完全相同输入**下三方数值一致——
> PyTorch oracle ↔ ddrive_jax NNX ↔ maxtext-fork(含 **MaxText 真·3B 完整前向** logits/loss vs PyTorch ~1e-4/1e-6)
> + 数据 prep→parquet→AR round-trip bit-exact;**GPU/CPU 一键 `SASD_MM_STEP_PARITY_PASS`(10/10)** + **真 v6e-1 TPU 复核**
> (data + NNX impl 两样本 PASS)。**独立 harness、未改任何现有代码**;数字/设计/容差/ViT-cuDNN-seed 根因(对照实验 Layer1-CTRL)
> 见 `jax_ddrive/scripts/mm_step_parity/README.md`。
> 📌 **历史基线(2026-06-08):** Phase 7(MaxText port)完成并在真实 v6e-1 上用真实权重跑通
> (loss 0.31/0.56)。详见 `docs/4collect/OVERNIGHT_TPU_PROGRESS{,-chn}.md`。
> 📌 **数据集(2026-06-12 白天):** 全量 **415,663 帧** parquet 构建完成;**Round-2 语义验证:10/10
> 样本与从 raw 重跑整条链 bit-exact**(全部列、answer 文本 round-trip、轨迹==真值@1s、像素重建);
> 可视化复查网站 `jax_ddrive/visualizations/`。*Phase 8 后更新:* round-2 与网站已升级到 **v2** ——
> **12/12 PASS(10 train + 2 val)**,含 `image_embeds` 从像素复算(两层 bf16 判据)与 PCA 三联图;
> 判据与复跑命令见 `DATASET_V2.md` §6。
> 📌 **状态(2026-06-12 overnight,当前真相):** 生产训练闭环在真实 TPU 上**端到端重验证通过**
> (`V2_TPU_VALIDATION_PASS`):**dataset v2**(ArrayRecord;12 数组含 pixel_values + 预计算
> frozen-ViT `image_embeds` bf16 [168,2048] 单份)× **AR reader**(`make_sasd_loader` 按扩展名
> 自动探测,公共 API 不变 → 三条训练路径同时获得全量数据能力,旧"尚无训练路径读 AR"的缺口已关闭)
> × **数据迭代器状态进 checkpoint**(`waymo_sasd` 加入 grain ckpt 家族;resume **续流不重放**,
> v6e-1 实测恢复后精确续训 12..17)。v6e-1 吞吐 **83.4 TFLOP/s/device(较 ViT-in-loop 的 65 提升
> 28%,pod 侧不再加载 ViT/transformers)**。v2 数据 train 全量/50k/400 + val(479,训练格式带
> target)均已构建、抽样审计(逐字节 vs parquet)、上传 GCS。MaxText fork 已 commit(`a645b25`,
> 含 vendored 自包含 `sasd_data/`)+ `PATCHES.md` 逐文件文档化。
> 📌 仍开放:**≥8-chip 多节点跑**(纯 GCP trial 容量问题,外部/瞬时;Waymo 容量下即验证阶梯 step 2)。
> 📌 **状态(2026-06-13,from-base + 推理):** 从**干净 base Qwen2.5-VL**(非 release)打通整条管线
> (详见 `docs/2implementation-details/INFERENCE_DEPLOY.md` + `docs/4collect/07_from_base_b1_b2_progress.md`):
> **A1** base ViT 与 release 不同(NVIDIA 微调过 ViT,367/390 张量超 bf16 舍入)→ from-base embeds 一律用 base ViT;
> **bf16 加载补丁**(base 快照全 BF16,三处 safetensors 读法加固,F32 路径逐位不变);
> **A2** base MaxText 参数 ckpt(434/434,3.086B);**A3** distilled-400 base-ViT v2 AR(400/7,eager 2.2e-5);
> **A4/A5** 本地从-base 训练(adafactor + `MEM_FRACTION=0.93`,42.7 TFLOP/s;A5 overfit loss 5.5→~0.36 进行中)。
> **B1** `maxtext_to_hf_export.py`(MaxText→bf16 HF):**round-trip 824/824 逐位通过**。
> **B2** `src/maxtext/diffusion/eval_sasd/`(vendored 验证版多模态采样器,自包含、PYTHONPATH=fork/src 零 ddrive_jax,
> bf16 加固):**自包含 import 测试通过**;fork-only 生成 / bf16 加载 / 嵌入彩排 = **post-A5 GPU 双检已挂载**。
> 推理输入离线预算成 npz(B3),内部侧只需 tokenizer 解码;只有标量验证日志带出。
> (注:B1/B2/B3 + 更新后的 PATCHES.md 目前是 fork **工作区文件、尚未 commit**;发布靠
> `upload_code_to_gcs.sh` 打包**工作区**而非 git,故不影响内部使用。)

---

## 0. TL;DR(逐点)

### Phase 1–5(模型 port + eval,此前已验证)
- **整个模型已 port 并对 PyTorch 做了 parity 验证**:text decoder **3.2e-5**、ViT **4.0e-5**、multimodal forward **7.7e-5**、SASD loss **7.9e-8**、attention mask 逐位一致(bit-identical)。
- **训练时 loss 下降**:text 从零 **3.84→1.47**;multimodal **0.999→0.701**;真实 Waymo **0.6815→0.6005**。
- **WOD-E2E eval 在两套栈上**:PyTorch ADE@3s **0.814** / RFS **7.914**;JAX **0.839** / **7.929**(持平,轨迹匹配 **0.01 m**)。完整 479-frame rated-val 集。
- **10-gate 验证套件 + 对抗 audit:0 defects**。

### Phase 6(scale-up + dataset,2026-06-05,新完成)
- **TPU-ready dataset 已构建并上传**:50,331 个 WOD-E2E train frames → 787 个 Apache Parquet shards(22 GB),私有 HF `kaiwen2/wod-e2e-fast-ddrive-sasd-50k`。0 conversion errors;bit-exact decode 已验证;MaxText `hf` data-path 兼容。
- **FSDP 训练 harness 已验证**:`shard_map`+`psum` FSDP、AdamW、Orbax `CheckpointManager`。FSDP-vs-single-device **|diff|=9.5e-7**;checkpoint resume **diff=0.0**;252/252 个 real-model kernels 已 sharded。
- **真实 3.09B 模型在 GPU 上通过 harness 训练**:loss **0.985→0.598**(40 steps,no NaN,24.8 GB VRAM)。
- **grain multi-host input pipeline**:deterministic、per-host sharding、online SASD noising、resumable。

### Phase 7(MaxText port —— next,文档撰写时尚未开始)
- **决策(2026-06-06)**:用 **MaxText** 作为生产训练框架。
- **要 graft 什么**:`diffusion/` 模块 + SASD `loss_fn`(~5 行 diff)+ bidirectional attention patch + Waymo grain data source。模板:`jax-mdlm-handoff`(LLaDA in MaxText)。
- **数据已就绪**:Parquet 在本地磁盘 + 私有 HF;pod 跑之前 `gsutil rsync` 到 GCS。

### Phase 8(dataset v2 + 生产数据路径,2026-06-12,当前真相)
- **Dataset v2(生产训练格式)**:ArrayRecord(tf.train.Example),12 个数组**全保留**(含
  `pixel_values`,使 embeds 可独立复验)+ 新增 **`image_embeds` [168,2048] bf16** = frozen-ViT
  离线预计算(fp32 highest 精度 → bf16;单份存储,loader `concat([ie,ie])` 双倍)。规格 SSOT:
  `docs/2implementation-details/DATASET_V2.md`。
- **四个 split 构建 + 审计 + 上传 GCS**:train 全量 **415,663/130 shards/369 G**、50k
  (50,331/787/45 G)、400(烟测)、**val 479(训练格式带 target,从 raw val `--with_target` 重建)**。
  审计 = 三方计数对账 + 抽样逐字节 vs parquet + embeds 独立复算(bf16 量化后 99.67% 逐位相同)。
- **AR reader**:`make_sasd_loader` 按扩展名自动探测 AR/parquet(公共 API 不变);grain 管线透传
  `image_embeds`;`tests/test_ar_pipeline.py` 证 AR↔parquet **batch 级 bit-exact**。
- **迭代器状态进 checkpoint**:`waymo_sasd` 加入 MaxText grain ckpt 家族 → **resume 续流不重放**。
- **真实 TPU 重验证**(v6e-1,`V2_TPU_VALIDATION_PASS`):预计算 embeds 路径(pod 不加载 ViT)
  **83.4 TFLOP/s/device(+28% vs 旧路径)**;恢复跑精确续训 12..17(零 step-0)。
- **MaxText fork 自包含化并 commit**(`a645b25`):vendored `input_pipeline/sasd_data/`,
  `PATCHES.md` 逐文件文档化(provenance、flag-gating、验证命令)。

### 硬件 + 存储
- RTX 5090(32 GB VRAM)。Host RAM 30 GB,**无 swap** —— 别在 CPU 8-device emulation 下加载完整 3.09B 模型(会 OOM)。
- 大文件放在 `/home/kaiwen/data/fast-ddrive/`(3.6 TB SSD)。Raw train:877 GB,raw val:226 GB。
  v2 AR:全量 369 G / 50k 45 G(均已镜像到 GCS bucket `gs://project-…-ddrive-sasd/`)。

### 诚实的开放项
- **Pseudo text labels（默认）+ teacher-distill 升级管线（已建）**:三档 parquet（400/50k/415k）默认是伪 `critical_objects`/`explanation`（trajectory 是真 GT）。teacher-distill（Route A）管线已构建并验证 → distill-400 完成（L=1280,已上传 GCS）;distill-50k **已暂停**（本里程碑不需要,2026-06-13 停,中间产物已清理）。详见 §2.4 与 `docs/2implementation-details/LABELING.md`。
- **≥8-chip 多节点跑未发生**:FSDP 数学在 CPU 8-device 仿真 + 单芯 TPU 已证;多节点只差 GCP trial 容量(Waymo 内部容量下跑验证阶梯 step 2 即可)。
- **B4 GPU↔TPU 生成数值(2026-06-16 彩排)**:eval_sasd sampler 经 backend 代理确认 **fp32 可移植**(GPU fp32 确定性;GPU==CPU fp32 在 20 val 逐 token 位等,含 bf16 下最难的 3 个样本)→ **TPU-fp32 推理预期与 GPU 逐位一致、无需改 TPU 码**;真 v6e 确认**仍卡 GCP trial 容量**(跨 zone 重试均无可用)。bf16 为 confidence-cascade 的设计性分歧(求可信用 fp32)。详见 `docs/2implementation-details/INFERENCE_DEPLOY.md` §B4。
- **训练循环 eval 未接线**(`eval_interval: 0`):val v2 AR(479,训练格式)已就绪,接上是小改动。
- HF datasets 是**私有的**(WOD license 禁止再分发);GCS bucket 属 $300 trial 项目(注意到期迁移)。
- 分支 `jax-ddrive-port` 与 maxtext fork 均**未推远端**(本地 + GCS tarball)。

---

## 1. 这是什么

**Fast-dDrive**(NVIDIA/`Efficient-Large-Model/Fast-dDrive`,在 HF 上)是一个 Qwen2.5-VL-3B-Instruct
backbone,被 fine-tune 成针对 **Waymo Open Dataset End-to-End Driving(WOD-E2E)** 挑战的
**masked-diffusion(MDM)block-diffusion** VLA。给定 3 路前视相机帧 + 一段文本 prompt(导航指令
+ 3 秒自车历史),它输出一个含四个 section 的 **JSON 答案**:

```json
{"critical_objects": {12 yes/no flags}, "explanation": "...scene reasoning...",
 "future_meta_behavior": {"longitudinal": "...", "lateral": "..."},
 "trajectory": "[[+14.70,-00.04], ... 5 waypoints @1 s ...]"}
```

它用 **SASD** 训练 = *Section-Importance-Weighted Loss*(per-section 权重
{critical_objects 1.5, explanation 1.0, future_meta_behavior 2.0, trajectory 3.0})+ 一个
*per-section Beta noise schedule*,作用在一个 **deep-JSON scaffold** 上(JSON 骨架固定;只有
value slot 被 mask 并去噪),在一个 doubled `[noisy | clean]` 序列上,配 **hybrid
block-causal attention mask**。推理时逐 block 去噪 scaffold。

**我们的任务**是把这一切 port 到 JAX/Flax-NNX(让它能在 Waymo TPU pods 上跑,MaxText 风格),
通过对 PyTorch release 的数值 parity、以及 loss 下降来在本地 5090 上证明正确性。JAX/MaxText 模式的
参考是之前的项目
`/home/kaiwen/Desktop/research/DLM-policy4AV/jax-mdlm-handoff`。

---

## 2. 我们构建并验证了什么

### 2.1 模型 port(`jax_ddrive/ddrive_jax/`)
纯 Flax-NNX,镜像 PyTorch `modeling.py` + `section_utils.py` + `generation_utils.py`:

| 组件 | 文件 | 对 PyTorch release 的 parity |
|---|---|---|
| Qwen2.5 text decoder(36L/2048d, GQA 16/2 heads, qkv bias, RMSNorm, SwiGLU, tied embeds, θ=1e6) | `models/qwen2_5_text.py` | logits rel-max **3.2e-5**, top-1 **100%** |
| RoPE + 3D **M-RoPE**(mrope_section [16,24,24]) | `models/{rope,qwen2_5_text}.py` | text / MM parity 的一部分 |
| Qwen2.5-VL **ViT**(Conv3D-as-linear patch embed, 2D RoPE, window attn @[7,15,23,31], spatial-merge 2, RMSNorm+SwiGLU, PatchMerger) | `models/vision_qwen25vl.py` | arch **4.0e-5**(end-to-end 2.6% 是良性的 cuDNN-Conv3d seed) |
| Vision↔text **fusion**(image-token scatter)+ M-RoPE | `models/qwen2_5_text.py` | multimodal forward **7.7e-5**, top-1 100% |
| HF safetensors → NNX weight load(text + ViT) | `convert/hf_to_jax.py` | 0 missing keys |
| **SASD loss**(section-weighted CE + complementary-mask causal CE) | `diffusion/sasd_loss.py` | total **7.9e-8** |
| **Hybrid block-causal mask**(training doubled-2L + eval) | `diffusion/masks.py` | **bit-identical**(0 mismatches) |
| Per-section **Beta noising**(scaffold freeze, always-mask `im_end`, complementary) | `diffusion/noise.py` | unit-tested |

### 2.2 训练(loss 下降)
- `train_overfit.py`(text)、`train_overfit_mm.py`(multimodal):从 base Qwen2.5-VL,SASD loss
  下降 **3.84 → 1.47**(text,从零学会任务)和 **0.999 → 0.701**(multimodal)。
- `train_waymo_sasd_jax.py`:在**真实 Waymo data** 上的生产多样本 loop ——
  stochastic per-section Beta noise、frozen ViT image embeds、Section-Importance-Weighted +
  complementary-mask loss、bf16 + per-layer `nnx.remat` + Optax Adafactor、**Orbax checkpoints**。
  Fixed-eval loss **0.6815 → 0.6005**,跑 400 steps、200 real frames(ckpts at step_200/400)。
- 单张共享 5090 的内存技巧:bf16 + Adafactor + per-layer remat(+ 可选 `lora.py` 里的 LoRA)。在 TPU pod
  上这些都不需要(FSDP 给 per-chip ~1/N 内存 → full-FT + AdamW)。

### 2.3 评测 pipeline(两套栈,一个 metric)
```
WOD-E2E tfrecords ──convert_wod_e2e.py──▶ val_rated.json (479) + front-cam JPEGs
        ┌────────────────────────────────┴───────────────────────────────┐
        ▼ PyTorch / 5090                                                  ▼ JAX
 batch_inference.py ─▶ predictions.json        prep_jax_eval.py ─▶ jax_batch_inference.py ─▶ predictions.json
        └────────────────▶ evaluate_waymo_metrics.py (autovla env) ◀──────┘
                              ADE_3s / ADE_5s / RFS   (一个共享 backend)
```
- **Converter** `fast_ddrive/data/convert_wod_e2e.py`(repo 承诺了但缺失的 preprocessor):
  解析 `E2EDFrame`,提取 3 路前视相机,构建 canonical prompt(**逐字节验证**对照
  `fast_ddrive/data/example/sample.json`),设 `sample_id = frame.context.name` 让 predictions
  能 join GT,`--rated_only` 保留 **479** 个官方 rater-scored frames。
- **PyTorch eval** = `fast_ddrive/eval/batch_inference.py`(`scaffold_spec`,paper canonical,bf16,
  ~1.7 s/sample)。
- **JAX eval** = `prep_jax_eval.py`(CPU prep:HF processor + deep-scaffold + numpy `get_rope_index`,
  全部对 PyTorch 内部做了 bit-exact 验证)→ `jax_batch_inference.py`(JAX ViT fuse + block-wise
  `section_diffusion` 去噪,经 `ddrive_jax/eval/mm_sampler.py`,~16 s/sample,无 KV-cache)。
  *(注:eval prep `prep_jax_eval.py` 默认 `min=max=200704`(paper-eval 高分辨率);而 §5(e) 的
  SASD-训练 prep `prep_train_jax.py` 用 `784/50176`(→ 168 image tokens)—— 两套是按用途有意区分的
  分辨率策略,不是不一致。)*
- **Metric** = `fast_ddrive/eval/evaluate_waymo_metrics.py`(autovla env)。它先把 5×1 s waypoints
  JMT-interpolate 到 20×4 Hz 再算 ADE,并在 ≤3 条 rater trajectories 上算官方的 trust-region **RFS**。
  真实 flags:`--pred_json --gt <tfrecord-glob|.pkl> --output_dir`。

**完整 479-frame rated-val 结果,两套栈,100% trajectory parse:**

| Metric | PyTorch `scaffold_spec` | JAX `section_diffusion` |
|---|---|---|
| ADE @3s (m) | 0.814 | 0.839 |
| ADE @5s (m) | 1.990 | 2.072 |
| RFS | 7.914 | 7.929 |

(JAX 和 PyTorch 用的是不同 decoder —— section-diffusion vs scaffold-speculative —— 却落在
持平水平;在匹配样本上 JAX 轨迹等于 PyTorch 到 **0.01 m**。)

### 2.4 训练文本标签 —— 伪标签 + teacher-distill 升级
原始 WOD-E2E tfrecords 含**图像、ego states、intent、trajectories —— 但没有文本标签**。所以
`convert_wod_e2e.py --with_target` 构建的训练 target 里,**trajectory 是真实 GT**,
`future_meta_behavior` 派生(longitudinal 来自 waypoint speeds,lateral 来自 `EgoIntent`),
`critical_objects`/`explanation` 是**伪标签**。

**升级管线已构建并验证(Route A,teacher-distill)**:用 release 3B checkpoint 给文本三段打标、
保留 GT 轨迹 + GT 派生的 longitudinal(**hybrid fmb** 策略,因为 release teacher 推理时看不到未来,
其 longitudinal 仅 4/10 与 GT 一致)。蒸馏答案更长 → 数据集为**统一 L=1280**(与旧 L=1184 不兼容;
loss-zero padding 已验证)。**当前状态(2026-06-13):distill-400 完成
(`train/distill_400/parquet_L1280`,400/400 验证,已上传 GCS);distill-50k **已暂停**(本里程碑不需要,
2026-06-13 停于 3/21 chunk,中间产物已清理;`distill_teacher_chunked.py`+`finalize_distill_50k.py` 可续跑)。**
论文真标签来自 dVLM-AD 的 **GPT-4.1** 标注(未开源)→ 论文级保真度走 Route B。

> 标签的概念/provenance/管线/脚本、以及 **400 vs 800 vs 50k vs 415k 身份图**,权威说明见
> **`docs/2implementation-details/LABELING.md`**。

### 2.5 验证 & audit
- `jax_ddrive/scripts/run_all_verification.sh` → **10 个 gate**:`cpu_mask_loss, cpu_lora, cpu_noise,
  cpu_sharding, cpu_eval_ports, phase1_text, phase2_sasd, phase4_vit, phase4b_mm_fwd,
  phase3_lora_train`。clean run 下全部通过。(`phase3_lora_train` 在密集背靠背套件里可能偶发 flake,
  因为 LoRA 在近最优的 trained ckpt 上只把 loss 移动 ~0.001;单独跑则可靠通过 —— 见该脚本里的注释。)
- **对抗式 code audit**(11 agents,review→verify)覆盖 eval/training 代码:**2 confirmed**
  (一个 latent 的 multi-`<image>` 处理 divergence —— **已修**为镜像 reference;一个 minor 的
  intent-only lateral 伪标签 —— **已文档化**),**5 refuted**,validated 路径里 **0 correctness
  defects**。见 `docs/2implementation-details/AUDIT.md`。
- **Parity 方法学**:每个组件都有一个 PyTorch "oracle" capture(`scripts/capture_oracle_*`,
  eager attention + 显式 masks/positions)和一个 JAX gate(`scripts/parity_*`,带
  `jax_default_matmul_precision=highest` 来关闭 TF32)。Gates 断言 rel-max < 1e-3。

---

## 3. 仓库 & artifact 布局

**代码(在 git repo 里,分支 `jax-ddrive-port`):**
```
fast_ddrive/                         # PyTorch release + 我们的新增
  data/convert_wod_e2e.py            # NEW: tfrecord -> Fast-dDrive JSON(val eval + train targets)
  data/example/sample.json           # canonical 2-sample 示例(prompt ground truth)
  eval/batch_inference.py            # PyTorch eval(release 的)
  eval/evaluate_waymo_metrics.py     # 官方 ADE/RFS(release 的)
  eval/waymo_rfs_utils.py            # RFS(纯 numpy)
jax_ddrive/
  to-host-chn.md                     # 本文件(中文版;英文 to-host.md 已于 2026-06-12 退役)
  README.md
  docs/3summary/{REPORT,FEATURES}.md
  docs/2implementation-details/{ARCHITECTURE,EVAL_PIPELINE,AUDIT,01_pytorch_reference_algorithm}.md
  docs/1plans/{00_PLAN,02_tpu_plan,03_scaleup_tpu_spec,04_tpu_smallscale_validation}.md   # frozen
  docs/4collect/{HANDOFF,OVERNIGHT_PROGRESS,05_maxtext_port_progress,OVERNIGHT_TPU_PROGRESS}.md   # frozen 日志
  ddrive_jax/
    models/{rope,qwen2_5_text,vision_qwen25vl,sharded}.py
    diffusion/{masks,sasd_loss,noise,sample_sd}.py
    eval/{rope_index,scaffold,mm_sampler}.py          # NEW: JAX inference building blocks
    convert/hf_to_jax.py  lora.py  sharding.py  checkpoint.py
    train_overfit.py  train_overfit_mm.py  train_waymo_sasd_jax.py   # 最后一个是 NEW(real data)
  eval/{prep_jax_eval,jax_batch_inference,prep_train_jax}.py          # NEW: JAX eval/train drivers
  scripts/{capture_oracle_*,parity_*,debug_vit,prep_*,verify_sd_mm,run_all_verification.sh,
           run_overnight.sh, ...}        # 列举式;Phase 6/8 的数据/AR 脚本也在此(如
  #          parquet_to_ar_with_embeds.py, parquet_to_arrayrecord.py, check_real_fsdp_shard.py,
  #          build_full_dataset.sh, verify_ar_round2.{py,sh} —— §0/§2.4 引用)
  tests/{test_mask_loss,test_lora,test_noising,test_sharding,test_eval_ports,
         test_ar_pipeline,test_grain_pipeline,test_harness_fsdp,test_multihost_datafeed}.py
```

**大 artifacts(不在 git —— 在 `/home/kaiwen/data/fast-ddrive/` 下):**
```
waymo/{train (877G, 263 shards), val (226G, 93 shards), meta/val_sequence_name_to_scenario_cluster.json (479 rated seqs)}
eval/{val_rated.json (479), val_images/, prep_val_full/ (479 npz), pt_val_full_ss/, jax_val_full_sd/}
train/{train_targets.json (800), train_images/, prep_train/ (400 npz), ckpt_jax/{step_200,step_400}}
ckpt/                # HF checkpoint cache
ref_logits/*.npz     # parity gates 的所有 PyTorch oracle captures
proto_build/         # 编译好的 WOD-E2E protobuf(见 §4)
logs/                # 每次跑的 log
RESTART_starVLA_server.txt   # 怎么重启我们释放掉的用户 GPU service
```
HF model snapshot:`/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f`(下称 `$SNAP`)。

---

## 4. 环境(3 个 conda/venv;永远先 `unset LD_LIBRARY_PATH`)

| Env | Python | 用途 |
|---|---|---|
| **ddrive** `/home/kaiwen/miniconda3/envs/ddrive/bin/python` | torch 2.11+cu128, transformers 4.57 | PyTorch oracle captures、PyTorch eval、所有 CPU prep(processor + section_utils) |
| **jax** `/home/kaiwen/jax-dlm-baseline/.venv/bin/python` | jax 0.10, flax 0.12.7 (nnx), optax 0.2.8, orbax;transformers(仅 tokenizer,无 torch) | 所有 JAX compute(parity gates、training、JAX eval)。设 `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| **autovla** `/home/kaiwen/miniconda3/envs/autovla/bin/python` | py3.10, tensorflow, waymo-open-dataset-tf-2-12-0, grpcio-tools | tfrecord parsing(converter)+ 官方 ADE/RFS metric |

**构建 metric proto(重要 —— 没有 pip wheel 提供它):** WOD-E2E proto
`end_to_end_driving_data_pb2` 在 `waymo-open-dataset-tf-2-12-0`(1.6.5 和 1.6.7)里缺失。我们
从源码、对照**已安装的** descriptors 编译它,使 descriptor pool 保持一致:
```bash
pip install grpcio-tools
mkdir -p proto_build/waymo_open_dataset/protos && cd proto_build
curl -fsSL https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/end_to_end_driving_data.proto \
     -o waymo_open_dataset/protos/end_to_end_driving_data.proto
# 1) 从已安装的包 dump 一个 dataset.proto + 传递依赖的 FileDescriptorSet
#    (这样编译出的 E2E pb2 是 descriptor-compatible 的 —— 无 descriptor-pool 冲突):
python - <<'PY'
from google.protobuf import descriptor_pb2
from waymo_open_dataset import dataset_pb2
seen = {}
def collect(fd):
    if fd.name in seen: return
    p = descriptor_pb2.FileDescriptorProto(); fd.CopyToProto(p); seen[fd.name] = p
    for d in fd.dependencies: collect(d)
collect(dataset_pb2.DESCRIPTOR)                 # DESCRIPTOR 已经是 FileDescriptor(无 .file)
open("deps.pb", "wb").write(descriptor_pb2.FileDescriptorSet(file=list(seen.values())).SerializeToString())
PY
# 2) 只编译 E2E proto,从 dumped descriptors 解析 imports,输出到 site-packages
python -m grpc_tools.protoc --proto_path=. --descriptor_set_in=deps.pb \
       --python_out="$(python -c 'import site;print(site.getsitepackages()[0])')" \
       waymo_open_dataset/protos/end_to_end_driving_data.proto
python -c "from waymo_open_dataset.protos import end_to_end_driving_data_pb2; print('OK')"
```
(也记录在项目 memory `autovla-e2e-proto-compile` 里;build dir 保留在
`/home/kaiwen/data/fast-ddrive/proto_build/`。)

---

## 5. 本地复现(RTX 5090)

```bash
cd /home/kaiwen/Desktop/research/Fast-dLLM
unset LD_LIBRARY_PATH
PT=/home/kaiwen/miniconda3/envs/ddrive/bin/python
JX=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
AV=/home/kaiwen/miniconda3/envs/autovla/bin/python
SNAP=/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f
D=/home/kaiwen/data/fast-ddrive
export PYTHONPATH=$PWD/jax_ddrive XLA_PYTHON_CLIENT_PREALLOCATE=false

# (a) 所有 parity + unit gates(~10 min,会多次加载 ckpt)
bash jax_ddrive/scripts/run_all_verification.sh        # -> "ALL_VERIFICATION_PASS"(10 gates)

# (b) 构建 rated val 集(autovla env)
$AV fast_ddrive/data/convert_wod_e2e.py \
   --tfrecords "$D/waymo/val/val_*.tfrecord*" \
   --out_json $D/eval/val_rated.json --image_root $D/eval/val_images --rated_only

# (c) PyTorch eval + 官方 metric
$PT fast_ddrive/eval/batch_inference.py --model_path $SNAP \
   --eval_json $D/eval/val_rated.json --image_root $D/eval/val_images \
   --output_dir $D/eval/pt_val --mode scaffold_spec --num_gpus 1
$AV fast_ddrive/eval/evaluate_waymo_metrics.py --pred_json $D/eval/pt_val/predictions.json \
   --gt "$D/waymo/val/val_*.tfrecord*" --output_dir $D/eval/pt_val      # -> ADE/RFS

# (d) JAX eval + 官方 metric(同一个 metric backend)
$PT jax_ddrive/eval/prep_jax_eval.py --eval_json $D/eval/val_rated.json \
   --image_root $D/eval/val_images --out_dir $D/eval/prep_val
$JX jax_ddrive/eval/jax_batch_inference.py --prep_dir $D/eval/prep_val --out_dir $D/eval/jax_val
$AV fast_ddrive/eval/evaluate_waymo_metrics.py --pred_json $D/eval/jax_val/predictions.json \
   --gt "$D/waymo/val/val_*.tfrecord*" --output_dir $D/eval/jax_val

# (e) 真实数据上的 JAX SASD 训练
$AV fast_ddrive/data/convert_wod_e2e.py --tfrecords "$D/waymo/train/training_*.tfrecord*" \
   --out_json $D/train/train_targets.json --image_root $D/train/train_images \
   --with_target --max_frames 800
$PT jax_ddrive/eval/prep_train_jax.py --train_json $D/train/train_targets.json \
   --image_root $D/train/train_images --out_dir $D/train/prep_train
$JX jax_ddrive/ddrive_jax/train_waymo_sasd_jax.py --prep_dir $D/train/prep_train \
   --samples 200 --steps 400 --lr 2e-5 --ckpt_dir $D/train/ckpt_jax   # -> WAYMO_SASD_JAX_TRAIN_PASS

# 一键:(c)+(e)+(d) 顺序跑
bash jax_ddrive/scripts/run_overnight.sh
```

---

## 6. 在 TPU 上部署

模型是纯 Flax-NNX,所以它接入 TPU mesh 的方式和 `jax-mdlm-handoff` 的 `LLaDAModel` 一样。
三个保真度递增的选项;按顺序做。

### 6.1 Mesh
2D mesh `('fsdp','tp')`,经 `jax.make_mesh((n_fsdp, n_tp), ('fsdp','tp'))`。从 **pure-FSDP
`(N,1)`** 开始(最简单;3.09 B 扩展得很好)。v5e-256 → 例如 `(64,4)`;v6e → 按 chips 调。把所有
jits 包在 `with jax.sharding.use_mesh(mesh):` 里(用 `use_mesh`/`set_mesh`,**不要**用裸的
`PartitionSpec` 而无 mesh context,否则 JAX 0.10 会 raise)。

### 6.2 Sharding(FSDP param map —— 已在 `ddrive_jax/sharding.py` 里 map 好)
| Param | PartitionSpec (fsdp,tp) |
|---|---|
| `embed_tokens.embedding` [V,D] | `P('fsdp', None)`(pure-FSDP)或 `P(None,'tp')` |
| `q/k/v/o_proj.kernel` | `P('fsdp','tp')` / `P('tp','fsdp')` |
| `gate/up/down_proj.kernel` | `P('fsdp','tp')` / `P('tp','fsdp')` |
| norms / biases | `P('tp')` 或 replicated |
| activations [B,L,D] | `P('fsdp', None, 'tp')`,经 `with_sharding_constraint` |

> 注:`ddrive_jax/sharding.py` 当前只产 **pure-FSDP 列**(`fsdp_pspec` 把单个最大轴放 `'fsdp'`、
> 其余 replicate,embeddings 强制 `P()`);带 `'tp'` 的列是 Option 2 的目标,需要
> `ShardedLinear`/`ShardedEmbedding`(见 `models/sharded.py`),`sharding.py` 本身不发出 `'tp'` spec。

**Option 1 —— 仅 FSDP,无模型改动(最快):** 把 `nnx.state(model, nnx.Param)` `device_put`
到 `NamedSharding(mesh, pspec)`(最大轴放 `'fsdp'`);jit train step,配匹配的
`in/out_shardings`。`ddrive_jax/sharding.py` 正是这么做,且**在 mesh=1 时验证为 no-op**
(5090 路径不变)。仅这一项就能在 pod 上跑完整 3.09 B。

**Option 2 —— 显式 TP(吞吐):** 在 decoder 上把 `Linear→ShardedLinear`、`Embed→ShardedEmbedding`
替换(~80 LOC;primitives 在 `ddrive_jax/models/sharded.py`,JAX-0.10 `out_sharding=` plumbing,
**在 `tests/test_sharding.py` 里 unit-tested 为 mesh=1 no-op**),并按表标注 kernels。

**Option 3 —— MaxText-native(生产):** 把 `diffusion/` + SASD `loss_fn` lift 进一个 MaxText
fork,完全照 `jax-mdlm-handoff/code-fork` 参考:一个 ~5 行 `loss_fn` diff + bidirectional/
block-causal attention-mode patch;经 **grain** wire WOD-E2E;checkpoints 经 MaxText 的 Orbax
pipeline。

### 6.3 TPU 上的数据 pipeline
复用 `convert_wod_e2e.py` 把 JSON + JPEGs materialize 到 **GCS**(或改它来 emit 一个 grain/
tfrecord shard 格式)。per-sample SASD tensors(`prep_train_jax.py` 输出:input_ids, labels,
rbi, turn, scaffold, weight_vec, block α/β, pixel_values, image_grid_thw, 3D position_ids,
vision_mask)是框架中立的 numpy → 经 grain 加载。ViT image embeds 可以 per sample 预计算一次
(frozen)并缓存,或 inline 计算。

### 6.4 Checkpointing
`ddrive_jax/checkpoint.py` 是简单的**单机 in-place `StandardCheckpointer`**(只存 `nnx.state`,
无 sharding/abstract-state plumbing、无 opt_state/grain;其 docstring 自称 multi-host-ready,但实际
不带分片管线)。本地跑已用它产出 checkpoints(`train/ckpt_jax/step_*`),restore 是同一个调用。
**真正的多节点路径走生产线(§6.6 的 MaxText fork + Orbax),或 `train/checkpoint_mgr.py` 的
`CheckpointManager`(params+opt+meta+grain 四件套、sharded restore)** —— 不是这个文件。

### 6.5 TPU 上的数值
- loss / log-softmax 保持 **fp32**(`sasd_loss.py` 已 upcast);bf16 params/activations OK。
- TPU XLA 比 GPU 更严格 —— eval 里加 NaN guards,并在大跑之前**在 TPU CPU/emulator 上重跑
  Phase-1/2 parity gates**(rel < 1e-3)。
- FSDP 跨 N chips 时 per-chip 内存 ~1/N → **full fine-tune + AdamW** 可行(扔掉
  5090-only 的 bf16+remat+Adafactor+LoRA 技巧;把 `remat` 作为超长序列的旋钮保留)。

### 6.6 代码与数据发布到 GCS + 内部 ingestion(code-publish,2026-06-13)

生产路径走 **MaxText fork**(非上面的 NNX-mesh 选项);内部侧的可执行 runbook 是
`docs/6for_internal/test_training.md`(从 GCS 拉 → 内部 TPU 训练 → 导出 → 推理)。
GCS 是中转桶 `gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd`。

**代码发布(自动化,带时间戳):**
```bash
bash jax_ddrive/scripts/upload_code_to_gcs.sh   # repo 内、版本化(旧 /home/kaiwen/upload_code_to_gcs.sh 已转 stub)
# 2026-06-16 起:maxtext-dlm-fork 已并入 Fast-dLLM repo → 打包同一个 repo 的两个子目录
#   maxtext-dlm-fork + jax_ddrive(排除 .git/__pycache__/visualizations),一个 git commit 描述整包。
# 上传 gs://…/code/fastddrive-<TS>-<sha7>[-dirty].tgz + 同名 .MANIFEST.json(记 git commit SHA + dirty),
# 刷新 gs://…/code/fastddrive-LATEST.txt(指向最新 .tgz)与 fastddrive-LATEST-MANIFEST.json(给出最新 commit)。
# 解包后结构:fastddrive-<TS>-<sha7>/{maxtext-dlm-fork, jax_ddrive, MANIFEST.json}
```
内部侧把它下载并解到 google3 源码树:
`/google/src/cloud/kaiwenh/fastddrive/google3/experimental/waymo/users/xqin/third_party/fastddrive-<TS>`
(命令见内部 runbook §4)。

**数据发布(已在 GCS):** `maxtext_sasd_params_base/`(base 参数)、
`wod_e2e_sasd_distilled_0613-400_baseViT_v2_ar/`(数据集)、`base_qwen25vl_3b_snapshot/`
(导出参照 + tokenizer)。**每个 artifact 同目录放一份 `DATA_MANIFEST.json`**(内容指纹 `digest`(GCS crc32c)+ git SHA;owner 跑 `bash jax_ddrive/scripts/data_manifest.py gs://…/<artifact> --upload` 生成)→ 运行时把 `digest` 记进 `validation_log` 的 `data_provenance` 事件,即可反查/校验用的是哪份数据(数据版本化,补齐"命名路径原地覆盖、无版本"的缺口)。**大数据在内部走 CNS**(不放 pod $HOME):经 Cloudtop 中转——
`gcloud storage cp -r gs://…/<artifact> ~/ddrive_stage/` → `fileutil cp -R -parallelism 50
~/ddrive_stage/<artifact> $DATA_ROOT/` → 删本地。**CNS 大池**(取代旧的 500G `/cns/sf-d/…`):
```
DATA_ROOT=/cns/is-d/home/chauffeur/perception_training/kaiwenh/data
```

---

## 7. 标准 —— 什么已验证 vs 什么是预期

### 7.1 本地已验证(RTX 5090)—— 验收证据
| 主张 | 怎么验证 | 结果 |
|---|---|---|
| Text decoder 匹配 PyTorch | `parity_text.py`(fp32, highest) | logits rel-max **3.2e-5**, top-1 **100%** |
| SASD loss + mask 匹配 | `parity_sasd.py` | loss **7.9e-8**, mask **bit-identical** |
| ViT 匹配(架构) | `parity_vit.py` | **4.0e-5**(end-to-end 2.6% = 良性 cuDNN seed) |
| Multimodal forward 匹配 | `parity_mm.py` | hidden 3.2e-5 / logits **7.7e-5** / top-1 100% |
| 训练在学(loss ↓) | `train_overfit{,_mm}.py` | text **3.84→1.47**, MM **0.999→0.701** |
| 真实数据训练(loss ↓)+ ckpt | `train_waymo_sasd_jax.py` | **0.6815→0.6005**, Orbax step_200/400 |
| JAX generation == PyTorch | `verify_sd_mm.py` | trajectory **0.01 m**, valid JSON |
| Eval pipeline 正确(两套栈) | 完整 479 rated val + 官方 metric | PyTorch **0.814/1.990/7.914**, JAX **0.839/2.072/7.929**, 100% parse |
| Converter prompt 正确 | 对 `sample.json` byte-compare | **exact**(1896 chars) |
| 无回归 / defects | 10-gate 套件 + 11-agent audit | gates pass;validated 路径里 **0 defects** |

### 7.2 TPU 上的预期 —— 给 host 的验收标准
跑这个**验证阶梯**(每一步 gate 下一步):
1. **mesh=1 单 chip**:对固定 batch,SASD loss 在 fp32 噪声内等于 5090 值
   (mesh=1 时 sharding 按构造是 no-op)。*预期:在 ~1e-4 内匹配。*
2. **FSDP `(N,1)` vs 1-chip**:单步 loss 相等(同 seed/batch)**在 1e-4 内**。*确认
   FSDP sharding 不改变数学。*
3. **Overfit 那 2 个示例样本**:loss 单调下降(复现本地 Phase 3)。
4. **真实 WOD-E2E 训练**:loss 在 multi-thousand-sample 跑里下降;checkpoints 能 restore。
5. **TPU/host 上 eval**:跑 inference → predictions.json → **同一个** `evaluate_waymo_metrics.py`。
   *预期 ADE/RFS 在本地数字的 run-to-run 噪声内*(PyTorch 参考 0.814/1.990/
   7.914;JAX section-diffusion 0.839/2.072/7.929)。RFS ≈ 7.9 和 ADE@3s < 1.0 m 是正确 setup
   的 sanity band。

**TPU 上"done"意味着什么:** 步骤 1–2 证明 port 在 sharding 下数值相同;
步骤 4 证明训练能 scale;步骤 5 证明 eval 能复现。吞吐随后是一个优化
轴(Option 2/3 TP、KV-cache decode)—— 不是 correctness gate。

### 7.3 已知 gap / non-goals(让 host 不被惊到)
- **JAX decode 无 KV-cache**(~16 s/sample,每个去噪步 full-seq recompute)。离线 eval 够用;
  做 serving 要 port 一个 block-wise KV-cache(PyTorch `scaffold_speculative_sample` 是
  参考)—— *未做*。
- **Whole-model TP** 在真实 mesh 上是*机械的但尚未执行*(primitives + no-op tests
  已有;上面 Option 2)。
- **训练文本标签**对非 trajectory sections 是伪标签(§2.4)—— trajectory 是
  真 GT。要生产精度,提供真标签或 teacher-distill。
- **未推到 GitHub** —— 分支 `jax-ddrive-port` 仅本地。

---

*本文档与 `docs/2implementation-details/EVAL_PIPELINE.md`(pipeline 细节)、
`docs/1plans/02_tpu_plan.md`(TPU 细节)、`docs/2implementation-details/AUDIT.md`(audit)、
`docs/3summary/REPORT.md` / `docs/4collect/HANDOFF.md`(build log)一起维护。
为 Fast-dDrive JAX port 生成,分支 `jax-ddrive-port`。*
