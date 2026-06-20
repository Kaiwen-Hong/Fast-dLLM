# 07 — Fidelity Fixes (2026-06-20)

<!-- FROZEN RECORD (1plans/ = write-once, per ../README.md rule 4). This is a dated record of
what happened on 2026-06-20, not a forward plan. Later corrections to numbers here = inline
bracketed [订正 YYYY-MM-DD: ...] annotations, never a rewrite. Downstream of 06_trainable_vit_plan.md. -->

Frozen record of the 2026-06-20 fidelity-fix pass — the faithfulness corrections made to the
trainable-ViT / SASD path after [`06_trainable_vit_plan.md`](06_trainable_vit_plan.md). Per the docs
convention `1plans/` is write-once; this is a record of what happened, not a forward plan.

## 0. TL;DR

我们把 JAX 数据 prep 改成在 **720 image tokens（200704 px = 原版 Fast-dDrive 分辨率）** 下对原始
Fast-dDrive 逐字节忠实，并修掉了 `EXP_BUDGET=192`（block_length×6）的 explanation padding bug —
此前 `EXP_BUDGET=32` 只 pad 到 ~1 个 block，训练看到的 scaffold 与 inference / 原版不一致。**Verdict：5
个 fidelity gap 全部修复并落地，且 5090 GPU 上 720 trainable 路径端到端 **PASS**（loss 7.945 / 3.086B params /
peak 19.6 GB / step-0 ckpt 保存）——证明 fixes + wiring 在 720 下正确。但同一 720 trainable step 在
**v5e-16（16G HBM/chip）compile 阶段 OOM**（HLO temporaries 17.11G > 可用 15.75G，`TRAINABLE_EXIT=1`）：这是
**v5e 容量问题、非保真度问题**，需更大 HBM 的 TPU（v6e 32G）或 remat/FSDP 才能在 TPU 上跑 720；168-res 的
trainable（run be47jjta8）此前在同款 v5e-16 是 PASS 的。** [订正 2026-06-20（同日）见 §9：OOM 根因其实是 loss 侧 fp32 全词表 log_softmax（随 2L 翻倍），**非** v5e 容量 / ViT；已用 chunked-CE 修复（bit-identical, loss 7.945），capped-GPU proxy 实测 step 真峰 ~12.66G < 15.75G，**v6e/remat/FSDP 非必需**；真实 v5e ~$5 确认仍 PENDING。] 衡量基准是原始 PyTorch dataloader（`CustomMultiModalDataset`）+
released ckpt 的 vision 分辨率。规范数字不在本文重述，见
[`../0overview/02_gotchas.md#规范数字框-canonical-numbers`](../0overview/02_gotchas.md#规范数字框-canonical-numbers)；本文 §6 给出本次 720 测量值。

## 1. Setup — what was reviewed / how (and against what oracle)

- **Who/what：** 本次 pass 由 Claude agent 在本地只读对照 + 直接编辑完成。对照的 oracle 是 HF
  snapshot `0fda810`（`Efficient-Large-Model--Fast-dDrive`）里原始 PyTorch dataloader 源码，以及原版
  train/eval 脚本里的分辨率常数。每个 finding 都先在源码里定位到 file:line 再改（adversarial-verify
  rule：先复现 / 锚定原始行为，再改 JAX 端）。
- **忠实度参考（每个 fix 的 oracle）：**
  - explanation budget / clean_nulls 顺序 → `third_party/lmflow/datasets/multi_modal_dataset_fast_ddrive.py`
    （`clean_nulls` 在 :793–804；`exp_total_budget = block_length*6` 注释 "Default: 32*6 = 192" 在 :818–820），
    以及 HF snapshot `0fda810` 的 `section_utils.py:153`（`exp_budget = explanation_block_size * explanation_max_blocks  # default 192`）。
  - vision 分辨率 → 原版 `fast_ddrive/train_scripts/finetune_fast_ddrive.py:159`（`min_pixels = max_pixels = 200704`）
    与 `fast_ddrive/eval/batch_inference.py:1169`（`MIN_PIXELS` 默认 200704）。
  - 两条 prep 路径不漂移 → 新增的 in-env gate `jax_ddrive/scripts/validate_prep_consistency.py`（CPU-only，PyTorch
    `ddrive` env，无模型权重；同一 toy json、同一分辨率跑 train+eval 两条 prep 并 assert 一致）。
- **范围：** 改的是 prep（training 侧 `prep_train_jax.py`、overfit `prep_overfit_data_mm.py`）+ MaxText
  trainable in-graph ViT 的 grid 几何（config-driven）+ AOT shape。inference prep（`prep_jax_eval.py`）只作为
  consistency gate 的对照被读取，不改。

## 2. Fidelity gaps found (what was unfaithful + why it matters)

| # | severity | gap (drift from reference) | where (file:line) | evidence it was real |
|---|---|---|---|---|
| 1 | **(HIGH)** | `EXP_BUDGET=32`：explanation 只被 NULL-pad 到 ~1 个 block，而原版固定 pad 到 6 个 block（192 token），训练 scaffold ≠ inference/原版 | `prep_train_jax.py:21`；同 bug 在 `scripts/prep_overfit_data_mm.py` 的常数行 | 原版 `multi_modal_dataset_fast_ddrive.py:818-820` 明写 `exp_total_budget = block_length*6` 注释 "Default: 32*6 = 192"；`section_utils.py:153` `exp_budget = ... # default 192` |
| 2 | **(HIGH)** | 缺 `clean_nulls` 归一：未在测量 section 预算前剥掉预存的 `<|NULL|>`，导致 per-section 长度量测偏离真实内容长度 | `prep_train_jax.py`（修前 `obj = json.loads(gpt)` 处，无 helper） | 原版 `multi_modal_dataset_fast_ddrive.py:793-804`：`def clean_nulls(obj): ... obj.replace('<|NULL|>','')` 且 `data_obj = clean_nulls(data_obj)` 在 exp padding(:815-828) **之前**执行 |
| 3 | **(HIGH)** | vision 分辨率跑的是已退役的 `784 / 784*64` 下采样（168 image tokens），偏离 released model 的 720-token（200704 px）分辨率 | `prep_train_jax.py:72-73`（`--min/max_pixels` 默认 784 / 784*64） | 原版 `finetune_fast_ddrive.py:159-160`（`min/max_pixels=int(os.environ.get(…,200704))`）；`batch_inference.py:1168`（字面量 `min_pixels = max_pixels = 200704`） |
| 4 | **(MED)** | trainable in-graph ViT 的 patch grid 硬编码成 `(1,16,14)×3`（672 patches / 168 tokens），无法跟随 720 分辨率；structural 几何与新分辨率不匹配 | `sasd_vit_ingraph.py:69`（`SASD_GRID_THW=((1,16,14)…)`）；`decoders.py` 实例化处 | 修前 grid 与 §3 的 200704 px / 720-token resolution 不一致；几何（window/2D-RoPE/mask）由 grid 决定 |
| 5 | **(MED)** | AOT `get_shaped_batch` 把 `sasd_pixel_values` 的 patch 轴硬编码成 `672`，绑死在退役的 168-token grid 上；720 下 AOT 抽象 shape 错 | `maxtext_utils.py:189`（修前 `(gbs, 672, 1176)`） | `N=sasd_num_image_tokens` 已是 doubled token count（=patches/2），正确应为 `2*N`（720 → 2880）；见 `maxtext_utils.py:173-180` |

> 说明：以上均为 **真实漂移（bug / 不忠实），不是 NUMERICAL finding**。168 vs 720（50176 vs 200704 px）本是
> `02_gotchas` 规范数字框里登记的"合理变体"，但 released model 的 vision 分辨率是 720；继续跑 168 即偏离 oracle。

## 3. Code fixes — committed to local working tree (branch `jax-ddrive-port`); bundle re-pack DEFERRED

| file | fix |
|---|---|
| `jax_ddrive/eval/prep_train_jax.py` | `EXP_BUDGET 32 → 32*6 (=192)`（:21，含解释 bug 的多行注释）；新增递归 `_clean_nulls(x)`（:30-39）镜像原版 `clean_nulls`；`obj = json.loads(gpt)` → `obj = _clean_nulls(json.loads(gpt))`（:47）；`--min_pixels/--max_pixels` 默认 `784 / 784*64` → `200704 / 200704`（:72-73）；docstring 改指向原始 `CustomMultiModalDataset` + 192/6-block 预算（:43-44） |
| `jax_ddrive/scripts/prep_overfit_data_mm.py` | 同步 `EXP_BUDGET 32 → 32*6 (=192)`（单行常数 + 注释 "block_length*6 (original); was 32 (bug)"），与 `prep_train_jax.py` 对齐 |
| `maxtext-dlm-fork/src/maxtext/configs/types.py` | `DiffusionObjective` 新增 `sasd_vit_grid_thw: str = Field('1,32,30', …)`（位于 `sasd_num_image_tokens` 与 `sasd_data_dir` 之间），描述 per-image ViT grid 't,h,w'；`'1,32,30'⇒2880 patches→720 tokens`，`'1,16,14'⇒672→168`(retired)；`n_patches == 2*sasd_num_image_tokens`；仅 trainable 路径读取 |
| `maxtext-dlm-fork/src/maxtext/diffusion/sasd_vit_ingraph.py` | `SASD_GRID_THW (1,16,14)×3 → (1,32,30)×3`（:69，注释 "2880 patches → 720 image tokens (200704 px = original Fast-dDrive res); retired downscale was (1,16,14)→672→168"）；新增 `sasd_grid_thw_from_str(grid_str='1,32,30', n_images=3)`（:72-76）从 config 字符串构建 n-camera grid tuple |
| `maxtext-dlm-fork/src/maxtext/layers/decoders.py` | import 增加 `sasd_grid_thw_from_str`；`SasdInGraphViT(…)` 增加 `grid_thw=sasd_grid_thw_from_str(getattr(cfg,'sasd_vit_grid_thw','1,32,30'))`，使 in-graph ViT 用 config-selected grid（getattr fallback `'1,32,30'` 保证 config 字段缺失时安全） |
| `maxtext-dlm-fork/src/maxtext/utils/maxtext_utils.py` | `sasd_pixel_values` 的 patch 轴 `672 → 2*N`（:189，`ShapeDtypeStruct((gbs, 2*N, 1176), f16)`），注释说明 `n_patches = 2*N`（`336→672 @168, 1440→2880 @720`，`dim 1176 = 3*2*14*14`），使 AOT 抽象 shape 随分辨率正确 |

新文件：

| file | purpose |
|---|---|
| `jax_ddrive/scripts/validate_prep_consistency.py` | CPU-only sanity gate（PyTorch `ddrive` env，无模型权重）：在**同一** toy json、**同一** `--pixels`(=min=max) 下 subprocess 跑 train prep(`eval/prep_train_jax.py`) + eval prep(`eval/prep_jax_eval.py`) 到两个 temp dir，再逐样本 assert：(a) `pixel_values` byte-identical(`np.array_equal`)、(b) `image_grid_thw` identical、(c) `train['input_ids']` 与 `eval['x_t0']` 的 prompt-token LCP > 50（二者只在 train 有 GT token vs eval 是 `MASK_ID=151665` 处分叉）。同时**报告**双方 `n_blocks`（train=实际 GT 长度；eval=`max(rbi)+1`=固定生成预算），仅当 block count ≤0 或 eval<train 才判 insane。打印逐样本 OK/MISMATCH + 末行 `PREP_CONSISTENCY_PASS` / `PREP_CONSISTENCY_FAIL`，exit 0/1 |

**Verified：** GPU 720 toy step-0 端到端 **PASS**（`/tmp/toy720_gpu.log`：3.086B params、lm_loss 7.945、total_weights
808.0、peak 19.6 GB、step-0 checkpoint 已保存、无 traceback；见 §6）。`sasd_vit_grid_thw=1,32,30` 在 GPU
config 中生效。**NOT yet run：** (a) `validate_prep_consistency.py` 的 `PREP_CONSISTENCY_PASS` 哨兵尚未在
本记录中跑出落账（脚本已就位，结果待补）；(b) 720 trainable 路径在 v5e-16 上**已跑、但 compile 阶段 OOM**（HLO temporaries 17.11G > v5e 单芯 15.75G HBM，
`TRAINABLE_EXIT=1`，见 §6.3）——**TPU 容量问题，非保真度问题** [订正 2026-06-20（同日）见 §9：根因是 loss 侧 fp32 `[N,V]` log_softmax，已用 chunked-CE 修复，v6e/remat/FSDP **非必需**；真实 v5e ~$5 确认仍 PENDING]；(c) GPU log **没有**显式 `GPU_EXIT` token —
日志在 step-0 metrics + ckpt save 后即结束，task-wrapper 的 `GPU_EXIT` 行未被这个 log 文件捕获（无 error + 干净
step-0 save ⇒ success，但 honest gap：缺显式 exit 哨兵）。bundle 未 re-pack / 未 republish 到 GCS LATEST（DEFERRED）。

## 4. Doc / script fixes

- `prep_train_jax.py` docstring（:43-44）改成引用权威源 `CustomMultiModalDataset` 与"FIXED 192-token / 6-block
  budget"，不再指向 sibling JAX 脚本 `prep_overfit_data_mm.py`。
- 规范数字（200704 / (1,32,30) / 2880 / 720 / EXP_BUDGET=192 / L=1856 / 2L=3712 / N=1440 / 3.086B …）的**唯一家**
  是 [`../0overview/02_gotchas.md#规范数字框-canonical-numbers`](../0overview/02_gotchas.md#规范数字框-canonical-numbers)（README rule 7，one-fact-one-home）。本次把 720-run 这组值在 §6 落账，并在
  `02_gotchas` 规范数字框**新增 720 canonical 块、把被取代的「720＝paper-eval 诊断值/从不断言」行订正为
  released-model 忠实分辨率、并把 168/784 一组标为 retired 下采样**（2026-06-20，附指回本文）；其它 living 文档只 LINK 不重述。
- 对 frozen 文档的更正以 inline `[订正 2026-06-20: …, 见 02_gotchas#规范数字框]` 注释方式追加，**不重写正文**：
  - [`06_trainable_vit_plan.md`](06_trainable_vit_plan.md)：其 §8 Companions 追加一条指向本文 07 的 bullet
    （companion-link 约定；§9 滚动 log 留给 run 进展，不塞 cross-ref）。
  - [`00_PLAN.md`](00_PLAN.md)：若正文残留 168-token/50176-px 作为"当前分辨率"的描述，加 `[订正 2026-06-20:
    released model 分辨率为 720 tokens / 200704 px；168 为已退役下采样，见 02_gotchas#规范数字框]`，保留原文。
- living/stable 文档（to-host-chn、DATASET_V2、`0overview/*`、`3summary/*`）在本次 landing 的**同一改动**里就地更新到现状
  （README rule 2），见 §5/§8。

## 5. Actions executed (2026-06-20)

1. 修复 `EXP_BUDGET 32→192` + 新增 `_clean_nulls` + 接线到 `process_gpt` + `min/max_pixels` 默认改 200704
   于 `jax_ddrive/eval/prep_train_jax.py`，并同步 overfit 脚本 `jax_ddrive/scripts/prep_overfit_data_mm.py` 的常数。
2. 把 trainable in-graph ViT 的 grid 改成 config-driven：`types.py` 加 `sasd_vit_grid_thw`，
   `sasd_vit_ingraph.py` 默认 grid 改 `(1,32,30)×3` + 加 `sasd_grid_thw_from_str()`，`decoders.py` 接线
   `grid_thw=sasd_grid_thw_from_str(getattr(cfg,'sasd_vit_grid_thw','1,32,30'))`，`maxtext_utils.py` 的
   AOT patch 轴 `672→2*N`。
3. 新增 CPU-only consistency gate `jax_ddrive/scripts/validate_prep_consistency.py`（train vs eval prep 同分辨率
   byte-level 对照，哨兵 `PREP_CONSISTENCY_PASS`/`FAIL`）。
4. 用修好的 prep 构建 720 toy AR（pixels-only，无 `image_embeds`）：
   `npz {00000,00001} → parquet/train-00000-of-00001.parquet → ar/train-00000-of-00001.arrayrecord`
   （本地 `/tmp/toy_ar_build/`），并镜像到 GCS
   `gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd/wod_e2e_sasd_toy720_v2_ar/train-00000-of-00001.arrayrecord`
   （`gsutil ls` 确认存在）。2 个样本，均 `L=1856 / n_blocks=11`，`image_grid_thw=[[1,32,30]×3]`，
   `pixel_values (2880,1176) f16`，`has image_embeds: False`。
5. 在 5090 GPU 上跑 720 toy step-0（`sasd_vit_trainable=true`），落账于 `/tmp/toy720_gpu.log`（§6 PASS）。
6. 在 v5litepod-16 @ us-south1-a（`--spot`，state=READY/WINNER）warm up SSH 并端到端跑 setup+RUN：ckpt 本地拉取、
   venv+deps、BASE restore 均 OK，但 720 trainable step 在 `compile()` 阶段 **OOM**（17.11G>15.75G HBM/chip，
   `TRAINABLE_EXIT=1`，见 §6.3）；pod 由 trap 删除，三 zone 0 残留。

## 6. Validation + canonical numbers

### 6.1 Canonical numbers (720) — SSOT 见 02_gotchas

> 规范数字的唯一家是 [`../0overview/02_gotchas.md#规范数字框-canonical-numbers`](../0overview/02_gotchas.md#规范数字框-canonical-numbers)；下表为本次 720-run 在本记录的落账副本；`02_gotchas` 规范数字框已据此**新增 720 canonical 块**（2026-06-20）。"规范值（canonical constant）" vs
> "测量值（不是代码常数）" 已逐行区分。

| name | value | kind | derivation | ground-truth ref |
|---|---|---|---|---|
| min_pixels = max_pixels | **200704** | 规范值 | 256·28·28 = 200704 px；原版 eval 分辨率（min=max 锁定 Qwen2.5-VL processor） | `prep_train_jax.py:72-73`；`scripts/capture_oracle_sd_mm.py:31-32`；`sasd_vit_ingraph.py:67` |
| image_grid_thw / cam | **(1, 32, 30)** | 规范值 | 200704 px 下 Qwen2.5-VL patcher 出 t=1,h=32,w=30；3 个 WOD-E2E 相机同 grid | `sasd_vit_ingraph.py:69`；数据 `/tmp/toy_ar_build/npz/00000.npz`；GPU log `sasd_vit_grid_thw=1,32,30` |
| patches / image | **960** | 规范值 | 1·32·30 = 960 | `sasd_vit_ingraph.py:69`；1·32·30=960 |
| TOTAL patches (3 cams) | **2880** | 规范值 | 3·960 = 2880（= 2·N，N=1440）；`pixel_values` 存为 [2880,1176] | `sasd_vit_ingraph.py:67-69`；npz `pixel_values (2880,1176) f16` |
| spatial_merge_size | **2**（merge unit = 2²=4） | 规范值 | Qwen2.5-VL ViT 2×2 spatial merge：tokens = patches/4 | `vision_qwen25vl.py:34` `spatial_merge_size:int=2`，:48 `return self.spatial_merge_size**2` |
| image tokens (single) | **720** | 规范值 | 2880 / 4 = 720（未 double） | `sasd_vit_ingraph.py:67-68`；2880/4=720 |
| sasd_num_image_tokens (N, DOUBLED) | **1440** | 规范值 | N 是已 double 的 token count = 2·720；doubled [2L] 行带 1440 IMAGE_TOK；in-graph embeds [2B,1440,D] | GPU log `sasd_num_image_tokens=1440`；`maxtext_utils.py:173-180` 注释 :175 |
| n_patches = 2·N | **2880** | 规范值 | trainable in-graph ViT 吃 pixels，patch 轴 = 2·N = 2·1440 = 2880；dim 1176 = 3·2·14·14 | `maxtext_utils.py:186-189`；npz `pixel_values (2880,1176)` |
| EXP_BUDGET | **192** | 规范值 | block_length·6 = 32·6 = 192（explanation NULL-pad 到固定 6 block）；WAS 32 (bug) | `prep_train_jax.py:21`；`scripts/capture_oracle_sasd.py:21` |
| block_length (BD) | **32** | 规范值 | semi-AR / per-block Beta 噪声的 block 大小；EXP_BUDGET = 6·BD | `prep_train_jax.py:21` |
| section weights | **{CO 1.5, exp 1.0, FMB 2.0, traj 3.0}** | 规范值 | per-section loss 权重，写入 `weight_vec`，gate `section_weighted_ce` | `prep_train_jax.py:25`；`ddrive_jax/train_waymo_sasd_jax.py:7` |
| per-section Beta (α,β) | **{CO (1,2), exp (1,1), FMB (1,1.5), traj (2,1)}** | 规范值 | 每 section 自带 Beta(α,β) per-block masking-rate；`t=rng.beta(α,β)→p_mask`，存 `block_alpha/beta` | `prep_train_jax.py:26-27`；`ddrive_jax/diffusion/noise.py:25`（`t = rng.beta(block_alpha, block_beta)`） |
| sasd_seq_len (L) | **1856** | 规范值（720 override） | 720-run 的 per-sample L（168-res 时为 928；image budget double 推到 1856）；npz L=1856 | GPU log `sasd_seq_len=1856`；npz manifest；`maxtext_utils.py:156` |
| max_target_length (2L) | **3712** | 规范值 | 2·L = 2·1856 = 3712（noisy\|clean concat）；SASD doubled batch [2B,2L] | GPU log `max_target_length=3712`；`maxtext_utils.py:159,163` |
| model parameters | **3.086 billion** | 测量值（trace 报） | Qwen2.5-3B text decoder + in-graph trainable Fast-dDrive ViT body | `/tmp/toy720_gpu.log:774` `number parameters: 3.086 billion` |

> 注：168 vs 720（50176 vs 200704 px）、L=928 vs 1856 是 `02_gotchas` 规范数字框登记的"合理变体"（分辨率相关），
> 不是"曾见错值"——不要把退役的 168 当错值去"修"它的历史记录；当前 truth 是 720。

### 6.2 GPU validation — PASS

- device：RTX 5090（5090，`sasd_vit_trainable=true`，BASE 加载 3.086B）。
- step-0 metrics：`lm_loss 7.945`（perplexity 2820.114）、`total_weights 808.0`。
- peak memory：**19.6 GB**（Output 5.8 / Temp 13.8 / Argument 5.8 / Host 0.0；device residency 5.8/29.16 GB）。
- exit：step 0 干净完成，checkpoint 在 step 0 已保存，无 traceback。**honest gap：** `/tmp/toy720_gpu.log` 中
  **没有** `GPU_EXIT` token（log 在 step-0 metrics + ckpt save 后即结束，wrapper 的 `GPU_EXIT` 行未被此 log 捕获）。
- **`TOY720_GPU_STEP0: PASS`**（720 trainable 路径端到端在 GPU 上跑通 step-0，loss 有限、ckpt 已存）。

### 6.3 TPU validation — v5e-16 上 720 trainable **OOM**（compile 阶段；非保真度问题）

- v5litepod-16 @ us-south1-a（`--spot`，16 chips × 16G HBM）已获取并端到端跑完：ckpt 本地拉取 OK（RAB workaround）、
  venv+deps 装好、`sasd_vit_trainable=true` 路径 engaged（`waymo_sasd_data_processing.py:174`）、BASE params restore OK
  （23 GiB，2.05s）。**但 `p_train_step.lower(...).compile()`（`train.py:677`）在 compile 阶段 OOM：**
  `RESOURCE_EXHAUSTED: HLO temporaries 17.11G > available HBM 15.75G`（per-chip），`TRAINABLE_EXIT=1`。
- **根因：v5e 单芯 16G HBM 装不下 720 trainable step 的 per-chip 临时量（17.11G）。** 当前 sharding 是纯 data-parallel
  `(1,1,1,16,…)`、per_device_batch_size=1、ViT params REPLICATED——activation/临时量没被切到 16 芯上。5090（32G）
  能装是因为单卡 32G > 17.11G（peak 实测 19.6G，见 §6.2）。168-res 的 trainable（run `be47jjta8`）此前在同款 v5e-16
  **PASS**——因为 168 的 ViT patches（672 vs 2880）+ L（928 vs 1856）小得多，临时量远低于 16G。
- 次要：output ckpt manager 对 `gs://ddrive-sasd-ussouth1-8a53f5ab/.../checkpoints` 报了两次 RAB
  `Precondition check failed`（400 FAILED_PRECONDITION），但**非致命**（found 0 steps 后继续）；致命的是上面的 HBM OOM。
- **`TPU_VIT_720: OOM (v5e-16, 17.11G>15.75G)`**。补救见 §7。 [订正 2026-06-20（同日）: 根因已查清 + 已修复 —— OOM 的真正大头不是 ViT/sharding，而是 loss 侧 `diffusion/sasd.py:_ce_per_token` 里那块 fp32 全词表 `[N,V]` log_softmax（N=2L=3712 随分辨率线性翻倍）；已用 chunked-CE 修掉，bit-identical（loss 7.945 不变），capped-GPU proxy 实测 step 真峰值降到 ~12.66G < 15.75G v5e 预算。详见同日追加的 §9。本段当时把 OOM 归到「v5e 容量 / ViT activation 没切片」的诊断**保留为历史记录**，但已被 §9 的根因订正。]

### 6.4 Reproduce

- toy AR（720，pixels-only）：本地
  `/tmp/toy_ar_build/ar/train-00000-of-00001.arrayrecord`（2.95 MB）；GCS 镜像
  `gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd/wod_e2e_sasd_toy720_v2_ar/train-00000-of-00001.arrayrecord`。
  2 样本，均 `L=1856 / n_blocks=11`，`image_grid_thw=[[1,32,30]×3]`，`pixel_values (2880,1176) f16`，`has image_embeds: False`
  （ViT 在 train 时 in-graph 跑，符合 dataset-v2 "pixels+structural, embeds computed in-graph" AR 契约）。
- 关键 override flags（720 run）：`sasd_vit_trainable=true sasd_seq_len=1856 sasd_num_image_tokens=1440
  max_target_length=3712 sasd_vit_grid_thw=1,32,30`（其余分辨率常数随 §6.1 推导）。

## 7. Open follow-ups (NOT done — flagged)

- ⏳ **DEFERRED：415k 全量重建 @ 720 + EXP_BUDGET=192 + pixels-only。** blocker：per-sample npz 中间产物
  ≈ **2.8 TB**（720 下每样本 `pixel_values (2880,1176) f16` 体积大），vs 本地仅 **~1.1 TB** 空闲盘。下一步：chunked /
  streaming build（边生成 npz → parquet → arrayrecord 边删中间 npz，不落全量 2.8 TB），落 GCS。会进 `6for_internal` 的
  大规模重建 runbook。
- ⏳ **TODO：跑 `validate_prep_consistency.py` 并落 `PREP_CONSISTENCY_PASS` 哨兵**（@ 200704 train-res 与 50176
  对照各一次），把结果回填本记录 §6 作为 prep 不漂移的 grep-able 证据。脚本已就位，结果未落账。
- ⏳ **TODO：让 720 trainable 在 TPU 上跑通（当前 v5e-16 compile OOM 17.11G>15.75G，见 §6.3）。** [订正 2026-06-20（同日）: OOM 根因已定位（loss 侧 fp32 `[N,V]` log_softmax，非 ViT/sharding）并以 chunked-CE 修复，capped-GPU proxy 预示 step 在 v5e 装得下（真峰 ~12.66G < 15.75G），见 §9；下面三条路是当时记录的候选补救，现已被「先修 CE 大头」取代——v6e/FSDP 不再是前置条件，仅真实 v5e ~$5 run 确认仍 PENDING（§9）。] 三条路（按优先级）：
  (1) 换更大 HBM 的 TPU——**v6e（32G/chip）** 最直接（GPU 32G 已证 19.6G peak 装得下）；(2) **activation remat /
  gradient checkpointing** 砍 compile-time 临时量；(3) **FSDP**（params+optimizer+activation 切到 fsdp 轴，而非纯
  data-parallel replicate）。这是 06 §0.1 "trainable step DOES execute on TPU" 在 720 下的复核——168 已 PASS
  （be47jjta8），720 待解容量。
- ⏳ **TODO：撰写 `transfer-dataset.md` 与 `wode2e-ar-pipeline.md`**（dataset-v2 720/192/pixels-only 的传输 + AR
  pipeline 文档），落 `6for_internal/`。
- ⏳ **TODO：归档 `7for_internal_inference/`**（inference runbook 收口，避免与 720 训练侧 prep 漂移）。
- ⚠️ benign smell（非 bug）：GPU log 缺显式 `GPU_EXIT` token（§6.2）——是日志捕获问题而非失败信号；下次跑确保 wrapper
  的 exit 行落进同一 log 文件。

## 8. Companions

- 里程碑日志：在 [`../4collect/08_trainable_vit_progress.md`](../4collect/08_trainable_vit_progress.md) append 一条 2026-06-20 dated 条目（README rule 2：frozen 1plans 留发现故事，living 文档留当前 truth）。
- 上游 plan：[`06_trainable_vit_plan.md`](06_trainable_vit_plan.md)（其 §8 Companions 追加指向本文的 bullet）；总线
  [`00_PLAN.md`](00_PLAN.md)（必要处加 `[订正 2026-06-20: …]` 注释，不重写）。
- 规范数字 / 新 gotcha：[`../0overview/02_gotchas.md#规范数字框-canonical-numbers`](../0overview/02_gotchas.md#规范数字框-canonical-numbers)（720 一组值已同步；新增 gotcha：trainable
  prep 必须 `EXP_BUDGET=192` + `clean_nulls` + 200704 px，否则训练 scaffold/分辨率偏离 released model）。
- runbook：`6for_internal/test_training.md`（720 override flags 见 §6.4）。
- memory keys：[[fast-ddrive-trainable-vit]]、[[fast-ddrive-trainable-vit-tpu-run]]、[[sasd-inference-b1-b2-gotchas]]
  #10、[[fast-ddrive-dataset-v2-spec]]（720/192/pixels-only AR 契约）。
## 9. Follow-up（同日 2026-06-20）：720 OOM 根因 + chunked-CE 修复

> 本节是**同日追加的 addendum**（append-only，newest-at-bottom；遵守 frozen 规则只追加、不重写正文，授权见 ../README.md），订正 §6.3 / §7 把 720 v5e OOM 当成「容量 / ViT
> sharding」问题的初判。规范数字仍只在 [`../0overview/02_gotchas.md#规范数字框-canonical-numbers`](../0overview/02_gotchas.md#规范数字框-canonical-numbers)，本节只 LINK 不重述（loss 7.945、N=2L=3712、V=151936 等已在 §6.1）。

### 9.1 诊断 —— OOM 的真正大头

- **真凶：loss 侧的 fp32 全词表 log_softmax。** `diffusion/sasd.py:_ce_per_token`（causal -1 shift + 2L packing 后）对一个
  flattened `[N, V]` logit 矩阵跑 `jax.nn.log_softmax(...astype(fp32))`，N=2L=3712、V=151936（Qwen2.5 词表）。un-chunked
  路径物化整块 fp32 `[N,V]`：前向 ~2.26G + 反向 softmax 梯度再 ~2.26G ≈ **单个瞬时 ~4.5G fp32 峰**（fp32 [3712,151936]×2 = 4.51G）。
  这块**随 2L 线性翻倍**：168-res 时 N=2L=1856 只 ~2.26G、整步装得下 16G v5e；翻到 720（N=2L=3712）就把 XLA 估的 HLO
  临时量推到 17.11G > 15.75G 而 OOM。机制写死在源码注释 `sasd.py:29-33`，并被数学对上（== 「~4.5G fp32-softmax peak」）。
- **死路一：ViT remat = 0 效果。** 对 vision body 加 remat 后 static metric **逐字节不变**（Total 19.6G / Temp 13.8G，与
  baseline 完全相同，`/tmp/toy720_gpu_remat.log`），loss 仍 7.945。证明 dominant tensor 根本不在 ViT —— remat vision encoder
  对这个 OOM 毫无帮助（订正 §6.3「ViT activation 没被切到 16 芯」的隐含归因：即便切了 ViT 也救不了，因为它不是大头）。
- **死路二：误信 MaxText static metric。** `utils/max_utils.py:796` 的 'Total memory size' / 'Temp size' 在 baseline/remat/chunk/bf16
  四个变体间**几乎不动**（13.8→13.6G），从不暴露真实 BFC 峰；在 32G GPU 上甚至从不触发 OOM —— 作为 v5e OOM 预测器是**误导**的。
  真信号是 BFC `MaxInUse`，只有把 pool capped 之后才看得到。
- **capped-GPU proxy 技术：** 用 `XLA_PYTHON_CLIENT_MEM_FRACTION` 把 32G 5090 的 BFC 人为 cap 到 ~15.68G 来**模拟 16G-HBM v5e
  单芯**，再读 BFC `MaxInUse`。这是绕过 static-metric 误导、在 GPU 上拿到「v5e 风格真实峰值」的关键手法。

### 9.2 修复 —— chunked CE

- **改动：`diffusion/sasd.py:_ce_per_token` 重写**为按行分块 + remat：`jax.lax.map(..., batch_size=_CE_ROW_CHUNK=512)`
  外加内层 `@jax.checkpoint` 的单行 helper，使峰值 fp32 张量从 `[N,V]` 收到 `[chunk=512, V]`（`sasd.py:34-53`）。
- **bit-identical：** log_softmax 逐行独立，故分块与一次算完**数值完全相同**，loss 仍 **7.945**（ppl 2820.114 不变；
  `/tmp/toy720_gpu_chunk.log`）。纯 memory knob，不动语义。
- **capped-15.7G proxy 结果（THE proxy，`preallocate=true`，`/tmp/toy720_gpu_cap16b.log`）：** BFC `Limit 15.68GiB`，
  step 真实峰 **`MaxInUse 12.66GiB` < 15.75G v5e 预算**（`preallocate=false` 的 cap16 给 12.82G），step 内**无 RESOURCE_EXHAUSTED**。
  对比 un-chunked 720 在 32G 上 HLO 临时量 17.11G > 15.75G —— chunking 拿掉了那 ~4.5G fp32-softmax 峰。
- **`TPU_VIT_720_FIX: CHUNKED_CE (peak 17.11G→~12.66G < 15.75G, loss 7.945 bit-identical)`**。

### 9.3 改动的代码（同日落地）

| file | change |
|---|---|
| `maxtext-dlm-fork/src/maxtext/diffusion/sasd.py` | **THE fix**：`_ce_per_token` 由一次性 fp32 `[N,V]` log_softmax 改成 `jax.lax.map(batch_size=_CE_ROW_CHUNK=512)` 逐行分块 + 内层 `@jax.checkpoint` remat（`sasd.py:29-53`）；峰值 fp32 张量 `[N,V]→[512,V]`，bit-identical |
| `ddrive_jax/models/vision_qwen25vl.py` + `sasd_vit_ingraph.py` + `decoders.py` + `types.py` | ViT per-block remat 基础设施（`VisionConfig.remat`、`body()` 用 `nnx.remat` gate、`sasd_vision_config(remat=)` 透传、`sasd_vit_remat` config 字段）—— **默认 OFF**，因实测 720 下 ~0 train-step memory 效果（大头在 loss 不在 ViT，见 §9.1），仅作休眠 knob |
| `types.py:sasd_vit_remat` Field | 默认 `False`，docstring 明记「measured ~0 train-step memory effect at 720 …addressed by chunked CE in diffusion/sasd.py:_ce_per_token」—— 把真正的 OOM 修复指回 CE，而非 ViT remat |
| backup lever（**未启用**） | 若 v5e run 偏紧，可切 **TRUE bf16 logits**：须**同时**翻 `logits_dot_in_fp32=False` 且 `cast_logits_to_fp32=False`（单翻一个无效——见 §9.4 死路三），把 dominant fp32 logits/softmax 块腰斩到 bf16，约释放整块 ~4.5G；代价是 loss 的 bf16 数值，仅作 fallback |

### 9.4 诚实的残留 / 死路

- **真实 v5e 确认仍 PENDING。** 9.2 的 proxy 是 **GPU BFC 实测峰值**，而原始 17.11G>15.75G 是 **v5e XLA compile-time HLO
  估计**——二者不等价。capped-GPU 证据**必要但不充分**；真正的一次 ~$5 v5e run 仍未跑，是唯一的硬确认。
- **「preempted」是 ckpt-save 碎片伪影，不是 step OOM。** capped proxy 里 step 本身**跑完了**
  （`jax.block_until_ready(state)` 通过 → `checkpointing.py:841` 'Waiting for step 0 …'），~26s 后才在 **checkpoint-SAVE**
  路径申请 13.58GiB 撞进**碎片化**的 capped pool（available 0B / LargestFreeBlock 0B）OOM，被 `except JaxRuntimeError`
  捕获并重抛成 `StopTraining('Job is preempted.')`（`train.py:758`）。即 v5e step 装得下，「preempted」是 save 路径碎片，**非前向/反向**。
- **死路三：bf16 logits 单翻无效。** 只设 `cast_logits_to_fp32=False`、留 `logits_dot_in_fp32=True`：Temp 仍 13.6G、loss
  7.945，fp32 `[N,V]` 块原封不动（matmul 仍被强制 fp32，`/tmp/toy720_gpu_bf16logit.log:72,367`）。两个 knob 必须**同时**翻（§9.3 backup lever）。
- **死路四：关 checkpointing 绕 save-OOM 被拒。** 试 `enable_checkpointing=False` 隔离 step，pydantic 在 device init 前
  报 `ValueError: You must set enable_checkpointing=True to load a checkpoint`（我们 restore base Qwen2.5-3B params，
  `/tmp/toy720_gpu_cap16c.log:52-53`）—— 反证 OOM 在 save 路径且无法这样回避。
- **`TPU_VIT_720_REAL_V5E: PENDING (~$5 run)`**。
