# 01 · CODEBASE MAP —— 代码怎么读

> 给要**读懂/改代码**的人。来源是对真实源码的逐文件精读（非二手转述）。配套：易误读组件 + 规范数字在 [`02_gotchas.md`](02_gotchas.md)。
> 所有 `file:line` 为撰写时（repo commit 见 git）实测；若对不上，**以代码为准**并回报。

---

## 1. 三个代码库的关系

同一个"模型 + 算法"有**三套并行实现**，各司其职，靠 parity gate 数值对齐。

```
   PyTorch oracle  ──ports──▶  ddrive_jax/ (NNX)  ──re-port/vendor──▶  maxtext-dlm-fork/
   数值锚（不可动）              算法真值/可执行规格           生产 / TPU
        ▲                              │                          │
        └────── gate 捕获 oracle tensor，JAX diff 之 ◀────────────┘
```

| 代码库 | 路径 | 角色 |
|---|---|---|
| **PyTorch oracle** | HF snapshot `0fda81009f4efa58a2debbb48c0c09818e45341f`（`modeling.py` / `generation_utils.py` / `section_utils.py`），**不在任一 JAX repo 内** | 上游参考、**数值真值**。每个 JAX 文件用 `file:line` 注明自己 port 的是它的哪段。跑在 conda env `ddrive`，用来**捕获 oracle tensor** 供 gate 对比 |
| **`ddrive_jax/`** | `/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/` | 手写 Flax-**NNX** 全栈 port + **parity gate harness**。**算法在这里定义并 bit-exact 验证**。单 GPU/CPU 优先 |
| **`maxtext-dlm-fork/`** | `Fast-dLLM/maxtext-dlm-fork/`（**2026-06-16 起并入 Fast-dLLM repo**，与 `jax_ddrive/` 同 repo） | Google MaxText fork（root `35dce93`）+ SASD graft。**生产/TPU** 路径 |

> **仓库结构（2026-06-16 起）**：`ddrive_jax/`（子目录）与 `maxtext-dlm-fork/`（2026-06-16 并入的兄弟子目录）同属 **Fast-dLLM 一个 git repo** → 一个 `git rev-parse HEAD` 即描述整套代码（发布脚本据此写 `MANIFEST.json`）；PyTorch oracle 是外部 HF snapshot。原独立 fork repo @ `da8c92b` 的历史备份在 `/home/kaiwen/data/fast-ddrive/maxtext-dlm-fork-history-2026-06-16.bundle`。

关键流向：
- **算法 A→B→C**：`ddrive_jax` 是可执行规格，MaxText-fork 是规模化部署，PyTorch 是不动的数值锚。
- **vendoring B→C**（推理）：`eval_sasd/*`（pin `4b0f4f2`）、`sasd_data/ar_dataset.py`（pin `b18e861`）从 `ddrive_jax` 原样拷入，带 `DO NOT EDIT logic here` 头——**改 `ddrive_jax` 再 re-copy**，规则见 fork `PATCHES.md`。
- C 里的**训练数学** `diffusion/sasd.py` 是独立 re-port，必须与 B 保持同步。
- **trainable in-graph ViT 路径（`sasd_vit_trainable=true`，默认 false）**：训练默认吃 frozen pre-baked `image_embeds`；置 true 后 ViT 改为每步在 `pixel_values` 上 in-graph 跑、params 进 MaxText train state（可训/分片/ckpt）。C 不重写 ViT —— `diffusion/sasd_vit_ingraph.py` 经 `ToLinen` 直接复用 B 的 `VisionTransformer.body`（host 几何由数据迭代器 `precompute_sasd_structural` 预算）。已 v5e-16 多 host 跑通（loss ↓、ckpt 落 GCS），TPU 坑/算法细节见 [`../1plans/06_trainable_vit_plan.md`](../1plans/06_trainable_vit_plan.md)。
- **验证 C/B→A**：`run_all_verification.sh` 在 `ddrive` env 捕获 oracle、在 `jax` venv 跑 gate。
- **eval 对称**：两套 JAX 栈都按 PyTorch-eval schema 写 `predictions.json`，**同一个官方 Waymo metric** 给三方打分。

---

## 2. 黄金阅读顺序（新 coding agent 的 7 个起手文件）

按依赖顺序，建立"后面一切都假设成立"的不变量：

1. **`ddrive_jax/ddrive_jax/diffusion/noise.py`** —— *从这里开始*。定义承重数据不变量：**doubled `[2,2L]` 序列**、**complementary 第 2 行**、**always-mask im_end**、**scaffold freeze**、EPS mask-floor、`num_items = 2×count(labels!=-100)`。`MASK_ID/IM_END/EPS` 的 SSOT 在此。
2. **`ddrive_jax/ddrive_jax/diffusion/sasd_loss.py`** —— 两个 CE 项（section-weighted MDM + clean causal），都 **shift-by-1**，都除以 `num_items`；返回 `{mdm, complementary, total}`。
3. **`ddrive_jax/ddrive_jax/diffusion/masks.py`** —— **训练** doubled `[2n,2n]` mask vs **推理** `[L,L]` mask（规则不同：block-diagonal vs block-causal）。
4. **`ddrive_jax/ddrive_jax/models/qwen2_5_text.py`**（+ 旁开 `rope.py`）—— 文本维度 `Qwen25TextConfig.fast_ddrive`、GQA + fp32 softmax、**两套 RoPE**（SPLIT vs FULL）、`mrope_cos_sin`（contiguous-chunk M-RoPE）。
5. **`ddrive_jax/ddrive_jax/models/vision_qwen25vl.py`** —— ViT config + window 重排/还原编排；理解 frozen image embeds。
6. **`ddrive_jax/scripts/run_all_verification.sh`** —— 可执行索引：哪个 PyTorch oracle 喂哪个 JAX gate、PASS marker（gate 图；**10 个 gate**，见 [`02_gotchas.md`](02_gotchas.md#10-个-gate可执行规格)）。
7. **`maxtext-dlm-fork/src/maxtext/diffusion/sasd.py`**（+ 旁开 `configs/sasd_waymo.yml`）—— 生产 re-port：`prepare_sasd_inputs` 怎么 flatten `flat = 2*b + r`、`sasd_loss_from_logits` 怎么选 loss 半段。

之后按任务分支：推理 → `eval/mm_sampler.py` + `eval_sasd/sampler_sasd.py`；权重 → `convert/hf_to_jax.py` + `param_mapping.py`；FSDP → `train/train_tpu.py` + `sharding.py`。

---

## 3. 子系统地图（概念 → 文件 → 关键符号）

> 深挖详细文档：模型/算法 → [`../2implementation-details/ARCHITECTURE.md`](../2implementation-details/ARCHITECTURE.md) + [`01_pytorch_reference_algorithm.md`](../2implementation-details/01_pytorch_reference_algorithm.md)；数据 → [`DATASET_V2.md`](../2implementation-details/DATASET_V2.md)/[`DATASET.md`](../2implementation-details/DATASET.md)；评测/部署 → [`EVAL_PIPELINE.md`](../2implementation-details/EVAL_PIPELINE.md)/[`INFERENCE_DEPLOY.md`](../2implementation-details/INFERENCE_DEPLOY.md)。

### `ddrive_jax/`（NNX 真值）

**models/** `ddrive_jax/ddrive_jax/models/`
| 文件 | 职责 | 关键符号 |
|---|---|---|
| `rope.py` | 文本路径 1D RoPE（**SPLIT/rotate-half** HF 约定） | `default_rope_params`、`apply_rope`(SPLIT)、`RoPE.__call__`(fp32 `Precision.HIGHEST`) |
| `qwen2_5_text.py` | Qwen2.5-3B 文本 decoder（GQA、qkv bias、无 qk-norm、tied embeds、SiLU、RMSNorm）+ 3D M-RoPE builder | `Qwen25TextConfig.fast_ddrive`、`Linear`(kernel `[in,out]`)、`Qwen25Attention.__call__`、`apply_rope_full`(FULL)、`mrope_cos_sin`(host numpy float64)、`attend`(tied)、`hidden_forward_mrope_cs` |
| `vision_qwen25vl.py` | Qwen2.5-VL ViT（depth 32, hidden 1280）；默认训练 frozen，**trainable 路径下 in-graph 可微** | `rot_pos_ids`、`get_window_index`、`VisionAttention.__call__`、`VisionTransformer.__call__`(`:277` 全程)、`precompute_structural`(`:220` host 端纯几何：window_index/cos/sin/mask/rev)、`body`(`:252` 纯 jax 可微半，trainable-ViT 用它)、`PatchMerger`(exact GELU) |
| `sharded.py` | TP mesh 的 sharding-aware drop-in（mesh size 1 时等价） | `ShardedLinear`、`ShardedEmbedding` |

**diffusion/** `ddrive_jax/ddrive_jax/diffusion/`
| 文件 | 职责 | 关键符号 |
|---|---|---|
| `sasd_loss.py` | 两个 SASD loss 项 + section-weight 向量 + 合并 | `section_weighted_ce`(MDM, shift-1)、`causal_ce`、`section_weight_vector`、`sasd_total_loss`(`{mdm,complementary,total}`) |
| `masks.py` | hybrid block-causal mask（训练 `[2n,2n]` + 推理 `[L,L]`） | `hybrid_block_causal_mask_dense`、`eval_hybrid_block_causal_mask_dense`、`compute_response_block_idx_simple`(非-deep fallback) |
| `noise.py` | host 端逐步随机加噪；doubled+complementary batch | `make_batch`、`num_items`(=2×) |
| `sample_sd.py` | 纯文本 section-diffusion 采样器（无 KV cache） | `section_diffusion_sample`(causal-shift slice `s-1:e-1`) |

**convert + data**
| 文件 | 职责 | 关键符号 |
|---|---|---|
| `convert/hf_to_jax.py` | HF safetensors → NNX（流式文本、eager ViT）；转置约定 | `_st_tensor_f32`(bf16-safe)、`load_fast_ddrive_text`(流式+tie-check)、`load_fast_ddrive_vit`(Conv3d→Linear) |
| `convert/prep_to_parquet.py` | npz → zstd Parquet；**12 个 array field**（SSOT） | `ARRAY_DTYPES`、`row_from_npz`、`parquet_schema` |
| `data/parquet_dataset.py` | decode 契约 SSOT | `decode_row` |
| `data/ar_dataset.py` | ArrayRecord 随机访问（v2 + image_embeds） | `decode_example`、`ArRecordSource` |
| `data/grain_pipeline.py` | 多 host 无限/确定性/可恢复 loader | `make_sasd_loader`、`_fold_rng`、`_collate` |

**train + sharding + lora + checkpoint**
| 文件 | 职责 | 关键符号 |
|---|---|---|
| `train_overfit.py` | 纯文本单 GPU overfit；**默认 LoRA**（除非 `--full_ft`） | `loss_fn`、`train_step`(`@nnx.jit`) |
| `train_overfit_mm.py` | 多模态单样本 overfit；**full-FT 文本、frozen ViT** | `main` |
| `train_waymo_sasd_jax.py` | 生产单 GPU 多样本循环 + Orbax | `step`(`@nnx.jit`)、`save_ckpt`(吞异常) |
| `train/train_tpu.py` | **Path-A FSDP driver**（CPU-8 仿真 / TPU），PoC | `make_train_step`(shard_map)、`build_harness` |
| `train/checkpoint_mgr.py` | Orbax CheckpointManager 4-item 复合 | `save_step`、`restore_latest` |
| `lora.py` / `sharding.py` / `checkpoint.py` | LoRA / **PartitionSpec spec-only** / 简单单-host StandardCheckpointer | `LoRALinear`；`fsdp_pspec`(最大轴)；`save_state` |

**eval/inference**（注意有**两个** eval 目录）
| 文件 | 职责 | 关键符号 |
|---|---|---|
| `ddrive_jax/eval/mm_sampler.py` | 多模态无-KV-cache section-diffusion 采样器 | `mm_section_diffusion_sample`(slice `s-1:e-1`)、`decode_generation` |
| `ddrive_jax/eval/rope_index.py` | 纯 numpy B=1 `get_rope_index` → `[3,S]` | `get_rope_index_numpy` |
| `ddrive_jax/eval/scaffold.py` | 深-JSON masked scaffold + 推理 rbi | `messages_from_prompt`、`build_scaffold` |
| `jax_ddrive/eval/prep_jax_eval.py` | PyTorch-env eval npz producer（paper-res 200704） | — |
| `jax_ddrive/eval/jax_batch_inference.py` | JAX-env eval driver → `predictions.json` | `parse_trajectory`（bf16 默认 / `--fp32`） |
| `jax_ddrive/eval/prep_train_jax.py` | PyTorch-env SASD 训练 npz producer（无权重） | `process_gpt`；**SECTION_W / NOISE_SCHED 字面值在此**（22-24） |
| `ddrive_jax/scripts/tpu_vit_body_smoke.py` | **trainable-ViT TPU 起步 smoke**：验 `VisionTransformer.body` 在真 TPU 上 compile+autodiff（有限非零梯度到每个 ViT param）；可选 `--snapshot/--ar` 加 cosine 对比 pre-baked embeds | `main`（random-init，仅需 ddrive_jax + jax[tpu]，无权重无数据） |

### `maxtext-dlm-fork/`（生产/TPU）

| 文件 | 职责 | 关键符号 |
|---|---|---|
| `src/maxtext/diffusion/sasd.py` | SASD 数学（loss/noising/mask/M-RoPE/embed-doubling/host prep/global loss） | `make_batch`、`num_items`(2×)、`mrope_cos_sin`(float64 contiguous)、`compute_fast_ddrive_image_embeds`、`prepare_sasd_inputs`(`flat=2b+r`)、`sasd_loss_from_logits`；`_ce_per_token`(fp32 per-token NLL，现 **按 `_CE_ROW_CHUNK=512` 行分块 + `@jax.checkpoint` remat**：把 fp32 `[N,V]` log_softmax 峰值压到 `[chunk,V]`，**数值 bit-identical**——纯显存，修 720-OOM；根因/数字见 [`../1plans/07_fidelity_fixes_2026-06-20.md`](../1plans/07_fidelity_fixes_2026-06-20.md)) |
| `src/maxtext/diffusion/load_fast_ddrive_maxtext.py` | HF 文本 → MaxText Linen tree（流式） | `build_maxtext_params_from_fast_ddrive`、`_StreamingTextGetter` |
| `src/maxtext/diffusion/sasd_vit_ingraph.py` | **trainable in-graph ViT**（`sasd_vit_trainable=true`）：每步在 pixels 上跑 ViT，params 进 train state（可训/可分片/可 ckpt）。复用 B 的 NNX `VisionTransformer.body`，经 `flax.nnx.bridge.ToLinen` 包成 Linen 子模块 | `sasd_vision_config`、`precompute_sasd_structural`(host 几何，仅依赖 grid_thw)、`SasdInGraphViT`(Linen：pixels `[B,N,1176]`→doubled embeds `[2B,2N,D]`)、`load_sasd_vit_leaves_in_order`(390↔390 in-order 填充)、`SASD_GRID_THW`(3×(1,16,14)→672 patch→168 tok) |
| `src/maxtext/diffusion/mdlm.py` | **独立 MDLM objective（≠ SASD）** | `mdlm_loss` |
| `src/maxtext/diffusion/eval_sasd/`（vendored @ `4b0f4f2`） | 自包含推理栈：`sampler_sasd.py`、`masks_eval.py`、`models/`、`driver.py`、`embedding_parity.py`、`hf_to_jax.py`（bf16-硬化） | `mm_section_diffusion_sample`、`run_eval`、`run_parity` |
| `src/maxtext/configs/sasd_waymo.yml` | 生产 SASD train config | `objective=sasd`、`sasd_mrope_section [16,24,24]`、`sasd_seq_len 1184`、`sasd_num_image_tokens 336`、`use_mrope false`、`max_target_length 2376` |
| `src/maxtext/configs/models/qwen2.5-3b.yml` | 模型 shape config | 2048/16/2/11008/36/head_dim 128/vocab 151936 |
| `src/maxtext/input_pipeline/sasd_data/ar_dataset.py`（vendored @ `b18e861`） | AR source | `decode_example`、`ArRecordSource` |
| `src/maxtext/checkpoint_conversion/utils/param_mapping.py` | QWEN MaxText↔HF 映射 + hook | `reshape_kernel`(`.T.reshape`)、`pad_embedding_layer`(identity) |
| `scripts/save_fast_ddrive_params_ckpt.py` | **B1 in**：HF → MaxText Orbax param ckpt | `main`（打印 `{n_el/1e9:.3f}B`） |
| `scripts/maxtext_to_hf_export.py` | **B1 out**：MaxText → bf16 HF safetensors | `main`（824 keys = 434 text + 390 visual，lm_head omit） |
| `scripts/prep_jax_eval_inputs.py` | 离线 eval npz builder（import ddrive_jax；默认 784/50176→168 tokens） | `main` |
| `PATCHES.md` | fork patch / provenance / vendor pin | — |

---

## 4. 接着深挖

- **算法每一步的数学**（loss/mask/noise/scaffold）→ [`../2implementation-details/01_pytorch_reference_algorithm.md`](../2implementation-details/01_pytorch_reference_algorithm.md)
- **NNX port 结构** → [`../2implementation-details/ARCHITECTURE.md`](../2implementation-details/ARCHITECTURE.md)
- **易误读的 ~25 个组件 + 规范数字 + 10 个 gate** → [`02_gotchas.md`](02_gotchas.md)
- **数据 schema / builder / reader** → [`../2implementation-details/DATASET_V2.md`](../2implementation-details/DATASET_V2.md)
- **B1/B2 部署 runbook** → [`../2implementation-details/INFERENCE_DEPLOY.md`](../2implementation-details/INFERENCE_DEPLOY.md)
- **fork vs upstream 的逐文件 diff** → `maxtext-dlm-fork/PATCHES.md`
