# 02 · GOTCHAS + 可执行规格 + 规范数字

> 三件别处没有的东西：① **易误读组件目录**（不警告就会读错的地方，带 `file:line`）；② **10 个 gate**（钉死行为的可执行规格）；③ **规范数字框**（每个关键数字的唯一权威值 —— 别的文档应链接这里，而不是各自重写）。
> 全部对真实源码核验过。对不上以代码为准并回报。代码地图见 [`01_codebase_map.md`](01_codebase_map.md)。

---

## 易误读组件目录（gotchas）

### 序列 & batch 结构
- **doubled 序列是 `[2,2L]`，不是 `[2,L]`。** 训练输入 = `concat([noisy(L), clean_ids(L)])` → 长 **2L**；前导轴 2 是 *(main 行, complementary 行)*，**不是**普通 batch。`noise.py:35,41,43`；`masks.py:19`；`grain_pipeline.py:124`；`sasd.py:94`。
- **两行语义不同。** 行 0 = MDM/noisy（随机 Beta-mask）；行 1 = **complementary**：`comp=(resp & ~mask)|(im_end & resp)` 再 `& ~scaff` —— 正好遮住行 0 没遮的 response token。`noise.py:37-41`。
- **loss 半段按行选取不同。** noisy 半段对**两行**都算；clean 半段**只对行 0(mdm)**算。`train_overfit.py:34`；`parity_sasd.py:50-52`；`sasd.py:330-335`。误以为 clean 用两行就错。
- **`num_items = 2 × count(labels!=-100)`** —— 因子 2 是 doubling 归一；两个 CE 项除以**同一个** doubled 分母（**不是** masked/valid 数）。`noise.py:50-51`；`sasd_loss.py:11`；`sasd.py:102`。
- **MaxText 行展平 `flat = 2*b + r`**：`[B,2,2L]→[2B,2L]`，cos/sin/mask/img 沿轴 0 ×2。`sasd.py:282-285,308-309`。
- **全局 DP-correct 归一**：各项 CE 用 `num_items=1.0` 求原始和，再 `loss=(sec+cau)/max(Σ num_items,1.0)` —— 单一全局分母，非 per-sample mean。`train_tpu.py:244-249`；`sasd.py:333-338`。

### RoPE / M-RoPE
- **两个不兼容的 RoPE helper 并存。** SPLIT-form `apply_rope` 收**半宽** sin/cos `[B,T,Dh/2]` 并把 x 切半（`rope.py:27-32`）；FULL-form `apply_rope_full` 收**全宽** cos/sin `[L,Dh]` 走 rotate-half（`qwen2_5_text.py:140-146`）。attention 按 `mrope_cs is None` 选分支（`:107-111`）。以为只有一个就错。
- **M-RoPE cos/sin 是全 head_dim `[L,128]`，不是半宽** —— `emb=concat([freqs,freqs],-1)`。`qwen2_5_text.py:155`；ViT `_rotary:215`。
- **M-RoPE 是 contiguous-chunk，不是 interleaved。** `sec=list(mrope_section)*2=[16,24,24,16,24,24]`，section i 取 channel `cos[i%3]` 拼连续片 → `[t..h..w..t..h..w]`。`qwen2_5_text.py:157-161`；`sasd.py:138-164`。**与 MaxText 自带 interleaved `use_mrope` 不同** → `use_mrope` 必须 false、手动注入 cos/sin。`sasd_waymo.yml:38`。
- **`mrope_cos_sin` 是 host 端 numpy float64，不可 jit。** 可 jit 的路径是 `hidden_forward_mrope_cs`，把预算 cos/sin 当 data 喂进去。`qwen2_5_text.py:152-153`。
- **`mrope_section` 有两个合法值。** 真模型 `(16,24,24)`（`2*(16+24+24)=128=head_dim`）；proxy harness/tests `(4,6,6)`（`2*16=32=proxy head_dim`）。`base.yml:390` 默认 `[4,6,6]` 是占位、被 `sasd_waymo.yml:22` 覆盖；`train_tpu.py:432-433` 强制重置回 `(16,24,24)`。**文档别把 `(4,6,6)` 当真值。**
- **纯文本路径坍缩成普通 RoPE：** 3 个 M-RoPE section 共享 `arange` 位置时 M-RoPE ≡ 1D RoPE —— 但**代码用的是两个不同函数**（文本 `RoPE`/`apply_rope`，多模态 `mrope_cos_sin`/`apply_rope_full`），是数值等价、非共享代码。`rope.py:4` docstring。
- **RoPE 精度技巧**：sinusoid einsum 强制 `Precision.HIGHEST`/fp32，避免 bf16 把位置 257 舍成 256。`rope.py:42-44`。

### causal shift / 采样
- **到处 causal shift-by-1，尽管这是扩散模型。** 两个 loss 都 `logits[:,:-1]` vs `labels[:,1:]`（权重也移位）。`sasd_loss.py:34-36,47-48`。
- **采样器移位切片是 `s-1:e-1`，不是 `s:e`。** 位置 p 由 p-1 的 logit/hidden 预测：`model.attend(hidden[0, s-1:e-1])` 填位置 `s..e`。`sample_sd.py:57`；`mm_sampler.py:80`；`sampler_sasd.py:83`。**最容易被误判为 off-by-one bug 的地方 —— 它是有意的。**
- **每块子步预算 = `n_mask + 5`**（不是恰好 n_mask），无 MASK 早停，全局上限 `steps>max_tokens(512)`。`sample_sd.py:52,67-68`。
- **置信回退**：unmask 所有 softmax conf>0.9 的位置；若无一超阈值，unmask 最自信的那一个（保证推进）；已解码位置强制 `-inf`。`sample_sd.py:61`；`mm_sampler.py:84-87`。
- **采样 PASS gate 不是 token 一致率。** bf16 会扰动低置信选择并级联；`ok = valid_json and has_traj`，一致率仅供参考。driver 的 T2 用**结构化指标**（轨迹数值匹配 + CO/FMB 相等），**非**位置 token 一致。`driver.py:131-148`。

### masks
- **训练 mask ≠ 推理 mask。** 训练 `hybrid_block_causal_mask_dense` 是同 turn 内**块-对角** + 对 clean key 的 offset-causal；推理 `eval_*` 是**块-因果**(`bq>=bk`)。`masks.py:29` vs `:50`。别混。
- **推理：response query 看到全部 prompt key 无位置约束**，但 prompt→prompt 严格因果。`masks.py:48-49`。
- **mask 是布尔(True=attend)，经 `jnp.where(mask4d, logits, finfo(f32).min)` 转加性。** `to_attn_mask4d` 只加两个前导单维 `[1,1,Q,K]`，**不**构造加性 -inf。`masks.py:35-37`；`qwen2_5_text.py:117-119`。
- **`compute_response_block_idx_simple` 是非-deep FALLBACK**（tests/data-prep），非生产。生产 rbi = `section_utils` deep-scaffold。`masks.py:54-55`。

### 权重 / 转换
- **Linear kernel 存 `[in,out]`、`x@kernel` 无转置。** PyTorch（存 `[out,in]`）转换时**必须转置**。`qwen2_5_text.py:73,77`；`sharded.py:22-28`。
- **bias/layernorm/embedding 不转置；所有 `*_proj.weight` 转置。** `hf_to_jax.py:57-72`；`param_mapping.py:779,785`。embedding（`[vocab,d_model]`）不转置。`hf_to_jax.py:106`。
- **tied embeddings → 没有 `lm_head`。** logits 经 `embed_tokens.attend`；存的 `lm_head` 仅在 untied 时加载，fast_ddrive 默认 `tie_embeddings=True` 故从不使用。B1 导出 omit lm_head。`qwen2_5_text.py:196-197`；`hf_to_jax.py:120-153`；`maxtext_to_hf_export.py:16-17`。
- **bf16 safetensors 陷阱：numpy framework 解不了 bf16。** 每个 loader/exporter 手写 raw-byte reader（`_st_tensor_f32`/`_st_tensor_native`）：解 8-byte LE header → JSON header → seek `data_offsets` → `frombuffer`（`BF16→ml_dtypes.bfloat16`）。release snapshot 是 F32、BASE Qwen2.5-VL snapshot 是 bf16。`hf_to_jax.py:29-43`；`parquet_to_ar_with_embeds.py:62-77`；`maxtext_to_hf_export.py:63-85`。**`f.keys()` 列举是 bf16-safe，只有解码不是。**
- **bf16 过不了 `np.savez`** —— 回来成结构化 void `|V2`；检测 `dtype.kind=='V'` 后 `raw.view(ml_dtypes.bfloat16)`。`driver.py:98-99`；`embedding_parity.py:71`。
- **Conv3d→Linear**：patch_embed `[out,in,T,H,W]→reshape(out,-1).T→[in_flat,out]`，`patch_dim=3*2*14*14=1176`。`hf_to_jax.py:177-179`。
- **ViT qkv 是融合的 `[3*hidden,hidden]`**（一次 `.T`），与文本塔分开的 q/k/v 不同。`hf_to_jax.py:185`。
- **流式文本 loader 是有意的 OOM 修复**（eager 在 30GB 机器峰值 ~25GB）；逐张量读 + `gc.collect()`。`hf_to_jax.py:89-95`。
- **ViT 张量数硬断言 `1 + 12*32 + 5 = 390`。** 数不对就 abort。`parquet_to_ar_with_embeds.py:126`。
- **导出 `--verify_against` 只对 BASE round-trip 有效**（bitwise 824/824）；TRAINED ckpt 文本权重变了会**假性 FAIL** → trained 导出去掉它。`maxtext_to_hf_export.py:22-28,259-281`。

### ViT
- **ViT 有两条路（`sasd_vit_trainable` 开关），别假设永远 frozen。** 默认 false=frozen/预烤 embeds（下面诸条）；true=**in-graph TRAINABLE**：Qwen2.5-VL ViT 每步在 `sasd_pixel_values` 上跑，param 进 MaxText train state（可训/可 shard/可 ckpt），经 `flax.nnx.bridge.ToLinen` 把 NNX ViT *body* 包成 Linen 子模块。`sasd_vit_ingraph.py:34,71,88`；`decoders.py:656,663`；`sharding.py:97,99`。
- **trainable 路把 ViT 切成 `body`（纯 jax，可 jit+autodiff）+ `precompute_structural`（host-only 几何，不可 jit）。** body 收 `(pixel_values, structural)`、是在图里求导的那段；structural=window 重排/2D-RoPE/seg mask，只依赖 (固定的) `grid_thw`。`vision_qwen25vl.py:220,252`；`sasd_vit_ingraph.py:41,58`。
- **structural 在 TRACE 时当图常量内嵌，故意 NOT 走 data pipeline。** grid_thw 全数据集恒定 → 几何是编译期常量；若塞进 batch dict 会被按 batch 轴 shard 而 mis-shard 这些 batch-共享数组。融合点在 caller（`SasdInGraphViT.__call__`），不在 `models/`。`sasd_vit_ingraph.py:68,87`。
- **ViT 用 `stop_gradient` 冻结 OUTPUT embeds（非 param freeze），随后 `del` 掉模块**省内存。`train_waymo_sasd_jax.py:45`；`train_overfit_mm.py:48-50`。
- **image embeds 在所有去噪步固定**（ViT 循环前跑一次，每步用相同预算值重新 scatter）。`mm_sampler.py:55,64`。
- **图像融合是 scatter 不是 concat**：`embed_tokens(ids).at[img_pos].set(image_embeds)` 覆写 `ids==IMAGE_TOK(151655)` 的行；融合在 **caller 侧**、不在 `models/`。`train_overfit_mm.py:60`；`sasd.py:215-220`。
- **image embeds 单份存 `[N_img,D]`；消费方 doubling 用 `concat([ie,ie],0)`**（单样本轴 0）/`axis=1`（batched）。`ar_dataset`(轴 0) 与 `grain_pipeline`(轴 1) 两处 docstring **不矛盾**——同操作不同 rank；doubling **不在**数据文件里做。`ar_dataset.py:6-8`；`grain_pipeline.py:166-167`；`parquet_to_ar_with_embeds.py:280-281`。
- **ViT window 重排在末尾用 `argsort(window_index)` 还原**（`spatial_merge_unit=4` 粒度）；忘了还原就乱序。`vision_qwen25vl.py:230,243-244`。
- **ViT 全注意力块恰为 `(7,15,23,31)`**；其余用 per-window seg mask；padding 哨兵 `-100`。`vision_qwen25vl.py:36,78,239`。
- **PatchMerger 用 exact GELU（`approximate=False`）；文本 MLP 用 SiLU。** 混合激活。`vision_qwen25vl.py:191` vs `qwen2_5_text.py:132`。
- **ViT 不能*整体* jit**（混了 host numpy index/rope/seg-mask 与 device 计算）；embeds builder 只 jit device 部分，并逐 shard 对 jit-vs-eager cross-check。`parquet_to_ar_with_embeds.py:167-171`。**但 trainable 路把可 jit 的 `body` 与 host-only `precompute_structural` 显式拆开** —— body 在真 v5e 上既 compile 又 autodiff（首步 ~24 min 是一次性 XLA 编译，非挂死）。`vision_qwen25vl.py:220,252`；`sasd_vit_ingraph.py:41`。

### 数值 / 精度策略
- **RMSNorm + 所有 attention logits/softmax 强制 fp32**（不论模型 dtype）。`qwen2_5_text.py:66-68,115-120`；`vision_qwen25vl.py:111-113,152-154`；loss cast logits 到 fp32 `sasd_loss.py:23`。
- **每个 parity 脚本 + embeds builder 设 `jax_default_matmul_precision='highest'`（关 TF32）。** 否则 ViT 系统性抬高 ~7.5e-4，被 rolling-median<1e-4 guard 抓到。`parquet_to_ar_with_embeds.py:35,239-243`；所有 `parity_*.py`。
- **section_weighted_ce 的分母（`num_items` 为 None 时）是无权 `valid.sum()`**，不是权重和 —— 有意匹配 PyTorch。`sasd_loss.py:38-41`。- **train-step 显存的 DOMINANT 项是 loss 侧的 fp32 全词表 log_softmax，不是 ViT 激活。** CE 在 causal-shift + 2L packing 后展平成 `[N,V]`（`N=2L`，`V=151936`），un-chunked 路一次性 materialize fp32 `[N,V]` log_softmax（forward + backward grad 各一份）—— 单个 ~4.5G fp32 transient。它随 **2L 线性增长**：168 配置 `N=2L=1856` 这块只 ~2.26G、整步装得下 v5e；720 配置 `N=2L=3712` 翻倍 → HLO temporaries 撑到 v5e 单芯 OOM（这正是「168 装得下 v5e 而 720 OOM」的机制）。**修复 = chunked CE**：`_ce_per_token` 用 `jax.lax.map(batch_size=_CE_ROW_CHUNK=512)` + `@jax.checkpoint` 把 fp32 log_softmax 按行分块 + remat，峰值 fp32 张量从 `[N,V]` 降到 `[chunk,V]`，**数值 bit-identical**（log_softmax 逐行独立，loss 7.945 不变）。`sasd.py:29-53`。**ViT remat（`sasd_vit_remat`）对此 OOM ~0 效果**（实测 Temp 13.8→13.8G bit-identical，dominant 张量不在 ViT 里）；canonical 数字 + v5e OOM 上下文见 [`../1plans/07_fidelity_fixes_2026-06-20.md`](../1plans/07_fidelity_fixes_2026-06-20.md) §6.3 / §7。
- **诊断技巧：MaxText 的静态 `Total memory size` 指标（`utils/max_utils.py:796`）查不出这类变化、是误导的 OOM 预测器。** 它在 baseline/ViT-remat/chunked-CE/bf16-logit 之间几乎不动（Temp 13.8↔13.6G），且在 32G GPU 上**永不触发** OOM → 对 16G v5e 毫无预测力。**真信号是 BFC `MaxInUse`，只有把池子 cap 住才看得见**：用 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.50` 把 GPU 压到 ~15.7G 当 v5e proxy，再读 `bfc_allocator.cc` 的 `MaxInUse`。**720 trainable step 在该 15.7G proxy 上的真实峰值 ~12.66G < 15.75G v5e 预算（step 本身 NO RESOURCE_EXHAUSTED；OOM 来自 step 后的 checkpoint-save 路径在 capped 池里碎片化，非 step OOM）；真 v5e 复跑 pending。** 链路与所有 dead-end（ViT-remat / 单独 `cast_logits_to_fp32=False` / `enable_checkpointing=false`）见 [`../1plans/07_fidelity_fixes_2026-06-20.md`](../1plans/07_fidelity_fixes_2026-06-20.md) §6.3 / §7。

### noise schedule / 权重出处
- **always-mask im_end(151645)** 在 response 位置、两行都遮、不管 Beta 抽样。`noise.py:31,37`。
- **scaffold freeze**：scaffold 位置永不加噪、永不入 masked 集（两行都 `& ~scaff`）。`noise.py:30,38`。
- **p_mask 经 EPS=1e-3 设地板**：`p=(1-EPS)*Beta+EPS`，永不恰好 0。`noise.py:11,26`。
- **section 权重 / Beta schedule 不在 diffusion/training 模块里** —— 是**per-sample 数据字段** `weight_vec`/`block_alpha`/`block_beta`，字面值只在 prep 脚本 `prep_train_jax.py:22-24`。说"diffusion 模块硬编码 {CO 1.5...}"是 drift。

### 分布式 / sharding
- **FSDP 用 `jax.shard_map` over `'fsdp'`，不是 vmap-over-mesh**（JAX-0.10 sharding-in-types 拒绝 vmap 切轴）。边界处 param reshard 到 `P()`(replicated) → all-gather；grad 在 optax 前 reshard 回。`train_tpu.py:218-227`。
- **`sharding.py` 是 SPEC-ONLY** —— 只给 PartitionSpec；物理 sharding 在 `train_tpu.py` 经 `shard_map`/`reshard`。embedding 永远 replicated `P()`。`sharding.py:1-9,41`。`fsdp_pspec` 只切**单个最大轴**。`sharding.py:26-28`。
- **三套 checkpoint 系统并存**：`checkpoint.py`（StandardCheckpointer，**单 host**，含 frozen Variables）；`train/checkpoint_mgr.py`（CheckpointManager 4-item，多 host）；`train_waymo_sasd_jax.py:159` 自己的 inline `save_ckpt`（无 opt_state、吞异常）。
- **multihost 修复**：旧 `jax.device_put(host_local_batch, global_sharding)` 在真 2-VM slice 上**错**；用 `jax.make_array_from_process_local_data`。`test_multihost_datafeed.py:5-8,65`；`train_tpu.py:368-390`。

#### TPU 运维坑（trainable-ViT v5e-16 run，已实测，不在源码里）
> 这些是**运维**教训（GCP/Orbax 行为），非代码 file:line gotcha。全程实测于 trainable-ViT 多机跑（run `be47jjta8`，loss 5.199→3.042，EXIT 0）。细节 → [`../1plans/06_trainable_vit_plan.md`](../1plans/06_trainable_vit_plan.md) §9。
- **GCS Regional Access Boundary(RAB) 是 REGION-scoped，会墙掉 TPU compute SA。** RAB 拦 TPU 服务账号读跨区桶（源桶在 us-east5、pod 在 us-south1 → 被拦）→ restore 要么从 ckpt 的**本地副本**读（先用 USER creds `gsutil` 拉下来），要么用**同区**桶；SAVE 必须用**同区**桶（如 `us-south1`）且给 TPU SA `roles/storage.admin`（否则 `403 storage.buckets.get`）。
- **multi-host Orbax checkpoint 必须有 SHARED filesystem。** 每 host 各自的本地盘会**逐层失败**（grain-iter 目录 → per-process 创建 → `array_metadatas`；mkdir 修复 + `primary_host=None` 只解前两层）→ **同区 GCS 才是正解**。别用 per-host 本地盘存多 host ckpt。
- **`enable_checkpointing=false` 在设了 `load_parameters_path` 时会被配置校验拒绝**；且 **step 0 必存**（`0 % checkpoint_period == 0`）—— 想完全不存 ckpt 走不通，得给一个能写的同区目的地。
- **fresh-pod SSH `Permission denied (publickey)` = key 还在传播**，不是配错 → 退避重试 warm-up 即可。
- **multi-host `process_state.cc Raising signal 6` / Shutdown-barrier abort 是 SYMPTOM**（某个 worker 先死了），不是 root cause → 两段式 SSH、把每 host 全量日志写盘、再开新 session 读那个真正先崩的 worker。signal-6 别当成代码 bug 去 debug。
- **trainable 路 ViT param 当前是 REPLICATED**（无 logical-axis sharding）—— 真正 multinode 前的 TODO；`sharding.py:97,99` 只把 `sasd_pixel_values` 按 batch 轴 shard，ViT 权重未切。

### LoRA / 其它
- **各脚本默认 trainability 不同**：`train_overfit.py` 默认 LoRA(rank 16) 除非 `--full_ft`；`train_overfit_mm.py`/`train_waymo_sasd_jax.py` 是 **full-FT 文本**（无 LoRA）。别假设都用 LoRA。`lora.py:18`；`train_overfit.py:75`。
- **`vision_mask` 在 schema 里但 grain 训练从不读** —— 携带后从 batch dict 丢弃。`prep_train_jax.py:117`；`grain_pipeline` 不读。说它在训练时门控 vision token 是**错的**。
- **`MDLM` 是独立 objective、不是 SASD**：token-归一 `(1/clip(t,1e-3))`-加权 CE。`mdlm.py:69,78`；`base.yml:376` 枚举 `ar|mdlm|sasd`。

---

## 10 个 gate（可执行规格）

> 想验证你对某模块的理解，就跑对应 gate。一键：`bash jax_ddrive/scripts/run_all_verification.sh`（在 `ddrive` env 捕获 PyTorch oracle、`jax` venv 跑 gate）。sentinel：全过 `ALL_VERIFICATION_PASS`，否则 `SOME_VERIFICATION_FAILED`。`run_all_verification.sh:51,55-57`。

| # | gate | pass marker | 命令 |
|---|---|---|---|
| 1 | cpu_mask_loss | `ALL CPU TESTS PASS` | `tests/test_mask_loss.py`(CPU) |
| 2 | cpu_lora | `ALL LORA TESTS PASS` | `tests/test_lora.py`(CPU) |
| 3 | cpu_noise | `ALL NOISING TESTS PASS` | `tests/test_noising.py`(CPU) |
| 4 | cpu_sharding | `ALL SHARDING TESTS PASS` | `tests/test_sharding.py`(CPU) |
| 5 | cpu_eval_ports | `ALL EVAL PORT TESTS PASS` | `tests/test_eval_ports.py`(CPU) |
| 6 | phase1_text | `PHASE1_PARITY_PASS` | `scripts/parity_text.py` |
| 7 | phase2_sasd | `PHASE2_PARITY_PASS` | `scripts/parity_sasd.py` |
| 8 | phase4_vit | `PHASE4_VIT_PASS` | `scripts/parity_vit.py` |
| 9 | phase4b_mm_fwd | `PHASE4b_MM_PASS` | `scripts/parity_mm.py` |
| 10 | phase3_lora_train | `PHASE3_PASS` | `train_overfit.py --source trained --fixed_batch --steps 30 --lr 1e-4` |

注：
- `phase3_lora_train` 在连跑套件里**会偶发 flake**（LoRA 在近最优 trained ckpt 上 30 步只动 ~0.001）；单跑稳过。`run_all_verification.sh:42-46`。
- `tests/test_harness_fsdp.py` 与 `tests/test_multihost_datafeed.py` **不在** `run_all_verification.sh` 里（需特定 CPU device-count / 多进程环境，单独跑）。

其它 grep-able pass marker：`WAYMO_SASD_JAX_TRAIN_PASS`、`V2_TPU_VALIDATION_PASS`、`AR_WITH_EMBEDS_DONE`、`SASD_{TRAIN_STEP,WEIGHT_PARITY,VLA_PARITY,TRAIN_LOSSDECREASE}_PASS`、`B1_ROUNDTRIP_PASS`(824/824)、`EVAL_SASD_SELFCONTAINED_PASS`、`SASD_EVAL_PASS`、`B2_BF16_LOAD_PASS`、`EMBED_PARITY_PASS/FAIL`、`MULTIHOST_DATAFEED_TEST_PASS`。

---

## 规范数字框 (canonical numbers)

> **数字的唯一权威值，全部对代码核验。** 别的文档若与此冲突，以此为准（这里又以代码为准）。"合理变体"=同一量在不同语境下的不同正确值；"曾见错值"=应被订正的写法。

### 模型（文本 decoder）`Qwen25TextConfig.fast_ddrive`
| 量 | 规范值 | 出处 |
|---|---|---|
| 参数量 | **3.086B**（≈3.09B 四舍五入）；**824 张量** = 434 text leaves + 390 visual | 运行时打印 `save_fast_ddrive_params_ckpt.py:72-73`；`maxtext_to_hf_export.py:5,11-14`。**曾见错值 `3.75B`**（过计数） |
| d_model / n_layers | 2048 / 36 | `qwen2_5_text.py:31,35` |
| n_heads / n_kv_heads / head_dim | 16 / 2 / 128（GQA n_rep=8） | `qwen2_5_text.py:32-34,87` |
| mlp_hidden / vocab | 11008 / 151936 | `qwen2_5_text.py:36,37` |
| rope_theta / rms_eps | 1e6 / 1e-6 | `qwen2_5_text.py:40,39` |
| tie_embeddings / qkv_bias / qk_norm | True / True / **False** | `qwen2_5_text.py:44,41,43` |
| release ckpt（fp32）大小 | ~16 GB | `run_all_verification.sh:3` |

### ViT（frozen 默认；`sasd_vit_trainable=true` 时同结构可训）
| 量 | 规范值 | 出处 |
|---|---|---|
| depth / hidden / heads / head_dim | 32 / 1280 / 16 / 80 | `vision_qwen25vl.py:28-30,42` |
| patch / temporal / spatial_merge / window | 14 / 2 / 2 / 112 | `vision_qwen25vl.py:32-35` |
| 全注意力块 | (7,15,23,31) | `vision_qwen25vl.py:36` |
| out_hidden / patch_dim | 2048 / 1176 | `vision_qwen25vl.py:37,197` |
| ViT 张量数 | 390 = 1+12*32+5 | `parquet_to_ar_with_embeds.py:126` |
| ViT rope_theta | 10000 | `vision_qwen25vl.py:40` |

### 扩散 / loss / token
| 量 | 规范值 | 出处 |
|---|---|---|
| MASK_ID / NULL_ID / IM_END | 151665 / 151666 / 151645 | `noise.py:11`；`sample_sd.py:91` |
| IMAGE_TOK / VSTART / VEND / VPAD | 151655 / 151652 / 151653 / 151654 | `rope_index.py:19,21`；`verify_ar_round2.py:63-64` |
| IM_START / video | 151644 / 151656 | `prep_train_jax.py:97`；`rope_index.py:20` |
| EPS(mask 地板) / ignore_index / causal shift | 1e-3 / -100 / 1 | `noise.py:11`；`sasd_loss.py:21,34` |
| num_items | 2 × count(labels!=-100) | `noise.py:51` |
| conf 阈值 / step 上限 / 内层步数 / bd_size | 0.9 / 512 / n_mask+5 / 32 | `sample_sd.py:40,52` |
| section 权重（CO/exp/fmb/traj）| 1.5 / 1.0 / 2.0 / 3.0 | `prep_train_jax.py:22`（**data-prep，不在模块**） |
| Beta(α,β)：CO/exp/fmb/traj | (1,2)/(1,1)/(1,1.5)/(2,1) | `prep_train_jax.py:23-24` |
| EXP_BUDGET（explanation NULL-pad 预算）| **192** = block_length(32)×6（固定 pad 到 6 block）| `prep_train_jax.py:21`。**曾见错值 32**（只 pad ~1 block → 训练 scaffold≠inference，[订正 2026-06-20，见 `1plans/07_fidelity_fixes_2026-06-20.md` §2]）|

### 序列长（**合理变体，别一刀切**）
| 量 | 值 | 语境 / 出处 |
|---|---|---|
| L（文本序列） | **1184** | WOD-E2E pseudo、生产默认。`sasd_waymo.yml:25`；`grain_pipeline.py:42` |
| L（distilled） | **1280** | distilled-from-base 覆盖（teacher 解释更长）；`max_target_length=2576`。`launch_..._frombase.sh` |
| L（一次性 parity 样本） | 1120 | 仅指 `HANDOFF.md` 那一个捕获样本，**非**规范 L |
| 2L（doubled） | 2368（=2×1184） | `sasd_waymo.yml:24` 注释 |
| max_target_length（config） | **2376**（≥2L） | `sasd_waymo.yml:63`。**曾见错值 2576**（那是 distilled 覆盖值，非committed config） |
| L / 2L / N @720（trainable-ViT, 200704px）| **1856 / 3712 / 1440** | 720-res override：image budget double 把 L 推到 1856，2L=3712=max_target_length。`1plans/07_fidelity_fixes_2026-06-20.md` §6.1；2026-06-20 GPU log |

### image tokens（**合理变体**；[订正 2026-06-20 见 `1plans/07_fidelity_fixes_2026-06-20.md`]：released-model 忠实分辨率为 **720 tokens / 200704 px**；168/784 一组为**已退役下采样**，committed `sasd_waymo.yml` 仍用之，trainable-ViT 路径迁向 720）
| 值 | 含义 | 出处 |
|---|---|---|
| **720** | **canonical 单份 train+infer**（200704 px，released model；grid (1,32,30)×3 → 2880 patch /4）；2026-06-20 toy GPU PASS | `sasd_vit_ingraph.py:67-69`；`1plans/07_fidelity_fixes_2026-06-20.md` §6.1 |
| **1440** | doubled config @720（2×720）= `sasd_num_image_tokens` | `1plans/07` §6.1；GPU log `sasd_num_image_tokens=1440` |
| 2880 | pixel patch 数 @720（merge 前；= 2×1440 = 3×960） | `sasd_vit_ingraph.py:67-69` |
| 168 | **retired** 单份 train-res（grid (1,16,14) @ 784/50176）；committed config / 旧 embeds-AR 仍用 | `parquet_to_ar_with_embeds.py:6`；`driver.py:128` |
| 336 | retired doubled config（2×168） | `sasd_waymo.yml:26` `sasd_num_image_tokens` |
| 672 | retired pixel patch 数 @168（merge 前） | `grain_pipeline.py:267` |
| image_embeds shape / dtype | `[168,2048]` / bfloat16（retired；720 走 pixels-only AR，不再 bake embeds） | `parquet_to_ar_with_embeds.py:206` |

### 图像分辨率（**合理变体，按用途**；[订正 2026-06-20 见 `1plans/07_fidelity_fixes_2026-06-20.md`]）
| 路径 | min/max pixels | 出处 |
|---|---|---|
| **canonical 训练 / 推理（trainable-ViT, released-model 忠实）** | **200704 / 200704 → 720 tokens** | `prep_train_jax.py:72-73`（默认已改 200704）；`finetune_fast_ddrive.py:159-160`（os.environ 默认 200704）；`batch_inference.py:1168`（字面量） |
| paper-eval（479 帧 ADE/RFS） | 200704 / 200704 | `prep_jax_eval.py:24-25` |
| **retired** 下采样（shared-GPU full-FT；committed config 仍用） | 784 / 50176 → 168 tokens | `sasd_waymo.yml`；`prep_jax_eval_inputs.py:53-54` |

### M-RoPE
| 量 | 值 | 出处 |
|---|---|---|
| mrope_section（**真**） | (16,24,24)，2*(16+24+24)=128 | `qwen2_5_text.py:200`；`sasd_waymo.yml:22` |
| mrope_section（proxy，**别当真值**） | (4,6,6)，=32 | `train_tpu.py:68`；`base.yml:390` |

### 数据 / 导出
| 量 | 值 | 出处 |
|---|---|---|
| array field 数 | **12**（input_ids,labels,rbi,turn,scaffold,weight_vec,block_alpha,block_beta,position_ids,vision_mask,pixel_values,image_grid_thw） | `prep_to_parquet.py:26-32`。**曾见错值 13** |
| 数据集规模 | 50k（50,331）/ full 415,663 | `grain_pipeline.py:267`；`ar_dataset.py:15` |
| 导出 key 数 | 824 = 434 text + 390 visual（lm_head omit） | `maxtext_to_hf_export.py:5,11-14` |

### 评测 / parity（**测量值；不是代码常数**）
| 量 | 值 | 出处 |
|---|---|---|
| 全 479 rated val（ADE3s/ADE5s/RFS） | PT 0.814/1.990/7.914 · JAX 0.839/2.072/7.929 | `REPORT.md`；轨迹 parity 0.01m |
| 52 帧（**早期 partial，非规范**） | JAX 0.853/2.196/8.10 · PT 0.888/2.250/7.913 | `FEATURES.md`（应标注为 preliminary） |
| 嵌入 parity gate | cosine ≥ 0.999 且 max_rel ≤ 5e-2；fp32_vs_ref cosine 1.00000 | `embedding_parity.py:19,57`（base ViT bf16 ~0.998 → demote 为 diagnostic） |
| parity 声明（source-comment） | text logits 3.2e-5 / MM 7.7e-5 / scaffold loss 7.9e-8 | `mm_sampler.py:12-13`；`scaffold.py:6` |
| gate tolerance | phase1/2/4b rel<1e-3；phase4 vit iso<1e-3 & pe_rel<5e-3（fallback e2e<1e-2）；round-trip 824/824 | `parity_*.py`；`maxtext_to_hf_export.py:281` |

### 出处 / pin
- HF snapshot：`0fda81009f4efa58a2debbb48c0c09818e45341f`。
- vendor pin：`eval_sasd @ 4b0f4f2`、`sasd_data @ b18e861`、MaxText root `35dce93`（`PATCHES.md:10-16`）。
