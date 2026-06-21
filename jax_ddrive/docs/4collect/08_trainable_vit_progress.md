# 08 — trainable in-graph ViT (`sasd_vit_trainable`) + multi-host TPU validation (2026-06-19)

*Frozen log of the 2026-06-19 session. Truth lives in `../1plans/06_trainable_vit_plan.md` (rolling
plan + §9 log) and `../0overview/02_gotchas.md` (trainable-ViT gotchas + TPU 运维坑); this records what
happened and in what order.*

## Goal (user-set)
Make the SASD ViT **not frozen**: change the implementation so the Qwen2.5-VL ViT runs **in-graph on
`pixel_values` every step** (optimized for multi-node TPU), with the ViT **trainable** (params in the
MaxText train state). Keep the frozen/pre-baked-`image_embeds` path too, behind a toggle. Sanity-check
against the pre-baked embeds (the in-graph ViT should reproduce them), then validate the pipeline on the
free-credit TPU (≈10 steps is enough for the pipeline-works bar). Document everything; work autonomously.

## Decisions (user-locked this session)
1. **Both paths, switched by an arg** — `sasd_vit_trainable` (bool, default `false` = frozen pre-baked
   embeds; `true` = in-graph trainable ViT). Keep frozen as the recommended default.
2. **TPU budget ≤ ~$40** on the free credit; tear pods down immediately, never leave idle.
3. **No git commit** (working tree only); but **keep `06_trainable_vit_plan.md` always updated**.
4. **争取在 TPU 上跑真 trainable 训练步** (strive to run a real trainable train step on TPU), not just
   the ViT body smoke.

## What landed

### Implementation (the toggle + in-graph ViT)
* **ViT split** (`jax_ddrive/ddrive_jax/models/vision_qwen25vl.py`): `VisionTransformer.__call__` split into
  `precompute_structural(grid_thw)` (host numpy geometry: window_index / cos / sin / seg-masks / rev —
  depends only on the grid → a compile-time CONSTANT) + `body(pixel_values, structural)` (pure-jax,
  differentiable, no host numpy). `__call__ = body(pv, precompute_structural(grid))` (bit-identical).
* **In-graph module** (NEW `maxtext-dlm-fork/src/maxtext/diffusion/sasd_vit_ingraph.py`): `SasdInGraphViT`
  (Linen) wraps the NNX ViT `body` via `flax.nnx.bridge.ToLinen` (390 ViT param leaves land in the
  MaxText train state — trainable / shardable / checkpointed). Per-sample ViT `[168,D]` → concat `[ie,ie]`
  → `[336,D]` → repeat ×2 over B → `[2B,336,D]` (the SASD doubling). `precompute_sasd_structural`
  (param-independent) builds the structural constant; `SASD_GRID_THW = 3×(1,16,14)` (672 patch → 168 tok).
* **Wiring (3 batch-key places + forward + config):** `layers/decoders.py` (`_apply_embedding` instantiates
  the in-graph ViT only when `sasd_vit_trainable and sasd_pixel_values is not None`), `input_pipeline/
  waymo_sasd_data_processing.py` (iterator emits `sasd_pixel_values [B,672,1176]` + `sasd_image_pos` when
  trainable, else the pre-baked `sasd_image_embeds`), `utils/maxtext_utils.py` (`get_shaped_batch`; `twoN=N`
  = the DOUBLED 336), `utils/sharding.py` (`_get_sasd_input_data_sharding` branches on the toggle),
  `trainers/pre_train/train.py` (threads `sasd_pixel_values` into `sasd_attention_metadata`),
  `configs/{types.py,sasd_waymo.yml}` (`sasd_vit_trainable: bool = False`).

### Sanity / numerics
* In-graph ViT reproduces the pre-baked embeds — **bf16 cosine 0.99922** (module level),
  **0.99925** with release weights loaded (`SasdInGraphViT`). Module + bf16 parity done.
* ViT **body compiles + autodiffs on a real v5e TPU** (smoke `jax_ddrive/scripts/tpu_vit_body_smoke.py`),
  jax 0.10.2 / flax 0.12.7 (same as the internal google3 env).

### GPU end-to-end train PASS (RTX 5090, real MaxText loop, `wod_e2e_sasd_v2_ar`)
* Trainable runs **3 real train steps**, loss finite, ViT params in the train state and **getting
  gradients**; the **frozen path is byte-unchanged** (regression PASS). The core correctness evidence.
* A bug fixed along the way: `inv_freq` leaked as an abstract `ShapeDtypeStruct(20,)` into the train state
  (a non-Param numpy attr on the NNX ViT). Fix: compute `inv_freq` LOCALLY in `_rotary` (no `self.inv_freq`).
  An UNCONDITIONAL-ViT variant (zero-pixel placeholder, tried for the ckpt-side snapshot-init) made the
  whole ViT subtree abstract → reverted to the CONDITIONAL form. Re-validated PASS.

### FULL MULTI-HOST TPU PASS (the deliverable)
* Run `be47jjta8`, **v5e-16 (4 hosts / 16 chips)** spot @ us-south1-a, jax 0.10.2. Sequence (all 4 workers,
  `EXIT 0`): local BASE-ckpt restore (`Finished load in 2.1 s`, 23 GiB) → compile (`number parameters
  3.086 billion`) → **3 real train steps, loss strictly DECREASING `5.199 → 3.921 → 3.042`** (perplexity
  `181 → 50 → 21`) → checkpoint saved to GCS → pod torn down. **The falling loss is the definitive proof:
  gradients flow through the in-graph trainable ViT and the optimizer updates it.**
* Getting there took several v5e-16 cycles, each peeling one obstacle (see Key numbers + the lessons
  below): RAB on ckpt restore → local-ckpt staging; SSH key-propagation race → warm-up retry; the
  multi-host checkpoint SAVE failing layer by layer on a non-shared FS → a **same-region GCS bucket**
  (`gs://ddrive-sasd-ussouth1-8a53f5ab`, us-south1) with the TPU SA granted `roles/storage.admin`.

## TPU operational lessons (hard-won; canonical copy in `0overview/02_gotchas.md` TPU 运维坑)
* **GCS Regional Access Boundary (RAB) is REGION-scoped** and walls the TPU compute SA. The source bucket
  is us-east5 but the pod is us-south1 → cross-region reads are blocked (`Regional Access Boundary ...
  Precondition`). Restore from a LOCAL ckpt copy (gsutil w/ USER creds, which aren't RAB-walled); SAVE to a
  SAME-REGION bucket with `roles/storage.admin` on the TPU SA.
* **Multi-host Orbax checkpointing requires a SHARED filesystem.** Per-host local disk fails layer by layer
  (grain-iter `process_N-of-4.json` dir → per-process creation needs `primary_host=None` → then
  `array_metadatas` still waits on `primary_process=0`). Same-region GCS is the fix, not Orbax hacks.
* **`enable_checkpointing=false` is rejected** when `load_parameters_path` is set, and step 0 always saves
  (`0 % checkpoint_period == 0`) → you can't simply skip the save; give it a writable same-region target.
* **Fresh-pod SSH `Permission denied (publickey)`** = key still propagating → warm-up retry loop.
* **A multi-host `process_state.cc Raising signal 6` / Shutdown-barrier abort is a SYMPTOM** (one worker
  died, coordination killed the rest) → two-phase SSH: write full per-host logs to disk, read them from a
  fresh session to find the worker that actually failed first.

## Key numbers
* In-graph vs pre-baked embeds: bf16 cosine **0.99922** (module) / **0.99925** (release-loaded).
* GPU 3-step trainable train: finite loss, frozen-path byte-unchanged; ViT 390 leaves get gradients.
* TPU `be47jjta8`: **3.086 B** params; loss **5.199 → 3.921 → 3.042**; perplexity **181 → 50 → 21**;
  ckpt save ~28 s to same-region GCS; first step 1455 s (one-time compile), then 30 s / 1.8 s; EXIT 0.
* Budget: ≈ **$40 of $40**, 0 pods left running (verified across us-south1-a / us-central1-a /
  asia-northeast1-b / us-east5-a).

## Open / next (still DEFERRED — confirm with owner before doing)
* **ViT param sharding:** in the trainable path ViT params are currently **REPLICATED** across the 16
  chips (no logical-axis annotations). Fine at this scale; annotate for real multinode throughput.
* **ckpt-side ViT snapshot-init:** init the in-graph ViT from base/release weights for real training (vs.
  today's random-init from the BASE text ckpt). The unconditional-ViT build leaked abstract params and was
  reverted; right next approach = pass a concrete pixel batch into the param-ckpt build, OR build the ViT
  params separately and `state.replace` the subtree post-restore. NOT a zero placeholder in the forward.
* Optional: cache/speed the first-step compile; full-model step-0 frozen-vs-trainable loss parity (frozen
  ref 0.981) — only the module-level numerics parity is done.

## Commits / locations
* `Fast-dLLM @ jax-ddrive-port` (working tree, **not committed** per owner): `jax_ddrive/ddrive_jax/models/
  vision_qwen25vl.py` (ViT split), `jax_ddrive/scripts/tpu_vit_body_smoke.py` (NEW).
* `maxtext-dlm-fork/src/maxtext`: `diffusion/sasd_vit_ingraph.py` (NEW), `layers/decoders.py`,
  `input_pipeline/waymo_sasd_data_processing.py`, `utils/{maxtext_utils,sharding}.py`,
  `trainers/pre_train/train.py`, `configs/{types.py,sasd_waymo.yml}`, `common/checkpointing.py`
  (multi-host local-FS save fix, gated to `process_count>1 and not gs://`), `diffusion/
  load_fast_ddrive_maxtext.py` (snapshot-init build, deferred). Bundle: `fastddrive-20260619_112401`.
* Launcher: `/tmp/tpu_trainable.sh` (v5e-16 spot @ us-south1-a, SSH warm-up, local data+ckpt, same-region
  GCS output, trap-deletes the pod). Same-region output bucket: `gs://ddrive-sasd-ussouth1-8a53f5ab`.
* Memory: `fast-ddrive-trainable-vit`, `fast-ddrive-trainable-vit-tpu-run` (full TPU gotchas + the PASS).

## 2026-06-20 — fidelity-fix pass (downstream of the trainable-ViT milestone)

After the multi-host TPU PASS, a faithfulness pass corrected the JAX prep to be byte-faithful to the
original Fast-dDrive at **720 image tokens (200704 px)** — the released-model resolution — and fixed the
`EXP_BUDGET=192` (block_length×6) explanation-padding bug (was 32 → train scaffold ≠ inference). Also made
the in-graph ViT grid config-driven (`sasd_vit_grid_thw`, default `(1,32,30)×3`) and pixels-only by default
(`image_embeds` baking kept but off). Validated the 720 trainable path end-to-end on a 5090 GPU
(**`TOY720_GPU_STEP0: PASS`** — 3.086B params, lm_loss 7.945, peak 19.6 GB, step-0 ckpt saved). The same 720
trainable step **OOM'd at compile on v5e-16** (`TPU_VIT_720: OOM` — HLO temporaries 17.11G > 15.75G HBM/chip,
`TRAINABLE_EXIT=1`): a v5e *capacity* limit, not a fidelity issue — needs v6e (32G) or remat/FSDP (the 168-res
trainable passed on v5e-16 earlier, run be47jjta8). The 168/784 downscale is now **retired**. Full frozen record:
[`../1plans/07_fidelity_fixes_2026-06-20.md`](../1plans/07_fidelity_fixes_2026-06-20.md).
## 2026-06-20 (cont.) — 720 OOM 根因定位 + chunked-CE 修复

承接上面的 720 保真度 entry：之前把 v5e-16 上的 `TPU_VIT_720: OOM` 归为「v5e 容量上限，需 v6e/remat/FSDP」。本轮把它**根因定位 (root-caused)** 并修掉——OOM 不在 ViT。

### 根因（OOM 的真正出处）
* 720 步的主导临时张量是 **loss 侧 `diffusion/sasd.py:_ce_per_token` 里整词表的 fp32 `log_softmax`**，不是 ViT 激活。causal `-1` shift + 2L packing 后 CE 作用在 flatten 的 `[N, V]` logit 上，`N = 2L = 3712`、`V = 151936`（Qwen2.5 词表）；未分块时要落地一块 fp32 `[N, V]` log_softmax（前向 + 反向 softmax-grad 各一份），单个瞬时 fp32 峰值约 **~4.5G**。它随 2L **线性**增长：168 配置 `N=2L=1856` 该块只有一半、整步能塞进 16G v5e；翻倍到 720（`N=2L=3712`）把这块也翻倍，正是把 XLA 估的 HLO temporaries 顶到 `17.11G > 15.75G` 而 OOM。机制由源码注释 `sasd.py:29-33` 记录，并被算式 `fp32 [3712,151936]×2 = 4.51G` 佐证。

### 修复（chunked CE，bit-identical）
* `_ce_per_token` 从「一次性 fp32 `[N,V]` log_softmax」改写为 **逐行分块 + remat**：`jax.lax.map(batch_size=_CE_ROW_CHUNK=512)` 包一个 `@jax.checkpoint` 的单行 helper，峰值 fp32 张量从 `[N,V]` 收到 `[512,V]`（`sasd.py:34-53`）。因 log_softmax 逐行独立，**数值 bit-identical**：loss 仍是 **7.945**（与未改前一致）。

### 容量验证（capped 15.7G v5e GPU-proxy）
* 用 `XLA_PYTHON_CLIENT_MEM_FRACTION` 把 BFC 压到 **15.68 GiB** 模拟 16G-HBM v5e（`preallocate=true` 为更干净的 proxy）：train step 真实峰值 **BFC MaxInUse 12.66G < 15.75G** v5e 预算，**整步内无 RESOURCE_EXHAUSTED**——`jax.block_until_ready(state)` 通过、step 0 完成。崩溃发生在**步后的 checkpoint-SAVE**（请求 13.58G 落进碎片化的受限池），被 `checkpointing.py` 捕获后以 `StopTraining('Job is preempted.')` 重抛，是 save 碎片伪影、不是 step OOM。

### 死胡同（记录以免重走）
* **ViT remat = 0 效果**：对 vision body 加 remat 后 Total/Temp 与 baseline **逐位相同**（19.6G/13.8G）、loss 同为 7.945——主导张量不在 ViT。已加 `sasd_vit_remat` 开关但**默认关 (off)**，仅作休眠旋钮（`models/vision_qwen25vl.py` 的 body 循环 + `configs/types.py:sasd_vit_remat` docstring 都注明这一点）。
* **MaxText 静态 `Total/Temp memory size`（`utils/max_utils.py:796`）不可信**：baseline/remat/chunk/bf16 之间几乎不动（13.8→13.6G），从不暴露真实 BFC 峰值；32G GPU 上压根不触发 OOM，做 v5e 预测器很误导。真信号是受限后才可见的 **BFC MaxInUse**。
* **`cast_logits_to_fp32=False` 单独翻 = 无效**：`logits_dot_in_fp32=True` 仍把最终 matmul（及 logits）强制 fp32，fp32 `[N,V]` 块不变。备用 bf16-logit 杠杆须**两个旋钮一起翻**，作为兜底保留（详见 1plans/07）。
* **`enable_checkpointing=false` 绕 save-OOM = 被拒**：加载 base ckpt 时 pydantic 报 `You must set enable_checkpointing=True to load a checkpoint`——反过来印证那次 OOM 在 save 路径、无法这样绕过。

### 结论与待办
* **`CHUNKED_CE_720: GPU PASS`**——chunked CE 移除 ~4.5G fp32-softmax 峰值、数值不变（loss 7.945），capped-proxy 强烈预测 720 step 能塞进 v5e。
* **Caveat / PENDING**：proxy 是 GPU BFC 实测，不等于 TPU 的 XLA 编译期估计（原 `17.11G>15.75G` 是 v5e HLO-temporary 估值）；GPU 证据**必要但不充分**，真正确认仍需一次实际 ~$5 的 **v5e run（PENDING）**。
* 代码改动：`sasd.py`（chunked CE，THE fix）、`vision_qwen25vl.py` / `sasd_vit_ingraph.py` / `decoders.py` / `configs/types.py`（`sasd_vit_remat` 开关 + `sasd_vit_grid_thw` 分辨率配置化，默认 720 grid `(1,32,30)`）。规范数字勿在此重述；见 [`../1plans/07_fidelity_fixes_2026-06-20.md`](../1plans/07_fidelity_fixes_2026-06-20.md) §9（及其 §6.3 的 720-OOM 记录，本条 root-cause 是其延续）与 `../0overview/02_gotchas.md#规范数字框-canonical-numbers`。

## 2026-06-21 — chunked-CE 真 v5e 证伪 + fused logsumexp CE 修复

承接上面的 `CHUNKED_CE_720: GPU PASS` / `PENDING` entry：那个「capped-proxy 强烈预测 720 step 能塞进 v5e」的结论，**这轮在真 v5e 上被证伪 (falsified)**——GPU 显存 proxy 对 TPU 是 necessary-not-sufficient，甚至误导。

### 真 v5e 复跑 → chunked-CE STILL OOM（GPU-proxy 结论被证伪）
* 跑了真 v5e-16 确认（run `vit720conf`，~$5）：chunked-CE 的 bundle **仍然 OOM、而且更糟**——`HLO temporaries 87.25G > 15.75G`，对比 pre-fix 的 17.11G **大 5 倍**。两次唯一差别就是 chunked-CE bundle。→ 上个 session「fits v5e / v6e·remat·FSDP 非必需」纯是 GPU-proxy 假象。
* **`CHUNKED_CE_TPU: FAIL(87.25G)`**（>17.11G pre-fix，5×）。证据：`/tmp/sanity_fix/tpu_720_confirm.log`。

### 根因（Codex 3-agent 审查 + 交叉验证）
* Codex 3 个并行 agent（memory / correctness / sharding）+ 交叉验证定位根因：**chunked-CE 的 `jax.lax.map(batch_size=512)` + per-row `@jax.checkpoint` 在 TPU XLA 上 lower 成病态**——保留多个完整行空间 fp32 `[~5565,V]=3.15G` 缓冲（≈22 个 → +70G temporaries）。GPU 的 `hlo_rematerialization` 救了它，TPU 不同。
* **FSDP 是开着的**（`base.yml ici_fsdp_parallelism=-1` 未被覆盖）→ params+优化器已分片 → 87.25G 是**纯 temporaries**，不是 persistent。
* **无正确性 bug**：doubled-row / noisy-clean 切片 / causal shift / 分母 / M-RoPE / mask 朝向 / ViT scatter 全部与原版 `modeling.py` 一致。证据：`/tmp/review2/out{1,2,3}.md`。

### 5 个修复（applied + GPU-verified）
* **FIX 1（CRITICAL，关键）** `diffusion/sasd.py:_ce_per_token` → **fused logsumexp CE**：`nll = logsumexp(logits.f32,-1) - take_along_axis(logits.f32, target)`（== `-log_softmax[target]`，无 lax.map、无 per-row checkpoint、不 materialize `[N,V]` log_softmax），删除 `_CE_ROW_CHUNK`，数学等价。
* FIX 2 `trainers/pre_train/train.py:382` → `if not is_sasd:` 才把 logits 塞进 `intermediate_outputs`（grad-accum liveness 隐患）。
* FIX 3 `configs/types.py:554` → `sasd_vit_remat` 默认 `False`→`True`（与 decoders.py getattr 默认对齐）。
* FIX 4 `configs/sasd_waymo_canonical.yml` → optimizer 加 `mu_dtype: "float32"`（AdamW 一阶矩 fp32，HF/DeepSpeed-faithful）。
* FIX 5 `configs/types.py:545` → `sasd_num_image_tokens` docstring 改成「已 double 的 2N 计数」与代码一致。
* **GPU 复验（免费，已完成）**：fused CE 重跑 `sasd_lossdecrease_test` → loss `0.983→0.637`、`SASD_TRAIN_LOSSDECREASE_PASS`、全 finite、peak 18.54G，与 pre-fix chunked 的 `0.980→0.637` 实质一致（差 ~0.002，bf16 噪声）→ loss 数学保持不变。87.25G 是 temporaries（FSDP 已开）。证据：`/tmp/sanity_fix/gpu_lossdecrease_fused.log`。
* **`FUSED_CE_GPU: PASS(0.983→0.637)`**。

### 结论与待办
* **`TPU_VALIDATION: IN_PROGRESS`（待确认）**——用修复后的 bundle（`fastddrive-20260621_060306-d627036-dirty`）在真 v5e-16 重跑同一 720 setup，**此刻仍在跑、结果未定**；不得预先声称 OOM 已在 TPU 解决。预期：无 `RESOURCE_EXHAUSTED`（87.25G→~12-15G）+ loss 下降，结果待补。
* **[回填 2026-06-21] `FUSED_CE_TPU: OOM(17.11G) — regression removed (was 87.25G) but floor 1.36G over`**（`TPU_VALIDATION: PARTIAL`）——修复 bundle（fused CE + `sasd_vit_remat=true`）真 v5e-16 复跑（run `vit720conf`）：`RESOURCE_EXHAUSTED: HLO temporaries 17.11G > 15.75G`、`TRAINABLE_EXIT=1`、pod 自动 clean 拆除。解读（PARTIAL，既非 fixed 亦非 failed）：fused CE **如期移除 chunked-CE 的 +70G TPU 回归**（87.25G→17.11G，正好退回 plain-log_softmax 基线），但 17.11G 仍**超 15.75G/chip 预算 1.36G**——回到原始 `bnu1tt5e9` 的 floor。`sasd_vit_remat=on` 对该数字 **~0 效果**（floor 是 loss 侧 fp32 `[N,V]`+logits，**不是** ViT）；fused CE 期望的「避开 `[N,V]` log_softmax」额外省显存在 TPU **未兑现**（XLA logsumexp 反向仍需 `[N,V]` softmax）。即 fused CE = **必要但不充分**。收尾最后 1.36G 的下一杠杆（决策 pending）：(a) bf16 logits（关 `logits_dot_in_fp32`+`cast_logits_to_fp32`，~−4.5G→~12.6G FITS，最便宜）/ (b) vocab-tiled / 选中位置从 hidden 算 loss（不物化 fp32 `[2B,2L,V]` logits，省最多、无保真度代价）/ (c) `ici_tensor_parallelism=4` / (d) v6e 32G。证据：`/tmp/sanity_fix/tpu_720_confirm2.log`。详见 [`../8doc_updates/2026-06-21_sasd-tpu-oom-fix.md`](../8doc_updates/2026-06-21_sasd-tpu-oom-fix.md) §5.2。
* 关键教训：**只在 GPU 验过的显存优化必须在真 TPU 复核**——XLA 对同一图在 GPU/TPU 上 lower 差异巨大（GPU 有 `hlo_rematerialization`，TPU 不同）；chunked-CE 是反例（GPU 19.6→13.6G，TPU 17→87G）。
* 详见 [`../8doc_updates/2026-06-21_sasd-tpu-oom-fix.md`](../8doc_updates/2026-06-21_sasd-tpu-oom-fix.md)；规范数字勿在此重述，见 `../0overview/02_gotchas.md#规范数字框-canonical-numbers`。

## 2026-06-21 (cont.) — bf16-alone NO-GO → vocab-tiled CE → 真 v5e OOM 已解决

承接上面的 `FUSED_CE_TPU: OOM(17.11G)` / `PARTIAL` entry：fused CE 把 chunked 的 +70G 回归移掉、退回 17.11G plain-log_softmax 基线，但仍超 15.75G/chip **1.36G**。本轮收尾这最后 1.36G。

### bf16-logits-ALONE 被否（NO-GO）
* 上一条列的「下一杠杆 (a) bf16 logits」经 **Codex 字节核算否掉**：关 `logits_dot_in_fp32`+`cast_logits_to_fp32` 让 `[2B,2L,V]` logits 变 bf16（4.2→2.1G），**但** `_ce_per_token` 紧接着对 shifted slice `.astype(fp32)` 又造出一块 2.1G fp32 `[N,V]` → **净省 ~0.001G，不是 1.36G**；只有 XLA 恰好把 convert 融进 logsumexp 才装得下——赌 XLA lowering（同 chunked-CE 翻车那类风险）。另：`float32_logits` 控的是**注意力** softmax 精度，不控词表 logits。证据：`/tmp/review2/decision_review.out`。
* **`BF16_LOGITS_ALONE: NO-GO`**（CE 的 `.astype(fp32)` 上转抵消了 logits 的 bf16 省显存）。

### 修复 — vocab-tiled online-logsumexp CE（标准大词表-CE 做法）
* `diffusion/sasd.py:_ce_per_token` → **vocab-tiled streaming logsumexp**：对 V 维分 `_CE_VOCAB_TILES=8` 块做 online logsumexp（逐块累积 running max `m` + running sum `s`，`m+log(s)-tgt == -log_softmax[target]`），fp32 峰值锁在 `[N, V_tile]` 而非 `[N,V]`，**无 `jax.lax.map`、无 per-row `jax.checkpoint`**（普通 Python 静态 for，XLA 友好）。签名 `(nll, valid)` 契约不变；配合 bf16 logits（`[2B,2L,V]` 也降 bf16）。这是 **LLM 训练教科书的大词表-CE 头号标准解**（MaxText 内置 `num_vocab_tiling`+`logits_from_hidden_states`；T5X/PaxML/Levanter 同；SOTA 变体 Cut Cross-Entropy）——我们只是 SASD 自定义 loss 路径之前绕过了内置 tiling。之前 chunked-CE 翻车是**实现错（分 row 维 + lax.map + per-row checkpoint，TPU 病态），不是思路错**；正解是分 vocab 维 + online-logsumexp + 不用 lax.map。
* **GPU re-verify（免费）**：vocab-tiled CE → loss `0.981→0.636`、`SASD_TRAIN_LOSSDECREASE_PASS`、全 finite、peak 18.54G——与 fused(`0.983→0.637`)/chunked(`0.981→0.637`) 实质一致（差 ~0.002 bf16 噪声）→ 数学保持不变。证据：`/tmp/sanity_fix/gpu_lossdecrease_vtile.log`。
* **`FUSED_VTILE_CE_GPU: PASS(0.981→0.636)`**。

### 真 v5e — OOM 已解决（loss-decrease 待回填）
* 真 v5e-16 复跑（run `vit720conf`，bundle `fastddrive-20260621_065405-d627036-dirty` = vocab-tiled CE，启动加 `logits_dot_in_fp32=false cast_logits_to_fp32=false`）：**编译通过，无 `RESOURCE_EXHAUSTED`**——这正是前三次（bnu1tt5e9 17.11G / vit720conf#1 chunked 87.25G / vit720conf#2 fused 17.11G）全 OOM 的那一步，这次**过了**；**step 0 已执行**（日志到 `checkpointing.py:841 Waiting for step 0 …`，在 train loop 里 train_step 之后）。⇒ vocab-tiled CE + bf16 logits 把 720 trainable step 装进 v5e 单芯 15.75G。
* **`TPU_OOM: SOLVED`**（vocab-tiled CE + bf16 logits → compile + step0 passed, no RESOURCE_EXHAUSTED）。证据：`/tmp/sanity_fix/tpu_720_vtile.log`。
* **`LOSS_DECREASE: pending`（run in flight）**——`completed step: N, loss:` 的下降数字在 step-0 GCS 存档之后才打印，run 仍在飞，loss 轨迹待回填；**不得**声称「训练已完整验证」，仅「OOM 已解决；loss-decrease 确认 pending」。
* 详见 [`../8doc_updates/2026-06-21_sasd-tpu-oom-fix.md`](../8doc_updates/2026-06-21_sasd-tpu-oom-fix.md)；规范数字勿在此重述，见 `../0overview/02_gotchas.md#规范数字框-canonical-numbers`。
