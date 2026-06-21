# 2026-06-21 · SASD 720 trainable-ViT 步真 v5e OOM 根因定位 + 5 修复（chunked-CE → fused logsumexp CE）

> **触发**：canonical「720」trainable-ViT SASD 训练步要在 **v5e-16（16GB HBM/chip）** 上跑。上个 session（2026-06-20）给 `_ce_per_token` 加了 chunked-CE 并据 GPU proxy 结论「fits v5e；v6e/FSDP/remat 都不需要」——但**只在 GPU 验过、真 v5e 标 PENDING**。本次拉真 v5e 确认时，chunked-CE **不仅没省、反而 5 倍恶化**（17.11G → 87.25G），推翻上个结论，换成 fused logsumexp CE 等 5 个修复。
>
> **诚实声明（grep-able）**：`TPU_VALIDATION: PARTIAL — FUSED_CE_TPU: 87.25G→17.11G (regression removed) but STILL OOM by 1.36G (floor remains)` —— fused CE **成功移除了 chunked-CE 的 +70G TPU 回归**（87.25G → 17.11G，正好退回 plain-log_softmax 基线），但 17.11G 仍**超 15.75G/chip v5e 预算 1.36G**——即回到了**原始 `bnu1tt5e9` 的 floor**。fused CE 是**必要但不充分**的修复：它消掉了回归，却**没**消掉 loss 侧 fp32 `[N,V]` floor（XLA logsumexp 反向仍需 `[N,V]` softmax）。`sasd_vit_remat=on` 对该数字 ~0 效果（floor 在 loss、不在 ViT）。本文**不**声称 OOM 已在 TPU 解决。
>
> 证据产物见 §7。本文只改文档、不碰代码。

---

## 1. 触发 / 背景

- canonical「720」trainable-ViT SASD 训练步：`sasd_seq_len=1856`、`2L=3712`、`sasd_num_image_tokens=1440`（**已 double 的 2N**）、`vocab=151936`。目标平台 **v5e-16（16GB HBM/chip）**。
- **2026-06-20（上个 session）**：720 步在 v5e-16 OOM（run `bnu1tt5e9`）：`HLO temporaries 17.11G > 15.75G`。
- 当时的修复假设：train-step 显存 DOMINANT 项是 loss 侧 fp32 全词表 log_softmax（`[2L,V]`），随 2L 线性增长 → 168 配置（`2L=1856`）装得下、720 配置（`2L=3712`）翻倍 OOM。于是给 `sasd.py:_ce_per_token` 加 **chunked-CE**：`jax.lax.map(batch_size=_CE_ROW_CHUNK=512)` + per-row `@jax.checkpoint`，把 fp32 log_softmax 按行分块 + remat。
- **GPU proxy 上有效**：`XLA_PYTHON_CLIENT_MEM_FRACTION=0.50` 压到 ~15.7G 当 v5e proxy，BFC `MaxInUse` 12.66G < 15.75G。
- **据此（仅 GPU proxy）下结论**：「chunked CE fits 720 on v5e；v6e/FSDP/remat 都不需要」—— 但**从未在真 v5e 确认**（标了 PENDING）。这条结论现已写进 `0overview/02_gotchas.md:68-69` / `1plans/07` §6.3。

---

## 2. 本次发现：GPU proxy 假象，在真硬件上被证伪

- **跑了真 v5e-16 确认**（run `vit720conf`，~$5）：chunked-CE bundle **仍然 OOM，而且更糟** —— `HLO temporaries 87.25G > 15.75G`。
- 对比 pre-fix（plain log_softmax）的 17.11G，chunked-CE **大了 5 倍**。两次 bundle 的唯一差别就是 chunked-CE。
- → **上个 session「fits v5e」的结论是 GPU-proxy 假象，在真 TPU 硬件上被直接证伪。** GPU 上 chunked-CE 好（19.6 → 13.6G），TPU 上灾难（17 → 87G）。
- 证据：`/tmp/sanity_fix/tpu_720_confirm.log`（真 v5e OOM 87.25G）。

`TPU_PREFIX_OOM: 17.11G` · `TPU_CHUNKED_CE_OOM: 87.25G (5x worse)`

---

## 3. 根因（Codex 3-agent 审查 + 交叉验证）

3 个并行 Codex agent（memory / correctness / sharding）+ owner 交叉验证定位，证据 `/tmp/review2/out{1,2,3}.md`：

| 维度 | 结论 |
|---|---|
| **FSDP 状态** | **开着的**（`base.yml ici_fsdp_parallelism=-1`，未被覆盖）→ 参数 + 优化器已分片 → 87.25G 是**纯 temporaries**，不是 persistent state。 |
| **根因（CRITICAL）** | chunked-CE 的 `jax.lax.map(batch_size=512)` + per-row `@jax.checkpoint` 在 **TPU XLA 上 lower 成病态**：保留多个完整行空间 fp32 `[~5565,V]≈3.15G` 缓冲（≈22 个 → **+70G**）。GPU 靠 `hlo_rematerialization` 救了它，TPU 的 lowering 不同、没救。 |
| 次要驱动 1 | 全量 fp32 `[2B,2L,V]` logits floor（≈4.2G/chip，其中 row-1 clean 半 25% 没用到）。 |
| 次要驱动 2 | `intermediate_outputs["logits"]` 在 grad-accum=4 下会堆到 ≈16.8G（liveness 隐患）。 |
| 次要驱动 3 | ViT remat（`sasd_vit_remat`）类型默认 `False`（虽 decoders.py 早已 getattr 默认 True）。 |
| 次要驱动 4 | AdamW `mu_dtype` 默认 bf16、非 fp32（与 HF/DeepSpeed 不一致）。 |
| **正确性（R2）** | **无正确性 bug**：doubled-row / noisy-clean 切片 / causal shift / 分母 / M-RoPE / mask 朝向 / ViT scatter **全部与原版 modeling.py 一致**。唯一 footgun：`sasd_num_image_tokens` 语义（types.py 说单份 N→2N，代码当作已 2N；720 用 1440 正确）。 |

`ROOT_CAUSE: chunked-CE lax.map+checkpoint TPU-pathological (+70G temporaries)` · `CORRECTNESS_BUG: NONE`

---

## 4. 5 个修复

Codex 用 `--sandbox workspace-write` 应用，owner 已 `git diff` + `py_compile` 验证；**只动了 4 个文件、无附带改动**。diff 见 `/tmp/review2/fix_apply.log`。

| # | 文件:行 | old → new 要点 | 类别 |
|---|---|---|---|
| **FIX 1** | `maxtext-dlm-fork/src/maxtext/diffusion/sasd.py` `_ce_per_token`（29-44） | chunked `lax.map(512)+@checkpoint` → **fused logsumexp CE**；删 `_CE_ROW_CHUNK` 常量 | CRITICAL / 显存 |
| FIX 2 | `maxtext-dlm-fork/src/maxtext/trainers/pre_train/train.py:382` | 无条件塞 `intermediate_outputs["logits"]=logits` → `if not is_sasd:` 才塞（SASD 不入 aux） | 显存 / grad-accum liveness |
| FIX 3 | `maxtext-dlm-fork/src/maxtext/configs/types.py:554` | `sasd_vit_remat` 默认 `False` → `True`（让类型默认与 decoders.py getattr 默认一致） | 激活显存 headroom |
| FIX 4 | `maxtext-dlm-fork/src/maxtext/configs/sasd_waymo_canonical.yml` | optimizer 段加 `mu_dtype: "float32"`（AdamW 一阶矩 fp32，HF/DeepSpeed-faithful） | 数值保真 |
| FIX 5 | `maxtext-dlm-fork/src/maxtext/configs/types.py:545` | `sasd_num_image_tokens` docstring 改成「已 double 的 2N 计数」，与代码一致 | 文档 / footgun |

### FIX 1 代码（fused logsumexp CE，`sasd.py:39-44`）

```python
valid = labels1d != ignore
safe = jnp.where(valid, labels1d, 0)
lg = logits2d.astype(jnp.float32)                                  # no-op if already fp32
tgt = jnp.take_along_axis(lg, safe[:, None], axis=-1)[:, 0]        # [N] logit at the target id
nll = jax.nn.logsumexp(lg, axis=-1) - tgt                          # [N] == -log_softmax[target]
return jnp.where(valid, nll, 0.0), valid
```

要点：`nll = logsumexp(logits) - logits[target] == -log_softmax(logits)[target]`，是**数学等价**的恒等式。logsumexp 是 fused 的逐行 reduction（`[N,V] → [N]`），**不 materialize `[N,V]` log_softmax**；**无 `jax.lax.map`、无 per-row `@jax.checkpoint`** → 没有那些 TPU 上病态保活的 `[chunk,V]` fp32 缓冲。fp32 峰值被界定为 `[N,V]` logits 读 + 一个 `[N]` reduction，比 plain 形式还少一份 `[N,V]` log_softmax 张量。

`FIX1: fused-logsumexp-CE (math-identical, no lax.map, no per-row checkpoint)`

---

## 5. 验证

### 5.1 GPU 验证（免费，已完成）—— PASS

- 用 fused CE 重跑 `sasd_lossdecrease_test` → loss `0.983 → 0.637`，全 finite，peak 18.54G。
- 与 pre-fix chunked 的 `0.980 → 0.637` 实质一致（差 ~0.002，bf16 run-to-run 噪声）→ **loss 数学保持不变**。
- 证据：`/tmp/sanity_fix/gpu_lossdecrease_fused.log`。

`SASD_TRAIN_LOSSDECREASE_PASS: 0.983 -> 0.637 (fused CE, math identical)`

### 5.2 TPU 验证 —— **PARTIAL（回归已除，floor 仍超 1.36G）**

- 用修复后的 bundle（`fastddrive-20260621_060306-d627036-dirty`，fused logsumexp CE + `sasd_vit_remat=true`）在真 v5e-16 重跑同一 720 setup。结果：`RESOURCE_EXHAUSTED: HLO temporaries 17.11G > 15.75G`、`TRAINABLE_EXIT=1`，pod 自动 clean 拆除。证据：`/tmp/sanity_fix/tpu_720_confirm2.log`。
- **解读（PARTIAL，既非 fixed 也非 failed）：** fused CE **按预期生效**——它消掉了 chunked-CE 的 +70G TPU 病态（**87.25G → 17.11G**，正好退回 plain-log_softmax 基线）。但 **17.11G 仍超 15.75G/chip v5e 预算 1.36G**，即回到了**原始 `bnu1tt5e9` 的 floor**。`sasd_vit_remat=on` 对该数字 **~0 效果**（floor 是 loss 侧 fp32 `[N,V]` + logits，**不是** ViT）。fused CE 期望的「避开 `[N,V]` log_softmax」的额外省显存在 TPU 上**没有兑现**（XLA 的 logsumexp 反向仍需 `[N,V]` softmax）。
- **STATUS：** fused CE = **必要修复**（移除回归）但**单独不充分**。**不声称 OOM 已在 TPU 解决。**

`TPU_VALIDATION: PARTIAL — FUSED_CE_TPU: 87.25G→17.11G (regression removed) but STILL OOM by 1.36G (floor remains)`

**收尾最后 1.36G 的下一个杠杆（每跑一次 ≈ 另一个 ~$5 v5e cycle；决策 pending）：**

- **(a) bf16 logits**（关 `logits_dot_in_fp32` + `cast_logits_to_fp32`；预计 ~−4.5G → ~12.6G **FITS**；代价是轻微 fp32-parity）—— **最便宜**。
- **(b) vocab-tiled / 仅按选中位置从 hidden state 算 loss**（不物化整块 fp32 `[2B,2L,V]` logits；**省得最多、无保真度代价**；代码量较大）。
- **(c) `ici_tensor_parallelism=4`**（把 `[N,V]` 沿 V 切到 4 芯）。
- **(d) v6e（32G/chip）**。

---

## 6. 关键教训（一条重要 gotcha）

**GPU 显存 proxy（BFC `MaxInUse`）对 TPU 是 necessary-not-sufficient，甚至会误导。** XLA 对同一图在 GPU/TPU 上 lower 方式差异巨大（GPU 有 `hlo_rematerialization`，TPU 不同）→ GPU proxy 上「装得下」**不能**推断 TPU 装得下。**只在 GPU 验过的显存优化，必须在真 TPU 复核才能下结论。** chunked-CE 正是反例：GPU 好（19.6 → 13.6G），TPU 灾难（17 → 87G）。

`LESSON: GPU mem proxy is necessary-not-sufficient (misleading) for TPU; GPU-only mem optimizations MUST be confirmed on real TPU.`

---

## 7. 证据日志

- `/tmp/sanity_fix/tpu_720_confirm.log` — 真 v5e chunked-CE OOM 87.25G（本次发现）
- `/tmp/sanity_fix/tpu_720_confirm2.log` — 真 v5e fused-CE 复跑 OOM **17.11G**（回归已除、floor 仍超 1.36G；§5.2 PARTIAL）
- `/tmp/review2/out1.md` / `out2.md` / `out3.md` — Codex 三份审查（memory / correctness / sharding）
- `/tmp/review2/fix_apply.log` — Codex 修复的 diff
- `/tmp/sanity_fix/gpu_lossdecrease_fused.log` — fused CE 的 GPU 复验（`0.983 → 0.637` PASS）
- `/tmp/review2/doc_brief.md` — 本文事实来源简报
- memory: `~/.claude/projects/-home-kaiwen-Desktop-research-Fast-dLLM/memory/fast-ddrive-trainable-vit-tpu-run.md`（已含 2026-06-21 更新）

---

## 8. 受影响文件 + 关联文档更新

### 代码改动（本次 doc-update 只记录、不执行）

- `maxtext-dlm-fork/src/maxtext/diffusion/sasd.py`（FIX 1）
- `maxtext-dlm-fork/src/maxtext/trainers/pre_train/train.py`（FIX 2）
- `maxtext-dlm-fork/src/maxtext/configs/types.py`（FIX 3 + FIX 5）
- `maxtext-dlm-fork/src/maxtext/configs/sasd_waymo_canonical.yml`（FIX 4）

### 关联文档更新（并行进行）

- `0overview/02_gotchas.md`（living，SSOT）：`:68-69` 的旧说法（「修复 = chunked CE / 12.66G proxy / 真 v5e pending」）**现在错了** → 就地纠正为「chunked-CE 在 TPU 病态（+70G）→ 已换 fused logsumexp CE；GPU proxy 误导」。
- `1plans/07_fidelity_fixes_2026-06-20.md`（frozen）：§6.3 / §7 的 chunked-CE「fits v5e」结论用内联 `[订正 2026-06-21: …]` 标注（不重写历史）。
- `4collect/`（append-only）：追加里程碑条目（`08`），记本次 TPU OOM 根因 + 5 修复 + TPU 验证进行中。

> 上述三处由其他子代理并行编辑；本文是「如何修这些 bug」的自含记录，与它们交叉引用、不重复其内容。

---

# 续章（2026-06-21 续）· bf16-alone NO-GO → vocab-tiled CE → 真 v5e OOM 已解决

> **承接 §5.2**：fused CE 把 chunked 的 +70G 病态退回 17.11G floor，但仍超 v5e 预算 1.36G，「下一个杠杆 TBD」。本续章记录后续：bf16-logits-ALONE 被字节核算否掉、改用 **vocab-tiled online-logsumexp CE（标准做法）+ bf16 logits**、GPU 复验数学不变、**真 v5e 编译 + step 0 通过 ⇒ OOM 已解决**。
>
> **诚实声明（grep-able）**：`TPU_OOM: SOLVED (vocab-tiled CE + bf16 logits — compile+step0 passed, no RESOURCE_EXHAUSTED); LOSS_DECREASE: pending (run in flight, 数字待回填)` —— OOM 已在真 v5e 解决（前三次全在编译/step0 之前就 `RESOURCE_EXHAUSTED`，这次编译过、step 0 已执行，train step 装进 15.75G/chip）。但 **loss 下降的数值确认仍在跑**（`completed step: N, loss:` 在 step-0 GCS 存档之后才打印，run 仍在飞）——**不声称训练已完整验证**。

## 9. bf16-logits-ALONE 被否（NO-GO，Codex 字节核算）

§5.2 给的「最便宜」杠杆 (a)「bf16 logits（关 `logits_dot_in_fp32` + `cast_logits_to_fp32`，预计 ~−4.5G）」被 Codex 决策审查**否掉**。字节核算：

- 关掉 `logits_dot_in_fp32` + `cast_logits_to_fp32` 让全量 `[2B,2L,V]` logits 变 bf16（4.2G → 2.1G，省 ~2.1G）。
- **但** `_ce_per_token` 紧接着 `.astype(fp32)` 把 shifted slice `[2,1855,V]` 又上转成 fp32 `[N,V]`（重造 ~2.1G）→ **净省 ~0.001G，不是 1.36G、更不是 4.5G**。
- 只有 XLA 恰好把那个 bf16→fp32 convert **融进 logsumexp 的逐行 reduction**（不物化中间 fp32 `[N,V]`）才装得下——这是赌 XLA lowering 的运气，正是 chunked-CE 翻车那一类（§6 教训）。
- **另一个常见误解纠正**：`float32_logits` 是**注意力 softmax** 的精度旗标，**不**控制词表（vocab）logits 的精度；想让 vocab logits 走 bf16 要关的是 `logits_dot_in_fp32` / `cast_logits_to_fp32`。

→ bf16-logits **单独**不是修复（NO-GO）；它只有和「不重造 fp32 `[N,V]`」的 loss 实现配对才有意义。证据：`/tmp/review2/decision_review.out`。

`BF16_LOGITS_ALONE: NO-GO (CE .astype(fp32) on [2,1855,V] cancels the bf16 saving → net ~0; fits only if XLA fuses the convert — a gamble)` · `float32_logits = attention-softmax precision, NOT the vocab-logit flag`

## 10. 决策：vocab-tiling 是标准做法（之前 chunked 是「对思路、错实现」）

大词表 CE 的 fp32 `[N,V]` 显存爆炸是 **LLM 训练教科书问题**，**vocab-tiling 是头号标准解**：

- **MaxText 本身内置** `num_vocab_tiling` + `logits_from_hidden_states`；T5X / PaxML / Levanter 同款；现代 SOTA 变体是 **Cut Cross-Entropy**。
- 我们的特殊点只是 **SASD 自定义 loss 路径绕过了内置 tiling**，所以要在 `_ce_per_token` 里手写一份等价的 vocab-tiled CE。
- **关键澄清**：之前 chunked-CE 翻车（§3 / §6，+70G TPU 病态）**不是思路错、是实现错** —— 它分的是 **row 维**（`lax.map(batch_size=512)` + per-row `@jax.checkpoint`），在 TPU XLA 上 lower 成病态保活。正解是分 **vocab 维**（online-logsumexp）+ **不用 `lax.map`、不用 per-row checkpoint**（普通 Python for 静态展开，XLA 友好）。

`DECISION: vocab-tiling is the STANDARD fix (MaxText num_vocab_tiling; T5X/PaxML/Levanter; Cut Cross-Entropy). Prior chunked-CE = right idea, BAD impl (row-tiling + lax.map + per-row checkpoint). Fix tiles the VOCAB dim with online logsumexp, no lax.map.`

## 11. 实现的修复：vocab-tiled `_ce_per_token`（替换 §4 FIX 1 的 fused 形式）

`maxtext-dlm-fork/src/maxtext/diffusion/sasd.py`，Codex 实现 A2，已 `git diff` + `py_compile` 验证。要点：

- `_CE_VOCAB_TILES = 8`：对 V 维（`vocab=151936`）静态分 8 块。
- **streaming / online logsumexp**：逐块维护 running max `m` 和 running sum `s`，恒等式 `m + log(s) - logits[target] == -log_softmax(logits)[target]`。fp32 峰值锁在 `[N, V_tile]`（一块 ≈ 1/8 词表）而**非** `[N,V]`。
- gather target logit **在 cast 之前**做（`take_along_axis` 后才 `.astype(fp32)`），避免为读 `[N]` 而把整块 bf16 logits 物化成 fp32。
- **无 `jax.lax.map`、无 per-row `jax.checkpoint`**：普通 Python `for` 静态展开 → 没有 chunked-CE 那种 TPU 病态保活缓冲。
- 签名 `_ce_per_token(logits2d, labels1d, ignore=-100)` 与 `(nll, valid)` 契约**不变**；`section_weighted_ce` / `causal_ce` / `sasd_loss_from_logits` 调用点不变。**数学等价**（与 plain / fused 同一恒等式）。
- 配对 launch flag `logits_dot_in_fp32=false cast_logits_to_fp32=false`（让那份 `[2B,2L,V]` 也是 bf16）—— 此时 vocab-tile 把每块独立上转 fp32，**无论 XLA 是否融合 convert 都安全**（不赌 XLA fusion 运气，这才是它优于 bf16-alone 的地方）。

### vocab-tiled online-logsumexp 循环（`sasd.py:51-65`，已落地）

```python
V = logits2d.shape[-1]
m = jnp.full(labels1d.shape, -jnp.inf, dtype=jnp.float32)
s = jnp.zeros(labels1d.shape, dtype=jnp.float32)

for tile_idx in range(_CE_VOCAB_TILES):
    start = (tile_idx * V) // _CE_VOCAB_TILES
    end = ((tile_idx + 1) * V) // _CE_VOCAB_TILES
    if start == end:
        continue
    tile = logits2d[:, start:end].astype(jnp.float32)            # fp32 peak = [N, V_tile]
    tile_max = jnp.max(tile, axis=-1)
    m_new = jnp.maximum(m, tile_max)
    old_scale = jnp.where(jnp.isneginf(m), 0.0, jnp.exp(m - m_new))
    shifted = jnp.where(jnp.isneginf(m_new)[:, None], -jnp.inf, tile - m_new[:, None])
    s = s * old_scale + jnp.sum(jnp.exp(shifted), axis=-1)
    m = m_new

nll = m + jnp.log(s) - tgt                                       # [N] == -log_softmax[target]
```

`FIX1-v2: vocab-tiled online-logsumexp CE (_CE_VOCAB_TILES=8, fp32 peak [N,V_tile], no lax.map, no per-row checkpoint, math-identical) + bf16 logits at launch`

## 12. 验证（续 §5）

### 12.1 GPU lossdecrease 复验（免费，已完成）—— PASS

- vocab-tiled CE 重跑 `sasd_lossdecrease_test` → loss `0.981 → 0.636`，`SASD_TRAIN_LOSSDECREASE_PASS`，全 finite，peak 18.54G。
- 与 fused（`0.983 → 0.637`）/ chunked（`0.981 → 0.637`）实质一致（差 ~0.002，bf16 run-to-run 噪声）→ **数学保持不变**。
- 证据：`/tmp/sanity_fix/gpu_lossdecrease_vtile.log`。

`SASD_TRAIN_LOSSDECREASE_PASS: 0.981 -> 0.636 (vocab-tiled CE, math identical)`

### 12.2 真 v5e-16 验证 —— **OOM 已解决（loss 下降 pending）**

- run `vit720conf`，bundle `fastddrive-20260621_065405-d627036-dirty`（= vocab-tiled CE），启动命令加 `logits_dot_in_fp32=false cast_logits_to_fp32=false`，跑同一 720 setup。
- ✅ **编译通过，无 `RESOURCE_EXHAUSTED`** —— 这正是前三次（`bnu1tt5e9` 17.11G / `vit720conf#1` chunked 87.25G / `vit720conf#2` fused 17.11G）**全都 OOM 失败**的那一步；这次**过了**。
- ✅ **step 0 已执行**（日志到 `checkpointing.py:841 Waiting for step 0 to finish before checkpoint`，在 train loop 里 `train_step` 之后）。
- **⇒ OOM 问题已解决**：vocab-tiled CE + bf16 logits 把 720 trainable step 装进了 v5e 单芯 15.75G。
- **⏳ 仍 pending**：`completed step: N, loss:` 的下降数字（在 step-0 GCS 存档之后才打印）；run 仍在飞，**loss 轨迹待回填**。
- 证据：`/tmp/sanity_fix/tpu_720_vtile.log`（SSH 实时偷看 worker-0）。

`TPU_OOM: SOLVED (vocab-tiled CE + bf16 logits — compile+step0 passed, no RESOURCE_EXHAUSTED); LOSS_DECREASE: pending (run in flight, 数字待回填)`

## 13. 一句话总结（续）

fused CE 去掉了 chunked 的 +70G 病态但回到 17.11G floor（超 1.36G）；bf16-alone 被 Codex 字节核算否掉（CE 的 `.astype(fp32)` 抵消了 bf16 省的那 ~2.1G）；**真正的修法 = vocab-tiled online-logsumexp CE（标准做法）+ bf16 logits**，GPU 复验 `0.981→0.636` PASS（数学不变），**真 v5e 编译 + step 0 通过 = OOM 已解决**（loss 下降确认 pending，run 在飞）。

## 14. 受影响文件 + 证据日志（续）

### 代码改动（本次 doc-update 只记录、不执行）

- `maxtext-dlm-fork/src/maxtext/diffusion/sasd.py`（FIX 1-v2：vocab-tiled online-logsumexp CE，`_CE_VOCAB_TILES=8`，替换上一轮 fused logsumexp 形式）

### 证据日志（续 §7）

- `/tmp/review2/decision_review.out` — Codex 决策审查（bf16-alone NO-GO 字节核算 + vocab-tiling 决策）
- `/tmp/review2/impl_A.log` — Codex 实现 A2（vocab-tiled `_ce_per_token`）的 diff + py_compile
- `/tmp/sanity_fix/gpu_lossdecrease_vtile.log` — vocab-tiled CE 的 GPU 复验（`0.981 → 0.636` PASS）
- `/tmp/sanity_fix/tpu_720_vtile.log` — 真 v5e vocab-tiled CE：编译 + step0 通过、无 `RESOURCE_EXHAUSTED`（OOM SOLVED；loss 下降 pending）

> **注**：§5.2 的 `TPU_VALIDATION: PARTIAL … STILL OOM by 1.36G` 是当时（fused CE）的诚实状态，**保留不改**作为历史；最新状态以本续章 §12.2 的 `TPU_OOM: SOLVED` 为准。
