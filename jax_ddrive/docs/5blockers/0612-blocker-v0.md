# 0612-blocker-v0 — 内部 TPU 部署(overfit 小数据集里程碑)的正式 Blocker 分析与执行方案

*版本 v0,2026-06-12。作者:Claude(与 Kaiwen 共同评审)。状态:**待拍板**(§9 决策清单)。*
*本文自包含:不依赖会话上下文即可读。配套阅读:`../README.md`(文档索引)、
`../2implementation-details/DATASET_V2.md`(数据格式)、`../2implementation-details/LABELING.md`
(标签来源)、`../../to-host-chn.md`(总交接)。*

---

## 0. 执行摘要(给只读一屏的人)

**当前里程碑目标**:在小数据集(distilled-400)上,从 **原始 Qwen2.5-VL base 权重**(非 NVIDIA
release 微调权重)训练出一个**过拟合模型**,以端到端证明训练设计正确;文本标签将来由 Waymo
内部模型重新填充。**硬约束:推理必须发生在内部 TPU 上(本地机器最终不可用于推理)。**

**结论先行**:
1. 训练侧无 blocker——MaxText SASD 路径已在真实 TPU 上端到端验证(含数据迭代器断点续训)。
2. **推理侧有 5 个 blocker(B1–B5)**,根因一句话:*Fast-dDrive 的推理是 block-wise
   section-diffusion 去噪,MaxText 没有这个算法;我们已验证的 JAX/NNX 采样器能跑 TPU,
   但吃不了 MaxText 格式的权重。*
3. 推荐路线 **D**(§4):MaxText 训练 → 写一个 MaxText→HF 权重导出工具(~半天)→ 把已验证的
   NNX 采样器 vendor 进 fork,在内部 TPU 上推理。备选 A/B/C 的取舍见 §4。
4. 从 base 训练牵出 4 个衍生事项(V1–V4),其中 **V1(ViT 一致性)是开工前必查项**。
5. 全程产出一份**可带出的简化验证日志**(§7):只含标量/判定/哈希,不含权重、图像、原始样本,
   用于在任何环境下快速裁定"设计是否正确"。

---

## 1. 背景与约束(2026-06-12 时点)

### 1.1 已验证的资产(证据见 `4collect/06_dataset_v2_progress.md`)
- **模型移植**:JAX/Flax-NNX 全模型 parity(text logits rel-max 3.2e-5、ViT 4.0e-5、MM 7.7e-5、
  SASD loss 7.9e-8、mask 逐位一致 [订正 2026-06-16: 这些 rel-max 微观数字是源码注释里的声明(`sample_sd.py:79`、`mm_sampler.py:12-13`、`scaffold.py:6`),非本文重测;下行 479 帧 ADE/RFS 则有 `EVAL_PIPELINE.md:11-16` 佐证]);完整 479 帧 eval 两栈持平(PyTorch ADE@3s 0.814 / JAX 0.839)。
- **生产数据**:dataset v2(ArrayRecord;12 数组含 `pixel_values` + 预计算 `image_embeds`
  bf16 [168,2048]);train 415,663/50k/400 + val 479 已构建、逐字节审计、上传 GCS。
- **生产训练**:MaxText fork(`a645b25`)在真实 v6e-1 上验证 `V2_TPU_VALIDATION_PASS`:
  预计算 embeds 路径 83.4 TFLOP/s(+28% vs ViT-in-loop);checkpoint 含 grain 迭代器状态,
  恢复后**续流不重放**(实测恢复后精确续训 step 12..17)。
- **标签**:三档 parquet(400/50k/415k)为伪文本标签(trajectory 是真 GT);teacher-distill
  (Route A)产出 **distilled-400(L=1280,parquet v1,完成)**;distill-50k 已暂停(本里程碑不需要)。

### 1.2 本里程碑的新约束(用户 2026-06-12 拍板)
| # | 约束 | 影响 |
|---|---|---|
| C1 | **推理只能在内部 TPU**(本地机器最终完全不可用于推理) | 推理栈必须能在内部 TPU 环境独立运行 → 本文 §3 的 blocker 分析 |
| C2 | **从原始 Qwen2.5-VL-3B base 训练**,不从 NVIDIA release 权重 | 权重来源干净(Apache-2.0,避开 release 微调权重的许可/出处问题);衍生 V1–V4(§5) |
| C3 | 数据用 **distilled-400**(L=1280);将来由内部模型重填标签 | distilled-400 的标签出自 release 模型(Route A),仅作管线占位;格式彩排了"换标签→重 token 化→L 变化→v2 AR"全流程 |
| C4 | 本地 5090 仅作开发/测试;overfit ckpt **不会**作为后续训练种子 | demo 工件隔离,不污染正式训练 |
| C5 | teacher-50k 暂停;不需要(本里程碑) | GPU 全空闲可用 |

### 1.3 推理算法的事实(为什么这事不平凡)
一次推理 = ①prompt + deep-JSON scaffold 构建(host/CPU,tokenizer+`section_utils`)→
②图像→ViT embeds → ③**逐 block 迭代去噪**:每个去噪步做一次**全序列 forward**(双向
block-causal mask + 3D M-RoPE + embeds scatter),取 masked 位置 logits,按 schedule
解掩码——重复直至四个 section 全部生成 → ④JSON 解析。

- **MaxText 没有 ③**:它的 decode 是标准自回归 + KV-cache,算法完全不同。
- 我们有两套已验证采样器:**PyTorch 版**(release 代码,吃 HF safetensors)和 **JAX/NNX 版**
  (`ddrive_jax/eval/mm_sampler.py` + `diffusion/sample_sd.py`,吃 NNX 权重)。后者是纯
  JAX——**硬件上天然能跑 TPU**;已对 PyTorch 验证到轨迹 0.01 m 一致。
- 性能现状:无 KV-cache,~16 s/样本(5090);40 样本 demo ≈ 10 min,479 帧 ≈ 2 h(可接受);
  **serving 级吞吐不在本里程碑范围**。

---

## 2. 问题陈述

> 在内部 TPU 上,对一个 **MaxText 训练产出的 SASD 模型** 执行 section-diffusion 推理,
> 并以可审计的方式证明"过拟合训练设计是正确的"。

---

## 3. Blocker 正式清单(B1–B5)

| # | Blocker | 本质 | 不解决的后果 | 解法(推荐) | 工作量 | 残余风险 | 验证方式 |
|---|---|---|---|---|---|---|---|
| **B1** | **权重格式断层**:MaxText ckpt(Orbax/MaxText param 树)两套采样器都读不了 | HF→MaxText 的映射已有(`save_fast_ddrive_params_ckpt.py`,434/434 leaves [订正 2026-06-16: 434/434 leaves 由其调用的 loader `load_fast_ddrive_maxtext.py:193` 打印;save 脚本本身只报 ~3.086B params / `SASD_PARAM_CKPT_SAVED`;leaf 数无误]);**反向工具不存在** [已解决 post-0612: `maxtext-dlm-fork/scripts/maxtext_to_hf_export.py` 已写好并 round-trip 验证(824/824 bitwise=434 text+390 vision)— 见 `4collect/07_from_base_b1_b2_progress.md` A6/B1] | 训出的模型永远取不出来做推理;只能拿 loss 曲线交差 | 写 `maxtext_to_hf_export.py`:现有映射表取逆(转置/重塑回写 safetensors) | ~0.5 天 | 低 | **round-trip 恒等**:HF→MaxText→HF 逐位一致(434/434),先过此关再碰真 ckpt |
| **B2** | **采样器代码不在内部交付物里**:NNX 采样器住在 `jax_ddrive` 仓,内部侧基线只拿 MaxText fork | 代码打包/依赖问题,非算法问题 | 内部环境无推理代码可跑 | **vendor 进 fork**(复制 `input_pipeline/sasd_data/` 的成功模式):`maxtext/diffusion/eval_sasd/` = NNX 模型定义 + 采样器 + 推理 driver;带来源 commit 头 [已解决 post-0612: 已 vendor 到 `src/maxtext/diffusion/eval_sasd/`(pin `4b0f4f2`,见 PATCHES.md);含 `sampler_sasd.py`+`driver.py`+`hf_to_jax.py`+`masks_eval.py`+`models/`] | ~0.5 天 | 低 | fork 单仓自包含测试(PYTHONPATH 只含 fork/src,跑通 1 样本) |
| **B3** | **前处理依赖**:scaffold/prompt 构建用 HF processor(传递依赖 torch);TPU host 默认无 torch | 环境依赖,非算法 | 推理输入没法在 TPU host 上现做 | 三选一:(a) **离线 prep、ship npz**(demo/固定评测集天然适配;479 帧 prep npz 已存在 [订正 2026-06-16: prep 机制(`eval/prep_jax_eval.py`,200704 定分辨率、确定性)已就绪,但当前 `eval_inputs/` 磁盘上只有 20+20 的 T2/T2' demo 子集(40 个 npz);完整 479 帧 npz 需按需重生成]);(b) host 装 torch-CPU(无害);(c) 远期做 torch-free prep。**本里程碑用 (a)** | (a) 0 天 | 低 | npz 与本地参考逐位一致(prep 本就是确定性的) |
| **B4** | **TPU 上的生成数值从未验证**:0.01 m 轨迹一致性是在 GPU/fp32 下验的;TPU 的 matmul 默认精度不同(bf16 倾向) | 数值风险 | 内部第一次跑就当小白鼠;若漂移无基线可比 | TPU 彩排:同一导出权重,TPU 跑 N≥10 样本 vs 本地 GPU 参考输出比对(轨迹 ≤0.1 m、结构化字段相等);保守起步用 **fp32** 跑 demo(慢无所谓),bf16 另测一组留档 | 含在彩排里 | 中→低 | §7 验证日志 S6 段 |
| **B5** | **吞吐(非本里程碑 blocker)**:无 KV-cache,~16 s/样本 | 工程优化项 | serving 不可用(但 demo/离线 eval 完全够) | 本里程碑**明确不做**;serving 阶段再评估"采样器移植进 MaxText + block KV-cache"(即 §4 Option B) | — | — | — |

**一句话总结**:B1+B2 是两个各半天的确定性工程;B3 用离线 prep 归零;B4 用一次彩排关闭;
B5 显式出范围。**没有未知数级别的 blocker。**

> **DiffusionGemma 交叉引用(2026-06-13,详见 `0613-diffusiongemma-insights-v0.md`)**:
> - **B1**:Google 的扩散网络**不能**用来论证"AR 权重 = 扩散权重"(它加载的是单独扩散预训练
>   ckpt,非 AR 权重复用);我们"网络侧零改"的结论**靠我们自己的代码证据**(434/434 映射完整、
>   SASD 零新参、MASK/NULL 是已有词表行),独立成立。
> - **B5 / §4 Option B**:DiffusionGemma `_sampler.py`(jitted `while_loop` + 跨块 KV-cache,
>   冻结前缀靠**自定义 attention mask** 而非 end_index override)是该终态的**官方可落地参考实现**——
>   未来做 serving 时照它走,本里程碑不动。

---

## 4. 备选方案对比与推荐

| 路线 | 内容 | 新代码量 | 交付时间 | 风险 | 评价 |
|---|---|---|---|---|---|
| **A** | MaxText 训 → B1 导出 → 在 TPU 上跑 `jax_ddrive` 的 NNX 采样器(双仓部署) | 0.5 天 | 快 | 低,但内部要装两个仓 | 可行,打包不如 D 干净 |
| **B** | 把 section-diffusion 采样器**移植进 MaxText**(训推一栈) | **数天级**:去噪循环、scaffold、eval mask、解掩码 schedule 全部在 MaxText 层上重写 + 整套**生成 parity 重验** | 慢 | **高**(最易引入细微数值分歧;历史上 JAX 采样器达到 0.01 m 一致花了完整验证轮) | 是 serving 的正确终态,**不是本里程碑的正确手段**。**官方参考实现见 DiffusionGemma `_sampler.py`(`0613-diffusiongemma-insights-v0.md`)** |
| **C** | 改用 NNX harness(`train_tpu.py`)在 TPU 上训练,训推同栈零转换 | 0 | 快 | 中:harness 从未上过真 TPU;放弃已验证的 MaxText 生产路径(iter-ckpt/多机都是 MaxText 给的) | 应急备胎,不建议主路线 |
| **D(推荐)** | = A + 打包:导出工具 + 采样器 **vendor 进 fork**,内部只拿一个仓 | ~1 天 | 快 | 低 | **训练用刚验证完的生产路径,推理用已验证 0.01 m 的采样器,只补一座格式桥** |

导出目标格式定为 **HF snapshot(safetensors)**:NNX 加载器(`load_fast_ddrive_text/vit`)
本来就吃它;顺带保留 PyTorch 栈可读性,未来任何对照实验零成本。

---

## 5. 从 base 训练的衍生事项(V1–V4)

| # | 事项 | 风险 | 处置 |
|---|---|---|---|
| **V1 🔴 ViT 一致性(开工前必查)** | dataset v2 的 `image_embeds` 用 **release 的 ViT** 预计算。若 NVIDIA 微调动过 vision tower(release ViT ≠ base ViT),这些 embeds 对 from-base 训练是**错误特征**——模型会对着一个它没有的视觉编码器的输出学习 | 高(若不查) | 第一步 bitwise 比对两 snapshot 的 `visual.*` 全部张量。**若不同:distilled-400 重算 embeds(base ViT,~20 min),并写入规范:from-base 时代的所有数据一律 base-ViT embeds** |
| V2 | 需要 base Qwen2.5-VL-3B-Instruct snapshot + 用它构建 MaxText param ckpt | 低 | 本地查有无(`train_overfit --base_snap` 历史用过);缺则下载 ~7 GB;builder 同一脚本换 snapshot 路径(同架构,映射表不变) |
| V3 | `\|<MASK>\|`(151665)/`<\|NULL\|>`(151666)在 base 里是**未训练 embedding** | 低 | 过拟合场景可学出(先例:from-base 文本 overfit 3.84→1.47、MM 0.999→0.701);若停滞,备用招 mask-token mean-init |
| V4 | **成功标准重校**:from-base 起点 loss ~1–4(模型需从头学会去噪行为),原"<0.05"不再合理 | — | 见 §6 阈值表;步数预算 12k 起步、按曲线**预授权延长至 ≤50k**(本地 $0,断点续训已验证) |

---

## 6. 过拟合"成功"的判定标准(v0 提案,待签字)

| 层级 | 判据 | 通过线 | 备注 |
|---|---|---|---|
| **T1 loss 级** | 固定噪声、固定 batch 的确定性 eval loss(SASD 逐步随机掩码,必须固定噪声才可比) | 较初值**下降 ≥95% 且平台化**(最后 500 步 Δ<0.005);全程无 NaN;aspiration < 0.1(如实报数,不硬卡) | from-base 起点高,百分比下降比绝对值更稳健 |
| **T2 记忆级(硬门槛)** | 20 个**训练**样本上生成 vs 训练标签 | `trajectory` **逐字符复现 ≥18/20**;`critical_objects`+`fmb` JSON 字段全等 ≥18/20;`explanation` 报 exact 率+编辑相似度,**不设门槛** | 需要 B1+B2 就位;这是"背下来了"的铁证 |
| **T2' 反塌缩** | 20 个 **val** 样本(没见过)上生成 | **20/20 合法 JSON、轨迹可解析**(内容允许错);且 val loss 显著高于 train loss(gap 本身就是过拟合证明) | 防"loss 低是因为塌缩到 scaffold 琐碎解" |
| **T3 管线级** | ckpt 中断恢复曲线连续;同 seed 同曲线 | 机制已验证,顺带在日志里确认 | 来自 grain 确定性 + iter-ckpt |

---

## 7. 简化验证日志(design-validation log)规格 v0

**动机**:训练/推理散落在本地 GPU、trial TPU、内部 TPU 三个环境;需要**一份紧凑、自包含、
可带出环境**的工件,让任何评审者(包括未跟进过程的人)在 5 分钟内裁定"设计是否正确"。

**原则**:
1. **只含标量、布尔判定、计数、哈希**——不含权重、图像、原始样本文本 → 跨环境携带的合规面最小;
2. **机器可读 + 人可读双件套**:`validation_log.jsonl`(逐事件)+ `VALIDATION_SUMMARY.md`(终表);
3. 每条事件**自带阈值与判定**(评审者不需要外部对照表);
4. 由 runner 自动 emit(计划提供 ~50 行的 `vlog.py` 小库),禁止人工填数。

**事件 schema(每行一个 JSON)**:
```json
{"ts": "...", "run_id": "overfit400-base-r1", "stage": "S2_train", "event": "fixed_eval",
 "metrics": {"step": 4000, "eval_loss": 0.211, "drop_pct": 91.2},
 "threshold": {"drop_pct": ">=95 at plateau"}, "verdict": "INFO",
 "refs": {"git": "fork@a645b25+", "dataset_sha": "dataset_info sha256 前 12 位", "ckpt_step": 4000}}
```

**覆盖的阶段与必含事件**:

| 阶段 | 事件 | 关键 metrics | 对应判定 |
|---|---|---|---|
| S1 数据构建 | `data_build`, `data_audit` | rows/shards、抽样数、`byte_exact`、embeds 复算 `{bitwise_frac, max_rel}`、loss-zero 检查 | 数据正确 |
| S2 训练 | `train_start`(init loss / config 指纹)、`step_metrics`(每 100 步)、`fixed_eval`(每 500 步)、`ckpt_save` / `resume`(含 `iter_state: bool`)、`train_final`(init/final/drop_pct/plateau/steps) | loss 曲线脱水版 | **T1、T3** |
| S3 导出 | `export_roundtrip` | `n_tensors`, `identical: bool`, `max_abs_diff` | B1 关闭 |
| S4 训练集推理 | `infer_train_sample` ×20 + `infer_train_summary` | 每样本 `{sid_hash, traj_exact, co_match, fmb_match, expl_sim}`;汇总 `{traj_exact: "19/20", ...}` | **T2** |
| S5 val 推理 | `infer_val_summary` | `{valid_json: "20/20", parse: "20/20", val_train_loss_gap}` | **T2'** |
| S6 TPU 数值彩排 | `tpu_parity` | `{n, traj_max_diff_m, struct_equal_n, dtype}` | **B4 关闭** |
| 终判 | `FINAL_VERDICT` | 各 gate 的 PASS/FAIL + 一行结论 | 设计正确与否 |

**带出与对账设计**:`refs.git` 锚定代码版本;`refs.dataset_sha` 用 `dataset_info_*.json` 的
sha256(数据指纹,不含数据本体);样本以 `sid_hash`(sample_id 的 sha256 前 12 位)出现,
内部持有 id↔hash 映射即可回查,日志本体不暴露 WOD 标识。

### 7.1 留给你拍板的日志问题(Q-log)
- **Q-log1 带出政策**:上述"仅标量/判定/哈希"的 JSONL 是否符合内部带出要求?`sid_hash`
  方案够不够,还是连哈希都不要(改用行号)?
- **Q-log2 格式**:JSONL+MD 双件套可以吗?是否还要 TensorBoard/内部度量系统兼容的副本?
- **Q-log3 阈值签字**:§6 阈值表以本文 v0 为基准生效,还是你要先改数?
- **Q-log4 曲线密度**:`step_metrics` 每 100 步(12k 步 ≈ 120 行)够吗,要不要全步数?
- **Q-log5 对账**:内部复跑时以 `dataset_sha + git + seed` 三元组为对账键,可以吗?

---

## 8. 执行计划(待批准后执行)

```
Phase A 本地(5090,$0;A5 与 A6/A7 并行)
  A1 ViT 一致性检查(release vs base 的 visual.* bitwise)→ 决定 embeds 用哪个 ViT  [10 min]
  A2 base snapshot 确认/下载 + 构建 base 的 MaxText param ckpt                    [~30 min]
  A3 distilled-400 → v2 AR(按 A1 选 ViT)+ AR 路径 loss-zero 抽查                [20 min]
  A4 MaxText GPU smoke 12 步(from-base,sasd_seq_len=1280)                       [15 min]
  A5 本地 overfit(12k 步起,按曲线 ≤50k;ckpt 每 1k 步;vlog S2 全程)            [数小时挂机]
  A6 ∥ B1:导出工具 + round-trip 恒等验证(vlog S3)                               [0.5 天]
  A7 ∥ B2:采样器 vendor 进 fork + 推理 driver(离线 prep 输入)                   [0.5 天]
  A8 本地推理 20 train + 20 val → T2/T2'(vlog S4/S5)→ 产出 TPU 比对用参考输出    [~30 min]
Phase B TPU 彩排(trial v6e-1 当内部替身,~$2–3)
  B1' 导出权重 + vendored 采样器上 TPU,≥10 样本 vs A8 参考(fp32 起步)(vlog S6) [~1 h]
  B2' 交付包:fork commit + PATCHES 更新 + runbook(内部 TPU 训练+推理操作序)
      + 标签接口契约一页(内部模型填标签用)+ validation_log/SUMMARY
```

**总预算**:本地机时 ~1 天(多为挂机)+ 我的工程 ~1.5 天 + trial TPU ~$2–3。

---

## 9. 待拍板决策清单

| # | 决策 | 推荐 | 状态 |
|---|---|---|---|
| D1 | 推理路线 = **D**(vendored NNX 采样器 + 导出工具;采样器进 MaxText 留给 serving) | 是 | ⬜ |
| D2 | 若 release ViT ≠ base ViT → embeds 一律 **base ViT** 重算 | 是 | ⬜ |
| D3 | §6 成功标准 v0(T1 ≥95% 下降+平台化;T2 ≥18/20;步数 ≤50k 预授权) | 是 | ⬜ |
| D4 | Phase B 用 trial TPU 做 B4 数值彩排(~$2–3);不做则 B4 留到内部首跑 | 做 | ⬜ |
| D5 | §7.1 的 Q-log1–5(日志格式/政策/阈值/密度/对账) | 见各问 | ⬜ |

---

## 10. 附录:关键事实速查

- 已验证数字:TPU `V2_TPU_VALIDATION_PASS`(83.4 TFLOP/s,+28%;恢复续训 12..17);
  两栈 eval ADE@3s 0.814/0.839、RFS 7.914/7.929;JAX 采样器 vs PyTorch 轨迹 0.01 m。
- from-base 先例:text overfit 3.84→1.47;MM overfit 0.999→0.701(均从 base Qwen2.5-VL)。
- distilled-400:L=1280 parquet v1(`train/distill_400/parquet_L1280`),GCS 已传;
  v2 AR **未做**(本计划 A3)[已解决 post-0612: distilled-400 base-ViT v2 AR 已构建(400/7 shards,全部审计通过)— 见 `4collect/07_from_base_b1_b2_progress.md` A3;目录 `wod_e2e_sasd_distilled_0613-400_baseViT_v2_ar`];标签 provenance = release 模型 teacher(Route A,hybrid fmb;
  teacher longitudinal 与 GT 一致率仅 4/10 → longitudinal 用 GT 派生)。
- 工具坐标:HF→MaxText = `maxtext-dlm-fork/scripts/save_fast_ddrive_params_ckpt.py`;
  v2 AR 构建 = `jax_ddrive/scripts/parquet_to_ar_with_embeds.py`;采样器 =
  `jax_ddrive/ddrive_jax/eval/mm_sampler.py` + `diffusion/sample_sd.py`;
  数据审计 = `jax_ddrive/scripts/verify_ar_v2_dataset.py`。
- 特殊 token:MASK=151665、NULL=151666(base 中未训练,V3)。
- 修订记录:v0(2026-06-12)首版。
