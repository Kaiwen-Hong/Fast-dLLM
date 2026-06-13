# 0613-diffusiongemma-insights-v0 — DiffusionGemma 代码精读:对我们推理(及训练)的可迁移洞见

*版本 v0,2026-06-13。作者:Claude(多 agent 工作流深读 + 对抗校验;与 Kaiwen 评审)。
状态:**分析完成,只读结论**(无代码改动)。*
*配套:`0612-blocker-v0.md`(B1–B5 正式 blocker 分析,本文为其补充)、`../README.md`(索引)、
记忆 `diffusiongemma-jax-reference.md`(仓库坐标与权重事实)。*

> **本文怎么来的**:对 `discreteGemma/gemma/gemma/diffusion/` 全子树(~7320 LOC)+ `hd/` 适配器 +
> docs,用 9 个并行 reader 深读 → 综合映射到 B1–B5 → 对 6 条最承重论断做**对抗校验**。
> 校验**推翻/修正了综合稿的两条论断**(§2),其余 file:line 均已核对。

---

## 0. 一句话结论

DiffusionGemma 是**强外部验证 +「采样器进生产 harness」终态(我们文档的 Option B / B5)的官方可落地蓝图**,
**但不是本次 overfit 里程碑的捷径**——它最值钱的机制(jitted 采样循环、跨块 KV-cache)正好落在我们
**明确出范围**的 B5;能立刻拿来的只有两个**会破坏 0.01 m greedy parity** 的可选旋钮。
**推理路线仍是路线 D,不变。**

---

## 1. DiffusionGemma 是什么(锚定事实)

gemma v4.1.0,2026-06-11 发布(`CHANGELOG.md`)。是一个 **"block-diffusion over AR" 两层文本解码器**,
建在标准 Gemma4 自回归 KV-cache 采样器之上:

- **外层 = 继承的 AR `SamplerLoop`**:`class DiffusionSampler(_sampler_loop.SamplerLoop)`
  (`_sampler.py:320`),但 `_sample_step`(`_sampler.py:490-543`)被 override 成"一步吐一整块
  canvas(默认 256 token)";每块去噪完成后用一次因果 forward 写进**共享 KV-cache**
  (`append_tokens_to_cache`,`_sampler.py:603-648`),后续块通过 cache attend 它
  (`state.step += canvas_length`,`:534`)。
- **内层 = `jax.lax.while_loop`**(`_sampler.py:486`,默认最多 48 步去噪),对单块 canvas 内部
  **全双向自注意力**(`_make_global_attention_mask` 对 canvas 段返回全 1,`:678`),带 `done` 早停。
- **去噪规则**:每步一次 forward → 退火温度整形 logits → categorical 采样 → 按**累积熵预算**保留一批
  低熵 token、其余**重噪成随机 token**(`SampleFromPredictions`,`:105-184`)。噪声调度是线性
  (`noise_proportions[i]=1-i/steps`,`:417`)。
- **校验确认(C4 SUPPORTED)**:`_sampler.py` 是**真正的发布版推理路径**
  (`__init__.py` → `_chat_sampler.Sampler._initialize_sampler_loop`(`_chat_sampler.py:74-106,:84`)
  → `DiffusionSampler`),**不是**旁路占位实现;`hd.sampling` 那套 `DiscreteDDIMStep` 属于
  **未 vendor 进来的 SFT/eval 适配器**(见 §9 caveat)。

与我们 SASD 的根本差异:它是 **uniform/multinomial(D3PM)扩散**(随机 token 起步、可重写已放 token);
我们是 **absorbing-MASK MDLM**(全-MASK 起步、单调解掩码、永不重噪)。这条差异是 §4 红线的根基。

---

## 2. ⚠️ 对抗校验推翻/修正的两条论断(最重要,先看)

### 修正 1 — "DiffusionGemma 复用 AR 权重 ⇒ 我们 B1 网络侧 no-op" 的**支撑类比被证伪**(C1 PARTIAL→REFUTED)

综合稿原想用"扩散网络与 AR 模型共用 checkpoint"佐证 B1。校验拆成两半:

- **架构(SUPPORTED)**:扩散网络确是 `WrappedDiffusionGemmaNetwork{gemma_model}`
  (`hd_gemma_network.py:115-129`,`gemma_param_path='gemma_network.gemma_model'`,
  `gemma_checkpointer.py:252`);模型 = 基础 Gemma4 transformer + **唯一**一个小
  `self_conditioner` 适配器(RMSNorm+FFW+RMSNorm,`_transformer.py:53-78`;`_models.py:21-42`)。
  **没有**学习式 time/noise embedding(`time` 是入参但 forward 里从不消费)。
- **权重(REFUTED,这才是承重的一半)**:配置加载的是**单独扩散预训练 + 指令微调的 ckpt**
  (`DIFFUSIONGEMMA_26B_A4B_IT = gs://gemma-data/checkpoints/diffusiongemma-26B-A4B-it`,
  `_paths.py:23-25`;`sft_pubmedqa.py:43`),**不是普通 AR Gemma 权重复用**;且 loader 对任何
  非-LoRA 的 model-only key 直接 `raise KeyError`(`gemma_checkpointer.py:136-140`),
  即 ckpt 里**必须已含** `self_conditioner` 参数 = 它是完整扩散模型,不是"AR 骨干 + 现加适配器"。

**净结论**:DiffusionGemma**不能**用来论证"AR 权重 = 扩散权重"。
**但我们自己的"B1 网络侧零改"结论仍独立成立——靠的是 C6(我们自己的代码证据),与 DiffusionGemma 无关**:
SASD 不新增任何网络参数,MASK=151665/NULL=151666 是现有 151936 词表的**已有行**
(`sasd.py:23,84`),冻结 ViT 在图外(预计算 embeds),434/434 前向映射完整
(`load_fast_ddrive_maxtext.py:162-164` 强制无未填 leaf)。
**B1 的真实工作 = 写反向导出工具(layout/reshape 取逆,`param_mapping.py:736,748-749` 已有反向分支)
+ 把循环搬进 harness;不是重导参数映射。**(与 `0612-blocker-v0.md` B1 行一致。)

### 修正 2 — 跨块"冻结前缀"的机制说错了(C2 PARTIAL)

核心论断(**复用同一个** Gemma KV-cache 结构、无专用 cache 类型)**成立**:扩散路径复用生产版
`{v,k,end_index,positions}` 每层 cache 字典与标准 `end_index` 写游标。
**但机制归因错了**:推理时冻结前缀**靠自定义 block-causal 注意力 mask
(`create_decoder_attention_mask`,`mask_helpers.py:247-309`)**,**不是** `end_index` override——
那是 SFT 训练路径的东西(`set_cache_end_index`,`mask_helpers.py:317-344`,仅 `sft_model.py:215` 调用);
推理路径明确写着 "Do **NOT** override end_index"(`hd_gemma_ar_state_handler.py:84-86`)。
**我们将来做 Option B 时,冻结前缀要照"自定义 attention mask"走,不要照 end_index 走。**

---

## 3. 能立刻低成本借用的(但有诚实代价)

| 机制 | 出处 | 评价 |
|---|---|---|
| **退火温度** 0.8→0.4 随噪声衰减 | `_sampler.py:217-289` | 纯 logit 变换、零重训。**代价**:改变 commit 的 token → **破 0.01 m greedy parity**,只能作 opt-in "fast/diverse mode",不进 parity 路径 |
| **`while_loop` 早停**(全 unmask 即停) | `_early_stopping.py` + `_sampler.py:452-465` | 概念可借。**但**我们当前 host-loop 因 argmax 兜底保证每步 ≥1 解掩码,**已 ≤n_mask 步终止**(`sample_sd.py:52`),早停在现循环里≈no-op;**只有把循环 jit 化后才有意义** |
| **熵预算解掩码**(累积熵阈值选一批) | `_sampler.py:105-184` | 比我们 max-prob>0.9 阈值(`sample_sd.py:60-65`)更"校准"。但同样破 parity;熵阈为 26 万词表调,我们数字轨迹 token 熵极低,需重调;且**只对 MASKED 位置排序**(它对全部排,因为它重噪所有) |
| **NaN-safe 熵** `lp=log_softmax(x); p=exp(lp); lp=where(p==0,0,lp); H=-sum(p*lp)` | `_early_stopping.py:114-121` | 唯一**严格安全**的小硬化,可无条件采纳 |

---

## 4. 🚫 红线:绝对不要移植的三样(否则静默打坏模型)

1. **随机 token 初始化**(`get_initial_sample` 用 `randint` 而非 MASK,`_sampler.py:62-76`)。
2. **每步把未选中位置重噪成随机 token**(`_sampler.py:174-182`)。
   —— 1+2 是 uniform/multinomial(D3PM)扩散(C3 SUPPORTED:`_sampler.py:56` docstring
   "multinomial diffusion";grep 全树无 absorbing-MASK 路径)。导入它会喂给我们 masked-CE 训练的
   模型从没见过的 OOD 随机 token,**破坏 forward**。我们部署的采样器依赖**从全-MASK 起步**。
3. **self-conditioning**(`_sampler.py:588-594`,`_transformer.py:53-78`):是**训练出来的组件**
   (额外 FFW+RMSNorm 参数),我们 ckpt 里没有,**不能推理时 bolt-on**;只能作未来 SASD-v2 重训候选。

---

## 5. 训练 / dataloader 诚实裁定(原本不确定是否有用)

**本里程碑无需改动;在你关心的两个轴上我们更强,不是更弱:**

- **噪声调度**:它一个样本一个全局连续 t(`UniformTimeSampler`,`sft_model.py:307`);
  我们 per-section Beta(按 block,`noise.py:25-30`),对 4-section JSON 更有表达力。
- **loss 加权**:它字面叫 `NoWeightDiscreteLoss`(无段权重,`sft_pubmedqa.py:126-133`);
  我们 `section_weighted_ce`(`sasd_loss.py:30-41`)更强。
- **dataloader**:它纯文本(sudoku/pubmedqa),**无图像路径**,整套 canvas/AR-target 机制对我们
  多模态(预计算 ViT embeds bf16 + doubled [noisy|clean] + 段权重)不适用;grain pipeline 我们更全。
- 两点旁证(非改动):它 per-token `canvas_id` == 我们 `rbi`(验证"批里带 block 索引"是对的);
  它 `SafeSpan(eps=1e-4)` ≈ 我们 Beta 的 `EPS=1e-3` 守卫。

**唯一不可证的潜在差距(C5 PARTIAL)**:它 loss 真正数学(`NoWeightDiscreteLoss.get_values`、
`corruption_process.convert_predictions`)在**未 vendor 的 `hackable_diffusion` 库**里(本地 import 不到,
§9)。所以"它是否在 CE 里藏了 t-依赖 ELBO 权重"**无法证实**。这是我们"段内不加权 CE"理论上
**唯一可能落后**的点——要钉死须拉那个库源码(§8 可选项)。"我们段加权更强"在**段维度**仍成立。

---

## 6. B1–B5 影响一览(经校验)

| | 效果 | 说明 |
|---|---|---|
| **B1** | reduces\* | 真实工作 = 反向导出工具 + 搬循环;"网络侧零改"靠 **C6 我们自己的证据**成立,**不是**靠已被证伪的 DiffusionGemma 类比(§2 修正 1)。可借鉴其 key 对账的宽容策略(丢弃 ckpt-only key 仅告警、model-only key 走 allowlist,`gemma_checkpointer.py:116-149`)给我们刚性 434/434 门加韧性 |
| **B2** | reduces | `eval_main.py:164-209` 给了"ckpt 经 harness restore → 采样器注册为 evaluator → device 只吐定形 token、host 上 parse"的整合契约,正是我们 vendor 进 fork 的模板 |
| **B3** | no-change | 它纯文本,无 Qwen-VL processor 类比;离线 npz prep 方案不变 |
| **B4** | no-change(+红线) | 它无 TPU 数值/sharding 指引(`docs/sharding.md` 是 stub);只强化"必须确认我们严格 absorbing-MASK"这条红线 |
| **B5** | reduces(未来) | 最强蓝图就在此:jitted `while_loop` + 跨块 KV-cache(经 §2 修正 2 的正确机制)。**但块内每步仍重算 canvas,cache 只省跨块**;我们 scaffold 短、块少,FLOP 省得有限——真正奖品是"循环进 harness"的架构解锁,与文档把 B5 出范围一致 |

---

## 7. 经校验的 claim → verdict 速查

| id | 论断 | verdict | 关键 |
|---|---|---|---|
| C1 | 扩散网络与 AR 共用 base ckpt,只加 self_conditioner | **PARTIAL**(架构✅/权重❌) | 加载的是**扩散预训练** ckpt(`_paths.py:23-25`),非 AR 权重复用 |
| C2 | 复用同一 KV-cache 结构 + `end_index` override 冻结前缀 | **PARTIAL** | 复用 cache✅;冻结靠**自定义 attention mask**,非 end_index override |
| C3 | uniform/multinomial 扩散(随机 init + 重噪 rejects) | **SUPPORTED** | `_sampler.py:56,62-76,174-182`;全树无 absorbing-MASK |
| C4 | 内层是 jitted `while_loop`,carry={step,canvas,sc,rng,done},且是发布版推理路径 | **SUPPORTED** | `_sampler.py:206-214,486`;`__init__`→`Sampler`→`DiffusionSampler` |
| C5 | SFT loss 是无权 masked CE、无 t-reweight | **PARTIAL**(段维度✅/t-权重不可证) | 真正 loss 数学在未 vendor 的 `hackable_diffusion` 库 |
| C6 | 我们 434/434 前向映射完整,仅缺反向导出工具,SASD 零新参 | **SUPPORTED** | `load_fast_ddrive_maxtext.py:162-164`;`param_mapping.py:736,748-749`;`sasd.py:23,84` |

---

## 8. 建议的下一步(仍是只读 / 计划,等拍板)

按"最便宜高价值"排序:

1. **(最该先做)再坐实 B1 地基**:读 `hf_to_jax.py` + `save_fast_ddrive_params_ckpt.py` +
   `load_fast_ddrive_maxtext.py`,确认"434/434 + SASD 零新参 + MASK/NULL 是已有词表行";
   与 **V3(from-base 时 MASK/NULL 行未训练,`0612-blocker-v0.md:108`)** 串起来——
   校验提醒:这改的是两行 embedding 的**值**,不改映射结构,但对 from-base 是独立风险。
2. **一个真正决定 Option B 价值的开放问题**:我们 4 个 JSON section 是否**严格左→右解码、
   且靠前 section 可冻结**?DiffusionGemma 的跨块 KV-cache 冻结只在**块间因果**时成立。
   若我们 section **相互依赖**(后段回头改前段),该 freeze 技巧对我们**可能不成立**,
   Option B 收益打折。不查清,B5 蓝图能否照搬就悬着。须看 `masks.py`
   (`eval_hybrid_block_causal_mask_dense`)与 scaffold 解码顺序。
3. **(可选,本里程碑不必要)** 拉 `gs://.../diffusiongemma-26B-A4B-it` 与未 vendor 的
   `hackable_diffusion` 库,钉死 C1(权重是否真与 AR 共享)与 C5(loss 是否藏 t-权重)。
   要下大文件 + 触网,只在想把这两条彻底定死时才值得。

---

## 9. 来源与方法 caveat

- **方法**:9 reader 并行深读 → 综合 → 6 条承重论断对抗校验(每条独立 agent 回查代码)。
  对抗校验**实际推翻了综合稿的 C1 权重论与 C2 机制论**——这正是不盲信单次综合的价值。
- **未 vendor caveat**:本 checkout 里**没有**顶层 `hackable_diffusion` 包(`import` 失败),
  只有 `hackable_diffusion_adapter/` 胶水 + 自包含的 `_sampler.py`。所以:
  - **推理侧**:`_sampler.py` 是完整且发布版的真路径(C4 已证),结论可靠;
  - **训练 loss 侧**:`NoWeightDiscreteLoss`/corruption 的真正数学不在树内,C5 的"无 t-权重"
    无法证实,只能凭类名推断。
- **框架差异**:官方是 Flax **linen + flax.struct**;我们 `jax_ddrive` 是 **nnx**——算法级借鉴,
  非 drop-in。

---

## 附. 关键 file:line 索引

**DiffusionGemma**(`discreteGemma/gemma/gemma/`):
`diffusion/_sampler.py` 320(DiffusionSampler)/347-488(内层 while_loop)/490-543(块 step,jit)/
603-648(append_to_cache)/62-76(随机 init)/105-184(熵预算+重噪)/217-289(退火温度)/452-465(早停冻结)/
588-594(self-cond);`diffusion/_transformer.py:53-78`(SelfConditioning);
`diffusion/_models.py:21-42`(Gemma4+self_conditioner);`diffusion/_paths.py:23-25`(扩散 ckpt 路径);
`diffusion/_early_stopping.py:114-121`(NaN-safe 熵);
`diffusion/hackable_diffusion_adapter/hd/hd_gemma_network.py:115-129`;`.../hd/gemma_checkpointer.py:116-149,252`;
`.../hd/hd_gemma_ar_state_handler.py:84-86`;`.../hd/mask_helpers.py:247-309,317-344`;
`.../hd/sft_model.py:215,307`;`.../configs/sft_pubmedqa.py:43,126-133`;`.../eval_main.py:164-209`。

**我们**(`jax_ddrive/`、`maxtext-dlm-fork/`):
`ddrive_jax/diffusion/sample_sd.py:52,60-65`;`ddrive_jax/eval/mm_sampler.py:79`;
`ddrive_jax/diffusion/noise.py:25-30`;`ddrive_jax/diffusion/sasd_loss.py:30-41`;
`ddrive_jax/diffusion/masks.py`(eval_hybrid_block_causal_mask_dense);
`...sasd.py:23,84`(MASK_ID/NULL);`load_fast_ddrive_maxtext.py:162-164`;
`param_mapping.py:590-733,736,748-749`。

---

*修订记录:v0(2026-06-13)首版。版本化约定:`5blockers/MMDD-<topic>-vN.md`,新版本新文件,旧版不改。*
