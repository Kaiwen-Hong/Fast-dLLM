# 内部文档 — 更新记录(changelog)

> `6for_internal/` 的**滚动更新记录**,最新在最上。**agent 不必读这篇**;要完整上下文读
> [`transfer-codebase.md`](transfer-codebase.md)(STEP 1)+ [`test_training.md`](test_training.md)(STEP 2)。
> 当出现新批次的 `updates-latest-<新日期>.md` 时,把旧的这份移到 [`history/`](history/)。
> 每条改动附了对应的 commit / bundle 时间戳;完整评审细节见 `../1plans/05_review_and_fixes_2026-06-14.md`。

---

## 2026-06-14 — codex/Claude 评审修复 + 预算嵌入设计定案

**当前发布**:bundle `fastddrive-20260614_024605.tgz`(`fastddrive-LATEST.txt` 已指向它) ·
fork commit `da8c92b` · Fast-dLLM commit `88bce4c`。内部拉 LATEST 即得全部修复。

**代码修复**
- 修(HIGH):`ml_dtypes.bfloat16` 过不了 `np.savez`(重载成 `|V2` void)→ `driver`/`embedding_parity`
  读 `image_embeds` 崩。现 `.view(ml_dtypes.bfloat16)` 再转。这是"TPU 跳过 ViT"的预算嵌入路。
- 修(HIGH):`ddrive_jax.load_fast_ddrive_vit`(`_load_all_tensors`)现 bf16-safe(base 快照可用)。
- `driver`:坏指标 `token_agreement`(scaffold vs flat 错位)仅从 **T2/target_ids 比较路径**移除 → 改 `co_match`/`fmb_match`/`traj_exact`(本地 `ref_output` 校验分支仍保留 `token_agreement`,见 `driver.py:153` 与 docstring `driver.py:17`)+ 标量诊断
  `L`/`n_image_tokens`(应 168)/`n_mask_remaining`(应 0);`--show_text` 默认**关**(只出标量)。
- `embedding_parity`:verdict 改 gate 在 `fp32_vs_reference`(预算嵌入复现 fp32 ViT,**PASS 1.00000**);
  `fp32_vs_bf16` 降级为诊断(base ViT bf16 漂移 ~0.998,正是不跑 bf16 ViT 的理由)。
- 导出脚本 docstring `float32→bf16`;`verify_against` 注明仅 base-ckpt。

**文档**
- 本目录两篇**去日期前缀**(`transfer-codebase.md` / `test_training.md`),living、原地维护,顶部加 Last-updated。
- `test_training` §6 训练导出去掉 `verify_against`(否则假性 FAIL);§7a 明确 **npz 由 owner 离线建好+发到
  `eval_inputs/`**(TPU 无 torch,跳过 prep);§8 parity 改为"gate 在 reference、bf16 是诊断"。
- `PATCHES.md` 双 vendoring pin + 验证指向 `launch_..._frombase.sh`;`INFERENCE_DEPLOY` parity/bucket/阈值同步。

**数据(GCS `gs://<bucket>/`)**
- `eval_inputs/` 新增 **40 个 npz**(20 train + 20 val);**20 train 用训练逐字 `image_embeds`**
  (从 v2 AR 数据集按样本下标取出,已逐样本核验),抹掉 T2 嵌入混淆。
- `launch_maxtext_sasd_tpu_frombase.sh` 改为新的 `fastddrive-LATEST` 取包方案(旧双包名已失配)。

**仍挂着的事**:overfit **尚未逐字**(30k 比 12k 在轨迹上回退,训练配方问题——resume 重算 LR),
不是管线问题。下一步:恒定-LR 干净重训拿下 T2。
