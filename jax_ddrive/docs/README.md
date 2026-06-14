# Docs index & maintenance rules

**Start here.** The living docs below are the sources of truth; everything else is either a
stable reference or a frozen historical log.

## Living (must stay current — update these when reality changes)

| doc | scope |
|---|---|
| [`../to-host-chn.md`](../to-host-chn.md) | **The handoff doc** (中文; the English `to-host.md` twin was retired 2026-06-12): what exists, what's verified (numbers), how to reproduce, how to deploy, acceptance criteria. The status banner at the top carries the latest date. |
| [`2implementation-details/DATASET_V2.md`](2implementation-details/DATASET_V2.md) | The production data format: v2 ArrayRecord schema (12 arrays + precomputed `image_embeds`), builder, verification chain, reader API, checkpoint/resume wiring, built-set inventory (local + GCS). |
| [`quick-refresh.md`](quick-refresh.md) | Distilled-dataset quick tracker: per-set location (local / GCS / CNS), build date, size, format/AR status. Update when a distilled set is built or moved. |
| [`5blockers/0612-blocker-v0.md`](5blockers/0612-blocker-v0.md) | 内部 TPU 部署的正式 blocker 分析(B1–B5)+ 备选路线 + from-base 衍生项(V1–V4)+ overfit 成功标准 + **可带出的简化验证日志规格** + 待拍板决策清单。版本化:`5blockers/MMDD-blocker-vN.md`,新版本新文件,旧版不改。 |
| [`5blockers/0613-diffusiongemma-insights-v0.md`](5blockers/0613-diffusiongemma-insights-v0.md) | DiffusionGemma(Google 2026-06-11 发布的离散扩散 LLM)代码精读对我们推理/训练的可迁移洞见:多 agent 工作流深读 + **对抗校验**(推翻了"AR 权重=扩散权重"类比、修正了跨块冻结机制);B1–B5 影响、红线清单、训练/dataloader 裁定。补充 `0612-blocker-v0.md`。 |
| [`6for_internal/0613-transfer-codebase.md`](6for_internal/0613-transfer-codebase.md) | **STEP 1(粘贴给内部 agent)**:自包含 bootstrap —— 从 GCS 拉带时间戳的代码包进 google3、写 `~/.fastddrive_env`、指向 repo 内文档。代码优先,先拿 full context。 |
| [`6for_internal/0613-test_training.md`](6for_internal/0613-test_training.md) | **STEP 2(repo 已在后)**:`source ~/.fastddrive_env` → 数据 ingestion(Cloudtop→CNS)→ 内部 TPU 从-base 训练 → B1 导出 → B2 推理(T2)→ 嵌入彩排 → 验证日志。含 gotchas、tested-vs-pending。 |

The MaxText fork side is documented in the fork itself:
`maxtext-dlm-fork/PATCHES.md` (file-by-file diff vs upstream, vendor sync rules, validation
commands).

## Stable references (update only when the underlying component changes)

| doc | scope |
|---|---|
| `2implementation-details/ARCHITECTURE.md` | JAX/Flax-NNX model port structure |
| `2implementation-details/01_pytorch_reference_algorithm.md` | the PyTorch SASD algorithm being ported |
| `2implementation-details/EVAL_PIPELINE.md` | WOD-E2E eval (two stacks, one metric) |
| `2implementation-details/INFERENCE_DEPLOY.md` | **internal-TPU inference**: B1 MaxText→HF export + B2 self-contained `eval_sasd` sampler + bf16 handling + offline prep (B3) + embedding parity + the deployment runbook. The doc the internal coding agent runs from. |
| `2implementation-details/DATASET.md` | v1 Parquet dataset (superseded for training by DATASET_V2, still the bit-exact source format) |
| `2implementation-details/LABELING.md` | where labels come from (pseudo vs real), annotation provenance (dVLM-AD / GPT-4.1), the teacher-distill pipeline + scripts, the 400/800/50k/415k identity map, L=1280 |
| `2implementation-details/AUDIT.md` | adversarial code-audit findings |
| `3summary/REPORT.md`, `3summary/FEATURES.md` | phase-completion summaries |

## Frozen (historical — never edit, append-only at the time they were live)

| doc | what it captured |
|---|---|
| `1plans/00_PLAN.md` | original port plan (Phases 1–5) |
| `1plans/02_tpu_plan.md` | TPU deployment plan |
| `1plans/03_scaleup_tpu_spec.md` | Phase 6 scale-up spec (dataset + FSDP harness) |
| `1plans/04_tpu_smallscale_validation.md` | $300-trial small-scale validation plan |
| `1plans/05_review_and_fixes_2026-06-14.md` | codex+Claude review pass: code/doc fixes (bf16-npz crash, metrics, bf16-safe ViT loader) + commit/publish/eval-npz actions |
| `4collect/HANDOFF.md`, `4collect/OVERNIGHT_PROGRESS.md` | Phase ≤6 build logs |
| `4collect/05_maxtext_port_progress.md` | Phase 7 MaxText port log |
| `4collect/OVERNIGHT_TPU_PROGRESS{,-chn}.md` | first real-TPU runs (v6e-1, 2026-06-08) |
| `4collect/06_dataset_v2_progress.md` | dataset v2 + AR reader + TPU re-validation (2026-06-12) |
| `4collect/07_from_base_b1_b2_progress.md` | from-base overfit pipeline + B1 export + B2 self-contained inference (2026-06-13) |

## Maintenance rules

1. **One fact, one home.** Current state lives in the living docs; don't restate it
   elsewhere (link instead). Frozen logs keep the *discovery* story, not the truth.
2. When a milestone lands: update the living docs **in the same change**, and append a
   dated entry to a `4collect/` log (create `NN_<topic>_progress.md`, numbered).
3. The handoff doc is `to-host-chn.md` (中文 only; the English `to-host.md` twin is retired).
4. Plans in `1plans/` are written once and frozen; deviations are recorded in the living
   docs, not by editing the plan.
5. Big artifacts (datasets, ckpts, oracles) live under `/home/kaiwen/data/fast-ddrive/`
   and on GCS — docs reference them by path; nothing heavy in git.
