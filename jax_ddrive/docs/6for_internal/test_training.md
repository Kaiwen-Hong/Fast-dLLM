# test_training.md — STEP 2: internal-TPU test training + inference

*Last updated: 2026-06-16 — changelog: [`updates-latest-0616.md`](updates-latest-0616.md); `history/` is archive (incl. old 0614).*

> **PREREQ — do STEP 1 first:** `transfer-codebase.md` (paste it to the agent) pulls the code
> into google3 and writes `~/.fastddrive_env`. Every block below starts with `source ~/.fastddrive_env`
> (→ `$FORK` / `$DDRIVE` / `$DATA_ROOT` / `$SRC`). If `~/.fastddrive_env` doesn't exist, you skipped STEP 1.

**Audience:** an autonomous coding agent running on the company's internal TPU. This is a
self-contained runbook + context to (1) train an **overfit** SASD model from the clean base
Qwen2.5-VL on a small dataset, (2) export it, (3) generate from it, and (4) emit a compact
validation log — all **inside** the internal TPU env (nothing leaves except scalar log values).

> **What this validates:** the from-base SASD **training pipeline** on the internal TPU, plus the
> **inference path** (export → sample → parse JSON). It is the "test training" milestone before any
> production run. Labels in this dataset are placeholders (teacher-distilled); they'll be refilled
> by the internal model later — so the *goal here is the pipeline + overfit behavior, not label
> quality.*

> **Status of upstream work (validated locally on an RTX 5090, 2026-06-13):** dataset built +
> byte-verified; base param ckpt 434/434; from-base training runs (42.7 TFLOP/s); B1 export
> round-trip 824/824 bitwise; B2 inference self-contained + fork-only generation `SASD_EVAL_PASS`;
> bf16 load PASS; embedding parity cosine 0.99966. **Not yet done on a real TPU** — that's this run.
> Authoritative design doc: `../2implementation-details/INFERENCE_DEPLOY.md`. Blocker analysis:
> `../5blockers/0612-blocker-v0.md`. Build log: `../4collect/07_from_base_b1_b2_progress.md`.

---

## 0. TL;DR — the four stages

```
STAGE 1  train (TPU, FSDP)      base params + distilled-400 base-ViT AR (L=1280) -> Orbax ckpt
STAGE 2  export (host CPU)      MaxText ckpt (items/) --B1--> bf16 HF snapshot          -> 824/824
STAGE 3  infer (1 chip)         offline npz + bf16 snapshot --B2--> 4-section JSON       -> SASD_EVAL_PASS
STAGE 4  embedding parity       ViT fp32 vs bf16 vs reference                            -> cosine>=0.999
                                                                  --> validation_log.jsonl (scalars only)
```

**Success (overfit "memorised"):** fixed-noise eval loss drops ≥95% and plateaus (T1); on ~20
**train** samples the generated `trajectory` reproduces the GT **char-exact ≥18/20** and
`critical_objects`+`future_meta_behavior` fields match ≥18/20 (T2); on ~20 **val** samples 20/20 are
valid parseable JSON (T2'). `explanation` is reported but not gated.

---

## 1. Context (what this model is)

Fast-dDrive = Qwen2.5-VL-3B fine-tuned into a **masked block-diffusion** VLA (scheme "SASD"). It
emits ONE JSON answer with 4 sections: `critical_objects` (12 yes/no), `explanation`,
`future_meta_behavior` (longitudinal/lateral), `trajectory` (5 waypoints @1s).
- **Training** = a single forward+backward on a pre-noised input (per-section Beta noise schedule,
  section-weighted CE on masked positions). No sampling loop. This is the MaxText path (FSDP-ready).
- **Inference** = a multi-step block-diffusion denoise loop (all-MASK scaffold → iteratively unmask
  by confidence → parse JSON). This is the vendored `eval_sasd` path (pure JAX, device-agnostic).
- We train from the **clean base Qwen2.5-VL** (Apache-2.0), not the released NVIDIA checkpoint.
- Hard constraint: inference must run **inside** the internal TPU; only scalar log values leave.

---

## 2. Artifacts on GCS (live, uploaded 2026-06-13)

Source bucket (transfer to CNS as needed): `gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd`
(referred to below as `$SRC`).

> **Where big data lives on the internal side:** put **all large artifacts** (datasets, the base
> snapshot, training checkpoints, exported HF snapshots) under CNS, NOT the pod's `$HOME` (ephemeral
> / small):
> ```
> DATA_ROOT=/cns/is-d/home/chauffeur/perception_training/kaiwenh/data
> ```
> The commands below use `$DATA_ROOT` for every big read/write; only the code tarball + venv stay in
> `$HOME`. MaxText reads/writes Orbax checkpoints to CNS paths directly (`base_output_directory=$DATA_ROOT/...`).

| path under `$SRC` | what | size |
|---|---|---|
| `maxtext_sasd_params_base/fast_ddrive_qwen25_3b_BASE_params/` | **base init weights** (Orbax, bf16, 434/434) | 4.6 GB |
| `wod_e2e_sasd_distilled_0613-400_baseViT_v2_ar/` | **dataset** (ArrayRecord, 7 shards, precomputed **base-ViT** image_embeds; full 4-section labels) | 364 MB |
| `code/fastddrive-<TS>.tgz` | **timestamped code bundle** = `maxtext-dlm-fork` (training + B1 export + B2 inference) **+** `jax_ddrive` (offline prep); `code/fastddrive-LATEST.txt` names the newest | ~38 MB |
| `base_qwen25vl_3b_snapshot/` | **base HF snapshot** (needed for B1 export ref + tokenizer/decode) | 7.0 GB |

Code is published with `bash jax_ddrive/scripts/upload_code_to_gcs.sh` (in-repo, version-controlled; re-run on
every code change → a new immutable `fastddrive-<TS>-<sha7>.tgz` + a `MANIFEST.json` recording the git commit,
bumped `LATEST` pointers). See `../../to-host-chn.md` §6.6.

---

## 3. Code layout (inside the bundle, at `fastddrive-<TS>/maxtext-dlm-fork/`)

```
maxtext-dlm-fork/
  src/maxtext/configs/sasd_waymo.yml            # the SASD config (objective=sasd, dot_product attn,
                                                #   scan_layers=false, remat_policy=full, weight_dtype=bf16)
  src/maxtext/diffusion/sasd.py                 # training: prepare_sasd_inputs / mrope / embeds / loss
  src/maxtext/diffusion/load_fast_ddrive_maxtext.py   # HF->MaxText param builder (bf16-hardened)
  src/maxtext/input_pipeline/sasd_data/         # vendored self-contained AR/grain data path
  scripts/save_fast_ddrive_params_ckpt.py       # build the param ckpt (already done -> on GCS)
  scripts/maxtext_to_hf_export.py               # B1: MaxText ckpt -> bf16 HF snapshot
  scripts/prep_jax_eval_inputs.py               # OFFLINE input-npz builder (imports ddrive_jax)
  src/maxtext/diffusion/eval_sasd/              # B2: self-contained inference (PYTHONPATH=src ONLY)
     models/{rope,qwen2_5_text,vision_qwen25vl}.py, hf_to_jax.py (bf16), masks_eval.py,
     sampler_sasd.py, driver.py (run_eval), embedding_parity.py (run_parity)
  src/maxtext/diffusion/tests/eval_sasd_import_test.py   # self-containment guard
  PATCHES.md                                    # file-by-file diff vs upstream MaxText
```

---

## 4. Environment setup — data ingestion + venv

> **Prerequisite: STEP 1 done** (`transfer-codebase.md`) — the code is in google3 and
> `~/.fastddrive_env` exists (defines `$SRC` / `$FORK` / `$DDRIVE` / `$DATA_ROOT`). Everything below
> `source`s it, so paths never need re-deriving.

**Data ingestion model:** big data goes to **CNS** via a Cloudtop hop —
`gcloud storage cp` to local disk → `fileutil cp -parallelism 50` to CNS → delete the local copy.

```bash
source ~/.fastddrive_env
fileutil mkdir -p $DATA_ROOT

# ---- data: GCS -> Cloudtop local -> CNS (50-thread fileutil), then clean local ----
for A in wod_e2e_sasd_distilled_0613-400_baseViT_v2_ar maxtext_sasd_params_base base_qwen25vl_3b_snapshot; do
  gcloud storage cp -r $SRC/$A ~/ddrive_stage/
  fileutil cp -R -parallelism 50 ~/ddrive_stage/$A $DATA_ROOT/
  rm -rf ~/ddrive_stage/$A
done

# ---- venv ----
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"
uv venv -p 3.11 ~/venv && source ~/venv/bin/activate
uv pip install -q -r $FORK/src/dependencies/requirements/generated_requirements/tpu-requirements.txt
uv pip install -q safetensors pyarrow transformers ml_dtypes flax       # flax/nnx required by eval_sasd

# sanity: B2 inference is self-contained (must import ZERO ddrive_jax):
PYTHONPATH=$FORK/src JAX_PLATFORMS=cpu python -m maxtext.diffusion.tests.eval_sasd_import_test
  # -> EVAL_SASD_SELFCONTAINED_PASS
```
The owner re-publishes code with `bash jax_ddrive/scripts/upload_code_to_gcs.sh` (packs the fork + jax_ddrive from
the single Fast-dLLM repo, uploads `code/fastddrive-<TS>-<sha7>.tgz` + `MANIFEST.json`, bumps `fastddrive-LATEST.txt`)
— see `../../to-host-chn.md` §6.6 (代码与数据发布).
`$FORK` / `$DATA_ROOT` are used in every command below.

---

## 5. STAGE 1 — TPU training (from base)

```bash
source ~/.fastddrive_env                       # -> $FORK $DDRIVE $DATA_ROOT $SRC
RUNOUT=$DATA_ROOT/run_overfit400_base          # Orbax checkpoints land on CNS
source ~/venv/bin/activate
export PYTHONPATH=$FORK/src
cd $FORK
python -m maxtext.trainers.pre_train.train src/maxtext/configs/sasd_waymo.yml \
  model_name=qwen2.5-3b hardware=tpu \
  load_parameters_path=$DATA_ROOT/maxtext_sasd_params_base/fast_ddrive_qwen25_3b_BASE_params \
  sasd_data_dir=$DATA_ROOT/wod_e2e_sasd_distilled_0613-400_baseViT_v2_ar \
  sasd_seq_len=1280 max_target_length=2576 \
  base_output_directory=$RUNOUT run_name=overfit400-base \
  steps=30000 checkpoint_period=3000 opt_type=adamw per_device_batch_size=1
```
- **`opt_type=adamw`** on TPU (FSDP shards the optimizer across chips). The data path loads the
  **precomputed base-ViT embeds** — the ViT is never loaded on the pod (you'll see
  `dataset carries precomputed image_embeds — ViT not loaded`).
- **Smoke first** (cheap, validates the path + iterator resume): set `steps=12 checkpoint_period=6`,
  run twice (2nd run with `steps=18`) — the 2nd must **restore + continue** (not restart from 0).
  The GCP-provisioning wrapper that does this end-to-end is the owner-local
  `/home/kaiwen/launch_maxtext_sasd_tpu_frombase.sh` (NOT shipped in the code bundle — used for the
  free 1-chip rehearsal; on the internal pod you run the python command above directly).
- **Resume** is automatic: re-run the same command (same `base_output_directory`/`run_name`) and it
  restores the latest checkpoint + grain iterator and continues.
- **Per-step loss bounces** (fresh random noise each step) — it is NOT the success metric. Judge by
  the **fixed-noise eval (T1, §9)** computed offline on checkpoints, and by **T2 generation (§7)**.
- Expect ~per-step throughput similar to the local 42 TFLOP/s/chip (faster on 8 chips). From-base
  start loss ≈ 4–6, trending down (local 12k run reached lows ~0.2).

---

## 6. STAGE 2 — B1 export (MaxText ckpt → bf16 HF snapshot)

Runs on the TPU **host CPU** (numpy/jax CPU; ~minutes). **Point `param_ckpt_dir` at the train-state
`items/` subdir** — the exporter restores just the 434 params from it (no extraction step needed).

```bash
source ~/.fastddrive_env
STEP=30000   # or whichever checkpoint you want to export
cd $FORK
PYTHONPATH=$FORK/src JAX_PLATFORMS=cpu \
python scripts/maxtext_to_hf_export.py src/maxtext/configs/sasd_waymo.yml model_name=qwen2.5-3b \
  param_ckpt_dir=$DATA_ROOT/run_overfit400_base/overfit400-base/checkpoints/$STEP/items \
  ref_snapshot=$DATA_ROOT/base_qwen25vl_3b_snapshot \
  out_dir=$DATA_ROOT/overfit400_base_hf
# do NOT add verify_against to a TRAINED export — the text weights changed, so the bitwise round-trip
# would spuriously FAIL. verify_against is ONLY for the base-ckpt sanity check (see note below).
```
- Writes a **bf16** HF snapshot: 434 trained text tensors (inverse-mapped) + 390 `visual.*` copied
  verbatim from the base ref + config/tokenizer. `lm_head` omitted (tied).
- `verify_against` is only meaningful when exporting the **base** param ckpt (expect
  `B1_ROUNDTRIP_PASS 824/824`); for a *trained* ckpt the text weights changed, so drop it.

---

## 7. STAGE 3 — inference + T2 (generate from the overfit model)

### 7a. Build inference inputs OFFLINE (torch host; produces an npz per sample)
**The npz inputs are OWNER-built and shipped** — they live in `$DATA_ROOT/eval_inputs/` (built locally
by the owner; the TPU host has no torch / HF processor / images). **If they're already there, SKIP to
§7b.** The command below is the OWNER's build reference (run off the TPU on a torch host with the WOD
sample JSON + camera JPEGs). `prep_jax_eval_inputs.py` imports `ddrive_jax` and torch. **CRITICAL:
match the TRAINING image resolution** or the image token count / mRoPE won't line up:
```bash
source ~/.fastddrive_env   # OFFLINE (Cloudtop / any torch host); $DDRIVE = jax_ddrive from the bundle
PYTHONPATH=$DDRIVE python $FORK/scripts/prep_jax_eval_inputs.py \
  --sample <WOD distilled targets JSON, e.g. train_targets_distilled_400.json> \
  --img_dir <WOD camera-JPEG dir for those samples> \
  --snapshot <base or release HF snapshot> \
  --out sample0.npz --idx 0
  # defaults already match training: --min_pixels 784 --max_pixels 50176  (-> 168 image tokens)
  # writes x_t0, rbi, position_ids, orig_len, pixel_values, image_grid_thw, target_ids
```
For ~20 train + ~20 val samples, loop `--idx`. (Optional `--with_embeds` runs the ViT offline and
stores bf16 `image_embeds` so the TPU skips the ViT entirely.) Ship the resulting npz(s) into
`$DATA_ROOT/eval_inputs/` on the internal side (that's where §7b/§8 read them).

### 7b. Generate on the TPU (fork-only; fp32 first, then bf16)
```bash
source ~/.fastddrive_env
PYTHONPATH=$FORK/src python -m maxtext.diffusion.eval_sasd.driver \
  --npz $DATA_ROOT/eval_inputs/sample0.npz --snapshot $DATA_ROOT/overfit400_base_hf --dtype fp32 \
  --tokenizer $DATA_ROOT/base_qwen25vl_3b_snapshot --vlog validation_log.jsonl --run_id overfit400-base --sample s0
# then --dtype bf16  (run BOTH precisions; record both)
# -> SASD_EVAL_PASS + metrics {valid_json, traj_parseable, L, n_image_tokens, n_mask_remaining, traj_exact, traj_max_abs_delta, co_match, fmb_match}
```
A correctly-overfit model reproduces the sample's GT trajectory (traj_exact True) and critical_objects.
> **Status (2026-06-14):** the pipeline runs end-to-end but the overfit is **not yet verbatim** — 12k
> gave traj Δ1.12m; **30k regressed** to Δ28.8m (a training-recipe issue: the cosine LR schedule is
> recomputed on resume, perturbing memorised digits — NOT a pipeline bug). The driver now emits
> `traj_exact` / `traj_max_abs_delta` / `co_match` / `fmb_match`; the old `token_agreement`-vs-target
> was a broken (scaffold-vs-flat misaligned) metric and was removed from the T2/target_ids
> comparison path (replaced by `traj_exact`/`co_match`/`fmb_match`); `token_agreement` still
> exists for the local `ref_output` parity branch (driver.py:153 + the module docstring driver.py:17).

---

## 8. STAGE 4 — embedding parity (precomputed-embeds validation + bf16-ViT drift diagnostic)

```bash
source ~/.fastddrive_env
PYTHONPATH=$FORK/src python -m maxtext.diffusion.eval_sasd.embedding_parity \
  --npz $DATA_ROOT/eval_inputs/sample0.npz --snapshot $DATA_ROOT/overfit400_base_hf --vlog validation_log.jsonl
# -> EMBED_PARITY_PASS  GATES on fp32_vs_reference cosine>=0.999 (the npz's precomputed embeds reproduce a
#    fresh fp32 ViT). fp32-vs-bf16 is a DIAGNOSTIC only: the BASE ViT's bf16-matmul cosine is ~0.998 (< 0.999),
#    which is exactly why the canonical path ships precomputed fp32->bf16 embeds and SKIPS the bf16 ViT
#    (see INFERENCE_DEPLOY §4). The eval_inputs npz already carry image_embeds, so this gates on the reference.
```
Report BOTH fp32 and bf16 numbers (abundant TPUs; the bf16-ViT diagnostic is a complete drift record).

---

## 9. Validation log (the only thing that leaves)

Append one JSON object per event to `validation_log.jsonl` — **scalars / booleans / counts / hashes
only**, no weights/images/raw text. `run_eval` and `run_parity` already append their events. Add the
training + T1 events yourself (schema in `../5blockers/0612-blocker-v0.md` §7). Minimum set:
- `data_provenance` (run start): the code bundle's `MANIFEST.json` `git_commit` + each consumed
  artifact's `DATA_MANIFEST.json` `digest` (dataset / base_params / base snapshot / eval_inputs) —
  records exactly which code + which data this run used. Each artifact ships its `DATA_MANIFEST.json`
  alongside it (owner-built via `jax_ddrive/scripts/data_manifest.py`), so the agent just reads the digest.
- `train_final`: init_loss, final fixed-noise eval loss, drop_pct, steps, nan_free.
- `fixed_eval` (T1, every ~Nk steps): deterministic eval loss with a **fixed mask** (the data
  `make_batch` supports `fixed_mask=` for a reproducible noise pattern) on a fixed batch.
- `export_roundtrip` (B1): identical bool + n_tensors (when exporting the base ckpt).
- `infer_sample` ×~40 + summary (T2/T2'): traj_exact, co/fmb field match, valid_json — counts like "19/20".
- `embed_parity` (S6): fp32+bf16 cosine / max_rel + verdict.
- `FINAL_VERDICT`: per-gate PASS/FAIL + one-line conclusion.

---

## 10. Gotchas (do NOT skip — each cost real time to find)

1. **bf16 everywhere.** The exported snapshot is bf16; safetensors' `framework="numpy"` cannot
   decode bf16 — the `eval_sasd` loaders already use a raw-byte `ml_dtypes` reader. Don't "fix" them
   back to `get_tensor`.
2. **`opt_type`**: `adamw` on TPU (FSDP). `adafactor` is the single-GPU choice (don't use on the pod).
3. **B1 reads the train-state `items/` dir** directly (restores just params). No separate extraction.
4. **Inference image resolution MUST equal training** (`min_pixels=784, max_pixels=50176` → 168 img
   tokens). The paper-eval resolution (200704 → 720 tokens) garbles the output (mRoPE/structure
   mismatch). Use each sample's own `image` paths.
5. **Do NOT build x_t0 by masking training `input_ids`** — that loses the deep-JSON NULL-placeholder
   structure → garbled. Only `prep_jax_eval_inputs.py` (`build_scaffold`) produces a valid scaffold.
6. **`flax.nnx` must be installed** (the sampler is NNX). It is a standard flax dependency.
7. **The data path needs NO ViT/torch** (precomputed embeds); inference needs NO ddrive_jax
   (self-contained `eval_sasd`). The offline prep (step 7a) is the only torch/ddrive_jax piece, and
   it runs off the TPU.

---

## 11. Tested locally (RTX 5090) vs to-validate on the internal TPU

| item | local | internal TPU (this run) |
|---|---|---|
| from-base data + base param ckpt 434/434 | ✅ | (consume from GCS) |
| from-base training runs (loss decreasing) | ✅ (42.7 TFLOP/s, L=1280) | ▶ Stage 1 |
| iterator-resume continues (not restart) | ✅ (earlier v6e-1) | ▶ Stage 1 smoke |
| B1 export round-trip 824/824 + from train-ckpt | ✅ | ▶ Stage 2 |
| B2 fork-only generation (valid JSON + 5-wp traj) | ✅ SASD_EVAL_PASS | ▶ Stage 3 |
| bf16 snapshot load | ✅ | ▶ Stage 3 |
| embedding parity cosine 0.99966 | ✅ | ▶ Stage 4 (fp32+bf16) |
| **overfit verbatim T2 (≥18/20)** | 🔬 not yet (12k under-trained; extending to 30k) | ▶ the goal |
| TPU generation numerics (fp32/bf16) | n/a | ▶ B4 rehearsal |

If T2 isn't met after 30k, extend training (≤50k authorized) and re-export/re-eval; the
trajectory digits + the long `explanation` section are the last to memorise.

---

## 12. 全量生产训练(变体 —— 与上面的 overfit 验证不同)

§5–§11 是 **from-base overfit 验证**(distilled-400、L=1280、逐字 T1/T2)。要在内部机器上跑**全量生产训练**
(dummy/pseudo 标签 OK),代码 + 数据**已就绪、无需改代码**,只改三处:

1. **数据**:`sasd_data_dir=$DATA_ROOT/wod_e2e_sasd_full_v2_ar`(**415,663 帧 / 130 shards / ~369G**,v2 AR
   + 预算 embeds,pseudo 标签,**L=1184**)。先按 §4 把它 ingest 到 CNS(§4 的 `for A in …` 里换/加成
   `wod_e2e_sasd_full_v2_ar`;369G,比 distilled 大得多,留足磁盘/时间)。
2. **序列长用默认值**:**不要**加 distilled 的 `sasd_seq_len=1280 max_target_length=2576` 覆盖 —— 全量是
   L=1184,`sasd_waymo.yml` 默认(`sasd_seq_len=1184 / max_target_length=2376`)正好匹配。
3. **规模 + 成功标准**:设 production `steps` / `checkpoint_period`(按算力/预算);**overfit 的 T1/T2 逐字标准
   不适用**(那是过拟合单批的判据)。改判:固定噪声 eval loss 持续下降 + 在 held-out(`wod_e2e_sasd_val_v2_ar`,
   479)上 loss/ADE 不发散。

```bash
source ~/.fastddrive_env && source ~/venv/bin/activate && export PYTHONPATH=$FORK/src && cd $FORK
RUNOUT=$DATA_ROOT/run_full_base
python -m maxtext.trainers.pre_train.train src/maxtext/configs/sasd_waymo.yml \
  model_name=qwen2.5-3b hardware=tpu \
  load_parameters_path=$DATA_ROOT/maxtext_sasd_params_base/fast_ddrive_qwen25_3b_BASE_params \
  sasd_data_dir=$DATA_ROOT/wod_e2e_sasd_full_v2_ar \
  base_output_directory=$RUNOUT run_name=full-base \
  opt_type=adamw per_device_batch_size=1 steps=<prod> checkpoint_period=<N>
  # 不加 sasd_seq_len/max_target_length —— 用默认 L=1184
```
B1 导出(§6)/ B2 推理(§7)/ 验证日志(§9)同上;推理输入仍按训练分辨率 784/50176 → 168 tokens。
> 注:全量是 **pseudo 标签**(`explanation` 等是伪标签,trajectory 真 GT)。这适合验证**全量训练能跑通 + 轨迹学习**;
> "推理是否真帮驾驶"需要 grounded 推理标签 + 消融,属另一阶段。
