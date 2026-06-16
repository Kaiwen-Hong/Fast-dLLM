# Fast-dDrive SASD — MaxText fork patch documentation

Everything this fork changes relative to its upstream MaxText snapshot, so the work can be
reviewed file-by-file or rebased onto another (e.g. internal) MaxText version.

## Provenance

* This repo began as a **MaxText snapshot taken ~2026-05-05** (root commit `35dce93` already
  contains the full tree; the exact upstream commit was not recorded — when rebasing, diff
  this tree against upstream history around that date, or simply re-apply the patches below).
* All changes are **flag-gated**: with the default `objective: "ar"` and any non-`waymo_sasd`
  `dataset_type`, every modified file behaves exactly as upstream.
* Reference implementation (numerics oracle, data-build toolchain, parity gates):
  the same `Fast-dLLM` repo, `jax_ddrive/` (sibling sub-tree). Vendored files are pinned at
  TWO commits: the `sasd_data` training path @ `b18e861`, and the `eval_sasd` (B2) inference stack
  @ `4b0f4f2` (each vendored file's header records its own pin).
* **Consolidated 2026-06-16:** this fork is now a sub-tree of the `Fast-dLLM` repo (sibling of
  `jax_ddrive/`), so a single `git rev-parse HEAD` describes both. It was previously a standalone git
  repo @ `da8c92b`; that full history is backed up at
  `/home/kaiwen/data/fast-ddrive/maxtext-dlm-fork-history-2026-06-16.bundle`.

## What runs in production

```
configs/sasd_waymo.yml          objective=sasd + dataset_type=waymo_sasd
  └─ WaymoSasdDataIterator      input_pipeline/waymo_sasd_data_processing.py
       └─ make_sasd_loader     input_pipeline/sasd_data/ (vendored, self-contained)
            └─ ArrayRecord v2 shards (12 arrays + precomputed image_embeds, bf16)
  └─ train step                 trainers/pre_train/train.py objective=="sasd" branch
       └─ maxtext.diffusion.sasd (loss + M-RoPE + mask prep; self-contained jax/np)
```

Dataset v2 contract (shapes, dtypes, embeds provenance & doubling rule) is recorded in each
dataset dir's `dataset_info_train.json` and in
`Fast-dLLM/jax_ddrive/docs/2implementation-details/DATASET_V2.md`.

## Modified upstream files

| File | Change | Why |
|---|---|---|
| `configs/types.py` | +`objective` (`ar\|mdlm\|sasd`), `WAYMO_SASD` dataset type, `sasd_*` fields (mrope_section, rope_theta, seq_len, num_image_tokens, data_dir, vit_snapshot), mdlm fields | config surface; defaults keep upstream behavior |
| `configs/base.yml` | default values + comments for the new fields (`objective: "ar"`, `sasd_*` zeros) | upstream-default config stays valid |
| `trainers/pre_train/train.py` | `objective=="sasd"` loss branch (doubled-sequence forward → `sasd_loss_from_logits`), SASD data-dict plumbing, mdlm branch (phase 23) | the SASD training objective |
| `input_pipeline/input_pipeline_interface.py` | dispatch `dataset_type=="waymo_sasd"` → `WaymoSasdDataIterator` | new data path entry |
| `layers/attentions.py` | when `attention_metadata` carries `sasd_rope_cos/sin`, apply Fast-dDrive 3D M-RoPE (`apply_rope_full`) to q/k instead of the standard rope | SASD uses precomputed doubled M-RoPE |
| `layers/attention_op.py` | optional `sasd_attn_mask` [2B,2L,2L] bool → additive large-negative mask on attn weights | hybrid block-causal training mask |
| `layers/decoders.py` | `attention_metadata` plumbing; scatter `sasd_image_embeds` into token embeddings at `sasd_image_pos` (`y.at[bidx, pos].set(...)`) | vision-token injection without a ViT inside MaxText |
| `layers/embeddings.py` | `attend()` dtype handling (fp32 table when attend_dtype fp32) | logits parity at fp32 loss |
| `optimizers/optimizers.py` | add `opt_type=adafactor` branch (factored, configurable) | memory-lean option used by smokes |
| `utils/maxtext_utils.py` | `get_shaped_batch()` SASD branch: abstract shapes for the doubled batch (`inputs` [2B,2L], `sasd_*` keys incl. optional image embeds) | static compile shapes |
| `utils/sharding.py` | `_get_sasd_input_data_sharding`: per-key data sharding (rank-1/2/3 SASD keys can't share the standard single 2-axis spec) | multi-device batch sharding correctness |
| `utils/max_utils.py` | defensive `except (ImportError, AttributeError, AssertionError)` in an optional-dep probe | robustness on minimal pods |
| `common/checkpointing.py` | `dataset_type in ("grain","waymo_sasd")` at the 3 grain-iterator sites (handler registration, save, restore) | data-iterator state in checkpoints → resume continues the stream instead of replaying (validated: handler save/restore round-trip is bit-identical) |

## New files

* `src/maxtext/diffusion/` — `sasd.py` (SASD loss + M-RoPE + mask/scatter prep +
  `compute_fast_ddrive_image_embeds`; pure jax/np, no ddrive_jax import),
  `load_fast_ddrive_maxtext.py` (HF snapshot → MaxText param ckpt builder),
  `tests/sasd_{parity,weight_parity,vla_parity,train_step,lossdecrease}_test.py`
  (parity vs recorded oracles + loss-decrease integration).
* `src/maxtext/input_pipeline/waymo_sasd_data_processing.py` — per-step host iterator:
  grain loader → (dataset-v2: pass through precomputed `image_embeds`, doubled by concat;
  v1 fallback: run the frozen ViT, lazy `ddrive_jax` import) → `prepare_sasd_inputs` →
  `_form_global_array`.  Exposes `local_iterator` (the grain `DatasetIterator`) for
  checkpointing.
* `src/maxtext/input_pipeline/sasd_data/` — **vendored** from `jax_ddrive @ b18e861`
  (`schema, parquet_dataset, ar_dataset, noise, grain_pipeline`).  Self-contained
  (numpy/grain/pyarrow; TF lazily for AR decode, pinned to CPU).  **Do not edit here** —
  edit `Fast-dLLM/jax_ddrive/ddrive_jax/...` and re-copy (only the import lines differ;
  see the header in each file).
* `src/maxtext/configs/sasd_waymo.yml`, `src/maxtext/configs/models/qwen2.5-3b.yml` —
  the production SASD config + Qwen2.5-3B model shape.
* `scripts/train_sasd_waymo.sh` (GPU smoke + TPU command), `scripts/save_fast_ddrive_params_ckpt.py`.
* `scripts/maxtext_to_hf_export.py` — **B1**: inverse of the param-ckpt builder. MaxText Orbax
  param ckpt → **bf16 HF safetensors snapshot** (434 text leaves inverse-mapped via the
  `QWEN_MAXTEXT_TO_HF_PARAM_*` hooks with `saving_to_hf=True`; 390 `visual.*` copied verbatim
  from a reference snapshot; `lm_head` omitted, tied). `--verify_against` does the round-trip
  identity check (824/824 bitwise vs the source HF snapshot). This is the bridge that lets the
  NNX sampler read a MaxText-trained model.
* `src/maxtext/diffusion/eval_sasd/` — **B2, vendored** from `Fast-dLLM jax_ddrive @ 4b0f4f2`
  (`models/{rope,qwen2_5_text,vision_qwen25vl}`, `hf_to_jax`, `masks_eval`, `sampler_sasd`).
  The validated multimodal section-diffusion **inference** stack, self-contained for the
  internal TPU (`PYTHONPATH=src` only; **no ddrive_jax**; mirrors the `sasd_data` pattern).
  Changes vs source: import-line rewrites + **bf16-hardened tensor reads** (`_st_tensor_f32`),
  because the B1 snapshot is bf16 and safetensors' numpy framework can't decode bf16.
  **Do not edit logic here** — edit `jax_ddrive/ddrive_jax/...` and re-copy.
  Plus (fork-authored, not vendored): `eval_sasd/driver.py` (`run_eval`: generate + T2 scalar
  metrics + exportable `validation_log.jsonl`), `eval_sasd/embedding_parity.py` (`run_parity`:
  ViT fp32-vs-bf16-vs-reference embedding parity), `tests/eval_sasd_import_test.py`
  (self-containment guard). `eval_harness.py` now re-exports `run_eval` / `run_parity`.
* `scripts/prep_jax_eval_inputs.py` — **OFFLINE** (imports ddrive_jax; runs locally, never on
  the internal TPU): builds the inference-input npz (scaffold + rope + pixels + optional bf16
  `image_embeds` + GT `target_ids`) that the fork-only driver consumes. Mirrors the input half
  of `capture_oracle_sd_mm.py` (parity-validated `build_scaffold` / `get_rope_index_numpy`).

## How to validate

```bash
# data-path self-containment + checkpoint round-trip (CPU, seconds):
PYTHONPATH=src JAX_PLATFORMS=cpu python -c "from maxtext.input_pipeline.sasd_data import make_sasd_loader"
# B2 inference self-containment guard — MUST import zero ddrive_jax (CPU, seconds):
PYTHONPATH=src JAX_PLATFORMS=cpu python -m maxtext.diffusion.tests.eval_sasd_import_test
# diffusion unit/parity tests (GPU box with oracles on disk):
python -m pytest src/maxtext/diffusion/tests -x -q
# B1 round-trip identity (CPU): export the base param ckpt back to HF, expect 824/824 bitwise:
PYTHONPATH=$DDRIVE:src JAX_PLATFORMS=cpu python scripts/maxtext_to_hf_export.py \
  src/maxtext/configs/sasd_waymo.yml model_name=qwen2.5-3b \
  param_ckpt_dir=<base param ckpt> ref_snapshot=<base HF snapshot> \
  out_dir=/tmp/rt verify_against=<base HF snapshot>   # -> B1_ROUNDTRIP_PASS
# B2 fork-only generation (GPU): generate from a B1 snapshot using the oracle inputs:
PYTHONPATH=src python -m maxtext.diffusion.eval_sasd.driver \
  --npz <inputs.npz> --snapshot <B1 HF snapshot> --dtype fp32   # -> SASD_EVAL_PASS
# end-to-end TPU smoke (provision → train + ckpt@6 → resume-and-continue → auto-deletes):
bash /home/kaiwen/launch_maxtext_sasd_tpu_frombase.sh   # default ACCEL=v6e-1 (1-chip smoke; pod: v6e-8/v6e-16)
```
