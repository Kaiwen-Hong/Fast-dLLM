All flagged claims confirmed against source. The image-token contradiction (168 in driver / 56 in prep / 336 in config) is real and unreconciled. The gate list matches exactly. I have enough verified ground truth to assemble the document.

---

# Fast-dDrive Codebase — Ground-Truth Map

This document is the assembled ground-truth map of the Fast-dDrive port across three codebases, built from 8 per-subsystem source digests and spot-verified against live source. Fast-dDrive is a Qwen2.5-VL-3B model adapted into a **Section-Aware Section-Diffusion (SASD)** driving model (discrete masked diffusion over a JSON-structured driving answer, with a frozen ViT image tower).

---

## 1. THREE-CODEBASE RELATIONSHIP

There are **three** parallel implementations of the same model+algorithm, each with a distinct job. They are kept numerically aligned by parity gates.

### A. The PyTorch oracle — *the upstream reference / source of truth for numerics*
- The original `Efficient-Large-Model/Fast-dDrive` HF release (snapshot revision `0fda81009f4efa58a2debbb48c0c09818e45341f`), plus its `modeling.py`, `generation_utils.py`, `section_utils.py`.
- It is **not in either JAX repo** — every JAX/NNX file cites it by file:line as the thing it ports (e.g. `sasd_loss.py:3-11 → modeling.py:2104-2766`; `masks.py:1-6 → modeling.py:178-279`; `noise.py → modeling.py:2251-2346`; `mm_sampler` → `mdm_sample_deep_scaffold`; `scaffold.py` → `generation_utils.py:L134-202`).
- Runs in the PyTorch conda env `ddrive`. Used to **capture oracle tensors** that the JAX gates diff against. The prep scripts (`prep_jax_eval.py`, `prep_train_jax.py`, `prep_jax_eval_inputs.py`) run in this env too — they use the HF processor/tokenizer/`section_utils` but **load no model weights**.

### B. `ddrive_jax/` (Fast-dLLM repo) — *NNX reference / algorithm truth*
Path root: `/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/`.
- Hand-written Flax-**NNX** port of the full stack: Qwen2.5 text decoder, Qwen2.5-VL ViT, SASD loss/masks/noise, two samplers (text-only `sample_sd.py`, multimodal `mm_sampler.py`), HF→JAX weight loaders, data pipeline (Parquet/ArrayRecord + grain), and the **parity gate harness** (`run_all_verification.sh`).
- This is where the **algorithm is defined and verified bit-exact** against the PyTorch oracle (claimed parity: text logits 3.2e-5, multimodal 7.7e-5, scaffold loss 7.9e-8). Single-GPU/CPU-first.
- `train/train_tpu.py` here is a *Path-A FSDP driver* (CPU 8-device emulation as a multi-host proxy; real on TPU) — a proof-of-concept, not the production trainer.

### C. `maxtext-dlm-fork/` — *production / TPU*
Path root: `/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/`.
- A fork of Google MaxText (snapshot ~2026-05-05, root commit `35dce93`) with a SASD graft. The SASD **math** lives in `src/maxtext/diffusion/sasd.py` (re-derived bit-exact from `ddrive_jax`), wired into MaxText's native Linen `qwen2.5-3b` model via `objective=sasd`.
- The **inference stack** is vendored-verbatim from `ddrive_jax` into `src/maxtext/diffusion/eval_sasd/` (pinned at commit `4b0f4f2`); the data source `sasd_data/ar_dataset.py` is pinned at `b18e861`. These carry `DO NOT EDIT logic here` headers — edit `ddrive_jax` and re-copy (governed by `PATCHES.md`).
- Two weight bridges:
  - **Forward (B1 in)**: HF snapshot → MaxText Orbax PARAM ckpt (`scripts/save_fast_ddrive_params_ckpt.py` → `load_fast_ddrive_maxtext.build_maxtext_params_from_fast_ddrive`).
  - **Reverse (B1 out)**: MaxText Orbax ckpt → bf16 HF safetensors (`scripts/maxtext_to_hf_export.py`), so the same official Waymo metric scores both stacks.

### How they relate
- **Algorithm flows A → B → C.** `ddrive_jax` is the executable spec; MaxText-fork is the scaled deployment; PyTorch is the immovable numeric anchor.
- **Vendoring direction is B → C** for inference (`eval_sasd/*`, `ar_dataset.py`). The SASD training math in C (`sasd.py`) is an independent re-port that must stay in sync with B.
- **Verification is C/B → A**: gates capture PyTorch oracle tensors and diff JAX against them (`run_all_verification.sh` captures in the `ddrive` env, runs gates in the `jax` venv).
- **Eval symmetry**: both JAX stacks emit `predictions.json` in the PyTorch-eval schema so the *same* official Waymo metric scores all three.

---

## 2. SUBSYSTEM MAP (file → responsibility → key symbols)

### Subsystem 1 — models (architecture + RoPE/M-RoPE), `ddrive_jax/ddrive_jax/models/`
| File | Responsibility | Key symbols (file:line) |
|---|---|---|
| `rope.py` | Standalone 1D RoPE (SPLIT/rotate-half HF convention) for text-only path | `default_rope_params` (16-21); `apply_rope` SPLIT-form (27-32); `RoPE.__call__` fp32 `Precision.HIGHEST` sinusoid (40-47) |
| `qwen2_5_text.py` | Qwen2.5-3B text decoder (GQA, qkv bias, no qk-norm, tied embeds, SiLU MLP, RMSNorm) + 3D M-RoPE builder | `Qwen25TextConfig.fast_ddrive` (47-56); `RMSNorm` (59-68); `Linear` kernel `[in,out]` (71-80); `Qwen25Attention.__call__` GQA/fp32 softmax/rope-branch (99-122); `apply_rope_full` FULL-form (140-146); `mrope_cos_sin` host numpy float64 (149-161); `attend` tied (196-197); `hidden_forward_mrope_cs` (207-214); `forward_with_embeds` (229-247) |
| `vision_qwen25vl.py` | Qwen2.5-VL ViT tower (depth 32, 1280 hidden); frozen at train | `rot_pos_ids` (53-63); `get_window_index` pad -100 (66-86); `VisionAttention.__call__` 2D-rope+seg mask (147-156); `_rotary` (202-216); `VisionTransformer.__call__` window reorder/reverse (218-244); `PatchMerger` exact GELU (182-191) |
| `sharded.py` | Sharding-aware drop-ins (TP mesh); identical at mesh size 1 | `ShardedLinear` out_sharding (19-29); `ShardedEmbedding` `.at[].get`/`embedding.T` attend (32-38) |

### Subsystem 2 — diffusion, `ddrive_jax/ddrive_jax/diffusion/`
| File | Responsibility | Key symbols (file:line) |
|---|---|---|
| `sasd_loss.py` | The two SASD loss terms + section-weight vector + combiner | `_ce_per_token` fp32 safe-gather (21-27); `section_weighted_ce` MDM term, shift-1 (30-41); `causal_ce` (44-52); `section_weight_vector` (55-66); `sasd_total_loss` keys `{mdm,complementary,total}` (69-84) |
| `masks.py` | Hybrid block-causal masks (training [2n,2n] + inference [L,L]) | `hybrid_block_causal_mask_dense` (13-32); `to_attn_mask4d` (35-37); `eval_hybrid_block_causal_mask_dense` (40-51); `compute_response_block_idx_simple` non-deep fallback (54-83) |
| `noise.py` | Host-side per-step stochastic noising; doubled+complementary batch | `make_batch` (14-47, verified); `num_items` = `2*count(labels!=-100)` (50-51, verified) |
| `sample_sd.py` | Text-only section-diffusion sampler (no KV cache) | `block_ranges_from_rbi` (25-36); `section_diffusion_sample` causal-shift slice `s-1:e-1` (39-69); `main` parity loader (72-106) |

### Subsystem 3 — convert + data
| File | Responsibility | Key symbols (file:line) |
|---|---|---|
| `convert/hf_to_jax.py` | HF safetensors → NNX (streaming text, eager ViT); transpose conventions | `_load_all_tensors` bf16-safe (28); `_name_map_text` (57); `_set` (75); `load_fast_ddrive_text` streaming+tie-check (84); `load_fast_ddrive_vit` Conv3d→Linear (163,179) |
| `convert/prep_to_parquet.py` | npz → zstd Parquet, raw LE bytes + shape cols | `row_from_npz` (35); `parquet_schema` (47) |
| `data/parquet_dataset.py` | SSOT decode contract; `ARRAY_DTYPES`/`ARRAY_FIELDS` | `decode_row` `frombuffer.reshape.copy` (18) |
| `data/ar_dataset.py` | ArrayRecord random-access (v2 + image_embeds) | `decode_example` (51); `ArRecordSource` (76) |
| `data/grain_pipeline.py` | Multi-host infinite deterministic resumable loader | `make_sasd_loader` (241); `_fold_rng` SeedSequence (104); `_per_sample_arrays` (121); `_collate` (144) |
| `eval/prep_train_jax.py` | PyTorch-env SASD-training npz producer (no weights) | `process_gpt` (27); response-span detection (96-104) |
| `scripts/parquet_to_ar_with_embeds.py` | Dataset v2 AR + frozen-ViT embeds writer | `_st_tensor_f32` (62-77); `load_vit_streaming` (80-127); `make_jit_fwd` (172-199); `embeds_fp32` (201-206); `to_example` (138-144) |
| `scripts/parquet_file_to_tfexample_ar.py` | v1 per-file AR writer | (TF-on-CPU pin line 8) |

### Subsystem 4 — training + sharding + lora + checkpoint
| File | Responsibility | Key symbols (file:line) |
|---|---|---|
| `train_overfit.py` | Text-only single-GPU SASD overfit; LoRA-by-default | `loss_fn` (34); `train_step` `@nnx.jit` (44); `main` (67) |
| `train_overfit_mm.py` | Multimodal single-sample overfit; full-FT text, frozen ViT | `main` (27) |
| `train_waymo_sasd_jax.py` | Production single-GPU multi-sample loop + Orbax | `static_tensors` (37); `step` `@nnx.jit` (88); `save_ckpt` swallows exceptions (159) |
| `train/dist.py` | Distributed bootstrap; JAX-0.10 mesh-context shim | `init_distributed` (35) |
| `train/train_tpu.py` | Path-A FSDP driver (CPU-8 emulation / TPU) | `HarnessConfig` (53); `per_sample_loss_sums` (189); `make_train_step` shard_map (217); `build_harness` (301); `_shard_inputs` (368) |
| `train/checkpoint_mgr.py` | Orbax CheckpointManager 4-item composite | `save_step` (41); `restore_latest` (66) |
| `lora.py` | LoRA for text backbone | `LoRALinear` (24); `apply_lora` (42); `freeze_non_lora_params` (64) |
| `sharding.py` | PartitionSpec pytree (spec-only, no physical layers) | `fsdp_pspec` (22); `param_pspecs` embed→P() (31) |
| `checkpoint.py` | Simple single-host StandardCheckpointer | `save_state` (15); `restore_into` (22) |

### Subsystem 5 — eval/inference (NNX), `ddrive_jax/ddrive_jax/eval/` + `ddrive_jax/eval/`
| File | Responsibility | Key symbols (file:line) |
|---|---|---|
| `eval/mm_sampler.py` | Multimodal no-KV-cache section-diffusion sampler | `block_ranges_from_rbi` (27); `mm_section_diffusion_sample` (43, loop 75); `hid_fwd` scatter img_emb (61-65); `decode_generation` (95) |
| `eval/rope_index.py` | Pure-numpy B=1 `get_rope_index` → `[3,S]` M-RoPE | `get_rope_index_numpy` (14) |
| `eval/scaffold.py` | Deep-JSON masked scaffold + inference rbi | `messages_from_prompt` (19); `build_scaffold` (46) |
| `eval/prep_jax_eval.py` | PyTorch-env eval npz producer | (min=max=200704 lines 24-25) |
| `eval/jax_batch_inference.py` | JAX-env eval driver → predictions.json | `parse_trajectory` (20-28); bf16 default / `--fp32` (40-42) |
| `eval/prep_train_jax.py` | (cross-ref, SASD training prep) | `process_gpt` (27); response-span (96-104) |

### Subsystem 6 — gates + key scripts + tests, `ddrive_jax/scripts/` + `ddrive_jax/tests/`
| File | Responsibility | Key symbols (file:line) |
|---|---|---|
| `scripts/run_all_verification.sh` | Single-command 10-gate driver | `gate()` (15-19); `clean()` (14); sentinel `ALL_VERIFICATION_PASS` (57, verified) |
| `scripts/parquet_to_ar_with_embeds.py` | Dataset v2 builder (see Subsystem 3) | `_st_tensor_f32` (62-77); `make_jit_fwd` (172-199); `to_example` (138-144) |
| `scripts/parity_sasd.py` | Phase-2 gate (mask exact + loss <1e-3) | `main` (24-63) |
| `scripts/parity_text.py` | Phase-1 gate (logits rel-max <1e-3) | `relmax`/`rel_l2` (24-29) |
| `scripts/parity_mm.py` | Phase-4b gate (fused embeds + M-RoPE) | `main` `hidden_forward_mrope` (19-31) |
| `scripts/parity_vit.py` | Phase-4 gate (ViT iso + conv-as-linear) | `main` two-tier (18-63) |
| `tests/test_mask_loss.py` | CPU mask + loss regression | `ref_hybrid` numpy ground truth (17-30) |
| `tests/test_eval_ports.py` | CPU eval-port regression | `_imgs` (17-18) |
| `tests/test_harness_fsdp.py` | FSDP harness mechanics (CPU-8) | `shrink_vocab` (44-64); `_cfg`/`_one_batch` (73-94) |
| `tests/test_multihost_datafeed.py` | True 2-process multihost | `_worker` (48-88) |

### Subsystem 7 — MaxText fork core (SASD graft)
| File | Responsibility | Key symbols (file:line) |
|---|---|---|
| `diffusion/sasd.py` | SASD math (loss, noising, mask, M-RoPE, embed-doubling, host prep, global loss) | `make_batch` (65); `num_items` 2× (101); `hybrid_block_causal_mask_dense` (108); `mrope_cos_sin` float64 contiguous-chunk (146); `compute_fast_ddrive_image_embeds` (192); `_image_positions` (214); `prepare_sasd_inputs` flat=2b+r (233); `sasd_loss_from_logits` (314) |
| `diffusion/load_fast_ddrive_maxtext.py` | HF text → MaxText Linen tree (streaming) | `_st_tensor_f32` (58); `_StreamingTextGetter` (92); `build_maxtext_params_from_fast_ddrive` (119); `_hf_config_dict_3b` (113) |
| `diffusion/mdlm.py` | Separate MDLM objective (NOT SASD) | `forward_mask` (26); `mdlm_loss_inner` (44); `mdlm_loss` (82) |
| `diffusion/eval_sasd/sampler_sasd.py` | Vendored multimodal sampler | `mm_section_diffusion_sample` (46); `block_ranges_from_rbi` (30); `decode_generation` (98) |
| `diffusion/eval_sasd/masks_eval.py` | Vendored masks | `eval_hybrid_block_causal_mask_dense` (42); `hybrid_block_causal_mask_dense` (15) |
| `diffusion/eval_sasd/models/qwen2_5_text.py` | Vendored text decoder | `Qwen25TextConfig.fast_ddrive` (49); `Qwen25Attention.__call__` (101); `attend` (198) |
| `configs/sasd_waymo.yml` | Production SASD train config | `model_name=qwen2.5-3b` (18); `objective=sasd` (21); `sasd_mrope_section [16,24,24]` (22) |
| `configs/models/qwen2.5-3b.yml` | Model-shape config | dims at lines 20-35 |
| `configs/base.yml` | MaxText defaults | `objective` enum (376); `mask_token_id` (377); default `sasd_mrope_section [4,6,6]` (390) |
| `input_pipeline/sasd_data/ar_dataset.py` | Vendored AR source | `decode_example` (53); `ArRecordSource` (78) |
| `checkpoint_conversion/utils/param_mapping.py` | QWEN MaxText↔HF mapping + hooks | `reshape_kernel` `.T.reshape` (773/779); `pad_embedding_layer` identity (758) |

### Subsystem 8 — MaxText fork eval_sasd + export (B1/B2)
| File | Responsibility | Key symbols (file:line) |
|---|---|---|
| `diffusion/eval_sasd/driver.py` | Fork-only SASD inference driver + scalar metrics | `run_eval` (79-164); `_EmbedShim` (45-54); `_parse_trajectory` (57-60) |
| `diffusion/eval_sasd/embedding_parity.py` | Frozen-ViT embedding parity gate | `run_parity` (57-103); `_cmp` (50-54) |
| `diffusion/eval_sasd/sampler_sasd.py` | (vendored sampler, see Sub 7) | `mm_section_diffusion_sample` (46) |
| `diffusion/eval_sasd/hf_to_jax.py` | Vendored + bf16-hardened HF→NNX loaders | `_st_tensor_f32` (29-43); `load_fast_ddrive_text` (86-164); `load_fast_ddrive_vit` (167-203); `_name_map_text` (59-74) |
| `diffusion/eval_sasd/models/{qwen2_5_text,vision_qwen25vl,rope}.py` | Vendored model + RoPE | `mrope_cos_sin` (151-163); `hidden_forward_mrope_cs` (209-216); `get_window_index` (68-88) |
| `scripts/maxtext_to_hf_export.py` | B1 exporter (MaxText → bf16 HF) | `main` (134-281); `_st_tensor_native` (75-85) |
| `scripts/prep_jax_eval_inputs.py` | Offline eval npz builder (imports ddrive_jax) | `main` (41-113) |
| `scripts/save_fast_ddrive_params_ckpt.py` | Forward bridge HF→MaxText ckpt | `main` (38-81) |
| `PATCHES.md` | Fork patch/provenance/vendor pins | (vendor pins 10-16) |

---

## 3. GOLDEN READ ORDER (for a fresh coding agent)

Read these 7 files in this order. They establish, in dependency order, the invariants everything else assumes.

1. **`ddrive_jax/ddrive_jax/diffusion/noise.py`** — *Start here.* Defines the load-bearing data invariants: the **doubled `[2,2L]` sequence**, the **complementary (2nd) row**, **always-mask im_end**, **scaffold freeze**, the EPS mask-floor, and `num_items = 2×count(labels!=-100)`. Constants `MASK_ID/IM_END/EPS` are SSOT here. (Verified.)
2. **`ddrive_jax/ddrive_jax/diffusion/sasd_loss.py`** — the two CE terms (section-weighted MDM + clean causal), both with **shift-by-1**, both dividing by `num_items`. Defines `{mdm,complementary,total}`.
3. **`ddrive_jax/ddrive_jax/diffusion/masks.py`** — the **training** doubled `[2n,2n]` mask vs the **inference** `[L,L]` mask (different rules: block-diagonal vs block-causal).
4. **`ddrive_jax/ddrive_jax/models/qwen2_5_text.py`** — canonical text dims (`Qwen25TextConfig.fast_ddrive`), GQA + fp32 softmax attention, the **two RoPE conventions** (SPLIT vs FULL), and `mrope_cos_sin` (the contiguous-chunk M-RoPE). Open `rope.py` alongside.
5. **`ddrive_jax/ddrive_jax/models/vision_qwen25vl.py`** — ViT config + window-reorder/reverse orchestration; needed to understand frozen image embeds.
6. **`ddrive_jax/scripts/run_all_verification.sh`** — the executable index: which PyTorch oracle feeds which JAX gate and the exact PASS markers (the gate graph).
7. **`maxtext-dlm-fork/src/maxtext/diffusion/sasd.py`** — the production re-port: how `prepare_sasd_inputs` flattens `flat = 2*b + r`, and how `sasd_loss_from_logits` selects loss halves. Read `configs/sasd_waymo.yml` next to it.

Then branch by task: inference → `eval/mm_sampler.py` + `eval_sasd/sampler_sasd.py`; weights → `convert/hf_to_jax.py` + `param_mapping.py`; FSDP → `train/train_tpu.py` + `sharding.py`.

---

## 4. SUBTLE-BEHAVIORS CATALOG (gotchas) — deduped, with evidence

This is the crown jewel. Each item is something a porter/agent is likely to get wrong.

### Sequence & batch structure
- **DOUBLED SEQUENCE `[2,2L]`, not `[2,L]`.** Training input is `concat([noisy(L), clean_ids(L)])` → length **2L**; the leading axis 2 is *(main row, complementary row)*, not a normal batch. `noise.py:35,41,43` (verified); `masks.py:19`; `grain_pipeline.py:124`; `sasd.py:94`.
- **The "batch-of-2" rows have distinct semantics.** Row 0 = MDM/noisy (random Beta-mask). Row 1 = **complementary**: `comp = (resp & ~mask_indices) | (im_end & resp)`, then `& ~scaff` — masks exactly the response tokens row 0 did *not*. `noise.py:37-41` (verified).
- **Loss half-selection differs by row.** noisy half = `attend(hidden[:, :L])` over **both** rows; clean half = `attend(hidden[:1, L:])` over **row-0 (mdm) only**. `train_overfit.py:34`; `parity_sasd.py:50-52`; `sasd.py:330-335`. Easy to wrongly assume clean uses both rows.
- **`num_items = 2 × count(labels!=-100)`** — the factor 2 is the doubling normalizer; **both** CE terms divide by the *same* doubled denominator (NOT by masked/valid count). `noise.py:50-51` (verified); `sasd_loss.py:11`; `sasd.py:102`.
- **Row-flatten convention `flat = 2*b + r`** in MaxText prep: `input_final [B,2,2L]→[2B,2L]`, with cos/sin/mask/img repeated x2 along axis 0. `sasd.py:282-285,308-309`.
- **Global DP-correct normalizer**: per-term CEs are summed with `num_items=1.0` (raw sums), then `loss = (sec+cau)/max(sum_b(num_items),1.0)` — a single global denominator, not a per-sample mean. `train_tpu.py:244-249`; `sasd.py:333-338`.

### RoPE / M-RoPE
- **TWO incompatible RoPE helpers coexist.** SPLIT-form `apply_rope` takes HALF-width sin/cos `[B,T,Dh/2]` and splits x into halves (`rope.py:27-32`); FULL-form `apply_rope_full` takes FULL-width cos/sin `[L,Dh]` and uses rotate-half (`qwen2_5_text.py:140-146`). Attention branches on whether `mrope_cs is None` (`qwen2_5_text.py:107-111`). Assuming one helper is wrong.
- **M-RoPE cos/sin are FULL head_dim `[L,128]`, not halved** — built via `emb=concat([freqs,freqs],-1)`. `qwen2_5_text.py:155` (verified); ViT `_rotary:215`.
- **M-RoPE is CONTIGUOUS-CHUNK, NOT interleaved.** `sec = list(mrope_section)*2 = [16,24,24,16,24,24]`; section i picks channel `cos[i%3]` and concatenates contiguous slices → `[t..h..w..t..h..w]`. `qwen2_5_text.py:157-161` (verified); `sasd.py:138-164`. **This is explicitly different from MaxText's built-in interleaved `use_mrope`** → `use_mrope MUST be false`; cos/sin injected manually. `sasd_waymo.yml:38`.
- **`mrope_cos_sin` is HOST-side numpy float64, NOT jit-able.** The jit-able path is `hidden_forward_mrope_cs`, fed precomputed cos/sin as data. `qwen2_5_text.py:152-153` (verified) docstring 201-202.
- **`mrope_section` has TWO valid values.** Real model = `(16,24,24)` → `2*(16+24+24)=128=head_dim`. Proxy harness/tests = `(4,6,6)` → `2*16=32=proxy head_dim`. Both internally consistent. `train_tpu.py:432-433` forcibly resets to `(16,24,24)`; base.yml default `[4,6,6]` is a placeholder overridden by `sasd_waymo.yml:22`. Docs must not cite `(4,6,6)` as the real section.
- **RoPE precision trick**: sinusoid einsum forced to `Precision.HIGHEST`/fp32 to avoid bf16 rounding position 257→256. `rope.py:42-44`.
- **Text default path collapses to plain RoPE.** For text-only, all 3 M-RoPE sections share `arange` positions, so M-RoPE ≡ 1D RoPE — but the *code* uses two *different functions* (`RoPE`/`apply_rope` for text, `mrope_cos_sin`/`apply_rope_full` for MM); the collapse is numeric equivalence, not shared code. `rope.py:4` docstring vs `forward_with_embeds`.

### Causal shift / sampling
- **CAUSAL SHIFT-BY-1 everywhere, even though this is a diffusion model.** Both losses do `logits[:, :-1]` vs `labels[:, 1:]` (weights also shifted). `sasd_loss.py:34-36,47-48`.
- **Sampler shifted-logit slice is `s-1:e-1`, not `s:e`.** Position p is predicted from the logit/hidden at p-1: `model.attend(hidden[0, s-1:e-1])` fills positions `s..e`. `sample_sd.py:57`; `mm_sampler.py:80`; `sampler_sasd.py:83`. **The single most likely thing to misread as an off-by-one bug** — it's intentional.
- **Per-block sub-step budget = `n_mask + 5`** (not exactly n_mask), early break when no MASK remains, global cap `steps > max_tokens(512)`. `sample_sd.py:52,67-68`; `mm_sampler.py:75,90`; `sampler_sasd.py:78,93-94`.
- **Confidence fallback**: unmask all positions with softmax conf > threshold(0.9); if NONE exceed, unmask the single most-confident masked position (so the loop always progresses). Already-decoded positions forced to `-inf`. `sample_sd.py:61`; `mm_sampler.py:84-87`.
- **Sampler PASS gate is NOT token agreement.** Default bf16 perturbs low-confidence choices that cascade; `ok = valid_json and has_traj`, agree% is informational. `sample_sd.py:78-81,104`. Driver T2 uses structure-aware metrics (numeric trajectory match + CO/FMB equality), **not** positional token agreement (interleaved NULL/MASK scaffold vs flat GT don't line up). `driver.py:131-148`.

### Masks
- **Training mask ≠ inference mask.** Training (`hybrid_block_causal_mask_dense`) is **block-DIAGONAL** within same turn (`turn_q==turn_kv`) + offset-causal to clean keys; inference (`eval_*`) is **block-CAUSAL** (`bq>=bk`). `masks.py:29` vs `masks.py:50`. Do not conflate.
- **Inference: response queries see ALL prompt keys with no positional constraint**, but prompt→prompt is strictly causal. `masks.py:48-49`.
- **Mask is BOOLEAN (True=attend), converted to additive via `jnp.where(mask4d, logits, finfo(f32).min)`.** `to_attn_mask4d` only adds two leading singleton dims `[1,1,Q,K]`; it does NOT build an additive -inf mask. `masks.py:35-37`; `qwen2_5_text.py:117-119,25`.
- **`compute_response_block_idx_simple` is the NON-deep FALLBACK** (tests/data-prep), not production. Production rbi = `section_utils` deep-scaffold. `masks.py:54-55`.

### Weights / conversion
- **Linear kernel stored `[in,out]`, applied `x@kernel` with NO transpose.** A PyTorch converter (which stores `[out,in]`) MUST transpose. `qwen2_5_text.py:73,77`; `sharded.py:22-28`. (Verified config context.)
- **Biases/layernorms/embedding are NOT transposed; all `*_proj.weight` ARE.** `hf_to_jax.py:57-72`; `param_mapping.py:779,785`.
- **Embedding NOT transposed** (`nnx.Embed` stores `[vocab,d_model]` = same as PyTorch). `hf_to_jax.py:106`.
- **Tied embeddings → there is NO `lm_head`.** Logits via `embed_tokens.attend`. A stored `lm_head` is loaded ONLY if `untied AND not cfg.tie_embeddings`; with the fast_ddrive default (`tie_embeddings=True`) it's silently never used. `qwen2_5_text.py:196-197`; `hf_to_jax.py:120-153`. The B1 export OMITS lm_head. `maxtext_to_hf_export.py:16-17`. MaxText mapping lists `logits_dense→lm_head.weight` but it's filtered out (no `logits_dense` leaf). `load_fast_ddrive_maxtext.py:17-24`.
- **bf16 safetensors gotcha: safetensors numpy framework CANNOT decode bf16.** Every loader/exporter hand-rolls a raw-byte reader (`_st_tensor_f32`/`_st_tensor_native`): parse 8-byte LE header length + JSON header, seek `data_offsets`, `np.frombuffer` with `{F32,F16,BF16→ml_dtypes.bfloat16}`. The release Fast-dDrive snapshot is F32; the BASE Qwen2.5-VL snapshot is bf16. `hf_to_jax.py:29-43`; `parquet_to_ar_with_embeds.py:62-77`; `load_fast_ddrive_maxtext.py:58-74`; `maxtext_to_hf_export.py:63-85`. **`f.keys()` listing is bf16-safe; only decoding isn't.**
- **bf16 does NOT survive `np.savez`** — comes back as structured void `|V2`; detect `dtype.kind=='V'` and `raw.view(ml_dtypes.bfloat16)`. `driver.py:98-99`; `embedding_parity.py:71`.
- **Conv3d→Linear reshape for patch_embed**: `[out,in,T,H,W] → reshape(out,-1).T → [in_flat,out]`, `patch_dim=3*2*14*14=1176`. `hf_to_jax.py:177-179`; `param_mapping` transform `'conv'`.
- **ViT qkv is a FUSED `[3*hidden,hidden]` matrix** (one `.T`), unlike the text tower's separate q/k/v. `hf_to_jax.py:185`; `vision_qwen25vl.py:143`.
- **Streaming text loader is a deliberate OOM fix** (eager peaked ~25GB on a 30GB box); reads one tensor at a time + `gc.collect()`. `hf_to_jax.py:89-95`; `load_fast_ddrive_maxtext.py:77-89`.
- **ViT tensor-count assert is exact: `1 + 12*32 + 5 = 390`.** Wrong count aborts. `parquet_to_ar_with_embeds.py:126`.
- **Double-nested params on disk**: `save_params_to_path` wraps `{'params':params}` where params is itself `freeze({'params':tree})` → `{'params':{'params':tree}}`; exporter defensively unwraps. `save_fast_ddrive_params_ckpt.py:74-79`; `maxtext_to_hf_export.py:177-184`.
- **Export `--verify_against` is ONLY valid for the BASE round-trip (bitwise 824/824)**; a TRAINED ckpt's text weights changed → would spuriously FAIL → drop it for trained exports. `maxtext_to_hf_export.py:22-28,259-281`.
- **Frozen ViT 390 `visual.*` tensors copied VERBATIM (native dtype, byte-for-byte)** in export. `maxtext_to_hf_export.py:230-237`.
- **MASK_ID/NULL rows are real embedding rows, copied verbatim (no mean-init); `pad_embedding_layer` is identity** since source vocab==target vocab==151936. `load_fast_ddrive_maxtext.py:17-24,758-764`.

### ViT specifics
- **ViT used FROZEN via `stop_gradient` on OUTPUT embeds (not param freeze); the ViT module is then `del`-eted** to free memory. `train_waymo_sasd_jax.py:45`; `train_overfit_mm.py:48-50`.
- **Image embeds are FIXED across all denoising steps** (ViT runs once before the loop; re-scattered each step with the same precomputed values). `mm_sampler.py:55,64`.
- **Image fusion = scatter, not concat**: `embed_tokens(ids).at[img_pos].set(image_embeds)` overwrites rows where `ids==IMAGE_TOK(151655)`. Fusion happens **caller-side**, not in `models/`. `train_overfit_mm.py:60`; `sasd.py:215-220`.
- **Image embeds stored SINGLE-COPY `[N_img,D]`; consumer doubles via `concat([ie,ie],0)`** (single sample axis 0) / `axis=1` (batched). The two doc strings (`ar_dataset` axis 0, `grain_pipeline` axis 1) are NOT contradictory — same op at different ranks. Doubling is NOT done in the data files. `ar_dataset.py:6-8`; `grain_pipeline.py:7-9,166-167`; `parquet_to_ar_with_embeds.py:280-281`.
- **ViT window reorder is REVERSED at the end via `argsort(window_index)`** at `spatial_merge_unit=4` granularity. Forgetting the reverse scrambles patch order. `vision_qwen25vl.py:230,243-244`.
- **ViT full-attention blocks are exactly `(7,15,23,31)`**; all others use per-window seg mask. Padding sentinel `-100`. `vision_qwen25vl.py:36,78,239`.
- **PatchMerger uses EXACT GELU (`approximate=False`); text MLP uses SiLU.** Mixed activations. `vision_qwen25vl.py:191` vs `qwen2_5_text.py:132`.
- **ViT can't be jitted as-is** (mixes host numpy index/rope/seg-mask with device compute); the embeds builder jits ONLY the device part and cross-checks the jitted mirror vs eager per shard. jit cache keyed on `grid_thw.tobytes()`. `parquet_to_ar_with_embeds.py:167-171,172-206`.

### Numerics / precision policy
- **RMSNorm + all attention logits/softmax forced fp32** regardless of model dtype. `qwen2_5_text.py:66-68,115-120`; `vision_qwen25vl.py:111-113,152-154`; losses cast logits to fp32 `sasd_loss.py:23`.
- **`jax_default_matmul_precision='highest'` (disables TF32) in EVERY parity script + embeds builder.** TF32 would cause ~7.5e-4 systematic ViT elevation that the rolling-median<1e-4 guard catches. `parquet_to_ar_with_embeds.py:35,239-243`; all `parity_*.py`.
- **`_ce_per_token` uses `jnp.where(valid, labels, 0)` as a SAFE gather index** before `take_along_axis`, then zeroes NLL at ignored positions. `sasd_loss.py:25-27`.
- **`section_weighted_ce` denominator (when `num_items` None) is UNWEIGHTED `valid.sum()`**, not sum of weights — a deliberate match to PyTorch, not a weighted mean. `sasd_loss.py:38-41`.

### Noise schedule / weights provenance
- **Always-mask im_end (151645)** in response positions in BOTH the main and complementary rows, regardless of Beta draw. `noise.py:31,37` (verified).
- **Scaffold freeze**: scaffold positions never noised, never in the masked set (`& ~scaff` in both rows). `noise.py:30,38` (verified).
- **p_mask FLOOR via EPS=1e-3**: `p=(1-EPS)*Beta_draw+EPS`, never exactly 0. `noise.py:11,26`.
- **`block_alpha`/`block_beta` (per-block Beta params) and `weight_vec` (section weights) are PER-SAMPLE DATASET fields**, NOT global constants in code. SECTION_W `{CO 1.5, exp 1.0, FMB 2.0, traj 3.0}` and NOISE_SCHED Betas live only in **prep scripts** (`prep_train_jax.py:22-24`), baked into per-sample arrays. The diffusion/training modules consume them generically. `noise.py:25`; `sasd.py:76`; `ar_dataset.py:48`.

### Distributed / sharding
- **FSDP uses `jax.shard_map` over `'fsdp'`, NOT vmap-over-mesh** (vmap over a mesh-sharded axis is rejected by JAX-0.10 sharding-in-types). Params resharded to `P()` (replicated) at the boundary → all-gather; grads resharded back before optax. `train_tpu.py:218-227,259,265`.
- **`sharding.py` is SPEC-ONLY** — provides PartitionSpecs only; physical sharding runs in `train_tpu.py` via `shard_map`/`reshard`. Embedding ALWAYS replicated `P()` (vocab axis never sharded, else `attend()` needs out_sharding). `sharding.py:1-9,41`; gate asserts ≥4 large kernels sharded + embedding `P()` `test_harness_fsdp.py:231,235`.
- **`fsdp_pspec` shards the SINGLE largest axis only** (argmax of shape); opt-state embedding-shaped leaf force-replicated by exact-shape match. `sharding.py:26-28`; `train_tpu.py:293-298`.
- **THREE checkpoint systems coexist**: `checkpoint.py` (StandardCheckpointer, full `nnx.state` incl. frozen Variables, single-host, in-place); `train/checkpoint_mgr.py` (CheckpointManager 4-item composite params/opt/meta/grain, sharded-abstract restore, multi-host); and `train_waymo_sasd_jax.py:159`'s own inline `save_ckpt` (no opt_state, swallows exceptions) — a third path.
- **Multihost fix**: OLD `jax.device_put(host_local_batch, global_sharding)` is WRONG on a true 2-VM slice; use `jax.make_array_from_process_local_data` (process p → global rows `[p*B:(p+1)*B]`). `test_multihost_datafeed.py:5-8,65`; `train_tpu.py:368-390`.
- **Sharding is on the SHUFFLED GLOBAL stream**: `shuffle → repeat → map_with_index → slice(process_index,None,process_count)`; host h gets `{h, h+H, ...}`; `global_step = gidx // process_count`. Deterministic noise keys on both step & gidx via `SeedSequence` spawn_key → resume replays identical noise from grain's `next_index` alone. `grain_pipeline.py:104-117,290-305`.
- **Ambiguous-source guard**: if BOTH `.arrayrecord` and `.parquet` shards exist, `make_sasd_loader` raises (no silent precedence). `grain_pipeline.py:251-252`.
- **Uniformity is PROBED, not scanned**: probes ≤16 linspaced indices and asserts single L/pixel-N/n-img; padding path is `NotImplementedError`. Non-uniform data silently relies on the 16-sample check. `grain_pipeline.py:264-288`.

### LoRA
- **Default trainability differs across scripts**: `train_overfit.py` defaults to LoRA (rank 16) unless `--full_ft`; `train_overfit_mm.py` and `train_waymo_sasd_jax.py` are FULL-FT text (no LoRA). Don't assume all use LoRA. `lora.py:18`; `train_overfit.py:75`.
- **`freeze_non_lora_params` destructively converts Param→Variable**; `lora_B` inits to ZEROS (initial LoRA delta = 0). LoRA wraps only modules whose attribute name is in `target_modules` AND have `.kernel`. `lora.py:30-31,54,80`.
- **`train_step` has `donate_argnums=(0,1)` and DELETES input params** — tests must snapshot before and reassign from the return. `test_harness_fsdp.py:106-113`.

### Misc carried-but-unused / dead paths
- **`vision_mask` is in the stored schema but NEVER consumed by grain training** — carried, then dropped from the batch dict. `prep_train_jax.py:117`; `grain_pipeline.py` omits it.
- **`PROXY_IMAGE_SEED=1234` is fixed & model-seed-independent** → proxy frozen embeds depend only on `step`, preserving ckpt-resume loss continuity across seeds. `train_tpu.py:157-159`.
- **`remat` (per-layer gradient checkpointing) applied only in TRAIN forward (`remat=True`), NOT eval.** `qwen2_5_text.py:211`; `train_waymo` eval line 116 omits it.
- **`compute_response_block_idx_simple` turn increments when rbi changes; prompt rbi=-1.** `masks.py:64-82`.
- **Test no-op**: `assert M[2,3] and not M[4,2] or True` always passes (`... or True` short-circuits) — only line-53 assertions constrain block-causality. `test_mask_loss.py:52`.
- **MDLM is a SEPARATE objective, not SASD**: token-normalized `(1/clip(t,1e-3))`-weighted CE, denominator `sum(loss_valid)`. base.yml objective enum `ar|mdlm|sasd`. `mdlm.py:69,78`; `base.yml:376`.

---

## 5. RAW NUMBERS INVENTORY (unreconciled — code says)

### Text config (Qwen25TextConfig.fast_ddrive) — verified
| Quantity | Value | Source |
|---|---|---|
| d_model | 2048 | `qwen2_5_text.py:31,51` |
| n_heads | 16 | `qwen2_5_text.py:32,51` |
| n_kv_heads | 2 | `qwen2_5_text.py:33,51` |
| head_dim | 128 | `qwen2_5_text.py:34,51` |
| n_layers | 36 | `qwen2_5_text.py:35,52`; `_hf_config_dict_3b` 36 `load_fast_ddrive_maxtext.py:116`; export hf_cfg 36 `maxtext_to_hf_export.py:200` |
| mlp_hidden_size | 11008 | `qwen2_5_text.py:36,52` |
| vocab_size | 151936 | `qwen2_5_text.py:37,52` |
| max_sequence_length | 128000 | `qwen2_5_text.py:38` |
| rms_norm_eps | 1e-6 | `qwen2_5_text.py:39,53` |
| rope_theta | 1_000_000.0 | `qwen2_5_text.py:40,53`; `rope.py:16` |
| include_qkv_bias | True | `qwen2_5_text.py:41,54` |
| include_bias (o_proj/mlp) | False | `qwen2_5_text.py:42,54` |
| use_q_k_norm | False | `qwen2_5_text.py:43,54` |
| tie_embeddings | True | `qwen2_5_text.py:44,55` |
| n_rep (GQA repeat) | 8 = 16//2 | `qwen2_5_text.py:87` |
| q_dim / kv_dim | 2048 / 256 | `qwen2_5_text.py:88` |
| attention scale | head_dim**-0.5 (fp32) | `qwen2_5_text.py:115` |
| LARGE_NEGATIVE | finfo(f32).min | `qwen2_5_text.py:25` |
| default kernel init | lecun_normal | `qwen2_5_text.py:73` |
| dtype default | jnp.float32 | `qwen2_5_text.py:30` |

MaxText echoes (`qwen2.5-3b.yml`): base_emb_dim 2048 (20), base_num_query_heads 16 (21), base_num_kv_heads 2 (22), base_mlp_dim 11008 (23), base_num_decoder_layers 36 (24), head_dim 128 (25), mlp_activations `[silu,linear]` (26), vocab_size 151936 (27), decoder_block qwen2 (28), eps 1e-6 (29), rope_max_timescale 1e6 (30), use_qk_norm False (31), attention_bias True (32), logits_via_embedding True (33), normalize_embedding_logits False (34/35).

### Vision config (VisionConfig)
| Quantity | Value | Source |
|---|---|---|
| depth | 32 | `vision_qwen25vl.py:28` |
| hidden_size | 1280 | `vision_qwen25vl.py:29` |
| num_heads | 16 | `vision_qwen25vl.py:30` |
| head_dim | 80 = 1280/16 | `vision_qwen25vl.py:42-44` |
| in_channels | 3 | `vision_qwen25vl.py:31` |
| patch_size | 14 | `vision_qwen25vl.py:32` |
| temporal_patch_size | 2 | `vision_qwen25vl.py:33` |
| spatial_merge_size | 2 | `vision_qwen25vl.py:34` |
| spatial_merge_unit | 4 = 2**2 | `vision_qwen25vl.py:47-48` |
| window_size | 112 | `vision_qwen25vl.py:35` |
| window (merged units) | 4 = 112//2//14 | `vision_qwen25vl.py:69` |
| fullatt_block_indexes | (7,15,23,31) | `vision_qwen25vl.py:36,239` |
| out_hidden_size | 2048 | `vision_qwen25vl.py:37` |
| intermediate_size | 3420 | `vision_qwen25vl.py:38` |
| rms_norm_eps | 1e-6 | `vision_qwen25vl.py:39` |
| rope_theta (ViT) | 10000.0 | `vision_qwen25vl.py:40` |
| attention scale | 80**-0.5 | `vision_qwen25vl.py:145` |
| patch_dim | 1176 = 3*2*14*14 | `vision_qwen25vl.py:197` |
| inv_freq dim d | 40 = head_dim//2 (len 20) | `vision_qwen25vl.py:202-203` |
| PatchMerger hidden | 5120 = 1280*4 | `vision_qwen25vl.py:184` |
| qkv fused out dim | 3840 = 1280*3 | `vision_qwen25vl.py:143` |
| padding sentinel | -100 | `vision_qwen25vl.py:78` |
| ViT tensor count | 390 = 1+12*32+5 | `parquet_to_ar_with_embeds.py:126` |

### Diffusion / loss
| Quantity | Value | Source |
|---|---|---|
| MASK_ID | 151665 | `noise.py:11`; `sample_sd.py:40`; `scaffold.py:14` |
| IM_END | 151645 | `noise.py:11`; `prep_train_jax.py:20` |
| EPS (mask floor) | 1e-3 | `noise.py:11` (verified) |
| ignore_index | -100 | `sasd_loss.py:21`; `noise.py:20` |
| causal shift | 1 | `sasd_loss.py:34-36,47-48`; sampler `s-1:e-1` |
| **num_items** | **2 × count(labels!=-100)** | `noise.py:51` (verified); `sasd_loss.py:11`; `sasd.py:102` — *factor 2 = doubling; same denom for both terms* |
| confidence threshold | 0.9 | `sample_sd.py:40`; `mm_sampler.py:44`; `sampler_sasd.py:47`; `driver.py:80` |
| max_tokens (step cap) | 512 | `sample_sd.py:40`; `mm_sampler.py:44`; `sampler_sasd.py:47` |
| inner-loop iters | n_mask + 5 | `sample_sd.py:52`; `mm_sampler.py:75`; `sampler_sasd.py:78` |
| NULL_ID (dropped in decode) | 151666 | `sample_sd.py:91`; `mm_sampler.py:24`; `scaffold.py:15` |
| _TIME_EPS (MDLM) | 1e-3 | `mdlm.py:23` |
| bd_size / BD | 32 | `prep_overfit_data_mm.py:13`; `grain_pipeline.py:242`; `prep_train_jax.py:20` |
| EXP_BUDGET | 32 | `prep_train_jax.py:20` |

Section weights (SECTION_W) — **data-prep only**: critical_objects 1.5, explanation 1.0, future_meta_behavior 2.0, trajectory 3.0 `prep_train_jax.py:22`. NOISE_SCHED Beta(α,β): CO (1.0,2.0), exp (1.0,1.0), FMB (1.0,1.5), traj (2.0,1.0) `prep_train_jax.py:23-24`. *Not hardcoded in diffusion/training modules.*

### Tokens / IDs (consistent across repos)
| Token | ID | Source |
|---|---|---|
| IMAGE_TOK / image_pad | 151655 | `prep_train_jax.py:21`; `mm_sampler.py:23`; `rope_index.py:19`; `sasd.py:189` |
| MASK_ID | 151665 | (see diffusion) |
| NULL_ID | 151666 | (see diffusion) |
| IM_END | 151645 | (see diffusion) |
| IM_START | 151644 | `verify_ar_round2.py:63`; assistant triple `prep_train_jax.py:97` |
| VSTART (vision_start) | 151652 | `rope_index.py:21`; `prep_train_jax.py:21` |
| VEND | 151653 | `verify_ar_round2.py:64` |
| VPAD / vision_end (in vmask) | 151654 | `verify_ar_round2.py:64`; `prep_train_jax.py:117` |
| video_token_id | 151656 | `rope_index.py:20` |
| assistant-header triple | (151644, 77091, 198) | `prep_train_jax.py:97` |

### Sequence lengths (single vs doubled — DO NOT reconcile)
| Quantity | Value | Source / note |
|---|---|---|
| Uniform L (text seq) | 1184 | `grain_pipeline.py:42,267`; `sasd_waymo.yml:25` |
| **2L (doubled)** | **2368** | `sasd_waymo.yml:24` comment — *L=1184 doubled* |
| input_final | [2, 2L] | `noise.py:43` (verified) |
| labels_final / weights | [2, L] | `noise.py:44,46` (verified) |
| original_labels | [1, L] | `noise.py:45` (verified) |
| position_ids (M-RoPE) | [3, L] | `prep_train_jax.py:116`; doubled to [3,2L] caller-side |
| sasd_seq_len (config) | 1184 | `sasd_waymo.yml:25` (verified) |
| max_target_length | 2376 (≥ 2L) | `sasd_waymo.yml:63` (verified) |
| training mask | [2n, 2n] | `masks.py:19` |
| inference mask | [L, L] | `masks.py:40` |

### Image tokens (THREE code values — UNRECONCILED, verified)
| Quantity | Value | Source |
|---|---|---|
| Uniform pixel N (patches) | 672 | `grain_pipeline.py:267`; `parquet_to_ar_with_embeds.py:6` |
| image_embeds N_img_tokens (after /4 merge) | 168 = 672/4 | `parquet_to_ar_with_embeds.py:6`; image_embeds shape `[168,2048]` |
| image_embeds D | 2048 | `parquet_to_ar_with_embeds.py:206` |
| 2N (doubled patches) | 672 (so N=336) | `sasd_waymo.yml:24-25` comment |
| **sasd_num_image_tokens (config)** | **336** | `sasd_waymo.yml:26` (verified) |
| driver diagnostic | "**expect 168** at train res" | `driver.py:128` (verified) |
| prep comment | "~64 merged tokens/img → grid [1,16,14] = **56**/img" | `prep_jax_eval_inputs.py:52` (verified) |
| train_overfit_mm single-image tokens | 1365 (×2 = 2730) | `train_overfit_mm.py:42,48` |

> **Note**: `n_image_tokens` is a reported diagnostic, never asserted in code. The numbers disagree: 168 = 3 cameras × 56; 336 = 2×168 (the doubled count). `parquet_to_ar_with_embeds.py` computes N per record from `grid_thw` (line 179) — these are grid-dependent, not model constants. **Flagged for audit; see §7.**

### Image resolution policy (eval ≠ train — verified)
| Path | min_pixels / max_pixels | Source |
|---|---|---|
| Train prep | 784 / 50176 (=784*64) | `prep_train_jax.py:56-57`; `prep_jax_eval_inputs.py:53-54` (verified) |
| Eval prep (NNX) | 200704 / 200704 (fixed) | `prep_jax_eval.py:24-25` |
| Eval prep (fork, paper) | 200704 / 200704 | `prep_jax_eval_inputs.py:52` (verified) |

### M-RoPE
| Quantity | Value | Source |
|---|---|---|
| mrope_section (real) | (16,24,24) | `qwen2_5_text.py:200`; callers `train_*`, `sample_sd.py:84`, `mm_sampler.py:50`, `parity_mm.py:29`; `sasd_waymo.yml:22` (verified `[16,24,24]`) |
| mrope_section (proxy) | (4,6,6) | `train_tpu.py:68`; `test_harness_fsdp.py:78`; `base.yml:390` default |
| mrope sum*2 = head_dim | 2*(16+24+24)=128 | `qwen2_5_text.py:157-161` (verified) |
| mrope internal dtype | float64 | `qwen2_5_text.py:152` (verified) |

### Thresholds / gate tolerances
| Gate | Threshold | Source |
|---|---|---|
| phase1 text | logits rel-max < 1e-3 | `parity_text.py:54` |
| phase2 sasd | total rel-diff < 1e-3 AND mask exact | `parity_sasd.py:61-62` |
| phase4b mm | logits rel-max < 1e-3 | `parity_mm.py:38` |
| phase4 vit | iso<1e-3 AND pe_rel<5e-3, OR fallback end-to-end<1e-2 | `parity_vit.py:59-60` |
| patch_embed conv-as-linear | seed ~6.5e-4; >5e-3 ⇒ bug | `parity_vit.py:44,58` |
| tie-check threshold / chunk | 1e-5 / 8192 rows | `hf_to_jax.py:150-151,143` |
| embeds jit-vs-eager per-sample | 1e-3 | `parquet_to_ar_with_embeds.py:238` |
| embeds rolling-median | 1e-4 over ≥8 shards | `parquet_to_ar_with_embeds.py:241` |
| embedding_parity cosine | ≥ 0.999 | `embedding_parity.py:19,57,112` |
| embedding_parity max_rel | 5e-2 | `embedding_parity.py:19` |
| FSDP parity tol | |l_fsdp - l_1dev| < 1e-4; init diff < 1e-6 | `test_harness_fsdp.py:120-122` |
| ckpt-resume | params < 1e-6; cont_loss < 1e-5 | `test_harness_fsdp.py:197,208` |
| multihost global-loss agreement | < 1e-6 | `test_multihost_datafeed.py:115` |
| round-trip export gate | bitwise 824/824 | `maxtext_to_hf_export.py:281` |

### Training hyperparams
| Param | Value | Source |
|---|---|---|
| LoRA rank / alpha / scale | 16 / 32.0 / 2.0 | `lora.py:18,19,32` |
| LoRA target_modules | q/k/v/o_proj, gate/up/down_proj | `lora.py:20-21` |
| Adafactor min_dim_to_factor | 128 | `train_overfit.py:113` (+ others) |
| Adafactor multiply_by_param_scale | True | `train_overfit.py:112` |
| clip_by_global_norm | 1.0 | `train_overfit.py:111`; `sasd_waymo.yml:69` |
| warmup / final lr | max(steps//10,1) / lr*0.1 | `train_overfit.py:109-110` |
| train_overfit steps/lr | 150 / 1e-4 | `train_overfit.py:71,72` |
| train_overfit_mm steps/lr | 80 / 5e-5 | `train_overfit_mm.py:29,30` |
| train_waymo samples/steps/lr/save | 200 / 400 / 2e-5 / 200 | `train_waymo_sasd_jax.py:55-59` |
| PASS threshold (mm/waymo) | final < init - 0.05 | `train_overfit_mm.py:102-103` |
| PROXY_IMAGE_SEED | 1234 | `train_tpu.py:157` |
| HarnessConfig proxy dims | d=128,h=4,kv=2,hd=32,layers=2,mlp=256 | `train_tpu.py:59-64` |
| HarnessConfig n_fsdp/n_tp | 8 / 1 | `train_tpu.py:71,72` |
| HarnessConfig opt/lr/clip | adamw / 1e-3 / 1.0 | `train_tpu.py:73,74,75` |
| HarnessConfig warmup/total | 4 / 40 | `train_tpu.py:76,77` |
| CPU emulation device count | 8 | `train_tpu.py:24` |
| DP loss formula | g_num / max(g_den, 1.0) | `train_tpu.py:249` |
| sasd_waymo opt/lr | adafactor / 1e-4 | `sasd_waymo.yml:67,68` (verified) |
| sasd_waymo per_device_batch | 1.0 | `sasd_waymo.yml:62` |
| sasd_waymo dtypes | bfloat16 (dtype/weight/grad) | `sasd_waymo.yml:57-59` |
| sasd_waymo attention/type | dot_product / full | `sasd_waymo.yml:36,37` |
| sasd_waymo use_mrope/scan/remat | false / false / full | `sasd_waymo.yml:38,39,40` |
| run_all MEM_FRACTION / PREALLOCATE | .9 / false | `run_all_verification.sh:11` |
| phase3_lora steps/lr | 30 / 1e-4 | `run_all_verification.sh:46` |

### Data pipeline
| Param | Value | Source |
|---|---|---|
| ARRAY_FIELDS count | 12 | `parquet_to_ar_with_embeds.py:52-54` |
| DTYPES | input_ids/labels int64; rbi/turn/position_ids int32; scaffold/vision_mask bool; weight_vec/block_alpha/block_beta float32; pixel_values float16; image_grid_thw int64; image_embeds bfloat16 | `parquet_to_ar_with_embeds.py:55-59`; `ar_dataset.py:44-50` |
| default shard_size (parquet) | 64 | `prep_to_parquet.py:61` |
| AR/parquet iter batch_size | 64 | `parquet_to_ar_with_embeds.py:220` |
| verify_every | 256 | `parquet_to_ar_with_embeds.py:153` |
| ArrayRecordWriter group_size | 1 | `parquet_to_ar_with_embeds.py:218` |
| grain ReadOptions threads/prefetch | 8 / 64 | `grain_pipeline.py:310` |
| LABEL_PAD / RBI_PAD | -100 / -1 | `grain_pipeline.py:66,67` |
| Dataset sizes | 50k and 415k | `grain_pipeline.py:267`; `ar_dataset.py:15` |
| export SHARD_BYTES | ~5 GB | `maxtext_to_hf_export.py:52` |
| export key counts | 824 = 434 text + 390 visual (lm_head omitted) | `maxtext_to_hf_export.py:5,11,12,14`; `PATCHES.md:73-76` |
| ckpt size referenced | 16GB | `run_all_verification.sh:3` |

### Snapshot / provenance
- HF snapshot revision: `0fda81009f4efa58a2debbb48c0c09818e45341f` (Efficient-Large-Model/Fast-dDrive) — `sample_sd.py:21-22`; `prep_jax_eval.py:13`; `save_fast_ddrive_params_ckpt.py:35`; `sasd_waymo.yml:32`.
- Vendor pins: `sasd_data @ b18e861`, `eval_sasd @ 4b0f4f2`; MaxText root `35dce93`, snapshot ~2026-05-05 — `PATCHES.md:10-16`.
- Claimed parity: Phase-1 logits 3.2e-5; Phase-4b multimodal 7.7e-5; scaffold loss 7.9e-8 — `mm_sampler.py:12-13`; `scaffold.py:6` (docstring claims, not measured here).

---

## 6. THE GATES (run_all_verification.sh) — verified against source

**True count: 10 gates** (sentinel: `ALL_VERIFICATION_PASS` if `$ok == $n`, else `SOME_VERIFICATION_FAILED`; `run_all_verification.sh:55-57`).

| # | Gate name | Pass marker | Command | Line |
|---|---|---|---|---|
| 1 | cpu_mask_loss | `ALL CPU TESTS PASS` | `tests/test_mask_loss.py` (CPU) | 22 |
| 2 | cpu_lora | `ALL LORA TESTS PASS` | `tests/test_lora.py` (CPU) | 23 |
| 3 | cpu_noise | `ALL NOISING TESTS PASS` | `tests/test_noising.py` (CPU) | 24 |
| 4 | cpu_sharding | `ALL SHARDING TESTS PASS` | `tests/test_sharding.py` (CPU) | 25 |
| 5 | cpu_eval_ports | `ALL EVAL PORT TESTS PASS` | `tests/test_eval_ports.py` (CPU) | 26 |
| 6 | phase1_text | `PHASE1_PARITY_PASS` | `scripts/parity_text.py` | 30 |
| 7 | phase2_sasd | `PHASE2_PARITY_PASS` | `scripts/parity_sasd.py` | 33 |
| 8 | phase4_vit | `PHASE4_VIT_PASS` | `scripts/parity_vit.py` | 37 |
| 9 | phase4b_mm_fwd | `PHASE4b_MM_PASS` | `scripts/parity_mm.py` | 40 |
| 10 | phase3_lora_train | `PHASE3_PASS` | `train_overfit.py --source trained --fixed_batch --steps 30 --lr 1e-4` | 46 |

Final summary loop iterates exactly these 10 keys (`run_all_verification.sh:51`).

Notes:
- **`phase3_lora_train` is documented as flaky** in the back-to-back suite (LoRA on the near-optimal trained ckpt moves loss only ~0.001 over 30 steps); passes reliably standalone. `run_all_verification.sh:42-46`.
- **`test_harness_fsdp.py` and `test_multihost_datafeed.py` are NOT invoked** by `run_all_verification.sh` — they are standalone (need CPU device-count / multi-process env).
- `clean()` strips noise lines (`external/`, `cuda_`, `warn`, `rope_param`, `layer_type`, `allocat`, `fragment`, `hlo_remat`, …) before marker matching. `run_all_verification.sh:14`.
- Capture step runs the PyTorch oracle in the `ddrive` conda env; gates run in the `jax` venv.

---

## 7. DRIFT SUSPICIONS (code-vs-doc contradictions + 404s)

1. **404 path: `diffusion/sampler_sasd.py` does NOT exist** at top-level. There are `diffusion/sampler.py` (generic) and `diffusion/eval_sasd/sampler_sasd.py` (the real SASD one). Any doc/path referencing `diffusion/sampler_sasd.py` is stale.

2. **Image-token count is internally contradictory and UNENFORCED** (verified live): `driver.py:128` says "expect **168** at train res"; `prep_jax_eval_inputs.py:52` says "**56**/img"; `sasd_waymo.yml:26` sets `sasd_num_image_tokens: 336`. `n_image_tokens` is a diagnostic, never asserted. Reconciliation hypothesis (not code-stated): 168 = 3 cameras × 56; 336 = 2×168 (doubled). The script computes N from `grid_thw` per record, so 168/672 are grid-dependent, NOT model constants. **Audit needed.**

3. **Eval vs train resolution mismatch** (verified): train uses 784/50176 → ~56–168 tokens; eval prep pins 200704/200704. MEMORY note ("prep resolution must match training 784/50176→168 tokens") conflicts with `prep_jax_eval.py:24-25` (200704). A real inconsistency — eval npz built at 200704 won't line up with a model trained at 784/50176.

4. **No YAML/JSON config in `ddrive_jax/`** — all model constants are inline dataclass defaults (`Qwen25TextConfig`, `VisionConfig`). Any doc claiming a `models/*.yaml` SSOT is drift. (The only YAMLs are in the MaxText fork.)

5. **No `train_waymo_sasd.sh` exists** in `jax_ddrive`. The "Canonical recipe (train_waymo_sasd.sh): section weights … bd_size 32" is a docstring claim (`train_waymo_sasd_jax.py:7-9`) with no corresponding script.

6. **`(4,6,6)` mrope_section must not be cited as real** — it's the proxy (head_dim 32). Real = `(16,24,24)` (head_dim 128). `base.yml:390` default `[4,6,6]` is a placeholder overridden by `sasd_waymo.yml:22`.

7. **base.yml default `sasd_mrope_section [4,6,6]` is internally invalid for 3B** (2*16=32≠128) — it's a placeholder; only valid after `sasd_waymo.yml` override.

8. **`rbi`/`position_ids` dtype docstring drift**: `driver.py:11` docstring says rbi/position_ids int32, but `prep_jax_eval_inputs.py:88-89` writes both as int64. Producer wins; harmless (downstream casts).

9. **`test_harness_fsdp.py` header says mesh `(8,1)` FSDP** but the parity code under test uses `n_fsdp=2` → mesh `(2,1)` (`loss_at(2)`, `test_harness_fsdp.py:116`). Docstring lines 4-14 contradict the code.

10. **Parity numbers (3.2e-5 / 7.7e-5 / 7.9e-8) are source-comment CLAIMS**, not measured in these files (`mm_sampler.py:12-13`; `scaffold.py:6`; `sampler_sasd.py:12-13`).

11. **`vision_mask` is a carried-but-unused schema field** — present in prep/parquet/AR, never read by grain training. A doc implying it gates vision tokens at train time is wrong.

12. **Section weights / Beta schedules are NOT in the diffusion or training module code** — they're baked into per-sample `weight_vec`/`block_alpha`/`block_beta` at data prep. A doc stating the diffusion module hardcodes `{CO 1.5, exp 1.0, FMB 2.0, traj 3.0}` is drift (those literals live in `prep_train_jax.py:22-24` / `prep_overfit_data*.py`).

13. **`ARRAY_DTYPES`/`ARRAY_FIELDS` are duplicated** (hardcoded) in `parquet_to_ar_with_embeds.py:52-59`, `parquet_file_to_tfexample_ar.py:19-21`, `ar_dataset.py:41-48` instead of importing from `prep_to_parquet` — drift risk if the SSOT changes.

14. **`MASK_ID` has a second source of truth**: defined in `noise.py:11` (canonical) and re-imported by grain, but `prep_train_jax.py:20` independently hardcodes `151665` (consistent today).

15. **`checkpoint.py:25` uses `ocp.utils.to_shape_dtype_struct`** — possibly a deprecated/moved Orbax API; worth verifying against the installed Orbax version.

16. **Export `--verify_against` only valid for BASE round-trip**; using it on a TRAINED export spuriously FAILs (text weights changed). `maxtext_to_hf_export.py:22-28`.

17. **hf_to_jax header comment** mentions `<|NULL|>=151666 already trained` but 151666 is never used in those files (prep uses the literal string `<|NULL|>`, not the id). `hf_to_jax.py:9`.

---

**Files of record (all absolute):**
- NNX reference: `/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/ddrive_jax/` (models, diffusion, eval, convert, data, train, lora.py, sharding.py, checkpoint.py) + `/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/{scripts,tests,eval}/`
- MaxText fork: `/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/src/maxtext/{diffusion,configs,input_pipeline,checkpoint_conversion}/` + `/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/scripts/` + `/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/PATCHES.md`
- PyTorch oracle: external HF snapshot `0fda81009f4efa58a2debbb48c0c09818e45341f` (not in either repo; cited by file:line throughout).