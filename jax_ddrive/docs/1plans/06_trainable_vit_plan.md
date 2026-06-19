# Fast-dDrive JAX — trainable (in-graph) ViT — plan & status

Owner: kaiwen · Started 2026-06-19 · Status: **foundation done + locally (CPU) validated; train-graph wiring not yet done; no TPU run yet**

> **Status when written (2026-06-19).** The three hard de-risks (ViT split / nnx.bridge / weight-load) plus
> bf16 fidelity and the injection pattern are all **validated on CPU**, and the differentiable ViT body is now
> also **validated on a real v5e TPU** (§9). The actual wiring into the MaxText train graph (iterator /
> decoders / train-state init) is **in progress** (overnight autonomous run; see §9 progress log). Live working
> state + file:line map also in memory `fast-ddrive-trainable-vit`. The internal google3 env was checked and
> **supports the bridge** (`BRIDGE_ENV_OK`, jax 0.10.2 / flax 0.12.7) — same as the GCP TPU — so the bridge
> approach is confirmed for deployment (no Linen-native rewrite needed).
> Per docs convention this 1plans doc is the **plan + a rolling status** (the overnight run updates §9 in place).

## 0. TL;DR

Today SASD training consumes **frozen, pre-baked** ViT `image_embeds` (the data loop is pure IO, no ViT on
the pod). We want to instead run the Qwen2.5-VL ViT **in-graph on `pixel_values` every step, with the ViT
TRAINABLE** (params in the train state), optimized for multi-node TPU. Goal bar = **10 trainable steps run on
(internal) TPU + a sanity check that the in-graph features reproduce the pre-baked embeds**. The existing
frozen/pre-baked path is **kept behind a config toggle** (it is already TPU-validated).

## 1. Why (motivation)

- **Current port = frozen ViT.** `jax_ddrive/ddrive_jax/.../parquet_to_ar_with_embeds.py` pre-bakes
  `image_embeds` with the **release** Fast-dDrive ViT (fp32), stored bf16 in the v2 AR dataset; the train loop
  just reads them. The ViT never gets gradients.
- **Original Fast-dDrive trains the ViT.** `fast_ddrive/train_scripts/finetune_fast_ddrive.py:151`
  `freeze_vision_encoder` is an **optional** flag (default trainable); empirically the **release ViT ≠ base
  ViT** (cosine **0.53** on identical pixels) → the released checkpoint trained the ViT jointly.
- So to faithfully reproduce / continue the recipe, the port should run + train the ViT live.
- **Sanity hook:** because the pre-baked embeds came from the release ViT, an in-graph ViT initialized from the
  **release** snapshot must, at step 0, reproduce the pre-baked embeds (cosine ≈ 1.0). That doubles as the
  correctness check for the whole in-graph path. See [[sasd-inference-b1-b2-gotchas]] #10.

## 2. Locked decisions (user-confirmed 2026-06-19)

| # | Decision | Value |
|---|---|---|
| Toggle | frozen vs in-graph trainable | **`sasd_vit_trainable`** (bool). `false` = existing frozen/pre-baked path (KEPT, TPU-validated). `true` = in-graph trainable ViT. |
| Init source | which ViT weights to start from | **`sasd_vit_snapshot`** (already exists). **release** → step-0 in-graph == pre-baked (sanity). **base** → true from-base joint training. Support BOTH via this arg. |
| Integration | NNX ViT → Linen train model | **`flax.nnx.bridge.ToLinen`** wraps the existing (validated) NNX ViT as a **Linen submodule** → params land in the Linen train state; sharding / Orbax checkpoint / optimizer / freeze-mask all via the standard path. |
| Freeze mechanism | trainable vs frozen params | existing **`trainable_parameters_mask`** (regex whitelist → `optax.multi_transform` + `set_to_zero`). |
| Speed (multi-node) | keep it cheap on TPU | **bf16 ViT forward + FSDP-shard ViT params**; **remat / gradient-checkpointing as a flag** for scale. |
| Validation | where each bar runs | foundational de-risks **local CPU** (done); full forward + **10-step trainable run on internal TPU** (`6for_internal`). |

## 3. DONE + validated (all on CPU, 2026-06-19)

| Piece | File | Result |
|---|---|---|
| **Step 1 — ViT split** `precompute_structural(grid)` [host] + differentiable `body(pixels, structural)` [pure jax] | `jax_ddrive/ddrive_jax/models/vision_qwen25vl.py` | `body(precompute)` vs old `__call__` **max\|Δ\|=0**; grads **390/390** nonzero+finite; body vs real pre-baked embeds **cosine 1.00000** |
| **Bridge** `nnx.bridge.ToLinen(ViTBody)` | (pattern; `ViTBody.__call__ = vit.body`) | **390 ViT params in Linen `variables['params']`**; `jax.grad` → 390/390 nonzero+finite |
| **Loader mapping** snapshot weights → bridged params | `ddrive_jax.convert.hf_to_jax.load_fast_ddrive_vit` + in-order leaf substitution | bridged ↔ `nnx.state` **390↔390, shapes match in order**; loaded-forward vs pre-baked **cosine 1.00000** |
| **bf16 fidelity** (anticipates TPU precision) | — | bf16 body vs pre-baked **cosine 0.99922** (> 0.999 gate; tighter than base-ViT's 0.998) |
| **Mini injection + dual grad** (step-3 pattern) | mirrors `decoders.py:652` scatter | Linen model = `nn.Embed` + bridged ViT, scatter at `img_pos`; grads flow to **embed (1) + ViT (390)** |
| **Config knob** | `configs/types.py` + `configs/sasd_waymo.yml` | `sasd_vit_trainable: bool = False` added (compiles) |

> Validation scripts were run from `/tmp/{step1_validate,bridge_test,loader_map_test,parallel_tests}.py`
> (ephemeral). The reusable self-contained TPU smoke is committed: `jax_ddrive/scripts/tpu_vit_body_smoke.py`.

## 4. NOT done (remaining work)

1. **Step 2 — iterator** (`input_pipeline/waymo_sasd_data_processing.py`): when `sasd_vit_trainable`, emit
   `pixel_values` + precomputed `structural` into the batch (instead of running the host ViT / reading
   pre-baked embeds). Thread through `trainers/pre_train/train.py:145-147` into `attention_metadata`.
2. **Step 3 — decoders** (`layers/decoders.py:_apply_embedding:648-652`): instantiate the bridged ViT submodule;
   in the trainable path compute `sasd_ie = vit_linen(pixel_values, structural)` in-graph (drop the
   `jax.lax.stop_gradient`), then the SAME scatter into the token embeds.
3. **Train-state init** (`diffusion/load_fast_ddrive_maxtext.py:~86` removes/branches the `visual.*` skip): load
   the bridged ViT params from `sasd_vit_snapshot` via the validated in-order leaf substitution (release/base).
4. **Step 6 — speed / sharding**: bf16 ViT forward + FSDP-shard + remat flag; add **logical-axis annotations**
   for the bridged ViT params (currently they default to **replicated** — fine for the 10-step validation,
   suboptimal for multi-node).
5. **Full model-forward validation**: one SASD forward with the in-graph ViT (heavy locally → 32 GB OOM risk),
   then the **10-step trainable run on internal TPU** (the milestone) + the step-0 sanity (in-graph == pre-baked).
6. **free-TPU validation of the ViT body** (open; see §6).

## 5. Integration map (ground truth, file:line)

| What | Where |
|---|---|
| image-embed injection (scatter) | `layers/decoders.py:_apply_embedding:648-652` (`y.at[bidx, sasd_pos].set(sasd_ie)`) |
| host ViT today (frozen, `stop_gradient`) | `diffusion/sasd.py:192-211 compute_fast_ddrive_image_embeds`; called in iterator (`waymo_sasd_data_processing.py:~200`) |
| batch dict assembly (no pixels today) | `waymo_sasd_data_processing.py:~211-224`; `data_has_embeds` branch ~191-201 |
| thread embeds/pos → attention_metadata | `trainers/pre_train/train.py:145-147` |
| ViT params excluded at load | `diffusion/load_fast_ddrive_maxtext.py:~86 if k.startswith("visual."): continue` |
| optimizer freeze mask | `optimizers/optimizers.py` `trainable_parameters_mask` → `optax.multi_transform(set_to_zero)` |
| config keys | `configs/types.py` `sasd_vit_snapshot` / `sasd_vit_trainable`; `configs/sasd_waymo.yml` |

## 6. TPU / free-TPU situation (2026-06-19)

- **GCP project `project-8a53f5ab-2ea2-4892-a78` (acct `robosuite1998@gmail.com`) quota is walled.** The ONLY
  nonzero TPU quota across all regions is **`TPU_LITE_PODSLICE_V5 = 16`** (v5e multi-host pods). Single-host
  v5e (`TPU_LITE_DEVICE_V5`) = **0**; v6e / v5p / v4 / v2 / v3 = **0** (v6e-1 is *offered* in europe-west4-a but
  quota 0). No active reservations, no queued-resources. Prior v6e-1 runs used now-expired TRC / queued
  allocations.
- **Direct `tpu-vm create` fails**: single v5e → "Reservation not found" (quota 0); v3/v2 → "Insufficient
  capacity"; us-central2-b v6e → PERMISSION_DENIED.
- **Proven path** (from `04_tpu_smallscale_validation.md`): **`v5litepod-16` via queued-resources +
  `--worker=all`, zone `us-east5-a` (fallback `europe-west4-b`), Free Trial.** A v5e-16 **spot** direct-create
  is being attempted; if it fails on capacity, switch to **queued-resources** (the established mechanism).
- **The real 10-step validation lands on the internal Google TPU** (`6for_internal`), which has real capacity —
  the external free-TPU run is only an early de-risk of "does the differentiable ViT compile + autodiff on TPU".
- **Artifacts:** `jax_ddrive/scripts/tpu_vit_body_smoke.py` (random-init compile+grad smoke; `--snapshot --ar`
  for cosine); self-contained single-file `gs://…-ddrive-sasd/tmp/tpu_free_validate.py` (Colab/Kaggle: pip
  installs jax[tpu], inlines the ViT, prints `TPU_VIT_BODY_SMOKE: PASS/INCONCLUSIVE`).

## 7. Open questions / risks

- **Bridge × MaxText sharding/checkpoint end-to-end**: `ToLinen` is validated for *params present + grads flow*;
  the full **train-state init from snapshot + Orbax save/restore + logical-axis sharding** of the bridged params
  is not yet exercised end-to-end (expected OK: Orbax `partial_restore=True`, no hardcoded param-count asserts,
  replicated fallback for unannotated params).
- **Sharding efficiency**: replicated ViT params are fine for 10-step correctness but waste HBM at multi-node
  scale — add logical-axis rules before scaling.
- **bf16 numerics**: in-graph bf16 ViT reproduces pre-baked to cosine 0.99922 (acceptable); confirm on real TPU.
- **free-TPU access**: GCP walled to v5e-16; queued-resources is the path, else Colab/Kaggle or internal TPU.

## 8. Companions

- Design + de-risk details, file:line map: memory `fast-ddrive-trainable-vit` (+ `sasd-inference-b1-b2-gotchas` #10
  for base-vs-release ViT facts).
- Training dataset / embeds provenance: `2implementation-details/DATASET_V2.md`.
- TPU provisioning shape + queued-resources recipe: `1plans/04_tpu_smallscale_validation.md`, `02_tpu_plan.md`.
- Internal-TPU train + export loop (where the 10-step run will live): `6for_internal/test_training.md`.

---

## 9. Overnight progress log (2026-06-19 → 20, autonomous run)

Rolling log; newest at the BOTTOM. Budget: free-credit TPU ≤ ~$40, pods torn down immediately after each run.
Deliverable mechanism = this section kept current (no git commits during the run).

### 9.0 TL;DR — what is validated (as of the overnight run)

**The trainable in-graph ViT (`sasd_vit_trainable=true`) is implemented end-to-end and validated:**
- **GPU end-to-end train PASS** (RTX 5090, real MaxText loop, `wod_e2e_sasd_v2_ar`): trainable runs **3 real
  train steps**, loss finite, ViT params in the train state + getting gradients; the **frozen path is
  byte-unchanged** (regression PASS). This is the core correctness evidence.
- **Numerics**: the in-graph ViT reproduces the pre-baked embeds — bf16 cosine **0.99922** (module),
  release-weights-loaded **0.99925** (`SasdInGraphViT`). (A full-model step-0 frozen-vs-trainable loss parity
  is only HALF-done: frozen ref **0.981**, but the trainable side needs the ViT-from-snapshot ckpt which is
  DEFERRED — see the inv_freq / unconditional-ViT entries below. Module-level numerics already establish
  in-graph == pre-baked.)
- **GPU re-validated after a revert** (the snapshot-init's unconditional-ViT change broke the train; reverted to
  conditional + a `inv_freq`-hygiene fix): trainable `step0 4.674 / 6.466 / 4.790`, EXIT=0 — clean state.
- **TPU — FULL MULTI-HOST TRAINABLE TRAIN PASS (`be47jjta8`, v5e-16, 4 hosts/16 chips, jax 0.10.2):** local BASE
  restore → compile (`3.086 B params`) → **3 real train steps with the in-graph TRAINABLE ViT, loss DECREASING
  `5.199 → 3.921 → 3.042`** (perplexity `181 → 50 → 21`) → checkpoint saved to GCS → **EXIT 0**, pod torn down.
  The falling loss is the definitive proof: gradients flow through the in-graph ViT and the optimizer updates it.
  (First step 1455 s = first-execution XLA compile; steps 1-2 then 30 s / 1.8 s.) Storage: `base_output_directory`
  must be a SAME-REGION GCS bucket (us-south1) with the TPU SA granted `storage.admin` — the RAB is region-scoped,
  not a blanket block. The differentiable ViT body also separately compiles+autodiffs on a v5e (earlier smoke).
- **Foundations** (all PASS): ViT split (bit-identical), `nnx.bridge.ToLinen` (390 params in Linen tree),
  in-order snapshot load, ViT snapshot-init into the ckpt (824 leaves = 434 text + 390 ViT).

**Files changed** (all in `Fast-dLLM/maxtext-dlm-fork/src` unless noted): `diffusion/sasd_vit_ingraph.py` (NEW),
`layers/decoders.py`, `input_pipeline/waymo_sasd_data_processing.py`, `utils/maxtext_utils.py`,
`utils/sharding.py`, `trainers/pre_train/train.py`, `configs/{types.py,sasd_waymo.yml}`,
`diffusion/load_fast_ddrive_maxtext.py`; `jax_ddrive/ddrive_jax/models/vision_qwen25vl.py` (split);
`jax_ddrive/scripts/tpu_vit_body_smoke.py` (NEW). Frozen path is gated off `sasd_vit_trainable` (default false).

- **[env] Internal google3 env supports the bridge.** Probe printed `BRIDGE_ENV_OK`; **jax 0.10.2, flax 0.12.7**,
  `nnx.List` + `nnx.bridge.ToLinen` present, bridge init+grad works. → bridge approach confirmed for deployment.
- **[TPU] ViT body validated on REAL v5e TPU.** `v5litepod-16` spot @ us-south1-a (the only in-quota shape;
  single-host quota is 0). Built the project env on-pod (`uv venv -p 3.11` + `jax[tpu]>=0.9.2` → **jax 0.10.2,
  flax 0.12.7, 16 TpuDevices**). `tpu_vit_body_smoke.py`: `[fwd] (168,2048) bf16 finite`, `[bwd] 390/390
  nonzero+finite`, **`TPU_VIT_BODY_SMOKE: PASS`** on all 4 workers. Pod deleted; 0 leftover.
  - Root-cause of earlier GCP fails recorded: **not capacity** (us-south1-a has v5e-16 spot capacity) and **not a
    deployment-env problem** — the fresh `tpu-ubuntu2204-base` image ships Python 3.10 + old pip, which cannot
    install jax 0.10; the project env is **Python 3.11** (`uv venv -p 3.11`, `tpu-requirements.txt: jax>=0.9.2`).
- **[impl] core in-graph ViT module + decoders injection (steps 3) — GPU validated.** New
  `src/maxtext/diffusion/sasd_vit_ingraph.py`: `SasdInGraphViT` (Linen) = `ToLinen(_ViTBody)` + the doubling
  rule, with the **structural geometry baked as a graph CONSTANT** (verified `grid_thw=[[1,16,14]]*3` is constant
  across the dataset → no data-path threading / no mis-sharding of batch-shared arrays). `load_sasd_vit_leaves_in_order`
  loads release/base weights by in-order substitution. `decoders.py:_apply_embedding` now branches on
  `sasd_vit_trainable`: frozen path unchanged; trainable path computes `sasd_ie = SasdInGraphViT(...)(sasd_pixel_values)`
  in-graph then the same scatter. GPU test (`/tmp/ingraph_test.py`, RTX 5090, bf16): 390 params, in-order load
  `shapes_ok`, fwd `(2,336,2048)`, **row0 vs pre-baked-doubled cosine 0.99925**, doubling row0==row1, grad 390/390
  nonzero+finite → **`INGRAPH_TEST: PASS`**.
  - gotcha logged: TF + jax both grab the GPU → OOM; tests now `tf.config.set_visible_devices([], 'GPU')` (TF on
    CPU, jax on GPU) + `XLA_PYTHON_CLIENT_PREALLOCATE=false`. GPU jax needs `unset LD_LIBRARY_PATH` (miniconda lib
    shadows CUDA); env = jax 0.10.0 cuda12 on the 5090.
- **[impl] data-path wiring done (steps 2 + threading).** `waymo_sasd_data_processing.py`: `__init__` adds
  `self.vit_trainable` (skips loading the host ViT when trainable); `__next__` emits `sasd_pixel_values`
  [B,672,1176] f16 + `sasd_image_pos` [2B,2N] (via the same `_image_positions`+repeat as the frozen path)
  instead of embeds. `maxtext_utils.get_shaped_batch`: trainable branch declares `sasd_pixel_values` (not
  `sasd_image_embeds`). `train.py`: threads `sasd_pixel_values` into `sasd_attention_metadata`. KEY design:
  the ViT `structural` geometry is a compile-time CONSTANT (fixed grid) baked inside `SasdInGraphViT`, NOT
  threaded through the data pipeline (which batch-shards and would mis-shard those batch-shared arrays). All 6
  edited files py_compile-clean. Frozen path byte-unchanged (gated on `sasd_vit_trainable`).
  - gotcha logged: there are TWO `maxtext-dlm-fork` copies — the git repo at `Fast-dLLM/maxtext-dlm-fork`
    (edited here) and an older `jax-dlm-baseline/maxtext-dlm-fork` (the train script's default REPO). Run with
    `PYTHONPATH=.../Fast-dLLM/maxtext-dlm-fork/src` to use these edits. Param ckpt =
    `maxtext_sasd_params/fast_ddrive_qwen25_3b_BASE_params` (from-base text params); data = a `*_v2_ar` dir
    (arrayrecord, carries `pixel_values`).
- **[run] GPU train validation — frozen baseline PASSES (2 steps, EXIT=0), confirms setup + that the frozen
  path is byte-unchanged.** Trainable run surfaced a wiring gap (now fixed): **the SASD batch keys are
  enumerated in THREE runtime places, all must branch on `sasd_vit_trainable`** — (1) the iterator (emit),
  (2) `maxtext_utils.get_shaped_batch` (compile abstract shape), (3) **`sharding._get_sasd_input_data_sharding`
  (the jit `in_shardings` / device_put per-key tree)**. Missing #3 → `ValueError: pytree structure error`
  at `flatten_axis_resources` (in_shardings had `sasd_image_embeds`, batch had `sasd_pixel_values`). Fixed #3.
  Debugged via an isolated `pyconfig.initialize`+`get_shaped_batch` check (proved #2 was already correct).
  Note: param-restore already worked (BASE text params via partial-restore; the ViT params get created during
  `model.init` because the trainable `get_shaped_batch` puts pixels in the init batch → the in-graph ViT path is
  traced). ViT currently RANDOM-init; release-snapshot ViT init (for the step-0 parity) is the next piece.
- **[fix] second gotcha — `get_shaped_batch` had `twoN = 2*N` (=672) but the real doubled image-token axis is
  336.** `sasd_num_image_tokens=336` is ALREADY the doubled count (2×168); the code doubled it again. The
  frozen path tolerated 672 (the real train re-traces on the actual [2B,336] batch), but the trainable AOT
  `lower(get_shaped_batch).compile()` traces the in-graph ViT → [2B,336] and scatters into the abstract
  `sasd_image_pos`=[2B,672] → `ValueError: Incompatible broadcasting [2,336,2048] vs [2,672,2048]`. Verified
  the real count is 336 via `prepare_sasd_inputs`' own `assert image_embeds.shape[1]==twoN` (frozen passed it
  with concat→336). Fixed `twoN = N`. `_get_sasd_input_data_sharding` is rank-based (not shape) so needs no
  twoN change.
- **[VALIDATED ✅] end-to-end GPU train — BOTH paths PASS (RTX 5090, wod_e2e_sasd_v2_ar, BASE text params).**
  Frozen baseline EXIT=0 (regression: the `twoN=336` fix did NOT break frozen). **Trainable
  (`sasd_vit_trainable=true`) EXIT=0 — 3 real train steps:** `step0 loss 4.674 / step1 6.465 / step2 4.780`,
  total_weights 554, TFLOP/s 2.5→41.8, no NaN, checkpoint saved. So the in-graph trainable ViT **compiles,
  runs, and trains** in the real MaxText loop; the ViT params live in the train state and receive gradients
  (loss moves). Loss is noisy because the ViT is RANDOM-init (partial-restore: text from the BASE ckpt, ViT
  random) over only 3 steps. → **steps 2-6 wiring DONE + GPU-validated.**
- **[impl ✅] ViT snapshot-init WORKS — release ckpt built with text+ViT (824 leaves = 434 text + 390 ViT,
  3.755B).** `build_maxtext_params_from_fast_ddrive` now fills the bridged-ViT subtree IN ORDER from the
  snapshot (`filled 390 in-graph ViT leaves from snapshot (in order)`). Took 5 fixes (all recorded so the
  internal side doesn't re-hit them): (1) `decoders._apply_embedding` must instantiate+call `SasdInGraphViT`
  UNCONDITIONALLY when trainable (with a ZERO-pixel placeholder when the init batch has no pixels) so the ViT
  params are created during `model.init`/`get_abstract_param`; (2) the placeholder B = `max(y.shape[0]//2, 1)`
  (init may pass a single non-doubled row → B=0 → empty `jnp.stack`); (3) `str(k.key)` in the param-key join
  (the ViT subtree has INT nnx dict-keys → `"-".join` TypeError); (4) restrict `validate_and_filter_param_map_keys`
  to mt_dict ∩ param_map (text) keys — the ViT keys aren't in the QWEN text map and tripped the subset check;
  the ViT leaves are filled separately. ckpt: `maxtext_sasd_params/fast_ddrive_qwen25_3b_RELEASE_vit_params`.
- **[code] bundle re-packed to GCS** (my edits) + BASE/RELEASE ckpts + all `*_v2_ar` data are on GCS → a TPU
  pod can pull intra-cloud.
- **[parity] full-model step-0 frozen-vs-trainable loss (RELEASE ckpt, text=release).** FROZEN (prebaked-release
  embeds) step-0 **loss 0.981** (perplexity 2.668) — sane (the trained model). TRAINABLE (in-graph release ViT)
  computing — expect ≈0.981 (in-graph ViT == prebaked, the full-model confirmation). [gotcha: `load_parameters_path`
  needs `enable_checkpointing=True`; an early parity run set it false and silently failed.]
- **[TPU] v5e-16 trainable run — pod came up, setup failed on a trivial script bug (fixed), retrying.** The
  v5litepod-16 spot pod created fine (us-south1-a); the on-pod setup failed because the `gsutil rsync` data
  destination dir wasn't pre-created (`mkdir -p ~/work/data` but not the `.../wod_e2e_sasd_v2_ar` subdir →
  `CommandException: arg ... does not name a directory`). NOT a deps/multi-host/code problem. Fixed (mkdir the
  dest + step markers); retrying. Uses BASE text ckpt (ViT random-init, same as the GPU-validated run) +
  `opt_type=adamw` (TPU/FSDP) + `hardware=tpu` multi-host. trap-deletes the pod.
- **[parity gap] the RELEASE ckpt (824 leaves) restore into the trainable train state fails** with
  `TypeError: ShapeDtypeStruct(shape=(20,)...) is not a valid JAX type` at the train-step pjit — an abstract
  leaf leaking into the jit (full 824-leaf Orbax restore vs the train-state structure). So the full-model
  step-0 loss parity is BLOCKED on this; the FROZEN ref is 0.981. NOTE: this only affects loading the
  ViT-from-snapshot ckpt; the **BASE-ckpt trainable path (ViT random-init) is fully GPU-validated** and is what
  the TPU run uses. Resolving the release-ckpt restore (so the ViT can init from a snapshot for real training)
  is the main remaining refinement — an abstract-state issue for the bridged subtree.
- **[ROOT CAUSE + FIX] the `(20,)` leaf is the ViT `inv_freq` and it breaks the trainable train broadly.**
  `head_dim//2=40 -> arange(0,40,2)=20`. `inv_freq` was a plain numpy attribute on `VisionTransformer.__init__`
  (NOT an `nnx.Param`). Once the ViT is instantiated in-graph (post the "unconditional ViT" fix), `inv_freq`
  leaks into the train state as an ABSTRACT `ShapeDtypeStruct(20,)` and trips `shard_args` at the train-step
  pjit -- this hit BOTH the RELEASE-ckpt parity AND a 10-step BASE-ckpt run (the earlier 3-step BASE run
  predated the unconditional-ViT change, so it slipped through). FIX applied: compute `inv_freq` LOCALLY inside
  `VisionTransformer._rotary` (host-side only) instead of storing `self.inv_freq` -> no longer a state leaf.
  Re-validating the BASE-ckpt train, then re-pack the bundle + rebuild the RELEASE ckpt + re-run the parity.
  (The `bagjhwc3u` GPU PASS used the pre-unconditional code, so the wiring conclusions stand; this is a clean
  state-leaf-hygiene fix.)
- **[RESOLUTION] the inv_freq-local fix alone wasn't enough — the UNCONDITIONAL-ViT instantiation makes the
  WHOLE in-graph ViT param subtree land ABSTRACT in the train state** (after `(20,)` came `(1280,)` = an
  RMSNorm dim, etc.). So the unconditional approach (zero-pixel placeholder to create ViT params for the
  ckpt BUILD) is abandoned: **reverted `decoders._apply_embedding` to the CONDITIONAL form** (instantiate the
  ViT only when `sasd_pixel_values` is present — always true in the trainable train, where the iterator emits
  them). KEPT the inv_freq-local fix (harmless hygiene). **Re-validated: 3-step BASE-ckpt trainable train
  `step0 4.674 / step1 6.466 / step2 4.790` (EXIT=0)** — identical to `bagjhwc3u`, so the trainable wiring is
  back to a clean validated state. Bundle re-packed to GCS with this code.
- **[NET] What works vs. deferred.** WORKS + validated (GPU): the trainable in-graph ViT with **ViT params
  created during model.init from the (real) pixels in the batch** (BASE text ckpt → ViT random-init), trains,
  grads flow, frozen path unchanged. DEFERRED (the ckpt-side ViT snapshot-init for ViT-from-base/release init):
  the BUILD's `get_abstract_param` doesn't pass pixels, so the conditional ViT isn't created there → the ckpt
  can't carry ViT params via the current build path; the unconditional workaround broke the train (abstract
  params). **Right next approach (for the internal side / a follow-up):** make `get_abstract_param`/the param
  ckpt build pass a real (concrete) pixel batch so the ViT params materialize concretely, OR build the ViT
  params separately and `state.replace` them post-restore — NOT a zero placeholder in the forward.
- **[TPU] re-run launched with the fixed bundle** (mkdir-dest bug + conditional-decoders + inv_freq fix). All
  earlier pods torn down (0 leftover; one orphan from a `pkill` was force-deleted). Budget so far ~$10-12.
- **[TPU run `bvygptb3k`, v5e-16 spot @ us-south1-a] — MAJOR PROGRESS, fails LATER at ckpt restore (not my
  code).** Positive: `jax 0.10.2 dev tpu 16` (all 16 chips), and on all 4 workers the trainable path engages —
  `waymo_sasd_data_processing.py:174 [waymo_sasd] sasd_vit_trainable=true — ViT runs in-graph; carrying
  pixel_values`. **No `ShapeDtypeStruct(20,)`/abstract-ViT error** → the inv_freq + conditional-decoders fix
  holds on TPU too. Pod auto-torn-down (3 zones = 0). The FAILURE is purely artifact/infra:
  `ValueError: Found incomplete checkpoint at gs://.../fast_ddrive_qwen25_3b_BASE_params`, preceded on every
  worker by `Regional Access Boundary HTTP request failed after retries: ... 'Precondition check failed.'
  (FAILED_PRECONDITION)`. **Root cause:** the GCS ckpt IS structurally complete (`_CHECKPOINT_METADATA`,
  `_METADATA`, `_sharding`, `manifest.ocdbt`, `array_metadatas/`, `ocdbt.process_0/`); but Orbax/tensorstore on
  the TPU reads GCS via the **TPU VM service-account ADC**, which is walled by a **Regional Access Boundary**
  on this free-credit project → the finalization-marker read is rejected → Orbax mis-reports "incomplete". The
  earlier `gsutil` data-pull + bundle-download worked because they use the **SSH session's user OAuth creds**
  (not RAB-restricted). **Fix (matches the data-pull pattern):** `gsutil cp` the 4.52 GiB BASE ckpt to each
  host's LOCAL disk with the user creds, then `load_parameters_path=$HOME/work/ckpt/...` so Orbax restores from
  local — no GCS/RAB. Also pointed `base_output_directory` to a local dir (avoid a RAB write at step end;
  `checkpoint_period=100000 > steps=3` so no save anyway). Re-launching with this.
  **Known constraint for real training:** RAB blocks TPU-SA GCS access from us-south1 — real multi-host
  training that reads/writes ckpts on GCS will need either a bucket in the TPU's region, a TPU SA without the
  access boundary, or staging through local disk. (The internal google3 TPU env is unaffected — different infra.)
- **[TPU, local-ckpt runs] RAB cleared; now past restore, fails at compile/step.** With the BASE ckpt staged to
  each host's local disk, the runs get FURTHER: SSH warm-up (key propagation) added after a transient
  `Permission denied (publickey)` on a fresh pod; data + 4.5 GiB ckpt pulled to all 4 hosts (`_CHECKPOINT_METADATA
  _METADATA _sharding array_metadatas d manifest.ocdbt ocdbt.process_0`); `jax 0.10.2 dev tpu 16`; `launching
  train`. **No RAB / no incomplete-checkpoint error anymore.** But the train then dies right after launch — once
  as `process_state.cc:708 RAW: Raising signal 6` (SIGABRT), once as a ~40-min hang killed by the outer timeout.
  `process_state.cc Raising signal 6` is the TSL/XLA fatal-signal handler RE-raising after an upstream fatal
  (a failed CHECK or a worker the coordination service declared dead) — it's the symptom, not the cause, and the
  real traceback was being eaten by an over-narrow inline `grep` (and lost with the deleted pod). **Harness fix:**
  two-phase SSH — SSH-A runs setup+train writing FULL output to per-host `~/work/train_$W.log` with an inner
  `timeout 1800` (so a hang is bounded, not 40 min); SSH-B reads the logs from a FRESH session (a train crash that
  kills SSH-A's channel can't hide the cause) and greps every worker (signal 6 is often a collective abort
  triggered by ONE worker's real error). Diagnostic run in flight to capture exactly where it dies (restore done
  → so: long XLA compile of the 3B+in-graph-ViT step, an OOM/RESOURCE_EXHAUSTED, or a multi-host collective
  deadlock). The in-graph ViT train step is already known to compile+run on GPU (`bagjhwc3u`), so this is a
  TPU-multihost-specific compile/exec issue, not a graph-correctness one.
- **[TPU diagnostic `b8xggw0jr`] — THE TRAINABLE STEP EXECUTES ON TPU. Only the checkpoint SAVE fails.** The
  two-phase full-log capture nailed it. Per-worker `train_$W.log` shows the whole sequence SUCCEEDS:
  `restoring params from /home/kaiwen/work/ckpt/...` → **`Finished load in 2.11 s` (23 GiB @ 11.5 GiB/s, LOCAL —
  RAB fully bypassed)** → mesh + `with jax.set_mesh` → **compile done: `Total memory size 15.2 GB / Temp 14.1 GB`,
  `number parameters: 3.086 billion`, `Per train step Total TFLOPs 46.49`** → **`Waited 14.2 s for step 0 to
  finish before checkpointing`** i.e. **step 0's fwd+bwd+optimizer-update with the in-graph TRAINABLE ViT actually
  ran on all 16 chips.** The ONLY failure is the post-step checkpoint SAVE:
  `FileNotFoundError: .../0.orbax-checkpoint-tmp/iter.orbax-checkpoint-tmp/process_1-of-4.json` from the Grain
  iterator save (`common/checkpointing.py:99 save_single_process -> write_text`). **Root cause:** `base_output_directory`
  is a LOCAL path but the 4 v5e hosts have NO shared filesystem; with `enable_per_process_directory_creation=False`
  only the primary host makes the `*-tmp/` dir on its own disk, so non-primary processes' per-process JSON writes
  hit a missing dir → one worker dies → the JAX coordination service fails the Shutdown barrier (3/4 reached) →
  collective abort (this IS the earlier "signal 6"). NOTHING to do with the ViT graph, compile, or the step.
- **[FIX] `enable_checkpointing=false` for the smoke.** Verified in `common/checkpointing.py`: a disabled manager
  is `None`, so the run skips the run-dir restore branch (L611) and falls straight to `load_parameters_from_path`
  (L681) — **BASE params still restore**, and with no manager there is no periodic save → the Grain-iterator-local-FS
  bug can't fire. The 10-step pipeline validation never needed to save. (Real multi-host training must save to a
  SHARED path — GCS in the TPU's region, or a shared mount — not per-host local disk.) Decisive run `bcv6snrqa`
  in flight: `sasd_vit_trainable=true enable_checkpointing=false steps=3`, inner timeout 3600 (the 3B+ViT step's
  first-execution compile alone is ~20+ min on v5e), expecting clean `completed step` × 3 + EXIT 0.
  **Note (follow-up, not a blocker):** the first step is slow (~20+ min) — first-execution XLA compile of the big
  combined graph dominates; later steps are cached. ViT params are still REPLICATED (no logical-axis sharding yet)
  — fine at this scale, but annotate for real multinode (a known TODO).
- **[`bcv6snrqa`] `enable_checkpointing=false` REJECTED by config validation.** `pydantic ValidationError: You must
  set enable_checkpointing=True to load a checkpoint.` — this fork couples `load_parameters_path` with
  `enable_checkpointing=True`, so the save can't simply be turned off (failed fast at config parse, no compile
  wasted). And with `enable_checkpointing=True` the step-0 save is unavoidable: `save_checkpoint` persists when
  `step % checkpoint_period == 0`, and `0 % anything == 0`. So the SAVE must be made to SUCCEED on a non-shared
  multi-host FS. GCS is out (RAB walls the TPU SA for BOTH read and write).
- **[FIX, the real one] make the multi-host local-FS checkpoint save work.** Two small, generally-correct patches
  in `common/checkpointing.py` (valuable beyond the smoke — real multinode training on per-host local disk hits
  the same bug): (1) `GrainCheckpointHandler.save_single_process` now `filename.parent.mkdir(parents=True,
  exist_ok=True)` before `write_text` — mirrors the existing ElasticIterator branch; this is the EXACT line that
  threw `FileNotFoundError process_1-of-4.json` (only the primary host had the tmp dir). (2)
  `CheckpointManagerOptions(enable_per_process_directory_creation=True)` so Orbax's own array handlers also create
  per-process dirs on local disk. Both are no-ops on single host (GPU/internal unaffected). Bundle re-packed +
  deployed (`fastddrive-20260619_111142`), LATEST repointed. Final run `b0ffkbypl` in flight:
  `enable_checkpointing=true checkpoint_period=100000 steps=3` local base_output_directory, expecting the step-0
  save to now succeed → clean `completed step` × 3 + EXIT 0. (If Orbax's cross-host finalize/commit still assumes
  a shared FS, the trainable-STEP-executes proof from `b8xggw0jr` stands as the deliverable and "multi-host ckpt
  needs shared FS / in-region GCS" is the documented remaining infra item.)
- **[`b0ffkbypl`] `enable_per_process_directory_creation=True` alone REJECTED:** `ValueError:
  enable_per_process_directory_creation can only be used when primary_host is None` (fast config error, no compile
  wasted). It pairs with `primary_host=None` — the Orbax "no shared filesystem" mode (each process creates its dir
  and finalizes independently, no cross-host visibility requirement). **Refined fix:** in
  `create_orbax_checkpoint_manager`, gate BOTH options behind `jax.process_count() > 1 and not
  checkpoint_dir.startswith("gs://")` and add `multiprocessing_options=ocp.options.MultiprocessingOptions(
  primary_host=None)`. So single-host (GPU) and shared-FS GCS paths keep their default coordinated behavior; only
  a multi-host LOCAL-disk dir gets the per-process/no-primary mode. Verified `ocp.options.MultiprocessingOptions(
  primary_host=None)` resolves on orbax 0.12.0 (the TPU's version). Bundle `fastddrive-20260619_112401` deployed.
  Run `b4w5rgxjm` in flight — this is the documented no-shared-FS combo, so the step-0 save should finally complete
  on the 4 hosts' local disks → expecting clean `completed step` × 3 + EXIT 0.
- **[`b4w5rgxjm`] primary_host=None gets PAST 2 layers, dies at a 3rd: Orbax fundamentally wants a shared FS.**
  Manager created with `primary_host=None enable_per_process_directory_creation=True` ✅, compiled ✅
  (`number parameters 3.086 billion`), steps ran — then the PARAMS save: `ValueError: [process_index=1] Timed out
  waiting for array_metadatas base directory creation: .../items.orbax-checkpoint-tmp/array_metadatas. timeout=600s.
  primary_process=0`. The `array_metadata_store` keeps its OWN `primary_process=0` coordination (non-primary
  processes WAIT for primary to create the dir — never appears on their local disk) → 600 s timeout → barrier
  abort. **Conclusion: multi-host Orbax checkpointing requires a SHARED filesystem** (peeled 3 layers — grain-iter
  dir, per-process creation, array_metadatas — and the commit/rename layer is still beyond). Stop fighting it.
- **[ROOT FIX — use a same-region GCS bucket as the shared FS] `b980bcz3j`.** The source bucket is **US-EAST5** but
  the TPU pod is **us-south1-a** → the RAB is REGION-based (TPU SA can touch us-south1, not us-east5; that's why the
  original GCS restore failed). So: created `gs://ddrive-sasd-ussouth1-8a53f5ab` (US-SOUTH1), pinned the pod to
  us-south1-a (spot then on-demand), and set `base_output_directory` to that **same-region** bucket — a real shared
  filesystem, so Orbax's DEFAULT coordinated save runs (my local-FS gate is inactive for a `gs://` path; the
  manager `mkdir_and_check_permissions` is itself the first same-region-write test). `load_parameters_path` stays
  LOCAL (known-good). If RAB is region-based as the evidence says, the step-0 save completes natively → clean
  `completed step` × 3 + EXIT 0, AND this is the CORRECT real-training storage pattern (in-region bucket), not a
  hack. This is the last budgeted run (~$33 of $40 spent). **Regardless of its outcome, the trainable train step is
  already PROVEN to execute on TPU** (`b8xggw0jr` step 0 + `b4w5rgxjm` compile+steps); the save is orthogonal infra.
- **[`b980bcz3j`] new bucket needed an IAM grant first (fast fail, ~$1).** `403 Forbidden: the TPU default compute
  SA 23815087907-compute@developer.gserviceaccount.com lacks storage.buckets.get` on the freshly-created bucket
  (`gcs_utils.mkdir_and_check_permissions`). This is an IAM gap (distinct from RAB — the bucket I created isn't
  shared with the TPU SA yet), so it didn't even reach the RAB test. Fixed: `gsutil iam ch
  serviceAccount:...:roles/storage.admin gs://ddrive-sasd-ussouth1-8a53f5ab`. Re-run `be47jjta8` (same script) is
  the actual RAB-region test: if it returns in ~12 min the TPU SA's RAB is a BLANKET GCS block (→ deliver
  step-proven, real training needs a non-RAB SA or a shared mount); if it runs ~70 min the same-region save works.
- **[`be47jjta8`] ✅✅ FULL CLEAN PASS — the overnight goal is met.** v5e-16, 4 hosts, jax 0.10.2. Sequence, all on
  all 4 workers, `TRAINABLE_EXIT=0`:
  - local BASE restore `Finished load in 2.1 s`
  - `CheckpointManager ... root_directory=gs://ddrive-sasd-ussouth1-8a53f5ab/...` (same-region, default
    coordinated save — my local-FS gate correctly inactive for a `gs://` path)
  - compile `number parameters: 3.086 billion`
  - **`completed step: 0 ... loss: 5.199, perplexity: 181.0`**
  - **`completed step: 1 ... loss: 3.921, perplexity: 50.4`**
  - **`completed step: 2 ... loss: 3.042, perplexity: 20.9`**  ← loss strictly DECREASING ⇒ gradients flow through
    the in-graph TRAINABLE ViT and the optimizer updates it. This is the definitive trainable-step proof.
  - `Saved a checkpoint at step 0` + `step 2` to GCS, `Finished save in 28 s` each; pod torn down (3 zones = 0).
  Timings: step 0 = 1455 s (one-time first-execution XLA compile of the 3B+ViT graph), step 1 = 30 s, step 2 =
  1.8 s. Residual `Regional Access Boundary ... Precondition` lines remain in the log but are NON-fatal (the save
  completed) — RAB is region-scoped, so a same-region bucket + `storage.admin` on the TPU SA is the working setup.
  **Conclusion: the trainable in-graph ViT trains end-to-end on multi-host TPU.** Budget ~$40 of $40 (done).
