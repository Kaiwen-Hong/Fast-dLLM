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
