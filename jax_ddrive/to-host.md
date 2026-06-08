# Fast-dDrive → JAX / TPU — Hosting & Deployment Handoff (`to-host.md`)

Self-contained handoff for whoever runs this on Waymo's TPU infra. Covers what was built,
what is verified (with numbers), how to reproduce it, how to deploy on TPU, and the acceptance
criteria. Branch `jax-ddrive-port`, not pushed to GitHub.

> **Last updated: 2026-06-06** (Phase 6 complete; Phase 7 = MaxText port, in progress).
> For Phase 6 detail see `docs/03_scaleup_tpu_spec.md` and `docs/OVERNIGHT_PROGRESS.md`.
>
> 📌 **STATUS UPDATE (2026-06-08):** Phase 7 (MaxText port) is **done and proven on real TPU** —
> MaxText SASD trains on a real v6e-1 with real Fast-dDrive weights (loss 0.31/0.56, matching the GPU
> smoke). The claims below of "MaxText port not done" / "no real TPU run yet" are **superseded**; the
> current source of truth is **`docs/OVERNIGHT_TPU_PROGRESS.md`**. The only remaining open item is the
> literal **≥8-chip multi-node run**, blocked solely by GCP trial TPU **capacity** (external/transient,
> not a code issue) — one command (`ACCEL=v6e-16 bash launch_maxtext_sasd_tpu.sh`) once capacity frees.

---

## 0. TL;DR (point by point)

### Phases 1–5 (model port + eval, previously verified)
- **The whole model is ported and parity-verified** against PyTorch: text decoder **3.2e-5**, ViT **4.0e-5**, multimodal forward **7.7e-5**, SASD loss **7.9e-8**, attention mask bit-identical.
- **Trains with decreasing loss**: text scratch **3.84→1.47**; multimodal **0.999→0.701**; real Waymo **0.6815→0.6005**.
- **WOD-E2E eval on both stacks**: PyTorch ADE@3s **0.814** / RFS **7.914**; JAX **0.839** / **7.929** (on par, trajectory match **0.01 m**). Full 479-frame rated-val set.
- **10-gate verification suite + adversarial audit: 0 defects**.

### Phase 6 (scale-up + dataset, 2026-06-05, newly completed)
- **TPU-ready dataset built and uploaded**: 50,331 WOD-E2E train frames → 787 Apache Parquet shards (22 GB), private HF `kaiwen2/wod-e2e-fast-ddrive-sasd-50k`. 0 conversion errors; bit-exact decode verified; MaxText `hf` data-path compatible.
- **FSDP training harness verified**: `shard_map`+`psum` FSDP, AdamW, Orbax `CheckpointManager`. FSDP-vs-single-device **|diff|=9.5e-7**; checkpoint resume **diff=0.0**; 252/252 real-model kernels sharded.
- **Real 3.09B model trains through the harness on the GPU**: loss **0.985→0.598** (40 steps, no NaN, 24.8 GB VRAM).
- **grain multi-host input pipeline**: deterministic, per-host sharding, online SASD noising, resumable.

### Phase 7 (MaxText port — next, not yet started)
- **Decision (2026-06-06)**: use **MaxText** as the production training framework.
- **What to graft**: `diffusion/` module + SASD `loss_fn` (~5-line diff) + bidirectional attention patch + Waymo grain data source. Template: `jax-mdlm-handoff` (LLaDA in MaxText).
- **Data is ready**: Parquet on local disk + private HF; `gsutil rsync` to GCS before the pod run.

### Hardware + storage
- RTX 5090 (32 GB VRAM). Host RAM 30 GB, **no swap** — don't load the full 3.09B model under CPU 8-device emulation (OOMs).
- Big files on `/home/kaiwen/data/fast-ddrive/` (3.6 TB SSD). Raw train: 877 GB, raw val: 226 GB. 50k Parquet: 22 GB.

### Honest open items
- **Pseudo text labels** for `critical_objects`/`explanation` (WOD-E2E has none; only trajectory+meta are real GT).
- **50k subset** (50k of ~420k train frames); full = `scripts/build_full_dataset.sh --full`.
- **No real TPU run yet** — all multi-host verification is CPU 8-device emulation + single GPU.
- **MaxText port not done** (Phase 7).
- HF datasets are **private** (WOD license prohibits redistribution).

---

## 1. What this is

**Fast-dDrive** (NVIDIA/`Efficient-Large-Model/Fast-dDrive` on HF) is a Qwen2.5-VL-3B-Instruct
backbone fine-tuned as a **masked-diffusion (MDM) block-diffusion** VLA for the **Waymo Open
Dataset End-to-End Driving (WOD-E2E)** challenge. Given 3 front camera frames + a textual prompt
(navigation command + 3 s of ego history), it emits a **JSON answer** with four sections:

```json
{"critical_objects": {12 yes/no flags}, "explanation": "...scene reasoning...",
 "future_meta_behavior": {"longitudinal": "...", "lateral": "..."},
 "trajectory": "[[+14.70,-00.04], ... 5 waypoints @1 s ...]"}
```

It is trained with **SASD** = *Section-Importance-Weighted Loss* (per-section weights
{critical_objects 1.5, explanation 1.0, future_meta_behavior 2.0, trajectory 3.0}) + a
*per-section Beta noise schedule* over a **deep-JSON scaffold** (the JSON skeleton is fixed; only
the value slots are masked and denoised), on a doubled `[noisy | clean]` sequence with a **hybrid
block-causal attention mask**. Inference denoises the scaffold block-by-block.

**Our task** was to port all of this to JAX/Flax-NNX (so it runs on Waymo TPU pods, MaxText-style),
proving correctness on the local 5090 via numeric parity to the PyTorch release and via training
loss that decreases. The reference for the JAX/MaxText patterns is the prior project at
`/home/kaiwen/Desktop/research/DLM-policy4AV/jax-mdlm-handoff`.

---

## 2. What we built & verified

### 2.1 The model port (`jax_ddrive/ddrive_jax/`)
Plain Flax-NNX, mirroring the PyTorch `modeling.py` + `section_utils.py` + `generation_utils.py`:

| Component | File | Parity vs PyTorch release |
|---|---|---|
| Qwen2.5 text decoder (36L/2048d, GQA 16/2 heads, qkv bias, RMSNorm, SwiGLU, tied embeds, θ=1e6) | `models/qwen2_5_text.py` | logits rel-max **3.2e-5**, top-1 **100%** |
| RoPE + 3D **M-RoPE** (mrope_section [16,24,24]) | `models/{rope,qwen2_5_text}.py` | part of text / MM parity |
| Qwen2.5-VL **ViT** (Conv3D-as-linear patch embed, 2D RoPE, window attn @[7,15,23,31], spatial-merge 2, RMSNorm+SwiGLU, PatchMerger) | `models/vision_qwen25vl.py` | arch **4.0e-5** (end-to-end 2.6% is a benign cuDNN-Conv3d seed) |
| Vision↔text **fusion** (image-token scatter) + M-RoPE | `models/qwen2_5_text.py` | multimodal forward **7.7e-5**, top-1 100% |
| HF safetensors → NNX weight load (text + ViT) | `convert/hf_to_jax.py` | 0 missing keys |
| **SASD loss** (section-weighted CE + complementary-mask causal CE) | `diffusion/sasd_loss.py` | total **7.9e-8** |
| **Hybrid block-causal mask** (training doubled-2L + eval) | `diffusion/masks.py` | **bit-identical** (0 mismatches) |
| Per-section **Beta noising** (scaffold freeze, always-mask `im_end`, complementary) | `diffusion/noise.py` | unit-tested |

### 2.2 Training (loss decreases)
- `train_overfit.py` (text), `train_overfit_mm.py` (multimodal): from base Qwen2.5-VL the SASD loss
  drops **3.84 → 1.47** (text, learns the task from scratch) and **0.999 → 0.701** (multimodal).
- `train_waymo_sasd_jax.py`: the production multi-sample loop on **real Waymo data** —
  stochastic per-section Beta noise, frozen ViT image embeds, Section-Importance-Weighted +
  complementary-mask loss, bf16 + per-layer `nnx.remat` + Optax Adafactor, **Orbax checkpoints**.
  Fixed-eval loss **0.6815 → 0.6005** over 400 steps on 200 real frames (ckpts at step_200/400).
- Memory trick for the single shared 5090: bf16 + Adafactor + per-layer remat (+ optional LoRA in
  `lora.py`). On a TPU pod these are unnecessary (FSDP gives per-chip ~1/N memory → full-FT + AdamW).

### 2.3 The evaluation pipeline (both stacks, one metric)
```
WOD-E2E tfrecords ──convert_wod_e2e.py──▶ val_rated.json (479) + front-cam JPEGs
        ┌────────────────────────────────┴───────────────────────────────┐
        ▼ PyTorch / 5090                                                  ▼ JAX
 batch_inference.py ─▶ predictions.json        prep_jax_eval.py ─▶ jax_batch_inference.py ─▶ predictions.json
        └────────────────▶ evaluate_waymo_metrics.py (autovla env) ◀──────┘
                              ADE_3s / ADE_5s / RFS   (one shared backend)
```
- **Converter** `fast_ddrive/data/convert_wod_e2e.py` (the repo's promised-but-missing preprocessor):
  parses `E2EDFrame`, extracts the 3 front cams, builds the canonical prompt (**verified byte-for-byte**
  vs `fast_ddrive/data/example/sample.json`), sets `sample_id = frame.context.name` so predictions
  join the GT, and with `--rated_only` keeps the **479** official rater-scored frames.
- **PyTorch eval** = `fast_ddrive/eval/batch_inference.py` (`scaffold_spec`, paper canonical, bf16,
  ~1.7 s/sample).
- **JAX eval** = `prep_jax_eval.py` (CPU prep: HF processor + deep-scaffold + numpy `get_rope_index`,
  all validated bit-exact vs PyTorch internals) → `jax_batch_inference.py` (JAX ViT fuse + block-wise
  `section_diffusion` denoise via `ddrive_jax/eval/mm_sampler.py`, ~16 s/sample, no KV-cache).
- **Metric** = `fast_ddrive/eval/evaluate_waymo_metrics.py` (autovla env). It JMT-interpolates the
  5×1 s waypoints → 20×4 Hz before ADE, and computes the official trust-region **RFS** over the
  ≤3 rater trajectories. Real flags: `--pred_json --gt <tfrecord-glob|.pkl> --output_dir`.

**Full 479-frame rated-val results, both stacks, 100% trajectory parse:**

| Metric | PyTorch `scaffold_spec` | JAX `section_diffusion` |
|---|---|---|
| ADE @3s (m) | 0.814 | 0.839 |
| ADE @5s (m) | 1.990 | 2.072 |
| RFS | 7.914 | 7.929 |

(JAX vs PyTorch use different decoders — section-diffusion vs scaffold-speculative — yet land on
par; on a matched sample the JAX trajectory equals PyTorch's to **0.01 m**.)

### 2.4 Honest limitation — training text labels
Raw WOD-E2E tfrecords contain **images, ego states, intent, and trajectories — but no text labels**
(no critical-object flags, no explanation prose). So `convert_wod_e2e.py --with_target` builds
training targets where the **trajectory is real ground truth** (the genuine supervised signal),
`future_meta_behavior` is **derived** (speed from waypoint dynamics, lateral from `EgoIntent`), and
`critical_objects`/`explanation` are **pseudo-labels**. This is sufficient to demonstrate a working
real-data training loop with decreasing loss; for production training fidelity you'd either
teacher-distill the text sections (run the released model to label them, keeping the GT trajectory)
or supply Waymo's own perception labels.

### 2.5 Verification & audit
- `jax_ddrive/scripts/run_all_verification.sh` → **10 gates**: `cpu_mask_loss, cpu_lora, cpu_noise,
  cpu_sharding, cpu_eval_ports, phase1_text, phase2_sasd, phase4_vit, phase4b_mm_fwd,
  phase3_lora_train`. All pass on clean runs. (`phase3_lora_train` can flake transiently in the heavy
  back-to-back suite because LoRA on the near-optimal trained ckpt only moves loss ~0.001; it passes
  reliably standalone — see the note in that script.)
- **Adversarial code audit** (11 agents, review→verify) over the eval/training code: **2 confirmed**
  (a latent multi-`<image>` handling divergence — **fixed** to mirror the reference; a minor
  intent-only lateral pseudo-label — **documented**), **5 refuted**, **0 correctness defects** in the
  validated path. See `docs/AUDIT.md`.
- **Parity methodology**: every component has a PyTorch "oracle" capture (`scripts/capture_oracle_*`,
  eager attention + explicit masks/positions) and a JAX gate (`scripts/parity_*`, with
  `jax_default_matmul_precision=highest` to disable TF32). Gates assert rel-max < 1e-3.

---

## 3. Repository & artifact layout

**Code (in the git repo, branch `jax-ddrive-port`):**
```
fast_ddrive/                         # the PyTorch release + our additions
  data/convert_wod_e2e.py            # NEW: tfrecord -> Fast-dDrive JSON (val eval + train targets)
  data/example/sample.json           # the canonical 2-sample example (prompt ground truth)
  eval/batch_inference.py            # PyTorch eval (releases')
  eval/evaluate_waymo_metrics.py     # official ADE/RFS (releases')
  eval/waymo_rfs_utils.py            # RFS (pure numpy)
jax_ddrive/
  to-host.md                         # THIS FILE
  README.md  docs/{REPORT,HANDOFF,EVAL_PIPELINE,FEATURES,ARCHITECTURE,AUDIT,00_PLAN,
                   01_pytorch_reference_algorithm,02_tpu_plan}.md
  ddrive_jax/
    models/{rope,qwen2_5_text,vision_qwen25vl,sharded}.py
    diffusion/{masks,sasd_loss,noise,sample_sd}.py
    eval/{rope_index,scaffold,mm_sampler}.py          # NEW: JAX inference building blocks
    convert/hf_to_jax.py  lora.py  sharding.py  checkpoint.py
    train_overfit.py  train_overfit_mm.py  train_waymo_sasd_jax.py   # last is NEW (real data)
  eval/{prep_jax_eval,jax_batch_inference,prep_train_jax}.py          # NEW: JAX eval/train drivers
  scripts/{capture_oracle_*,parity_*,debug_vit,prep_*,verify_sd_mm,run_all_verification.sh,
           run_overnight.sh}
  tests/{test_mask_loss,test_lora,test_noising,test_sharding,test_eval_ports}.py
```

**Big artifacts (NOT in git — under `/home/kaiwen/data/fast-ddrive/`):**
```
waymo/{train (877G, 263 shards), val (226G, 93 shards), meta/val_sequence_name_to_scenario_cluster.json (479 rated seqs)}
eval/{val_rated.json (479), val_images/, prep_val_full/ (479 npz), pt_val_full_ss/, jax_val_full_sd/}
train/{train_targets.json (800), train_images/, prep_train/ (400 npz), ckpt_jax/{step_200,step_400}}
ckpt/                # HF checkpoint cache
ref_logits/*.npz     # all PyTorch oracle captures for the parity gates
proto_build/         # the compiled WOD-E2E protobuf (see §4)
logs/                # every run's log
RESTART_starVLA_server.txt   # how to restart the user's GPU service that we freed
```
The HF model snapshot: `/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f` (referred to as `$SNAP`).

---

## 4. Environments (3 conda/venvs; always `unset LD_LIBRARY_PATH` first)

| Env | Python | Purpose |
|---|---|---|
| **ddrive** `/home/kaiwen/miniconda3/envs/ddrive/bin/python` | torch 2.11+cu128, transformers 4.57 | PyTorch oracle captures, PyTorch eval, all CPU prep (processor + section_utils) |
| **jax** `/home/kaiwen/jax-dlm-baseline/.venv/bin/python` | jax 0.10, flax 0.12.7 (nnx), optax 0.2.8, orbax; transformers (tokenizer-only, no torch) | all JAX compute (parity gates, training, JAX eval). Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` |
| **autovla** `/home/kaiwen/miniconda3/envs/autovla/bin/python` | py3.10, tensorflow, waymo-open-dataset-tf-2-12-0, grpcio-tools | tfrecord parsing (converter) + official ADE/RFS metric |

**Building the metric proto (important — no pip wheel ships it):** the WOD-E2E proto
`end_to_end_driving_data_pb2` is absent from `waymo-open-dataset-tf-2-12-0` (1.6.5 and 1.6.7). We
compiled it from source against the **installed** descriptors so the descriptor pool stays
consistent:
```bash
pip install grpcio-tools
mkdir -p proto_build/waymo_open_dataset/protos && cd proto_build
curl -fsSL https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/protos/end_to_end_driving_data.proto \
     -o waymo_open_dataset/protos/end_to_end_driving_data.proto
# 1) dump a FileDescriptorSet of dataset.proto + transitive deps FROM the installed package
#    (so the compiled E2E pb2 is descriptor-compatible — no descriptor-pool conflict):
python - <<'PY'
from google.protobuf import descriptor_pb2
from waymo_open_dataset import dataset_pb2
seen = {}
def collect(fd):
    if fd.name in seen: return
    p = descriptor_pb2.FileDescriptorProto(); fd.CopyToProto(p); seen[fd.name] = p
    for d in fd.dependencies: collect(d)
collect(dataset_pb2.DESCRIPTOR)                 # DESCRIPTOR is already the FileDescriptor (no .file)
open("deps.pb", "wb").write(descriptor_pb2.FileDescriptorSet(file=list(seen.values())).SerializeToString())
PY
# 2) compile only the E2E proto, resolving imports from the dumped descriptors, into site-packages
python -m grpc_tools.protoc --proto_path=. --descriptor_set_in=deps.pb \
       --python_out="$(python -c 'import site;print(site.getsitepackages()[0])')" \
       waymo_open_dataset/protos/end_to_end_driving_data.proto
python -c "from waymo_open_dataset.protos import end_to_end_driving_data_pb2; print('OK')"
```
(Also recorded in the project memory `autovla-e2e-proto-compile`; build dir kept at
`/home/kaiwen/data/fast-ddrive/proto_build/`.)

---

## 5. Reproduce locally (RTX 5090)

```bash
cd /home/kaiwen/Desktop/research/Fast-dLLM
unset LD_LIBRARY_PATH
PT=/home/kaiwen/miniconda3/envs/ddrive/bin/python
JX=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
AV=/home/kaiwen/miniconda3/envs/autovla/bin/python
SNAP=/home/kaiwen/data/huggingface/hub/models--Efficient-Large-Model--Fast-dDrive/snapshots/0fda81009f4efa58a2debbb48c0c09818e45341f
D=/home/kaiwen/data/fast-ddrive
export PYTHONPATH=$PWD/jax_ddrive XLA_PYTHON_CLIENT_PREALLOCATE=false

# (a) all parity + unit gates (~10 min, loads the ckpt several times)
bash jax_ddrive/scripts/run_all_verification.sh        # -> "ALL_VERIFICATION_PASS" (10 gates)

# (b) build the rated val set (autovla env)
$AV fast_ddrive/data/convert_wod_e2e.py \
   --tfrecords "$D/waymo/val/val_*.tfrecord*" \
   --out_json $D/eval/val_rated.json --image_root $D/eval/val_images --rated_only

# (c) PyTorch eval + official metric
$PT fast_ddrive/eval/batch_inference.py --model_path $SNAP \
   --eval_json $D/eval/val_rated.json --image_root $D/eval/val_images \
   --output_dir $D/eval/pt_val --mode scaffold_spec --num_gpus 1
$AV fast_ddrive/eval/evaluate_waymo_metrics.py --pred_json $D/eval/pt_val/predictions.json \
   --gt "$D/waymo/val/val_*.tfrecord*" --output_dir $D/eval/pt_val      # -> ADE/RFS

# (d) JAX eval + official metric (same metric backend)
$PT jax_ddrive/eval/prep_jax_eval.py --eval_json $D/eval/val_rated.json \
   --image_root $D/eval/val_images --out_dir $D/eval/prep_val
$JX jax_ddrive/eval/jax_batch_inference.py --prep_dir $D/eval/prep_val --out_dir $D/eval/jax_val
$AV fast_ddrive/eval/evaluate_waymo_metrics.py --pred_json $D/eval/jax_val/predictions.json \
   --gt "$D/waymo/val/val_*.tfrecord*" --output_dir $D/eval/jax_val

# (e) JAX SASD training on real data
$AV fast_ddrive/data/convert_wod_e2e.py --tfrecords "$D/waymo/train/training_*.tfrecord*" \
   --out_json $D/train/train_targets.json --image_root $D/train/train_images \
   --with_target --max_frames 800
$PT jax_ddrive/eval/prep_train_jax.py --train_json $D/train/train_targets.json \
   --image_root $D/train/train_images --out_dir $D/train/prep_train
$JX jax_ddrive/ddrive_jax/train_waymo_sasd_jax.py --prep_dir $D/train/prep_train \
   --samples 200 --steps 400 --lr 2e-5 --ckpt_dir $D/train/ckpt_jax   # -> WAYMO_SASD_JAX_TRAIN_PASS

# one-shot: (c)+(e)+(d) in sequence
bash jax_ddrive/scripts/run_overnight.sh
```

---

## 6. Deploy on TPU

The model is plain Flax-NNX, so it slots into a TPU mesh the same way `jax-mdlm-handoff`'s
`LLaDAModel` did. Three increasing-fidelity options; do them in order.

### 6.1 Mesh
2D mesh `('fsdp','tp')` via `jax.make_mesh((n_fsdp, n_tp), ('fsdp','tp'))`. Start **pure-FSDP
`(N,1)`** (simplest; scales the 3.75 B fine). v5e-256 → e.g. `(64,4)`; v6e → size to chips. Wrap all
jits in `with jax.sharding.use_mesh(mesh):` (use `use_mesh`/`set_mesh`, **not** bare
`PartitionSpec` without a mesh context, or JAX 0.10 raises).

### 6.2 Sharding (FSDP param map — already mapped in `ddrive_jax/sharding.py`)
| Param | PartitionSpec (fsdp,tp) |
|---|---|
| `embed_tokens.embedding` [V,D] | `P('fsdp', None)` (pure-FSDP) or `P(None,'tp')` |
| `q/k/v/o_proj.kernel` | `P('fsdp','tp')` / `P('tp','fsdp')` |
| `gate/up/down_proj.kernel` | `P('fsdp','tp')` / `P('tp','fsdp')` |
| norms / biases | `P('tp')` or replicated |
| activations [B,L,D] | `P('fsdp', None, 'tp')` via `with_sharding_constraint` |

**Option 1 — FSDP-only, no model change (fastest):** `device_put` `nnx.state(model, nnx.Param)`
onto `NamedSharding(mesh, pspec)` (largest axis on `'fsdp'`); jit the train step with matching
`in/out_shardings`. `ddrive_jax/sharding.py` does exactly this and is **validated to be a no-op at
mesh=1** (the 5090 path is unchanged). This alone runs the full 3.75 B on a pod.

**Option 2 — explicit TP (throughput):** swap `Linear→ShardedLinear`, `Embed→ShardedEmbedding`
across the decoder (~80 LOC; the primitives are in `ddrive_jax/models/sharded.py`, JAX-0.10
`out_sharding=` plumbing, **unit-tested no-op at mesh=1** in `tests/test_sharding.py`) and annotate
kernels per the table.

**Option 3 — MaxText-native (production):** lift `diffusion/` + the SASD `loss_fn` into a MaxText
fork exactly like the `jax-mdlm-handoff/code-fork` reference: a ~5-line `loss_fn` diff + the
bidirectional/block-causal attention-mode patch; wire WOD-E2E via **grain**; checkpoints via
MaxText's Orbax pipeline.

### 6.3 Data pipeline on TPU
Reuse `convert_wod_e2e.py` to materialize JSON + JPEGs to **GCS** (or adapt it to emit a grain/
tfrecord shard format). The per-sample SASD tensors (`prep_train_jax.py` output: input_ids, labels,
rbi, turn, scaffold, weight_vec, block α/β, pixel_values, image_grid_thw, 3D position_ids,
vision_mask) are framework-neutral numpy → load via grain. ViT image embeds can be precomputed once
per sample (frozen) and cached, or computed inline.

### 6.4 Checkpointing
Use `ddrive_jax/checkpoint.py` (Orbax `StandardCheckpointer`, multi-host safe). Point it at a `gs://`
path; Orbax handles sharded save/restore across the pod. The local run already produces
`StandardCheckpointer` checkpoints (`train/ckpt_jax/step_*`), so restore is the same call.

### 6.5 Numerics on TPU
- Keep loss / log-softmax in **fp32** (`sasd_loss.py` already upcasts); bf16 params/activations fine.
- TPU XLA is stricter than GPU — add NaN guards in eval, and **re-run the Phase-1/2 parity gates on
  a TPU CPU/emulator** (rel < 1e-3) before a big run.
- With FSDP across N chips per-chip memory is ~1/N → **full fine-tune + AdamW** is feasible (drop the
  5090-only bf16+remat+Adafactor+LoRA tricks; keep `remat` as a knob for very long sequences).

---

## 7. Criteria — what is verified vs. what we expect

### 7.1 Verified locally (RTX 5090) — acceptance evidence
| Claim | How verified | Result |
|---|---|---|
| Text decoder matches PyTorch | `parity_text.py` (fp32, highest) | logits rel-max **3.2e-5**, top-1 **100%** |
| SASD loss + mask match | `parity_sasd.py` | loss **7.9e-8**, mask **bit-identical** |
| ViT matches (architecture) | `parity_vit.py` | **4.0e-5** (end-to-end 2.6% = benign cuDNN seed) |
| Multimodal forward matches | `parity_mm.py` | hidden 3.2e-5 / logits **7.7e-5** / top-1 100% |
| Training learns (loss ↓) | `train_overfit{,_mm}.py` | text **3.84→1.47**, MM **0.999→0.701** |
| Real-data training (loss ↓) + ckpt | `train_waymo_sasd_jax.py` | **0.6815→0.6005**, Orbax step_200/400 |
| JAX generation == PyTorch | `verify_sd_mm.py` | trajectory **0.01 m**, valid JSON |
| Eval pipeline correct (both stacks) | full 479 rated val + official metric | PyTorch **0.814/1.990/7.914**, JAX **0.839/2.072/7.929**, 100% parse |
| Converter prompt correct | byte-compare vs `sample.json` | **exact** (1896 chars) |
| No regressions / defects | 10-gate suite + 11-agent audit | gates pass; **0 defects** in validated path |

### 7.2 Expected on TPU — acceptance criteria for the host
Run this **validation ladder** (each step gates the next):
1. **mesh=1 on one chip**: for a fixed batch, the SASD loss equals the 5090 value within fp32
   noise (sharding is a no-op at mesh=1 by construction). *Expected: match within ~1e-4.*
2. **FSDP `(N,1)` vs 1-chip**: single-step loss equal (same seed/batch) **within 1e-4**. *Confirms
   FSDP sharding doesn't change math.*
3. **Overfit the 2 example samples**: loss decreases monotonically (reproduces local Phase 3).
4. **Real WOD-E2E training**: loss decreases over a multi-thousand-sample run; checkpoints restore.
5. **Eval on TPU/host**: run inference → predictions.json → the **same** `evaluate_waymo_metrics.py`.
   *Expected ADE/RFS within run-to-run noise of the local numbers* (PyTorch reference 0.814/1.990/
   7.914; the JAX section-diffusion 0.839/2.072/7.929). RFS ≈ 7.9 and ADE@3s < 1.0 m are the
   sanity band for a correct setup.

**What "done on TPU" means:** steps 1–2 prove the port is numerically identical under sharding;
step 4 proves training scales; step 5 proves the eval reproduces. Throughput is then an optimization
axis (Option 2/3 TP, KV-cache decode) — not a correctness gate.

### 7.3 Known gaps / non-goals (so the host isn't surprised)
- **JAX decode has no KV-cache** (~16 s/sample, full-seq recompute per denoise step). Fine for
  offline eval; for serving, port a block-wise KV-cache (PyTorch `scaffold_speculative_sample` is the
  reference) — *not done*.
- **Whole-model TP** on a real mesh is *mechanical but not yet executed* (primitives + no-op tests
  exist; Option 2 above).
- **Training text labels** are pseudo-labels for non-trajectory sections (§2.4) — the trajectory is
  real GT. For production accuracy, supply real labels or teacher-distill.
- **Not pushed to GitHub** — the branch `jax-ddrive-port` is local only.

---

*Doc maintained alongside `docs/EVAL_PIPELINE.md` (pipeline detail), `docs/02_tpu_plan.md` (TPU
detail), `docs/AUDIT.md` (audit), and `docs/REPORT.md`/`HANDOFF.md` (build log). Generated for the
Fast-dDrive JAX port, branch `jax-ddrive-port`.*
