# Fast-dDrive — WOD-E2E evaluation + JAX real-data training pipeline

Built 2026-06-03. The whole open-loop evaluation pipeline (official Waymo ADE + Rater
Feedback Score) now runs on **both** stacks — PyTorch on the RTX 5090 and the JAX port —
sharing one metric backend, plus JAX SASD training on real Waymo data.

## TL;DR results — official rated val set, **full 479 frames, both stacks**

| Metric | PyTorch `scaffold_spec` | JAX `section_diffusion` |
|---|---|---|
| ADE @3s (m) | 0.814 | 0.839 |
| ADE @5s (m) | 1.990 | 2.072 |
| RFS         | 7.914 | 7.929 |
| parse rate  | 100% (479/479) | 100% (479/479) |

Both paper-consistent and **on par** (RFS 7.93 ≈ 7.91; ADE within ~3%). The JAX multimodal
section-diffusion sampler reproduces the PyTorch trajectory to **0.01 m** on a matched sample
(bf16). Results: `eval/pt_val_full_ss/` and `eval/jax_val_full_sd/waymo_eval_results.json`.

**JAX real-data SASD training**: fixed-eval loss **0.6815 → 0.6005** (−0.08) over 400 steps on
200 real Waymo frames (bf16+remat+Adafactor), Orbax ckpts at step_200/step_400. `WAYMO_SASD_JAX_TRAIN_PASS`.

## The pipeline (5 stages)

```
WOD-E2E tfrecords ──convert_wod_e2e.py──▶ val_rated.json (479) + front-cam JPEGs
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼ (PyTorch / 5090)                                                ▼ (JAX)
   batch_inference.py  ──▶ predictions.json          prep_jax_eval.py ─▶ jax_batch_inference.py ─▶ predictions.json
              │                                                                 │
              └────────────────▶ evaluate_waymo_metrics.py (autovla env) ◀──────┘
                                   ADE_3s / ADE_5s / RFS  (one shared backend)
```

### 1. Environment — `autovla` (metrics + tfrecord parsing)
`/home/kaiwen/miniconda3/envs/autovla` (py3.10): `tensorflow`, `waymo-open-dataset-tf-2-12-0`,
`grpcio-tools`. The E2E proto `end_to_end_driving_data_pb2` is **not in any pip wheel** — it
was compiled from source against the installed descriptors (see
`docs/` memory `autovla-e2e-proto-compile`; build dir `/home/kaiwen/data/fast-ddrive/proto_build/`).

### 2. Converter — `fast_ddrive/data/convert_wod_e2e.py`
The repo's promised-but-missing preprocessor. `E2EDFrame` → Fast-dDrive JSON.
- **Prompt reproduces `data/example/sample.json` byte-for-byte** (verified, 1896 chars).
  Fixed 4-task instruction + `<image>` (3 front cams) + `High-level navigation command:`
  (= `EgoIntent.Intent` name: GO_STRAIGHT/GO_LEFT/GO_RIGHT) + 7-pt historical ego state
  (past_states idx 3,5,7,9,11,13,15 @0.5s).
- `sample_id = frame.context.name` → predictions join GT in the metric.
- `--rated_only` → the **479** rater-scored frames = the official eval subset (the meta json
  lists 479 sequences across scenario clusters). `--with_target` → training targets.
```bash
AV=/home/kaiwen/miniconda3/envs/autovla/bin/python
$AV fast_ddrive/data/convert_wod_e2e.py \
   --tfrecords '/home/kaiwen/data/fast-ddrive/waymo/val/val_*.tfrecord*' \
   --out_json  /home/kaiwen/data/fast-ddrive/eval/val_rated.json \
   --image_root /home/kaiwen/data/fast-ddrive/eval/val_images --rated_only
```

### 3a. PyTorch eval (5090, multimodal)
```bash
PT=/home/kaiwen/miniconda3/envs/ddrive/bin/python
$PT fast_ddrive/eval/batch_inference.py --model_path $SNAP \
   --eval_json .../val_rated.json --image_root .../val_images \
   --output_dir .../pt_val_full_ss --mode scaffold_spec --num_gpus 1
```
`scaffold_spec` (paper canonical), bf16, ~5 s/sample. 100% trajectory parse.

### 3b. JAX eval (multimodal section-diffusion)
Two steps — prep (PyTorch `ddrive` env, CPU, **no model weights**: HF processor + section_utils
scaffold + numpy `get_rope_index`, all validated against PyTorch internals) then JAX compute:
```bash
$PT jax_ddrive/eval/prep_jax_eval.py --eval_json .../val_rated.json \
   --image_root .../val_images --out_dir .../prep_val_full
XLA_PYTHON_CLIENT_PREALLOCATE=false \
$JX jax_ddrive/eval/jax_batch_inference.py --prep_dir .../prep_val_full --out_dir .../jax_val_full_sd
```
JAX runs the ViT once, scatters image embeds into the text stream, then iteratively denoises
the deep-JSON scaffold block-by-block (`ddrive_jax/eval/mm_sampler.py`). bf16, ~17 s/sample
(no KV-cache). Writes the SAME `predictions.json` schema → SAME metric.

### 4. Official metric (both stacks)
```bash
$AV fast_ddrive/eval/evaluate_waymo_metrics.py --pred_json .../predictions.json \
   --gt '/home/kaiwen/data/fast-ddrive/waymo/val/val_*.tfrecord*' --output_dir .../
```
(`--gt` takes a tfrecord glob or a `.pkl`; the README's `--gt_tfrecords` flags are stale.)
ADE interpolates the 5×1 s waypoints → 20×4 Hz via JMT before comparing to the log GT; RFS
is the official trust-region score over the (≤3) rater trajectories.

### 5. JAX SASD training on real Waymo data
- Train data: `convert_wod_e2e.py --with_target` → trajectory = **real GT** (5 wp @1s);
  meta derived from intent + speed; CO/explanation pseudo-labeled (raw WOD-E2E has no text
  labels — the trajectory is the genuine supervised signal).
- Prep: `jax_ddrive/eval/prep_train_jax.py` → per-sample npz (doubled SASD structures + frozen
  ViT inputs). 400 samples prepped, all L=1184 / 7 blocks (constant → single XLA compile).
- Train: `jax_ddrive/ddrive_jax/train_waymo_sasd_jax.py` — multi-sample, stochastic per-section
  Beta noise, Section-Importance-Weighted + complementary-mask loss, bf16 + remat + Adafactor,
  Orbax checkpoints. Canonical recipe weights {CO 1.5, exp 1.0, FMB 2.0, traj 3.0}.

## One-command overnight runner
`jax_ddrive/scripts/run_overnight.sh` — PyTorch-479 eval+metric → JAX training →
JAX-479 eval+metric, each with a stage marker. Logs in `/home/kaiwen/data/fast-ddrive/logs/`.

## Files added
| File | Role |
|---|---|
| `fast_ddrive/data/convert_wod_e2e.py` | tfrecord → Fast-dDrive JSON (val eval + train targets) |
| `jax_ddrive/ddrive_jax/eval/rope_index.py` | numpy `get_rope_index` (3D M-RoPE), validated vs PyTorch |
| `jax_ddrive/ddrive_jax/eval/scaffold.py` | deep-JSON scaffold + rbi (generation_utils replica) |
| `jax_ddrive/ddrive_jax/eval/mm_sampler.py` | JAX multimodal section-diffusion sampler |
| `jax_ddrive/eval/prep_jax_eval.py` | eval prep (processor + scaffold + posids → npz) |
| `jax_ddrive/eval/jax_batch_inference.py` | JAX eval driver → predictions.json |
| `jax_ddrive/eval/prep_train_jax.py` | training-data prep → per-sample SASD npz |
| `jax_ddrive/ddrive_jax/train_waymo_sasd_jax.py` | JAX SASD training on real data + Orbax ckpt |
| `jax_ddrive/scripts/{capture_oracle_sd_mm,verify_sd_mm}.py` | PyTorch reference + JAX parity gate |
| `jax_ddrive/scripts/run_overnight.sh` | full-val both-stacks eval + training runner |
```
