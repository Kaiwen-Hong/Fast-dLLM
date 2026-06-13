# LABELING.md — where the training labels come from, and how we upgrade them

Stable reference for the **label side** of the SASD dataset: what each of the four
output sections is, which are real vs pseudo, where the paper's real labels came from,
and the teacher-distill pipeline that upgrades the pseudo text sections. Read this first
whenever the "is this dataset annotated?" / "what is 400 vs 800?" confusion comes up.

> **Scope split (one fact, one home):** this doc owns the *stable* labeling concepts,
> provenance, pipeline, and the dataset-identity map. The **live distill progress** (which
> sets are distilled right now) lives in the handoff doc `../../to-host-chn.md` (§2.4).
> Dataset *format/schema/inventory* live in `DATASET.md` / `DATASET_V2.md`.

---

## 1. The four output sections

The model emits one JSON answer with four section, in causal order:

```json
{"critical_objects": {12 yes/no flags}, "explanation": "...scene reasoning...",
 "future_meta_behavior": {"longitudinal": "...", "lateral": "..."},
 "trajectory": "[[+14.70,-00.04], ... 5 waypoints @1 s ...]"}
```

SASD weights them unequally (trajectory 3.0 / fmb 2.0 / critical_objects 1.5 /
explanation 1.0) — see `DATASET.md` §1 for the full token-level schema.

## 2. Label provenance — raw WOD-E2E has NO text labels

Raw WOD-E2E tfrecords contain images, ego states, intent, and **trajectories — but no
text labels** (no critical-object flags, no explanation prose). So our converter
`fast_ddrive/data/convert_wod_e2e.py --with_target` produces three tiers of label quality:

| Section | Source in our converter | Quality |
|---|---|---|
| `trajectory` | **real GT** (5 wp @1 s, indices 3/7/11/15/19 of the 4 Hz future) | ✅ genuine supervised signal |
| `future_meta_behavior.longitudinal` | derived from GT waypoint speeds (speed up / slow down / come to stop) | ⚠️ weak but grounded in the real trajectory |
| `future_meta_behavior.lateral` | from `EgoIntent` (go straight / turn left / turn right only) | ⚠️ coarse; misses lane-follow / lane-change / yield |
| `critical_objects` | all `"no"` | ❌ pseudo (no perception labels in raw WOD-E2E) |
| `explanation` | fixed template | ❌ pseudo |

This is enough to demonstrate a loss-decreasing real-data training loop; it is **not**
production-faithful for the reasoning sections.

## 3. Where the paper's REAL labels come from (and why we can't just download them)

Fast-dDrive (arXiv:2605.23163) §4.1 says it **adopts the chain-of-thought annotations from
dVLM-AD** (Ma et al. 2025, **arXiv:2512.04459**). dVLM-AD §4.2 *Dataset Construction* states
the annotator is **GPT-4.1**:

> "we construct reasoning annotations tailored to trajectory prediction by employing
> **GPT-4.1** as an automatic annotator"

GPT-4.1 is conditioned on the same signals as the predictor **plus the ego's future GT
waypoints**, with **2D bounding boxes overlaid** on the current frame for grounding; the
four-field structure is "inspired by Poutine". **These annotations are NOT open-sourced**
(dVLM-AD ships "Code (Coming Soon)", no dataset; Fast-dDrive ships no annotations). The
released checkpoint encodes the capability; the example `fast_ddrive/data/example/sample.json`
shows the rich real format (e.g. `critical_objects.nearby_vehicle: "yes"` + a grounded
explanation). **So real labels must be re-generated, not downloaded.**

## 4. Two routes to real labels

| Route | Annotator | Conditioning | Fidelity | Cost | Status |
|---|---|---|---|---|---|
| **A — teacher-distill** (what we do) | released **Fast-dDrive 3B** | image + past (no future) | a lossy copy of GPT-4.1 | local GPU, $0 API | implemented, validated |
| **B — replicate dVLM-AD** | **GPT-4.1 / current GPT-vision** | image + past + **future GT** + 2D bbox | highest (paper method) | OpenAI API ($) + a 2D detector | not done (deferred) |

Key asymmetry that motivates the **hybrid fmb policy** (§5): the GPT-4.1 annotator *sees the
future trajectory*; our released-3B teacher does **not** (it only sees one front image at
inference), so the teacher's `longitudinal` is a guess and measured only 4/10 agreement with
the GT-derived label — GT-derived is more accurate there.

## 5. Teacher-distill pipeline (Route A)

Upgrades `critical_objects` + `explanation` + `future_meta_behavior.lateral` with the
released checkpoint's output, **keeps the real-GT trajectory and the GT-derived
longitudinal**. Four stages, every script resumable / single-clean-command launchable:

| Stage | Script | What |
|---|---|---|
| 1 convert | `fast_ddrive/data/convert_wod_e2e.py --with_target` (autovla env) | tfrecord → JSON + JPEGs (image+prompt+GT) |
| 2 teacher | `fast_ddrive/data/distill_teacher_chunked.py` (ddrive env) | chunked, resumable `batch_inference.py scaffold_spec`; ~1.6 s/sample on the 5090 |
| 3 merge | `fast_ddrive/data/merge_distilled_labels.py` | hybrid: teacher CO/explanation/lateral + **GT longitudinal** + real-GT trajectory |
| 4 tokenize | `jax_ddrive/eval/prep_train_jax.py` → `jax_ddrive/scripts/pad_npz_uniform.py` → `prep_to_parquet.py` | npz → **uniform L=1280** → Parquet |
| orchestrate | `fast_ddrive/data/finalize_distill_50k.py` | waits for Stage 2, then runs 3+4 + writes `FINALIZE_REPORT.json` |
| inspect | `fast_ddrive/data/viz_distill.py` | HTML: 3 cams + BEV + pseudo-vs-distilled per sample |

**`--fmb_mode` policy** (in `merge_distilled_labels.py`, default `hybrid`): `hybrid` = GT
longitudinal + teacher lateral; `teacher` = whole fmb from teacher; `gt` = keep pseudo fmb.

### Why distilled datasets are L=1280 (NOT 1184)
Teacher explanations are longer than the pseudo template, so re-tokenizing pushes the
sequence past the old uniform **L=1184** (measured: 95% of distilled samples exceed 1184,
max needs **L=1280**, n_blocks up to 10). The distilled set is therefore a **separate
uniform-L=1280 dataset, shape-incompatible with the existing L=1184 50k/415k** (you cannot
label-swap in place). `pad_npz_uniform.py` right-pads to L=1280 using the documented
`grain_pipeline._pad_sample` contract (input_ids→MASK 151665, labels→-100, rbi/turn→-1,
scaffold→True, weight_vec→0.0); the **loss-zero invariant** (padded positions have
labels=-100 AND weight=0 → contribute exactly zero loss) is verified (400/400, 150/150).
Training accepts L=1280 with no code change (grain stacks any uniform L).

## 6. Dataset identity map — the "what is 400 vs 800?" answer

Three canonical *sizes* (small / middle / full) PLUS one intermediate source artifact:

| Name | Path | Rows | Format | What it is |
|---|---|---|---|---|
| **400 "small"** | `hf/wod_e2e_sasd/` | 400 | tokenized parquet | the canonical small/dev subset (pseudo labels) |
| **800 (NOT a size)** | `train/train_targets.json` | 800 | **JSON + raw JPEGs** | early Stage-1 train subset; the **only artifact with raw images** → used as the teacher-distill SOURCE. **400 ⊂ 800.** |
| **50k "middle"** | `hf/wod_e2e_sasd_50k/` | 50,331 | tokenized parquet | first 32 train shards (pseudo labels) |
| **415k "full"** | `hf/wod_e2e_sasd_full_packed/` | 415,663 | tokenized parquet (+ AR) | all 263 train shards (pseudo labels) |

The three canonical parquet sets above carry **pseudo** text labels (built before distill).
The **distilled** outputs are separate (L=1280): `train/distill_400/` (done, on GCS),
`train/distill_50k/` (stopped 2026-06-13, not needed for this milestone — intermediates
cleaned), `train/train_targets_distilled_{10,800}.json`. Because
the teacher needs raw images and only `train_targets.json` (800) keeps them, distilling the
800 automatically covered the 400 (400 ⊂ 800) — that is why a fully-annotated **400** can be
built in minutes with no teacher run. **Live distill status: `../../to-host-chn.md` §2.4.**

## 7. Distilled → v2 AR (the production-format redo)

The distilled sets are **Parquet v1, L=1280** (§5). Production training consumes **v2
ArrayRecord with precomputed `image_embeds`** (`DATASET_V2.md`). So a distilled set is not
yet on the production loader path — it needs the **last builder stage run on the distilled
parquet**:

```bash
# jax venv, GPU; first: unset LD_LIBRARY_PATH; export XLA_PYTHON_CLIENT_PREALLOCATE=false
JX=/home/kaiwen/jax-dlm-baseline/.venv/bin/python
$JX jax_ddrive/scripts/parquet_to_ar_with_embeds.py \
   /home/kaiwen/data/fast-ddrive/train/distill_400/parquet_L1280 \
   /home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd_distilled_400_v2_ar --split train
# ~19 samples/s on the 5090; resumable; writes dataset_info_train.json.
# 50k distill was stopped 2026-06-13 (not needed this milestone); if revisited, rebuild
# its distilled parquet first (distill_teacher_chunked.py -> finalize_distill_50k.py), then run this.
```

**Why this is cheap / low-risk:**
- **`image_embeds` are unchanged by distillation.** Distillation edits only the *text*
  arrays (input_ids/labels/scaffold/weight_vec/rbi/turn/block α,β); `pixel_values` +
  `image_grid_thw` come from the same images, and embeds are computed from `pixel_values`
  → the v2 builder recomputes embeds **bit-identical to the pseudo set's**. No new embeds risk.
- **No model/reader change.** `make_sasd_loader` auto-detects `.arrayrecord`; the distilled
  v2 AR is just **uniform L=1280** instead of 1184 — its own dataset (cannot be streamed in
  the same batch as the 1184 sets; the reader handles any uniform L).

**Verification caveat (important):** `verify_ar_round2.sh` re-derives the row **from raw
tfrecords**, where the converter regenerates *pseudo* text — so for a distilled set its
**text-array compare will (correctly) MISMATCH**. For distilled v2 AR, verify instead via
(a) the builder's build-time AR-vs-source-parquet bit-exact + embeds recompute checks
(automatic), (b) the distilled parquet's own loss-zero check (already done, §5). Use
round-2 only for the **embeds + pixel + structure** parts, not the text round-trip.

**After building:** name `wod_e2e_sasd_distilled_<date>-<size>_v2_ar`, mirror to GCS +
CNS like the parquet (commands in `../quick-refresh.md`), and add the row to that tracker.
**Status: not done — owner will run it.**

> Convention reminder: do not restate live status here — link to the handoff doc. Update
> this file only when the labeling *concepts/pipeline/identity* change.
