# Docs index & maintenance rules

**Start here.** Two documents are the living sources of truth; everything else is either a
stable reference or a frozen historical log.

## Living (must stay current — update these when reality changes)

| doc | scope |
|---|---|
| [`../to-host.md`](../to-host.md) / [`../to-host-chn.md`](../to-host-chn.md) | **The handoff doc** (En/中文 twins — update both): what exists, what's verified (numbers), how to reproduce, how to deploy, acceptance criteria. The status banner at the top carries the latest date. |
| [`2implementation-details/DATASET_V2.md`](2implementation-details/DATASET_V2.md) | The production data format: v2 ArrayRecord schema (12 arrays + precomputed `image_embeds`), builder, verification chain, reader API, checkpoint/resume wiring, built-set inventory (local + GCS). |

The MaxText fork side is documented in the fork itself:
`maxtext-dlm-fork/PATCHES.md` (file-by-file diff vs upstream, vendor sync rules, validation
commands).

## Stable references (update only when the underlying component changes)

| doc | scope |
|---|---|
| `2implementation-details/ARCHITECTURE.md` | JAX/Flax-NNX model port structure |
| `2implementation-details/01_pytorch_reference_algorithm.md` | the PyTorch SASD algorithm being ported |
| `2implementation-details/EVAL_PIPELINE.md` | WOD-E2E eval (two stacks, one metric) |
| `2implementation-details/DATASET.md` | v1 Parquet dataset (superseded for training by DATASET_V2, still the bit-exact source format) |
| `2implementation-details/AUDIT.md` | adversarial code-audit findings |
| `3summary/REPORT.md`, `3summary/FEATURES.md` | phase-completion summaries |

## Frozen (historical — never edit, append-only at the time they were live)

| doc | what it captured |
|---|---|
| `1plans/00_PLAN.md` | original port plan (Phases 1–5) |
| `1plans/02_tpu_plan.md` | TPU deployment plan |
| `1plans/03_scaleup_tpu_spec.md` | Phase 6 scale-up spec (dataset + FSDP harness) |
| `1plans/04_tpu_smallscale_validation.md` | $300-trial small-scale validation plan |
| `4collect/HANDOFF.md`, `4collect/OVERNIGHT_PROGRESS.md` | Phase ≤6 build logs |
| `4collect/05_maxtext_port_progress.md` | Phase 7 MaxText port log |
| `4collect/OVERNIGHT_TPU_PROGRESS{,-chn}.md` | first real-TPU runs (v6e-1, 2026-06-08) |
| `4collect/06_dataset_v2_progress.md` | dataset v2 + AR reader + TPU re-validation (2026-06-12) |

## Maintenance rules

1. **One fact, one home.** Current state lives in the two living docs; don't restate it
   elsewhere (link instead). Frozen logs keep the *discovery* story, not the truth.
2. When a milestone lands: update the living docs **in the same change**, and append a
   dated entry to a `4collect/` log (create `NN_<topic>_progress.md`, numbered).
3. `to-host.md` and `to-host-chn.md` are twins — never update one without the other.
4. Plans in `1plans/` are written once and frozen; deviations are recorded in the living
   docs, not by editing the plan.
5. Big artifacts (datasets, ckpts, oracles) live under `/home/kaiwen/data/fast-ddrive/`
   and on GCS — docs reference them by path; nothing heavy in git.
