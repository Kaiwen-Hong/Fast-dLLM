# Overnight autonomous job — MaxText SASD on multi-node TPU (2026-06-07)

**Task (user, going to sleep):** use the GCP TPU credit (up to $300) to test the Fast-dDrive SASD
functionality **on multi-node TPU, using the MaxText framework**, autonomously overnight.

**Money rules I'm holding to:** spend only what the smokes need (target ≪ $300; a smoke never needs
that), every TPU is auto-deleted (`trap` + a cron **reaper** + the free-trial hard cap), and I
**stop + document rather than flail** if I hit a deployment wall. Account stays on Free Trial (hard
cap; Google won't charge the card).

This is the live report; updated as each step lands. Final summary + cost at the bottom.

---

## ☀️ MORNING SUMMARY (read this first)

**You asked:** test Fast-dDrive SASD on **multi-node TPU via MaxText**, overnight, ≤$300.

**Result: ✅ the MaxText SASD model TRAINS on real TPU (v6e), with real Fast-dDrive weights, at the
expected loss (~0.3–0.6) and 65 TFLOP/s/device.** The whole stack is proven on TPU end-to-end:
provisioning, install (uv + Python 3.11), frozen ViT (390 tensors), the Waymo SASD grain pipeline,
real-weight param restore from GCS, and the section-weighted SASD train step compiling on TPU XLA.

**The one thing I could NOT complete: the literal "multi-node" (≥2-host) run** [完成 2026-06-19: 多主机 v5e-16（4 hosts/16 chips）trainable-ViT 跑通了，run `be47jjta8`，loss 5.199→3.921→3.042，EXIT 0 — 见 `../1plans/06_trainable_vit_plan.md` §9 与 `HANDOFF.md` 的 2026-06-19 段；用的是另一份 free-credit 项目的 v5e-16 spot @ us-south1-a，绕开了本夜的容量墙] — because **GCP did not
allocate ≥8-chip TPU capacity to this trial account all night.** Every `v6e-8` / `v6e-16` / `v5e-16`
request returned no-capacity / "try again at a later time"; only single-chip (`v6e-1`) had capacity.
This is an **external, transient GCP capacity constraint — not a code/config/quota/permission issue.**
Everything needed for the multi-node run is built and verified; it's purely waiting on capacity.

**Armed for you:** `/home/kaiwen/mt_multinode_catcher.sh` is running (up to 6 h) and will fire the
multi-node run automatically if capacity opens. Or run it yourself when capacity is free:
`ACCEL=v6e-16 NAME=sasd-m16 bash /home/kaiwen/launch_maxtext_sasd_tpu.sh`

**Money:** ≈ **$5–8 spent** (small probes + single-chip runs + a brief debug VM). **0 TPUs left
running** (verified both zones). Reaper cron + $80 budget alert remain active. Nowhere near $300.

**Big surprise (good):** the MaxText SASD port **already existed** (a prior session, "phase 33") and
trains on GPU — so tonight was about getting it running on **TPU**, not building the port.

---

## TL;DR status (updated live)

| Step | State |
|---|---|
| Path-A multi-host data-feeding fix (`train_tpu.run_step`) + verify | ✅ done (CPU 2-process: `MULTIHOST_DATAFEED_TEST_PASS`) |
| **MaxText SASD port already exists** (prior session, phase 33) | ✅ discovered |
| MaxText SASD trains on GPU (gate) | ✅ re-confirmed: 12 steps, loss ~0.6–0.9, ckpt@11 (5.8 GiB), EXIT=0 |
| Money safety net (reaper cron + $80 budget alert) | ✅ installed |
| Stage artifacts → GCS (param ckpt, parquet, ViT, code) | ✅ done |
| TPU launch + install probe (v6e-1: trial-launch + uv/py3.11 + MaxText import) | ✅ `MAXTEXT_TPU_IMPORT_OK` |
| **MaxText SASD TRAINS on real TPU (v6e-1)** | ✅ 12 steps on `TFRT TPU v6 lite`, 3.086 B params, ViT loaded, finite loss, 65 TFLOP/s/device (random init; real-weight loss↓ shown on GPU; real-weight TPU run building now) |
| Single/multi-host SASD on TPU (≥8 chips) | ⏸ blocked **that night** by **trial TPU capacity** [✅ 完成 2026-06-19 on a separate free-credit project: multi-host v5e-16 / 4 hosts / 16 chips trainable-ViT train, loss 5.199→3.921→3.042, EXIT 0 — `../1plans/06_trainable_vit_plan.md` §9] — `mt_multinode_catcher.sh` ran its **full 6 h (14 retries) and capacity never opened**; ≥8-chip slices were unavailable all night, all zones. Multi-node is built+ready; needs capacity (paid account / reservation / TRC, or retry off-peak). Launch: `ACCEL=v6e-16 NAME=sasd-m16 bash /home/kaiwen/launch_maxtext_sasd_tpu.sh` |

---

## Major finding — the MaxText port is NOT "unstarted"

`to-host.md`/`03_scaleup_tpu_spec.md` said the MaxText port was Phase 7, unstarted. **It's actually
well underway** in `/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/` (a prior agent session, through
"phase 33"):

- `src/maxtext/diffusion/sasd.py`, `load_fast_ddrive_maxtext.py`, `sampler.py`, `mdlm.py` + a SASD
  test suite (`sasd_{parity,lossdecrease,train_step,vla_parity,weight_parity}_test.py`).
- `src/maxtext/input_pipeline/waymo_sasd_data_processing.py` — wraps **the same `ddrive_jax`
  `make_sasd_loader`** (per-host sharded) + frozen ViT + `prepare_sasd_inputs`, and assembles global
  arrays via MaxText's own `_form_global_array` (so **multi-host data feeding is already handled
  correctly** in the MaxText path).
- `configs/sasd_waymo.yml` + `configs/models/qwen2.5-3b.yml` + `scripts/train_sasd_waymo.sh`
  (the deployable entry; documents the TPU-pod reuse: `hardware=tpu`, `opt_type=adamw`, gs:// paths).
- The MaxText Orbax **param checkpoint is already saved** at
  `/home/kaiwen/data/fast-ddrive/maxtext_sasd_params/fast_ddrive_qwen25_3b_params`.
- **GPU smoke (task 23, 03:51 today) passed**: 12 steps through MaxText's standard loop, finite loss,
  ckpt + resume. I re-ran it tonight — green.

So the multi-node TPU run is genuinely within reach: the port runs locally; what's missing is exactly
the TPU execution (this job).

## Key facts for the run

- **Install:** MaxText is used via `PYTHONPATH=src` (not pip-installed). Deps come from the fork's
  `src/dependencies/requirements/generated_requirements/tpu-requirements.txt` + SASD extras
  (`safetensors pyarrow transformers`). Runs on Python 3.11 locally (so the `>=3.12` in pyproject is
  not enforced — TPU's 3.10/3.11 should work). `PYTHONPATH` must include **both** the fork `src` and
  `jax_ddrive` (the data path imports `ddrive_jax.*`).
- **Artifacts staged to GCS** (`gs://project-8a53f5ab-2ea2-4892-a78-ddrive-sasd`): `maxtext_sasd_params/`
  (param ckpt, load via `load_parameters_path=gs://...`), `wod_e2e_sasd/` (parquet → each worker local,
  grain globs local paths), `vit_snap.tgz` (ViT HF snapshot, deref'd → each worker local for on-the-fly
  image embeds), `code/{maxtext_fork,jax_ddrive}.tgz`.
- **Launch scripts** (in `/home/kaiwen/`): `tpu_install_probe.sh` (v5e-1 make-or-break),
  `launch_maxtext_sasd_tpu.sh` (`ACCEL=v5litepod-8|v5litepod-16` [SUPERSEDED later same night: switched to v6e / us-east5-a / runtime v2-alpha-tpuv6e — see Update log lines 111-118; canonical is `ACCEL=v6e-16`]), `tpu_reaper.sh` (cron safety net).
- **Account:** robosuite1998@gmail.com, project `project-8a53f5ab-2ea2-4892-a78`, v5e quota 16/zone,
  zone `us-east5-a`.

## Cost log
<!-- [SUPERSEDED: the v5litepod rows below are pre-switch placeholders; the actual probes/runs that night were v6e-1 / v6e-8 / v6e-16 in us-east5-a — see Update log lines 111-118, 130, 136. v5e @ us-east5-a was PERMISSION_DENIED; v6e was the permitted queue.] -->
| Run | Slice | Purpose | Cost (est.) |
|---|---|---|---|
| (pending) | v5litepod-1 | install/import probe | ~$0.5 |
| (pending) | v5litepod-8 | single-host SASD smoke | ~$2.5 |
| (pending) | v5litepod-16 | multi-node SASD (deliverable) | ~$7 |

## Safety
- Reaper: `*/10 * * * * bash /home/kaiwen/tpu_reaper.sh` — force-deletes any `ddrive/sasd/mt-*` TPU
  past `/tmp/tpu_reaper_deadline`. Each launch script writes the deadline before create + self-deletes
  via `trap`.
- Verify-nothing-alive: `gcloud compute tpus queued-resources list --zone=us-east5-a` and
  `gcloud compute tpus tpu-vm list --zone=us-east5-a` — both should be EMPTY by morning.

## Update log (live)

- **TPU provisioning access — important.** The trial account's quota shows 16 v5e/v6e chips, but
  there's a **separate per-(type,zone)-queue access gate**. Probed combos (failed submits are free):
  - `v5e @ us-east5-a` → ❌ `PERMISSION_DENIED` ("not permitted to submit into this queue").
  - `v5e @ europe-west4-b` → ✅ permitted.
  - **`v6e @ us-east5-a` → ✅ permitted** (chosen — same region as the GCS bucket, no cross-region egress).
  - Direct `gcloud compute tpus tpu-vm create` → ❌ "Reservation not found" for all; **must use
    queued-resources (CQR)**.
  - **Switched all scripts to `v6e` / `us-east5-a` / runtime `v2-alpha-tpuv6e`.** (v6e on-demand ~$2.7/chip-hr:
    probe v6e-1 ~$0.9, single-host v6e-8 ~$9, multi-node v6e-16 ~$21 — all ≪ $300.)
  - Reaper zone is `us-east5-a` (matches). If a run ever uses europe-west4-b, update the reaper zone.

- **TPU install chain — solved layer by layer (v6e-1 probe, ~$3 total across attempts):**
  1. ❌→✅ TPU VM's default compute SA (`23815087907-compute@…`) lacked GCS read → granted
     `roles/storage.objectAdmin` on the bucket (`gcloud storage buckets add-iam-policy-binding`).
  2. ❌→✅ TPU runtime ships **Python 3.10**, but MaxText `tpu-requirements` needs 3.11
     (`array-record>=0.8.3`). Fix: **`uv venv -p 3.11`** (the VM has `/usr/bin/python3.11`), install
     into that venv, run with `~/venv/bin/python`.
  3. ✅ **PROBE_PASS**: `DEVICES [TpuDevice(id=0…)]` (jax[tpu] sees the TPU) + MaxText/SASD/ddrive_jax
     all import. Install path is confirmed; baked into `launch_maxtext_sasd_tpu.sh`.
- **Single-host `v6e-8` SASD run launched** (NAME=sasd-h8) — provision + uv/py3.11 install + 12 steps,
  auto-delete. Tests real-TPU SASD training (multi-device batch=8, gs:// param restore, on-VM ViT)
  before the multi-node deliverable. (next: `v6e-16` multi-node.)

- **🚧 CAPACITY WALL (the real blocker tonight).** A **1-chip** v6e provisions instantly, but **multi-chip
  slices have no capacity** for this trial right now — confirmed across:
  - `v6e-8` on-demand @ us-east5-a → `WAITING_FOR_RESOURCES` (queued, never allocated).
  - `v6e-16` **spot** @ us-east5-a → `WAITING_FOR_RESOURCES` (no spot capacity).
  - `v5e-16` on-demand @ europe-west4-b → `code 8: Insufficient capacity. Try again ... at a later time.`
  This is an **external, transient GCP capacity constraint** (Google isn't allocating ≥8-chip TPU
  slices to this trial account at this hour), not a config/quota/permission issue. The error itself
  says "try again at a later time."
  - **Implication:** the true multi-node (≥2 host = 16-chip) run can't provision right now. Everything
    *else* for it is proven/ready (access, GCS, install, import, the SASD GPU train, the multi-host
    data path). The launch is one command once capacity frees up:
    `ACCEL=v6e-16 NAME=sasd-m16 bash /home/kaiwen/launch_maxtext_sasd_tpu.sh` (add `POOL=spot` or try
    `ZONE=europe-west4-b RUNTIME=v2-alpha-tpuv5-lite ACCEL=v5litepod-16`).
  - **Pivot:** prove SASD trains on the largest slice that *does* have capacity (trying `v6e-4`,
    single host, 4 chips — exercises real-TPU multi-device FSDP). Best achievable functional proof
    tonight; multi-node is gated only on capacity.
  - Even **`v6e-4` had no capacity** → only **1-chip (`v6e-1`)** is available. Running SASD on `v6e-1`
    (real-TPU XLA compile of the SASD graph; single chip, mesh=1 — the TPU analog of the GPU smoke).

- **TPU run config fix (found cheaply on `v6e-1`):** first `v6e-1` attempt **provisioned + installed
  (`WORKER_SETUP_OK`) + started MaxText**, then aborted on a config validation: `enable_checkpointing=false`
  (my smoke override) conflicts with `load_parameters_path` — MaxText needs `enable_checkpointing=True`
  to restore the param ckpt. Fixed in `launch_maxtext_sasd_tpu.sh` (drop the override, set
  `checkpoint_period=999999` so it loads params without saving mid-smoke). **This bug would have hit
  the multi-node run too** — caught for ~$1 instead of ~$21. Re-running v6e-1 to confirm the SASD
  train step compiles + runs on TPU.

- **Autonomous multi-node catcher** (`/home/kaiwen/mt_multinode_catcher.sh`): since capacity is
  transient, this retries `v6e-16` every ~20 min and runs the full multi-node SASD the moment capacity
  is granted, then stops (bounded to ONE provisioned run). Launches after the v6e-1 confirmation.

- **✅✅ SASD TRAINS ON REAL TPU — random init AND real weights:**
  - *Random init* (no restore): 12 steps on v6e, finite loss, 65 TFLOP/s — proves the SASD train step
    (forward + section-weighted loss + backward + optimizer) compiles + runs on TPU XLA.
  - *Real pretrained weights*: built the MaxText param ckpt **on the VM** from the on-disk snapshot
    (434/434 leaves, 3.086 B, bf16), loaded it, trained → **loss 0.308 / 0.556** at steps 10/11 —
    matching the GPU smoke (~0.6). **The Fast-dDrive SASD model trains on real TPU via MaxText with
    real weights and the expected loss.** This is the complete single-chip functional proof.
  - **GCS "incomplete checkpoint" root cause:** `gsutil rsync` of an Orbax OCDBT checkpoint produces a
    copy Orbax won't load (a copy artifact, not missing files). **Fix:** write the ckpt directly to GCS
    via Orbax (`save_fast_ddrive_params_ckpt.py out_ckpt_dir=gs://...`, run on the VM which has the SA's
    GCS access) → `maxtext_sasd_params_v2/`. Launch script now points there, so the multi-node catcher
    restores real weights cleanly.

### Bottom line (so far)
The **full MaxText SASD stack is proven on real TPU**: provision (v6e), uv/py3.11 install, ViT load,
data pipeline, param restore, and the SASD train step with real weights (loss ~0.3-0.56). The **only**
missing piece for the literal "multi-node" ask is **≥8-chip TPU capacity**, which Google did not
allocate to this trial tonight (every ≥4-chip request returned no-capacity / "try again later"). The
multi-node run is one command (`ACCEL=v6e-16 bash /home/kaiwen/launch_maxtext_sasd_tpu.sh`) and is
armed to fire automatically via the catcher if capacity opens up. Total spend so far ≈ a few dollars.
