# Fast-dDrive JAX — small-scale 2-node TPU validation (handoff runbook)

**Purpose.** Run the existing FSDP harness (`ddrive_jax/train/train_tpu.py`) on a **real 2-host TPU
slice** to close the only gaps that can't be tested locally: multi-device physical sharding of the
real harness, **cross-host `jax.distributed`**, and TPU XLA numerics. Stay **within the $300 GCP
free-trial credit** (hard cap — do not burn extra money).

> **Status when this was written (2026-06-06):** account + quota verified, **local validation done**
> (see §3). **No TPU has been launched yet.** This doc is the execution plan for the next session.
> Target shape chosen by the user: **`v5litepod-16` (2 hosts × 8 v5e chips), `--proxy` model.**

This is a focused companion to `02_tpu_plan.md` (sharding theory), `03_scaleup_tpu_spec.md` (full
scale-up), and `to-host.md` §6 (deploy). Update `to-host.md` §6 once this run passes.

---

## 1. Locked decisions

| # | Decision | Value |
|---|---|---|
| Shape | true 2-node slice | **`v5litepod-16`** = 2 VMs × 8 chips (mesh `(16,1)` pure-FSDP) |
| Model | mechanics smoke, no 6 GB download | **`--proxy`** (small Qwen config, FULL vocab 151936) |
| Provisioning | matches `launch_tpu.sh` template | **queued resources** + `--worker=all` |
| Zone | v5e availability (verified) | **`us-east5-a`** (fallback `europe-west4-b`) |
| Account | personal gmail (NOT the .edu) | **`robosuite1998@gmail.com`** — stay on **Free Trial** (hard cap) |
| Cost | one ~15–20 min run | **~$5–7 on-demand** (`v5litepod-16` ≈ $19/hr); credit barely touched |

Why proxy and not the real 3.09B model: the real model trains fine through this exact driver on the
5090 (§3); the *only* new thing the TPU adds is cross-host + multi-chip mechanics, which the proxy
exercises identically while skipping the 6 GB snapshot staging. A real-model TPU run is a later step.

---

## 2. Verified GCP account facts (don't re-discover these)

Established this session via read-only `gcloud` (active account already authed on the 5090 box):

| Fact | Value |
|---|---|
| Login with the $300 credit | **`robosuite1998@gmail.com`** (`gcloud config set account` to switch) |
| Project | **`project-8a53f5ab-2ea2-4892-a78`** (proj #`23815087907`, "My First Project") |
| Billing account | **`01A61E-C8CE7D-05C484`** — `OPEN: true`, `billingEnabled: true` |
| APIs | `compute.googleapis.com` + `tpu.googleapis.com` **already enabled** |
| **v5e quota** | **`tpu-v5s-litepod` effectiveLimit = 16 chips/zone** (on-demand AND preemptible) — **no request needed** |
| v6e quota | `tpu-v6e` = 16 chips/zone (alt) |
| queued-resources | 5 per zone |
| v5e zones offered | `us-east5-a`, `us-east5-b/c`, `europe-west4-a/b`, `us-west4-b`, … |

> ⚠️ Do NOT use `kaiwen2@illinois.edu` — it's UIUC-org-managed and **cannot start the individual
> $300 trial** (org policy). `kaiwenh.17@gmail.com` only has **closed/expired** billing.
> ⚠️ **Open question to confirm before spending:** glance at
> <https://console.cloud.google.com/billing> and verify the credit is really **~$300, "Free trial",
> N days left**. Quota=16 ≠ credit present.
> ⚠️ Quota 16 means *creation is allowed up to 16 chips*; it does NOT 100% guarantee a trial account
> can launch (rare extra trial gates). That's exactly why §5 step B is a $0.3 probe **first**.

Re-pull the quota any time:
```bash
gcloud config set account robosuite1998@gmail.com
gcloud config set project project-8a53f5ab-2ea2-4892-a78
gcloud alpha services quota list --service=tpu.googleapis.com \
  --consumer=projects/23815087907 2>/dev/null \
  | grep -iA6 'tpu-v5s-litepod' | grep -E 'metric:|effectiveLimit'
```

---

## 3. What is ALREADY validated locally (do not redo)

All free; on the 5090 + CPU 8-device emulation. The whole production code path works at mesh=1, so the
TPU run is only about *multi-host + multi-chip*.

| Check | How | Result |
|---|---|---|
| Real driver `train_tpu.py main()` (real 3.75B + ViT + grain + Orbax), real mode | 5090, `--n_fsdp 1 --batch 1 --opt adafactor --bf16` | 6 steps 0.985→0.638, no NaN, ~15 s/step |
| Checkpoint **save → restore → resume** on the REAL model | 5090, 2nd invocation | `restored step=6` → ran 7–10, clean |
| grain multi-batch loop on real model | 5090 | ✅ (distinct samples each step) |
| FSDP math `(2,1)` vs `(1,1)` | CPU-8 emulation, `tests/test_harness_fsdp.py` | `GRAIN/HARNESS_FSDP_TESTS_PASS`, parity **9.5e-7**, ckpt-resume **0.0** |
| grain determinism / disjoint sharding / resume | CPU-8, `tests/test_grain_pipeline.py` | `GRAIN_PIPELINE_TESTS_PASS` |

GPU run env (for reference / a real-model TPU run later):
```bash
unset LD_LIBRARY_PATH; source jax_ddrive/scripts/jax_gpu_env.sh     # $JAXPY binds the 5090
export PYTHONPATH=$PWD/jax_ddrive XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# do NOT set XLA_PYTHON_CLIENT_PREALLOCATE=false -> fragmentation OOM on the 8.73 GB backward alloc
```
Real-model checkpoint ≈ **5.2 GB** each (matters for GCS time/cost on a real-model run).

---

## 4. Gaps / risks the TPU run must handle (READ THIS — saves the next session hours)

1. **🔴 Multi-host data feeding is the #1 likely failure.** `train_tpu.run_step` does
   `jax.device_put(host_local_batch, global_data_sharding)`. On the single-process CPU-8 emulation all
   8 devices are local so this works — but on a **true 2-host slice each process only addresses its own
   8 of the 16 devices**, so `device_put` of a host-local array onto a *global* `(16,1)` sharding is
   likely wrong/raises. The grain loader already yields **per-host shards** (`process_index/count`
   slicing), so the fix is to assemble a global array from process-local data:
   replace the `device_put` in `run_step` with
   `jax.make_array_from_process_local_data(sharding, local_np_array)` for each batch leaf (one global
   array of shape `[batch*process_count, ...]`). Expect to apply this; it's the main code change.
2. **grain reads LOCAL paths, not `gs://`.** `parquet_dataset.shard_paths` uses `glob.glob`. The
   400-sample Parquet (175 MB) must sit on **each worker's local disk**; `--parquet_dir` points there.
   The runbook stages it via `scp --worker=all`. (Do NOT pass a `gs://` parquet_dir — it globs to [].)
3. **Multi-host Orbax must checkpoint to `gs://`.** A local `--ckpt_dir` can't coordinate across 2
   VMs. Use a bucket (runbook creates one). Or skip saving with `--save_every 99999` if you only want
   the forward/backward smoke.
4. **Force `jax.distributed` init.** `dist.init_distributed()` only calls `jax.distributed.initialize()`
   if `DDRIVE_MULTIHOST=1` or a TPU pod env var is present. **Set `DDRIVE_MULTIHOST=1`** in the run
   command so both hosts join one job (else each host runs independently = wrong). On TPU,
   `initialize()` auto-detects topology (no coordinator args needed).
5. **`launch_tpu.sh` is the WRONG size as-is**: it targets `v5litepod-256` (~$300/h) and its **teardown
   is commented out**. Use the runbook in §5 instead (small slice + `trap` auto-delete). Its
   `_real_image_embeds_fn is a stub` comment is stale (it's implemented).
6. Numerics: proxy runs **fp32** (default) → TPU loss should track the CPU-emulation proxy closely.
   For a real-model TPU run, keep loss/log-softmax fp32 (already upcast in `sasd_loss.py`); bf16 params ok.

---

## 5. Runbook (copy-paste; nothing bills until step C)

```bash
# ---- 0. context ----
gcloud config set account robosuite1998@gmail.com
gcloud config set project project-8a53f5ab-2ea2-4892-a78
PROJECT=project-8a53f5ab-2ea2-4892-a78
BILLING=01A61E-C8CE7D-05C484
ZONE=us-east5-a
ACCEL=v5litepod-16
RUNTIME=v2-alpha-tpuv5-lite
QR=ddrive-smoke
REPO=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
DATA=/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd      # 7 parquet shards, L=1184
BUCKET=gs://${PROJECT}-ddrive-smoke                      # globally-unique; created in step C
```

### A. Budget alert (free, do once)
```bash
gcloud services enable billingbudgets.googleapis.com
gcloud billing budgets create --billing-account=$BILLING \
  --display-name="ddrive-tpu-smoke-cap-50" --budget-amount=50USD \
  --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
```

### B. De-risk probe — confirm a trial account can actually launch a TPU (~$0.04, ~2 min)
```bash
gcloud compute tpus tpu-vm create ddrive-probe --zone=$ZONE \
  --accelerator-type=v5litepod-1 --version=$RUNTIME    # succeeds in ~60-90s if trial can launch
gcloud compute tpus tpu-vm delete ddrive-probe --zone=$ZONE --quiet   # DELETE IMMEDIATELY
```
If create fails with a quota/permission/trial error → stop and resolve before the 16-chip slice
(no charge for a failed create). If it succeeds + deletes cleanly → proceed.

### C. The 2-host validation (auto-teardown script; ~$5–7, ~15–20 min)
Save as `jax_ddrive/scripts/launch_tpu_smoke.sh` and run it. The `trap` deletes the slice on ANY exit.
```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT=project-8a53f5ab-2ea2-4892-a78; ZONE=us-east5-a
ACCEL=v5litepod-16; RUNTIME=v2-alpha-tpuv5-lite; QR=ddrive-smoke
REPO=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
DATA=/home/kaiwen/data/fast-ddrive/hf/wod_e2e_sasd
BUCKET=gs://${PROJECT}-ddrive-smoke

cleanup(){ echo ">>> CLEANUP deleting $QR"; gcloud compute tpus queued-resources delete "$QR" \
           --zone="$ZONE" --quiet --force 2>/dev/null || true; }
trap cleanup EXIT INT TERM

gsutil ls "$BUCKET" 2>/dev/null || gsutil mb -l us-east5 "$BUCKET"

# 1. provision 2-host slice
gcloud compute tpus queued-resources create "$QR" --node-id="$QR" --zone="$ZONE" \
  --accelerator-type="$ACCEL" --runtime-version="$RUNTIME"
# 2. wait for ACTIVE (FAILED -> capacity/quota; try europe-west4-b or v5litepod-8)
until gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" \
        --format="value(state.state)" | grep -q ACTIVE; do
  s=$(gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" --format="value(state.state)")
  echo "state=$s ..."; [ "$s" = "FAILED" ] && { echo "PROVISION FAILED"; exit 1; }; sleep 20
done
# 3. stage code + LOCAL parquet (grain can't read gs://) to BOTH hosts, install deps
tar czf /tmp/ddrive_run.tgz  -C "$(dirname "$REPO")" jax_ddrive
tar czf /tmp/ddrive_data.tgz -C "$(dirname "$DATA")" "$(basename "$DATA")"
gcloud compute tpus tpu-vm scp /tmp/ddrive_run.tgz /tmp/ddrive_data.tgz "$QR":~ \
  --zone="$ZONE" --worker=all
gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --worker=all --command "
  mkdir -p ~/run ~/data && tar xzf ~/ddrive_run.tgz -C ~/run && tar xzf ~/ddrive_data.tgz -C ~/data
  pip install -q -U 'jax[tpu]==0.10.0' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install -q 'flax==0.12.7' 'optax==0.2.8' 'orbax-checkpoint==0.11.37' 'grain==0.2.16' pyarrow numpy
"
# 4. run the SAME driver on all workers; DDRIVE_MULTIHOST=1 forces jax.distributed
gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --worker=all --command "
  cd ~/run && PYTHONPATH=\$PWD DDRIVE_MULTIHOST=1 python ddrive_jax/train/train_tpu.py \
    --proxy --parquet_dir ~/data/wod_e2e_sasd --split train \
    --n_fsdp 16 --n_tp 1 --batch 16 --steps 10 --opt adamw --lr 1e-3 \
    --ckpt_dir '$BUCKET/ckpt' --save_every 5 --keep 2
"
echo ">>> RUN DONE (trap will delete the slice)"
```

### D. Teardown + verify-nothing-alive (run even if the script crashed)
```bash
# jax_ddrive/scripts/tpu_teardown.sh
gcloud compute tpus queued-resources delete ddrive-smoke --zone=us-east5-a --quiet --force || true
# jax_ddrive/scripts/tpu_check.sh  -> MUST both be empty afterwards
gcloud compute tpus queued-resources list --zone=us-east5-a
gcloud compute tpus tpu-vm list --zone=us-east5-a
# optional: drop the tiny bucket
gsutil -m rm -r gs://project-8a53f5ab-2ea2-4892-a78-ddrive-smoke || true
```

---

## 6. Acceptance criteria (what "passed" means)

From the harness stdout + a couple of checks, confirm:
- **Cross-host up:** the `[harness] ... devices=16 procs=2` line (i.e. `jax.device_count()==16`,
  `jax.process_count()==2`). This is the core new proof vs. local.
- **Runs:** 10 steps complete, all losses **finite, no NaN**.
- **FSDP real:** params carry a `'fsdp'` NamedSharding across the 16 chips (the harness already shards;
  `scripts/check_real_fsdp_shard.py` logic, or inspect `h.params` sharding).
- **Multi-host Orbax:** checkpoint written to `$BUCKET/ckpt` at step 5/10; a re-run `restored step=10`.
- Maps to `to-host.md` §7.2 ladder: this is **steps 1–2 (FSDP parity) + 4 (training runs)** on real TPU.
  Step 5 (ADE/RFS eval on TPU) and a real-model (non-proxy) run remain as later increments.

After it passes: update `to-host.md` §6 (replace the 256-chip template note with this verified
small-scale recipe + the §4 fixes), and `03_scaleup_tpu_spec.md` §4 (mark "TPU pod end-to-end" partial).

---

## 7. Money-safety checklist (the real risk is a forgotten VM)

- `v5litepod-16` bills **~$19/hr while ACTIVE** (a forgotten slice ≈ **$460/day**). The clock starts at
  ACTIVE and runs through staging+run; the `trap` in §5C deletes on any exit.
- After EVERY session touching TPUs, run `tpu_check.sh` (§5D) and confirm **both lists are empty**.
- Budget alert (§5A) emails at $25/$45/$50. Stay on **Free Trial** = Google won't charge the card
  (hard cap); do **not** "upgrade to full account" (that removes the cap) — quota=16 means you don't
  need to.
- Cheaper variants if desired: **spot** (`--accelerator-type` unchanged but use spot/preemptible flag;
  ~60% off, ~$2.5) accepting rare preemption; or **`v5litepod-8`** single host (~$2.5, but NOT a
  cross-host test); or **2×`v5litepod-1`** (~$0.7, true 2-node but needs manual coordinator wiring —
  not the queued-resource path above).
- $0 alternative for repeated runs: **TRC (TPU Research Cloud)** — free researcher TPUs, fits the
  `.edu` identity; apply separately if this becomes frequent.

---

## 8. Key code pointers

- Driver: `ddrive_jax/train/train_tpu.py` (`main()`, `run_step` ← apply §4.1 fix here), `build_harness`.
- Distributed init / mesh: `ddrive_jax/train/dist.py` (`init_distributed`, `DDRIVE_MULTIHOST`).
- Multi-host ckpt: `ddrive_jax/train/checkpoint_mgr.py` (Composite save: params+opt+meta+grain).
- Data: `ddrive_jax/data/grain_pipeline.py` (`make_sasd_loader`, per-host slice),
  `ddrive_jax/data/parquet_dataset.py` (`shard_paths` ← local-glob, §4.2).
- Sharding rule: `ddrive_jax/sharding.py` (`param_pspecs`, FSDP largest-axis), `models/sharded.py` (TP, later).
- Existing oversized template to replace: `ddrive_jax/train/launch_tpu.sh`.
- Env versions (from the verified jax venv): jax 0.10.0, flax 0.12.7, optax 0.2.8,
  orbax-checkpoint 0.11.37, grain 0.2.16, pyarrow.

*Generated 2026-06-06 from the local verification session; branch `jax-ddrive-port` (local only).*
