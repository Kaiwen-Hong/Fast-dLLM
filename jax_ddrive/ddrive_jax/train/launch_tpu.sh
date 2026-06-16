#!/usr/bin/env bash
# Multi-host TPU launch template for the Fast-dDrive FSDP harness (Phase 3).
#
# STATUS: TEMPLATE — NOT yet run on a real TPU pod (no TPU on the dev box). The harness
# mechanics are verified on CPU 8-device emulation (tests/test_harness_fsdp.py: FSDP-vs-1device
# parity 9.5e-7, ckpt resume 0.0). Two things must be done before this trains the REAL model:
#   1. wire the real-mode ViT image-embeds precompute in train_tpu.py (_real_image_embeds_fn is
#      currently a stub; mirror train_waymo_sasd_jax.static_tensors -> ie = vit(pv,grid); doubled).
#   2. point --parquet_dir at the dataset on GCS (gsutil rsync the HF/local Parquet to gs://).
#
# Mesh: pure-FSDP (n_fsdp = total chips, n_tp = 1) scales the 3.086B model fine; add TP later.
# jax.distributed.initialize() is auto-called on the pod via dist.init_distributed() (TPU_WORKER_ID).
set -euo pipefail

# ---- configure these -------------------------------------------------------------------
PROJECT="${PROJECT:-your-gcp-project}"
ZONE="${ZONE:-us-central2-b}"
TPU_NAME="${TPU_NAME:-ddrive-fsdp}"
ACCEL="${ACCEL:-v5litepod-256}"          # v5e-256 (64 hosts x 4 chips) or a v6e slice
RUNTIME="${RUNTIME:-v2-alpha-tpuv5-lite}"
DATA_GCS="${DATA_GCS:-gs://your-bucket/wod_e2e_sasd_50k}"     # Parquet shards (gsutil rsync first)
CKPT_GCS="${CKPT_GCS:-gs://your-bucket/ckpt/ddrive_fsdp}"
N_FSDP="${N_FSDP:-256}"                   # = total chips in the slice
BATCH="${BATCH:-256}"                     # global per-host batch * hosts; multiple of N_FSDP
STEPS="${STEPS:-2000}"
REPO_TARBALL="${REPO_TARBALL:-jax_ddrive.tar.gz}"   # tar of jax_ddrive/ staged to each worker

# ---- 1. provision (queued resource) ----------------------------------------------------
gcloud compute tpus queued-resources create "$TPU_NAME" \
  --node-id="$TPU_NAME" --project="$PROJECT" --zone="$ZONE" \
  --accelerator-type="$ACCEL" --runtime-version="$RUNTIME"
# (wait until ACTIVE: gcloud compute tpus queued-resources describe "$TPU_NAME" ...)

# ---- 2. deps on all workers ------------------------------------------------------------
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command "
  pip install -q -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install -q flax optax orbax-checkpoint grain pyarrow transformers pillow numpy
"

# ---- 3. stage code (+ model snapshot if not already on each worker) ---------------------
gcloud compute tpus tpu-vm scp "$REPO_TARBALL" "$TPU_NAME":~/ --zone="$ZONE" --worker=all
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command "
  mkdir -p ~/run && tar xzf ~/$REPO_TARBALL -C ~/run
"

# ---- 4. run training on ALL workers (jax.distributed auto-inits on the pod) -------------
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" --worker=all --command "
  cd ~/run && PYTHONPATH=\$PWD python ddrive_jax/train/train_tpu.py \
    --parquet_dir '$DATA_GCS' --split train \
    --n_fsdp $N_FSDP --n_tp 1 --batch $BATCH --steps $STEPS \
    --opt adamw --lr 1e-4 --bf16 \
    --ckpt_dir '$CKPT_GCS' --save_every 200 --keep 3
"
# NOTE: omit --proxy for the real model. On a pod, FSDP shards params+optimizer ~1/N_FSDP, so
# full fine-tune + AdamW is affordable (no LoRA/remat needed; keep remat as a long-seq knob).

# ---- 5. teardown -----------------------------------------------------------------------
# gcloud compute tpus queued-resources delete "$TPU_NAME" --project="$PROJECT" --zone="$ZONE" --force
