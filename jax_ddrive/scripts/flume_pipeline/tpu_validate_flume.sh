#!/usr/bin/env bash
# TPU validation of the NEW flume_pipeline dataloader on a REAL v6e-1.
#  (1) TRAINING: real Fast-dDrive 3B + frozen ViT consumes the new-dataloader batches ->
#      1-step loss+logits (both samples, bs=1) + 10 optimizer steps, on TPU silicon.
#  (2) INFERENCE: jax_batch_inference (Fast-dDrive release ckpt) on the 2 examples -> trajectory.
# Batches/prep are pre-materialized by the new msgpack dataloader (deterministic), so the TPU
# venv needs NO torch/grain/msgpack — just jax[tpu]+flax+transformers (same as tpu_mm_validate).
# Reuses the established free-trial queued-resource provisioning + auto-reaper (money-safe).
#
# Run:  bash tpu_validate_flume.sh
# Teardown any time: gcloud compute tpus queued-resources delete flume-val --zone=<z> --quiet --force
set -uo pipefail
PROJECT=project-8a53f5ab-2ea2-4892-a78
RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}"
ACCEL="${ACCEL:-v6e-1}"                  # 32GB HBM: bf16 3B + 2L logits fit
NAME="${NAME:-flume-val}"
ZONES=(${ZONES_OVERRIDE:-us-east5-a us-east5-b us-east5-c us-central1-a us-south1-a})
RETRY_DEADLINE=$(( $(date +%s) + ${RETRY_H:-2}*3600 ))
REAP_H="${DEADLINE_H:-3}"
POLL_ITERS="${POLL_ITERS:-60}"           # 20min patient wait per zone (QR free until ACTIVE)
SRC=gs://${PROJECT}-ddrive-sasd
FL=$SRC/flume
W=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/flume_pipeline
REPO=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive
DATA=/home/kaiwen/data/flume_pipeline
LOCALOUT=$DATA/tpu_results
mkdir -p "$LOCALOUT"
LOG=$W/tpu_flume.log; : > "$LOG"
say(){ echo "$(date +%T) $*" | tee -a "$LOG"; }
del_qr(){ gcloud compute tpus queued-resources delete "$NAME" --zone="$1" --quiet --force >/dev/null 2>&1 || true; }

# ---------------- PRE: stage code + batches + prep to GCS (local, no TPU) ----------------
prestage(){
  say "PRE: building code tgz + uploading artifacts to $FL ..."
  local tgz=/tmp/flume_jax_ddrive.tgz
  tar czf "$tgz" -C /home/kaiwen/Desktop/research/Fast-dLLM \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='jax_ddrive/docs' \
      --exclude='.git' jax_ddrive
  gsutil -q cp "$tgz" "$FL/code/flume_jax_ddrive.tgz"
  gsutil -m -q rsync -r "$DATA/batches_samplejson"     "$FL/batches_samplejson"
  gsutil -m -q rsync -r "$DATA/raw_track/batches"      "$FL/batches_raw"
  gsutil -m -q rsync -r "$DATA/infer_samplejson/prep"  "$FL/prep_samplejson"
  say "PRE done."
}

try_zone(){
  local z=$1 st
  gcloud compute tpus queued-resources create "$NAME" --node-id="$NAME" --zone="$z" \
     --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" >>"$LOG" 2>&1 \
     || { say "  [$z] create rejected"; del_qr "$z"; return 1; }
  for i in $(seq 1 "$POLL_ITERS"); do
    st=$(gcloud compute tpus queued-resources describe "$NAME" --zone="$z" --format="value(state.state)" 2>/dev/null)
    [ "$st" = ACTIVE ] && { say "  [$z] ACTIVE (~$((i*20))s)"; return 0; }
    case "$st" in FAILED|SUSPENDED) say "  [$z] $st"; del_qr "$z"; return 1;; esac
    [ $((i % 9)) -eq 0 ] && say "  [$z] ${st:-none} (~$((i*20))s)"
    sleep 20
  done
  say "  [$z] timeout -> next"; del_qr "$z"; return 1
}

arm_reaper(){
  local z=$1; echo $(( $(date +%s) + REAP_H*3600 )) > /tmp/tpu_reaper_${NAME}.deadline
  cat > $W/tpu_flume_reaper.sh <<RPR
#!/usr/bin/env bash
while true; do
  d=\$(cat /tmp/tpu_reaper_${NAME}.deadline 2>/dev/null || echo 0); [ "\$d" = 0 ] && exit 0
  if [ "\$(date +%s)" -ge "\$d" ]; then
    gcloud compute tpus queued-resources delete "${NAME}" --zone="${z}" --quiet --force >> $W/tpu_flume_reaper.log 2>&1
    rm -f /tmp/tpu_reaper_${NAME}.deadline; exit 0
  fi; sleep 60
done
RPR
  nohup bash $W/tpu_flume_reaper.sh >/dev/null 2>&1 &
  say "reaper armed (+${REAP_H}h) zone $z (pid $!)"
}

stage(){
  local z=$1
  say "stage on VM @ $z (timeout 40m) ..."
  timeout 2400 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$z" --worker=all --command '
    set -e; cd ~
    gsutil -q cp '"$FL"'/code/flume_jax_ddrive.tgz . && rm -rf ~/flume && mkdir -p ~/flume && tar xzf flume_jax_ddrive.tgz -C ~/flume
    mkdir -p ~/flume/batches_samplejson ~/flume/batches_raw ~/flume/prep_samplejson ~/flume/out
    gsutil -m -q rsync -r '"$FL"'/batches_samplejson ~/flume/batches_samplejson
    gsutil -m -q rsync -r '"$FL"'/batches_raw        ~/flume/batches_raw
    gsutil -m -q rsync -r '"$FL"'/prep_samplejson    ~/flume/prep_samplejson
    rm -rf ~/release_snapshot && mkdir -p ~/release_snapshot && gsutil -m -q rsync -r '"$SRC"'/release_fast_ddrive_snapshot ~/release_snapshot
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"; [ -d ~/venv ] || uv venv -p 3.11 ~/venv; source ~/venv/bin/activate
    uv pip install -q "jax[tpu]==0.10.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
        flax==0.12.7 transformers ml_dtypes safetensors numpy pyarrow jaxtyping optax grain array_record
    # grain+array_record: train_jax_driver imports train_tpu -> grain_pipeline (import grain) at
    # module load, even though batches are pre-materialized and the loader is not run on TPU.
    python -c "import jax; print(\"TPU_DEVICES\", jax.devices())"; echo WORKER_STAGE_OK
  ' >>"$LOG" 2>&1 && grep -q WORKER_STAGE_OK "$LOG"
}

run_validate(){
  local z=$1
  say "RUN flume validation on TPU @ $z (timeout 50m) ..."
  timeout 3000 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$z" --worker=all --command '
    set -uo pipefail
    source ~/venv/bin/activate; export PATH="$HOME/.local/bin:$PATH"
    export FASTDDRIVE_REPO=$HOME/flume/jax_ddrive FASTDDRIVE_SNAP=$HOME/release_snapshot
    export PYTHONPATH=$HOME/flume/jax_ddrive
    FP=$HOME/flume/jax_ddrive/scripts/flume_pipeline
    echo "===== TPU TRAIN (sample.json pair): 1-step both + 10-step ====="
    python $FP/train_jax_driver.py --batch_dir ~/flume/batches_samplejson \
      --out_dir ~/flume/out/train_samplejson --opt adafactor --lr 2e-5 --steps 10 --bf16 2>&1 \
      | grep -E "\[1-step\]|\[train\]|\[post-10\]|TRAIN_JAX_DRIVER_DONE|Error|error" || true
    echo "===== TPU TRAIN (raw-proto pair): 1-step both + 2-step ====="
    python $FP/train_jax_driver.py --batch_dir ~/flume/batches_raw \
      --out_dir ~/flume/out/train_raw --opt adafactor --lr 2e-5 --steps 10 --bf16 2>&1 \
      | grep -E "\[1-step\]|\[train\]|\[post-10\]|TRAIN_JAX_DRIVER_DONE|Error|error" || true
    echo "===== TPU INFERENCE (Fast-dDrive ckpt) on the 2 examples ====="
    python $HOME/flume/jax_ddrive/eval/jax_batch_inference.py \
      --prep_dir ~/flume/prep_samplejson --out_dir ~/flume/out/infer_samplejson 2>&1 \
      | grep -E "valid trajectories|JAX_BATCH_INFERENCE_DONE|Error|error" || true
    gsutil -m -q cp -r ~/flume/out/* '"$FL"'/results/ 2>/dev/null || true
    echo TPU_FLUME_VALIDATE_RAN
  ' 2>&1 | tee -a "$LOG"
  grep -q TPU_FLUME_VALIDATE_RAN "$LOG"
}

prestage
say "provision retry: zones=${ZONES[*]} accel=$ACCEL until +${RETRY_H:-2}h"
round=0
while [ "$(date +%s)" -lt "$RETRY_DEADLINE" ]; do
  round=$((round+1)); say "=== round $round ==="
  for z in "${ZONES[@]}"; do
    say " try $z"
    if try_zone "$z"; then
      arm_reaper "$z"
      if ! stage "$z"; then say "STAGE_FAILED @ $z"; tail -25 "$LOG"; exit 3; fi
      run_validate "$z" || say "RUN non-zero (see log)"
      gsutil -m -q cp -r "$FL/results/*" "$LOCALOUT/" 2>/dev/null || true
      say "TPU_FLUME_VALIDATE_DONE zone=$z (results -> $LOCALOUT and $FL/results ; reaper deletes in +${REAP_H}h)"
      del_qr "$z"; rm -f /tmp/tpu_reaper_${NAME}.deadline   # explicit teardown now that we're done
      say "TPU torn down."
      exit 0
    fi
    [ "$(date +%s)" -ge "$RETRY_DEADLINE" ] && break
  done
  say "round $round: no capacity; backoff 8m"; sleep 480
done
say "TPU_NO_CAPACITY (no $ACCEL within +${RETRY_H:-2}h)"; exit 2
