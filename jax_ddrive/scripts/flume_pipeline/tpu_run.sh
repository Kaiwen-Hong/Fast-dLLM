#!/usr/bin/env bash
# Reusable v6e-1 TPU runner for the flume_pipeline deliverable.
# Provisions a v6e-1 (queued-resource, cross-zone retry, auto-reaper), stages code + a JOB_DIR
# (containing batches/ and/or prep/) + the Fast-dDrive release snapshot, runs training and/or
# inference of the NEW dataloader's outputs, ships results to GCS, then tears down. Money-safe.
#
# Config via env:
#   JOB_DIR   local dir with batches/ (train) and/or prep/ (infer)   [required]
#   GCS_OUT   gs:// prefix to write results to                       [required]
#   MODE      train | infer | both                                   [default both]
#   STEPS     training steps                                         [default 10]
#   NAME      TPU/queued-resource name                               [default flume-ex]
set -uo pipefail
PROJECT=project-8a53f5ab-2ea2-4892-a78
RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}"; ACCEL="${ACCEL:-v6e-1}"
NAME="${NAME:-flume-ex}"; MODE="${MODE:-both}"; STEPS="${STEPS:-10}"
JOB_DIR="${JOB_DIR:?set JOB_DIR}"; GCS_OUT="${GCS_OUT:?set GCS_OUT}"
ZONES=(${ZONES_OVERRIDE:-us-east5-a us-east5-b us-east5-c us-central1-a us-south1-a})
RETRY_DEADLINE=$(( $(date +%s) + ${RETRY_H:-2}*3600 )); REAP_H="${DEADLINE_H:-3}"; POLL_ITERS="${POLL_ITERS:-60}"
SRC=gs://${PROJECT}-ddrive-sasd; FL=$SRC/flume
W=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/flume_pipeline
LOG=$W/tpu_run_${NAME}.log; : > "$LOG"
say(){ echo "$(date +%T) $*" | tee -a "$LOG"; }
del_qr(){ gcloud compute tpus queued-resources delete "$NAME" --zone="$1" --quiet --force >/dev/null 2>&1 || true; }

say "PRE: upload code tgz + JOB_DIR($JOB_DIR) -> $FL/jobs/$NAME"
tar czf /tmp/flume_${NAME}.tgz -C /home/kaiwen/Desktop/research/Fast-dLLM --exclude='__pycache__' --exclude='*.pyc' --exclude='*.log' --exclude='jax_ddrive/docs' --exclude='.git' jax_ddrive
gsutil -q cp /tmp/flume_${NAME}.tgz "$FL/jobs/$NAME/code.tgz"
[ -d "$JOB_DIR/batches" ] && gsutil -m -q rsync -r "$JOB_DIR/batches" "$FL/jobs/$NAME/batches"
[ -d "$JOB_DIR/prep" ]    && gsutil -m -q rsync -r "$JOB_DIR/prep"    "$FL/jobs/$NAME/prep"

try_zone(){ local z=$1 st
  gcloud compute tpus queued-resources create "$NAME" --node-id="$NAME" --zone="$z" --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" >>"$LOG" 2>&1 || { say "  [$z] create rejected"; del_qr "$z"; return 1; }
  for i in $(seq 1 "$POLL_ITERS"); do st=$(gcloud compute tpus queued-resources describe "$NAME" --zone="$z" --format="value(state.state)" 2>/dev/null)
    [ "$st" = ACTIVE ] && { say "  [$z] ACTIVE (~$((i*20))s)"; return 0; }
    case "$st" in FAILED|SUSPENDED) say "  [$z] $st"; del_qr "$z"; return 1;; esac; sleep 20; done
  say "  [$z] timeout"; del_qr "$z"; return 1; }

arm_reaper(){ local z=$1; echo $(( $(date +%s) + REAP_H*3600 )) > /tmp/tpu_reaper_${NAME}.deadline
  cat > $W/reaper_${NAME}.sh <<RPR
#!/usr/bin/env bash
while true; do d=\$(cat /tmp/tpu_reaper_${NAME}.deadline 2>/dev/null || echo 0); [ "\$d" = 0 ] && exit 0
  if [ "\$(date +%s)" -ge "\$d" ]; then gcloud compute tpus queued-resources delete "${NAME}" --zone="${z}" --quiet --force >/dev/null 2>&1; rm -f /tmp/tpu_reaper_${NAME}.deadline; exit 0; fi; sleep 60; done
RPR
  nohup bash $W/reaper_${NAME}.sh >/dev/null 2>&1 &  say "reaper armed (+${REAP_H}h) @ $z"; }

stage(){ local z=$1; say "stage @ $z (40m) ..."
  timeout 2400 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$z" --worker=all --command '
    set -e; cd ~; gsutil -q cp '"$FL"'/jobs/'"$NAME"'/code.tgz . && rm -rf ~/j && mkdir -p ~/j && tar xzf code.tgz -C ~/j
    mkdir -p ~/j/batches ~/j/prep ~/j/out
    gsutil -m -q rsync -r '"$FL"'/jobs/'"$NAME"'/batches ~/j/batches 2>/dev/null || true
    gsutil -m -q rsync -r '"$FL"'/jobs/'"$NAME"'/prep ~/j/prep 2>/dev/null || true
    rm -rf ~/snap && mkdir -p ~/snap && gsutil -m -q rsync -r '"$SRC"'/release_fast_ddrive_snapshot ~/snap
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
    [ -d ~/venv ] || uv venv -p 3.11 ~/venv; source ~/venv/bin/activate
    uv pip install -q "jax[tpu]==0.10.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html flax==0.12.7 transformers ml_dtypes safetensors numpy pyarrow jaxtyping optax grain array_record
    python -c "import jax; print(\"TPU\", jax.devices())"; echo STAGE_OK
  ' >>"$LOG" 2>&1 && grep -q STAGE_OK "$LOG"; }

run(){ local z=$1; say "RUN mode=$MODE steps=$STEPS @ $z ..."
  timeout 3000 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$z" --worker=all --command '
    set -uo pipefail; source ~/venv/bin/activate; export PATH="$HOME/.local/bin:$PATH"
    export FASTDDRIVE_REPO=$HOME/j/jax_ddrive FASTDDRIVE_SNAP=$HOME/snap PYTHONPATH=$HOME/j/jax_ddrive
    FP=$HOME/j/jax_ddrive/scripts/flume_pipeline
    if [ "'"$MODE"'" = train ] || [ "'"$MODE"'" = both ]; then echo "===TRAIN==="
      python $FP/train_jax_driver.py --batch_dir ~/j/batches --out_dir ~/j/out/train --opt adafactor --lr 2e-5 --steps '"$STEPS"' --bf16 2>&1 | grep -E "\[1-step\]|\[train\]|\[post|TRAIN_JAX_DRIVER_DONE|Error" || true; fi
    if [ "'"$MODE"'" = infer ] || [ "'"$MODE"'" = both ]; then echo "===INFER==="
      python $HOME/j/jax_ddrive/eval/jax_batch_inference.py --prep_dir ~/j/prep --out_dir ~/j/out/infer 2>&1 | grep -E "valid trajectories|JAX_BATCH_INFERENCE_DONE|Error" || true; fi
    gsutil -m -q cp -r ~/j/out/* '"$GCS_OUT"'/ 2>/dev/null || true; echo RUN_DONE
  ' 2>&1 | tee -a "$LOG"; grep -q RUN_DONE "$LOG"; }

while [ "$(date +%s)" -lt "$RETRY_DEADLINE" ]; do
  for z in "${ZONES[@]}"; do say "try $z"
    if try_zone "$z"; then arm_reaper "$z"
      if ! stage "$z"; then say "STAGE_FAILED"; tail -20 "$LOG"; del_qr "$z"; exit 3; fi
      run "$z" || say "RUN non-zero"
      gsutil -m -q cp -r "$GCS_OUT/*" "$JOB_DIR/tpu_out/" 2>/dev/null || true
      say "TPU_RUN_DONE @ $z (results -> $GCS_OUT and $JOB_DIR/tpu_out)"
      del_qr "$z"; rm -f /tmp/tpu_reaper_${NAME}.deadline; say "torn down"; exit 0
    fi; [ "$(date +%s)" -ge "$RETRY_DEADLINE" ] && break; done
  say "no capacity; backoff 8m"; sleep 480; done
say "TPU_NO_CAPACITY"; exit 2
