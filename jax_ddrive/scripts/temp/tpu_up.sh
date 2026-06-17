#!/usr/bin/env bash
# Provision a free-trial v6e-1 with CROSS-ZONE + BACKOFF RETRY (GCP trial capacity is transient),
# then stage the eval_sasd parity run from GCS. fp32 needs v6e (32 GB HBM) so we only try v6e-1.
# On success: arms a detached reaper (force-deletes the QR after REAP_H h, survives session) and
# writes the winning zone to tpu_active_zone.txt (tpu_run.sh/tpu_down.sh read it). Failed attempts
# are deleted (zero cost). Gives up after RETRY_H hours -> TPU_NO_CAPACITY.
set -uo pipefail
PROJECT=project-8a53f5ab-2ea2-4892-a78
RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}"
ACCEL="${ACCEL:-v6e-1}"
NAME="${NAME:-sasd-parity}"
ZONES=(${ZONES_OVERRIDE:-us-east5-a us-east5-b us-east5-c us-east1-d us-central1-a us-south1-a})
RETRY_DEADLINE=$(( $(date +%s) + ${RETRY_H:-5}*3600 ))
REAP_H="${DEADLINE_H:-4}"
SRC=gs://${PROJECT}-ddrive-sasd
W=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/temp
LOG=$W/tpu_up.log; : > "$LOG"
ZFILE=$W/tpu_active_zone.txt; rm -f "$ZFILE"
say(){ echo "$(date +%T) $*" | tee -a "$LOG"; }

del_qr(){ gcloud compute tpus queued-resources delete "$NAME" --zone="$1" --quiet --force >/dev/null 2>&1 || true; }

try_zone(){
  local z=$1 st
  gcloud compute tpus queued-resources create "$NAME" --node-id="$NAME" --zone="$z" \
     --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" >>"$LOG" 2>&1 \
     || { say "  [$z] create rejected"; del_qr "$z"; return 1; }
  for i in $(seq 1 8); do
    st=$(gcloud compute tpus queued-resources describe "$NAME" --zone="$z" --format="value(state.state)" 2>/dev/null)
    [ "$st" = ACTIVE ] && { say "  [$z] ACTIVE"; return 0; }
    case "$st" in FAILED|SUSPENDED|"") say "  [$z] ${st:-none}"; del_qr "$z"; return 1;; esac
    sleep 20
  done
  say "  [$z] still ${st} after ~2.7m"; del_qr "$z"; return 1
}

arm_reaper(){
  local z=$1
  echo "$z" > "$ZFILE"
  echo $(( $(date +%s) + REAP_H*3600 )) > /tmp/tpu_reaper_${NAME}.deadline
  cat > $W/tpu_reaper_${NAME}.sh <<RPR
#!/usr/bin/env bash
while true; do
  d=\$(cat /tmp/tpu_reaper_${NAME}.deadline 2>/dev/null || echo 0); [ "\$d" = 0 ] && exit 0
  if [ "\$(date +%s)" -ge "\$d" ]; then
    echo "\$(date +%T) REAPER force-delete QR ${NAME} @ ${z}" >> $W/tpu_reaper.log
    gcloud compute tpus queued-resources delete "${NAME}" --zone="${z}" --quiet --force >> $W/tpu_reaper.log 2>&1
    rm -f /tmp/tpu_reaper_${NAME}.deadline; exit 0
  fi; sleep 60
done
RPR
  nohup bash $W/tpu_reaper_${NAME}.sh >/dev/null 2>&1 &
  say "reaper armed (+${REAP_H}h) for zone $z (pid $!)"
}

stage(){
  local z=$1
  say "stage on VM @ $z (model+code+npz+venv, timeout 35m) ..."
  timeout 2100 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$z" --worker=all --command '
    set -e; cd ~
    gsutil -q cp '"$SRC"'/parity/parity_code.tgz . && rm -rf ~/parity && mkdir -p ~/parity && tar xzf parity_code.tgz -C ~/parity
    gsutil -q cp '"$SRC"'/parity/parity_val_npz.tgz . && rm -rf ~/eval_inputs && mkdir -p ~/eval_inputs && tar xzf parity_val_npz.tgz -C ~/eval_inputs
    mkdir -p ~/release_snapshot && gsutil -m -q rsync -r '"$SRC"'/release_fast_ddrive_snapshot ~/release_snapshot
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"; [ -d ~/venv ] || uv venv -p 3.11 ~/venv; source ~/venv/bin/activate
    uv pip install -q "jax[tpu]==0.10.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html flax==0.12.7 transformers ml_dtypes safetensors numpy
    python -c "import jax; print(\"TPU_DEVICES\", jax.devices())"; echo WORKER_STAGE_OK
  ' >>"$LOG" 2>&1 && grep -q WORKER_STAGE_OK "$LOG"
}

say "provision retry: zones=${ZONES[*]} accel=$ACCEL until +${RETRY_H:-5}h"
round=0
while [ "$(date +%s)" -lt "$RETRY_DEADLINE" ]; do
  round=$((round+1)); say "=== round $round ==="
  for z in "${ZONES[@]}"; do
    say " try $z"
    if try_zone "$z"; then
      arm_reaper "$z"
      if stage "$z"; then say "TPU_UP_DONE zone=$z"; exit 0; fi
      say "STAGE_FAILED @ $z -> delete + give up"; rm -f /tmp/tpu_reaper_${NAME}.deadline; del_qr "$z"; tail -20 "$LOG"; exit 3
    fi
    [ "$(date +%s)" -ge "$RETRY_DEADLINE" ] && break
  done
  [ "$(date +%s)" -ge "$RETRY_DEADLINE" ] && break
  say "round $round: no capacity anywhere; backoff 8m"
  sleep 480
done
say "TPU_NO_CAPACITY (no v6e-1 capacity within retry window)"
exit 2
