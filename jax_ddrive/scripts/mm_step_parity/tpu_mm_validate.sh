#!/usr/bin/env bash
# OVERNIGHT PARITY — TPU validation of the multimodal SASD step on a REAL v6e-1.
# Validates on TPU hardware: (1) DATA matches (parquet decode_row == prep npz, bit-exact),
# (2) NNX IMPLEMENTATION matches the PyTorch oracle (parity_nnx: Layer2/Layer1/ctrl/det/loss).
# Why TPU: XLA lowers differently on TPU vs GPU (project lesson: GPU-validated != TPU-validated),
# so the fp32 forward+loss must be re-confirmed on real TPU silicon.
#
# Reuses the established free-trial provisioning pattern (cross-zone retry + auto-reaper). Own
# NAME / GCS prefix so it never collides with the eval `sasd-parity` workflow.
#
# Run:  bash jax_ddrive/scripts/mm_step_parity/tpu_mm_validate.sh
# Teardown any time:  NAME=mmstep-parity bash jax_ddrive/scripts/temp/tpu_down.sh
set -uo pipefail
PROJECT=project-8a53f5ab-2ea2-4892-a78
RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}"
ACCEL="${ACCEL:-v6e-1}"                 # 32GB HBM — fp32 3B + 2L logits fit (as on the 5090)
NAME="${NAME:-mmstep-parity}"
ZONES=(${ZONES_OVERRIDE:-us-east5-a us-east5-b us-east5-c us-central1-a us-south1-a})  # dropped us-east1-d (perm denied)
RETRY_DEADLINE=$(( $(date +%s) + ${RETRY_H:-2}*3600 ))
REAP_H="${DEADLINE_H:-3}"
POLL_ITERS="${POLL_ITERS:-60}"          # 60*20s = 20min patient wait per zone (QR queues free; bill only when ACTIVE)
SRC=gs://${PROJECT}-ddrive-sasd
MM=$SRC/mmstep
W=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/mm_step_parity
LOCALOUT=/home/kaiwen/data/parity_overnight/tpu
mkdir -p "$LOCALOUT"
LOG=$W/tpu_mm.log; : > "$LOG"
ZFILE=$W/tpu_mm_active_zone.txt; rm -f "$ZFILE"
say(){ echo "$(date +%T) $*" | tee -a "$LOG"; }
del_qr(){ gcloud compute tpus queued-resources delete "$NAME" --zone="$1" --quiet --force >/dev/null 2>&1 || true; }

try_zone(){
  local z=$1 st
  gcloud compute tpus queued-resources create "$NAME" --node-id="$NAME" --zone="$z" \
     --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" >>"$LOG" 2>&1 \
     || { say "  [$z] create rejected"; del_qr "$z"; return 1; }
  # Patient: a QR that reaches WAITING_FOR_RESOURCES/PROVISIONING is queued for capacity (free until
  # ACTIVE) — keep waiting. Only abandon on a terminal FAILED/SUSPENDED or after the full window.
  for i in $(seq 1 "$POLL_ITERS"); do
    st=$(gcloud compute tpus queued-resources describe "$NAME" --zone="$z" --format="value(state.state)" 2>/dev/null)
    [ "$st" = ACTIVE ] && { say "  [$z] ACTIVE (after ~$((i*20))s)"; return 0; }
    case "$st" in FAILED|SUSPENDED) say "  [$z] $st"; del_qr "$z"; return 1;; esac
    [ $((i % 9)) -eq 0 ] && say "  [$z] ${st:-none} (~$((i*20))s, still waiting)"
    sleep 20
  done
  say "  [$z] still ${st:-none} after ~$((POLL_ITERS*20/60))m -> release + next zone"; del_qr "$z"; return 1
}

arm_reaper(){
  local z=$1; echo "$z" > "$ZFILE"; echo $(( $(date +%s) + REAP_H*3600 )) > /tmp/tpu_reaper_${NAME}.deadline
  cat > $W/tpu_mm_reaper.sh <<RPR
#!/usr/bin/env bash
while true; do
  d=\$(cat /tmp/tpu_reaper_${NAME}.deadline 2>/dev/null || echo 0); [ "\$d" = 0 ] && exit 0
  if [ "\$(date +%s)" -ge "\$d" ]; then
    gcloud compute tpus queued-resources delete "${NAME}" --zone="${z}" --quiet --force >> $W/tpu_mm_reaper.log 2>&1
    rm -f /tmp/tpu_reaper_${NAME}.deadline; exit 0
  fi; sleep 60
done
RPR
  nohup bash $W/tpu_mm_reaper.sh >/dev/null 2>&1 &
  say "reaper armed (+${REAP_H}h) zone $z (pid $!)"
}

stage(){
  local z=$1
  say "stage on VM @ $z (code+bundle+snapshot+venv, timeout 40m) ..."
  timeout 2400 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$z" --worker=all --command '
    set -e; cd ~
    gsutil -q cp '"$MM"'/mmstep_code.tgz . && rm -rf ~/mmstep && mkdir -p ~/mmstep && tar xzf mmstep_code.tgz -C ~/mmstep
    rm -rf ~/mmdata && mkdir -p ~/mmdata/bundles ~/mmdata/prep ~/mmdata/parquet
    gsutil -m -q cp '"$MM"'/bundles/* ~/mmdata/bundles/
    gsutil -m -q cp '"$MM"'/prep/* ~/mmdata/prep/
    gsutil -m -q cp '"$MM"'/parquet/* ~/mmdata/parquet/
    mkdir -p ~/release_snapshot && gsutil -m -q rsync -r '"$SRC"'/release_fast_ddrive_snapshot ~/release_snapshot
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"; [ -d ~/venv ] || uv venv -p 3.11 ~/venv; source ~/venv/bin/activate
    uv pip install -q "jax[tpu]==0.10.0" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
        flax==0.12.7 transformers ml_dtypes safetensors numpy pyarrow jaxtyping
    python -c "import jax; print(\"TPU_DEVICES\", jax.devices())"; echo WORKER_STAGE_OK
  ' >>"$LOG" 2>&1 && grep -q WORKER_STAGE_OK "$LOG"
}

run_validate(){
  local z=$1
  say "RUN validation on TPU @ $z (timeout 40m) ..."
  timeout 2400 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$z" --worker=all --command '
    set -uo pipefail
    source ~/venv/bin/activate
    export PATH="$HOME/.local/bin:$PATH"
    export MMSTEP_REPO=$HOME/mmstep FASTDDRIVE_SNAP=$HOME/release_snapshot
    S=$HOME/mmstep/scripts/mm_step_parity
    echo "===== TPU DATA MATCH ====="
    python $S/verify_dataset.py --prep_dir ~/mmdata/prep --parquet_dir ~/mmdata/parquet \
      --report ~/mmdata/tpu_verify_dataset.json 2>&1 | grep -E "\[D\]|SASD_MM_DATASET" || true
    for tag in 00000 00001; do
      echo "===== TPU NNX IMPL MATCH $tag ====="
      python $S/parity_nnx.py --bundle ~/mmdata/bundles/$tag.bundle.npz \
        --report ~/mmdata/tpu_parity_nnx_$tag.json 2>&1 \
        | grep -E "determinism|=== |relmax|loss primary|->|SASD_MM_NNX" || true
    done
    # ship reports back via GCS
    gsutil -q cp ~/mmdata/tpu_verify_dataset.json '"$MM"'/results/ 2>/dev/null || true
    gsutil -q cp ~/mmdata/tpu_parity_nnx_*.json '"$MM"'/results/ 2>/dev/null || true
    echo TPU_VALIDATE_RAN
  ' 2>&1 | tee -a "$LOG"
  grep -q TPU_VALIDATE_RAN "$LOG"
}

say "provision retry: zones=${ZONES[*]} accel=$ACCEL until +${RETRY_H:-3}h"
round=0
while [ "$(date +%s)" -lt "$RETRY_DEADLINE" ]; do
  round=$((round+1)); say "=== round $round ==="
  for z in "${ZONES[@]}"; do
    say " try $z"
    if try_zone "$z"; then
      arm_reaper "$z"
      if ! stage "$z"; then say "STAGE_FAILED @ $z"; tail -25 "$LOG"; exit 3; fi
      run_validate "$z" || say "RUN had non-zero (see log)"
      # pull reports
      gsutil -m -q cp "$MM/results/tpu_*.json" "$LOCALOUT/" 2>/dev/null || true
      say "TPU_MM_VALIDATE_DONE zone=$z (reports -> $LOCALOUT ; reaper will delete in +${REAP_H}h, or run tpu_down.sh)"
      exit 0
    fi
    [ "$(date +%s)" -ge "$RETRY_DEADLINE" ] && break
  done
  [ "$(date +%s)" -ge "$RETRY_DEADLINE" ] && break
  say "round $round: no capacity; backoff 8m"; sleep 480
done
say "TPU_NO_CAPACITY (no $ACCEL within +${RETRY_H:-3}h)"; exit 2
