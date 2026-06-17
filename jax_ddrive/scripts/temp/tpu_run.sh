#!/usr/bin/env bash
# Run the eval_sasd parity eval on the (already-up, staged) TPU: per-sample process for fp32 + bf16
# over the 20 val samples, merge, push results+tokens to GCS, pull to local. Re-runnable for
# iteration (after tpu_push.sh ships fixed code).
set -uo pipefail
PROJECT=project-8a53f5ab-2ea2-4892-a78
W=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/temp
ZONE="${ZONE:-$(cat $W/tpu_active_zone.txt 2>/dev/null || echo us-east5-a)}"
NAME="${NAME:-sasd-parity}"
SRC=gs://${PROJECT}-ddrive-sasd
TAG="${TAG:-r1}"           # iteration tag, so results don't clobber across rounds
say(){ echo "$(date +%T) $*"; }

say "run parity on TPU $NAME (per-sample fp32+bf16, tag=$TAG) — timeout 45m ..."
timeout 2700 gcloud compute tpus tpu-vm ssh "$NAME" --zone="$ZONE" --worker=all --command '
  set -uo pipefail
  source ~/venv/bin/activate
  export PYTHONPATH=$HOME/parity
  rm -rf ~/tokens_tpu ~/persample_tpu; mkdir -p ~/tokens_tpu ~/persample_tpu
  for i in $(seq -w 0 19); do S=val_s$i; for DT in fp32 bf16; do
    echo "=== $S $DT ==="
    timeout 300 python ~/parity/parity_eval.py --snapshot ~/release_snapshot --npz_dir ~/eval_inputs \
      --samples $S --dtype $DT --out ~/persample_tpu/${S}_${DT}.json --tokens_dir ~/tokens_tpu \
      2>&1 | grep -vE "deprecated|FutureWarning|warnings.warn|rope_|layer_type|Unrecognized|PyTorch was not|got .key=" || echo "  [$S $DT] nonzero/timeout"
  done; done
  python - <<PY
import json, glob, os
H = os.path.expanduser("~")
for dt in ("fp32", "bf16"):
    o = []
    for f in sorted(glob.glob(os.path.join(H, "persample_tpu", "val_s*_%s.json" % dt))):
        o += json.load(open(f))
    json.dump(o, open(os.path.join(H, "tpu_%s.json" % dt), "w"), indent=2)
    print("tpu_%s: %d ok=%d" % (dt, len(o), sum("error" not in r for r in o)))
PY
  gsutil -q cp ~/tpu_fp32.json '"$SRC"'/parity/tpu_'"$TAG"'_fp32.json
  gsutil -q cp ~/tpu_bf16.json '"$SRC"'/parity/tpu_'"$TAG"'_bf16.json
  tar czf ~/tokens_tpu.tgz -C ~/tokens_tpu . && gsutil -q cp ~/tokens_tpu.tgz '"$SRC"'/parity/tokens_tpu_'"$TAG"'.tgz
  echo TPU_RUN_OK
' 2>&1 | tee -a "$W/tpu_run_${TAG}.log"

say "pull results to local ..."
gsutil -q cp "$SRC/parity/tpu_${TAG}_fp32.json" "$W/tpu_${TAG}_fp32.json"
gsutil -q cp "$SRC/parity/tpu_${TAG}_bf16.json" "$W/tpu_${TAG}_bf16.json"
gsutil -q cp "$SRC/parity/tokens_tpu_${TAG}.tgz" "$W/tokens_tpu_${TAG}.tgz" && \
  mkdir -p "$W/tokens_tpu_${TAG}" && tar xzf "$W/tokens_tpu_${TAG}.tgz" -C "$W/tokens_tpu_${TAG}"
say "TPU_RUN_DONE tag=$TAG → $W/tpu_${TAG}_{fp32,bf16}.json + $W/tokens_tpu_${TAG}/"
