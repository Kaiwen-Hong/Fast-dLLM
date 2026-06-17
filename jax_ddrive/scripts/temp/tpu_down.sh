#!/usr/bin/env bash
# Tear down the parity TPU: stop the reaper (clear deadline) and force-delete the queued-resource.
# Idempotent. Run when done, or any time to guarantee no VM is left running.
set -uo pipefail
PROJECT=project-8a53f5ab-2ea2-4892-a78
W=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/temp
ZONE="${ZONE:-$(cat $W/tpu_active_zone.txt 2>/dev/null || echo us-east5-a)}"
NAME="${NAME:-sasd-parity}"
echo "$(date +%T) clearing reaper deadline + deleting QR $NAME ..."
rm -f /tmp/tpu_reaper_${NAME}.deadline    # stops the detached reaper loop on its next tick
gcloud compute tpus queued-resources delete "$NAME" --zone="$ZONE" --quiet --force 2>&1 | tail -3 || true
echo "verify (should be empty):"
gcloud compute tpus queued-resources list --zone="$ZONE" 2>&1 | grep -E "$NAME|Listed 0" || true
gcloud compute tpus tpu-vm list --zone="$ZONE" 2>&1 | grep -E "$NAME|Listed 0" || true
echo "TPU_DOWN_DONE"
