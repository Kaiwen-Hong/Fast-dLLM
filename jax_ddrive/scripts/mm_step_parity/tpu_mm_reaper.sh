#!/usr/bin/env bash
while true; do
  d=$(cat /tmp/tpu_reaper_mmstep-parity.deadline 2>/dev/null || echo 0); [ "$d" = 0 ] && exit 0
  if [ "$(date +%s)" -ge "$d" ]; then
    gcloud compute tpus queued-resources delete "mmstep-parity" --zone="us-east5-b" --quiet --force >> /home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/mm_step_parity/tpu_mm_reaper.log 2>&1
    rm -f /tmp/tpu_reaper_mmstep-parity.deadline; exit 0
  fi; sleep 60
done
