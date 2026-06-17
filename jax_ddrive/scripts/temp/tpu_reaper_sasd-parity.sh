#!/usr/bin/env bash
while true; do
  d=$(cat /tmp/tpu_reaper_sasd-parity.deadline 2>/dev/null || echo 0)
  [ "$d" = "0" ] && exit 0
  if [ "$(date +%s)" -ge "$d" ]; then
    echo "$(date +%T) REAPER force-delete QR sasd-parity" >> /home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/temp/tpu_reaper.log
    gcloud compute tpus queued-resources delete "sasd-parity" --zone="us-east5-a" --quiet --force >> /home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive/scripts/temp/tpu_reaper.log 2>&1
    rm -f /tmp/tpu_reaper_sasd-parity.deadline; exit 0
  fi
  sleep 60
done
