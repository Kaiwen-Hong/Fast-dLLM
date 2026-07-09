#!/bin/bash
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "[$(date)] START downloads"
echo "=== FP8 (primary, ~26GB) ==="
hf download RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic && echo "=== FP8_DONE $(date) ===" || echo "=== FP8_FAIL rc=$? ==="
df -h / | tail -1
echo "=== BF16 full (~52GB) ==="
hf download google/diffusiongemma-26B-A4B-it && echo "=== BF16_DONE $(date) ===" || echo "=== BF16_FAIL rc=$? ==="
df -h / | tail -1
echo "[$(date)] ALL_DOWNLOADS_DONE"
