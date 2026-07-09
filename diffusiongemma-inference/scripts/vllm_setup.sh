#!/bin/bash
echo "[$(date)] creating clean venv"
python3 -m venv /workspace/vllm-venv
V="/workspace/vllm-venv/bin"
$V/pip install -q -U pip 2>&1 | tail -1
echo ">>> installing vllm==0.24.0 into clean venv (isolated torch)"
$V/pip install vllm==0.24.0 2>&1 | tail -4
echo "=== vllm version ==="
$V/python -c "import vllm; print('vllm', vllm.__version__)" 2>&1
echo "=== does vLLM know DiffusionGemma? ==="
$V/python - <<'PY' 2>&1
try:
    from vllm.model_executor.models.registry import ModelRegistry as R
    archs = R.get_supported_archs()
    hit = [a for a in archs if "iffusion" in a or "DiffusionGemma" in a]
    print("DiffusionGemma supported:", hit if hit else "NO -- not in registry")
except Exception as e:
    print("registry check error:", repr(e))
PY
echo "[$(date)] VLLM_SETUP_DONE"
