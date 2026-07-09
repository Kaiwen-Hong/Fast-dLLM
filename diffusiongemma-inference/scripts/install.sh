#!/bin/bash
echo "[$(date)] START install"
python3 -m pip install -U pip 2>&1 | tail -1
echo ">>> installing vllm==0.24.0 (this pulls torch, ~several min)"
python3 -m pip install vllm==0.24.0 2>&1 | tail -3
echo ">>> installing latest transformers + accelerate"
python3 -m pip install -U transformers accelerate 2>&1 | tail -3
echo "=== VERSIONS ==="
python3 -c "import torch,transformers; print('torch',torch.__version__,'| transformers',transformers.__version__)" 2>&1
python3 -c "import vllm; print('vllm',vllm.__version__)" 2>&1
echo "=== Q1: does stock vLLM know DiffusionGemma? ==="
VLLM_DIR=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null)
echo "vllm dir: $VLLM_DIR"
grep -rl "DiffusionGemma" "$VLLM_DIR/model_executor/models/" 2>&1 | head || echo "VLLM_NO_DIFFUSIONGEMMA"
python3 -c "from vllm.model_executor.models.registry import ModelRegistry as R; a=R.get_supported_archs() if hasattr(R,'get_supported_archs') else []; print('DiffusionGemmaForBlockDiffusion in registry:', 'DiffusionGemmaForBlockDiffusion' in a)" 2>&1 | tail -2
echo "=== Q2: does transformers know DiffusionGemma? ==="
python3 -c "from transformers import DiffusionGemmaForBlockDiffusion; print('transformers HAS DiffusionGemmaForBlockDiffusion')" 2>&1 | tail -3
echo "[$(date)] INSTALL_DONE"
