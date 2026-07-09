#!/bin/bash
cd /workspace/ddgemma
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
run(){ echo "########## $1 ##########"; DDG_MODEL="$2" python3 bench.py --task "$3" --n "$4" --bs 4 --max_new_tokens "$5" --think off --tag "$6" 2>&1 | grep -vE "Fetching|Loading weights|Warning: You are sending|Downloading|Generating|Map:|Resolving|processor_kwargs|W0709|CUDACachingAllocator"; }
echo "===== BF16 reference ====="
run "bf16 gsm8k" google/diffusiongemma-26B-A4B-it gsm8k 120 640 _ablbf16
run "bf16 math"  google/diffusiongemma-26B-A4B-it math500 60 1280 _ablbf16
echo "===== FP8-dynamic ====="
run "fp8 gsm8k" RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic gsm8k 120 640 _ablfp8
run "fp8 math"  RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic math500 60 1280 _ablfp8
echo "ABLATION_DONE"
