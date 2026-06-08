#!/usr/bin/env bash
# Source this to run the jax-dlm-baseline venv on the local RTX 5090 GPU.
#
# Fixes: "RuntimeError: Unable to load cuSPARSE. Is it installed?" / CPU fallback.
# Cause: the jax-cuda12 plugin must load CUDA from the venv's bundled nvidia-*-cu12 pip
# packages, but the inherited LD_LIBRARY_PATH points at a mismatched system /usr/local/cuda.
# We prepend the venv's nvidia */lib dirs so the right libcusparse/libcudnn/... are found.
#
# Usage:
#   source jax_ddrive/scripts/jax_gpu_env.sh
#   "$JAXPY" jax_ddrive/ddrive_jax/train/train_tpu.py ...      # runs on the 5090
#
# For CPU multi-host emulation instead, do NOT source this; use:
#   JAX_PLATFORMS=cpu XLA_FLAGS="--xla_force_host_platform_device_count=8"
export JAX_VENV=/home/kaiwen/jax-dlm-baseline/.venv
export JAXPY="$JAX_VENV/bin/python"
_nvlibs=$(echo "$JAX_VENV"/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="${_nvlibs}:${LD_LIBRARY_PATH:-}"
unset _nvlibs
# quick check: "$JAXPY" -c "import jax; print(jax.devices())"  -> [CudaDevice(id=0)]
