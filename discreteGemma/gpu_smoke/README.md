# DiffusionGemma — JAX/GPU smoke tests (RTX 5090 / sm_120)

Proves the vendored DeepMind **DiffusionGemma** implementation (`discreteGemma/gemma/gemma/diffusion`)
can **train and infer on the RTX 5090 in JAX**, using a shrunk (random-init) config — no pretrained weights.

> **This is the clean baseline.** `discreteGemma/gemma` is the pristine upstream framework (0 edits,
> byte-identical to the vendored original). New work builds on top of this dir; the VQA/ChartQA
> experiment lives separately under `discreteGemma-modified/`.

## Environment (conda env `dgemma-jax`, verified 2026-07-01)

- Python **3.12** (gemma requires >=3.12)
- `jax[cuda12] == 0.6.2` (jaxlib 0.6.2, jax-cuda12-plugin/pjrt 0.6.2)
- pip NVIDIA CUDA wheels: CUDA **12.9** era + cuDNN 9.23 (works on this box's **570.153 / CUDA 12.8** driver)
- model deps: `kauldron==1.4.4`, `flax==0.11.2`, `optax`, `einops`, `jaxtyping`, `sentencepiece`, `dialog`, `etils[all]`

### Recreate
```bash
conda create -y -n dgemma-jax python=3.12
conda activate dgemma-jax
pip install "jax[cuda12]==0.6.2" jaxlib==0.6.2
pip install kauldron einops jaxtyping sentencepiece "etils[all]" absl-py dialog
# --- for the SFT path (official training via hackable_diffusion) ---
pip install "hackable-diffusion @ git+https://github.com/google/hackable_diffusion.git"  # v1.0.1
pip install "numpy<2.5"   # numpy 2.5 breaks mediapy (NDArray subclassing) in kauldron.train import chain
```

### Critical fix — LD_LIBRARY_PATH (else JAX falls back to CPU)
jaxlib 0.6.2's plugin can't dlopen the pip cuSPARSE (cuSPARSE→libnvJitLink chain not on the loader
path) and silently drops to CPU. Baked into env activation at:
`$CONDA_PREFIX/etc/conda/activate.d/zz_nvidia_libs.sh` — it prepends every
`site-packages/nvidia/*/lib` dir to `LD_LIBRARY_PATH`. After that, `conda activate dgemma-jax`
gives GPU automatically.

## Run
```bash
conda activate dgemma-jax
python gpu_check.py            # Step 1: JAX sees the 5090, runs fp32+bf16 matmul on sm_120
python tiny_dgemma_smoke.py    # Step 2: tiny DiffusionGemma init + infer + 1 train step, all on GPU
python sft_smoke.py            # Step 3: OFFICIAL SFT path (hackable_diffusion) train step on GPU
```

`sft_smoke.py` uses the real `SFTDiffusion` + `hackable_diffusion` uniform categorical
corruption + `UniformTimeSampler` + `NoWeightDiscreteLoss` + `EncoderARLoss` (the exact
pieces wired in `configs/sft_sudoku.py`), on the tiny gemma backbone with synthetic data —
proving the released SFT training graph (KV-cache prefill + self-conditioning double-pass +
diffusion loss + encoder AR loss) runs and is differentiable on GPU. Full-parameter SFT;
LoRA (`lora.LoRA`, the sudoku default) is an additional wrapper not exercised here.
Last verified: `103,106 params | total 10.98->9.92 (diffusion+encoder both decrease) | PASS`.

## Expected (last verified 2026-07-01)
```
gpu_check.py       -> backend gpu | CudaDevice(id=0) | device_kind NVIDIA GeForce RTX 5090 | PASS
tiny_dgemma_smoke  -> 103,106 params | self_conditioner present: True
                      inference OK (base AR forward)   : (2,16,256) on CudaDevice(id=0)
                      inference OK (diffusion forward) : (2,16,256) on CudaDevice(id=0)
                      training OK: loss 5.1967 -> 4.4060 (decreased) | PASS
```

## What the tiny config exercises (real gemma code, shrunk)
- `gemma.diffusion.DiffusionGemma_26B_A4B` with a custom `TransformerConfig`:
  2 local-sliding layers, `embed_dim=64`, `head_dim=16`, vocab=256, **`enable_moe=False`** (dense FFW),
  `vision_encoder=None`. No TPU-only kernels are touched (attention = softmax/einsum).
- Base AR forward **and** the diffusion-specific `call_with_self_conditioning`
  (self-conditioning FFW + bidirectional canvas attention).
- Training grads flow through the full diffusion forward incl. the self-conditioner.

## Not covered here (by design)
- MoE path (`jax.lax.ragged_dot`) — disabled for the smallest smoke; can be enabled by
  `enable_moe=True, num_experts=…` to also test ragged_dot on GPU.
- Full `kauldron.main` Trainer runs / real bagz datasets / released gs:// checkpoints — the smokes
  use synthetic data and a hand-rolled train loop. (`tiny_dgemma_smoke.py` uses a self-written CE;
  `sft_smoke.py` DOES use the real pip-installed `hackable_diffusion` library — see env above.)
- Real weights / tokenizer vocab (not needed to prove GPU train/infer mechanics).
</content>
