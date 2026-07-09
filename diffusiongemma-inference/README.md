# DiffusionGemma 26B-A4B — Inference, Benchmarks & Explainers

An overnight study (2026-07-09) of Google's **DiffusionGemma 26B-A4B-it** — a block-diffusion
LLM (dLLM), the first natively supported in vLLM — run and instrumented on a single **H100 80GB**
(vast.ai). Contains the inference/eval scripts, raw results, captured denoising trajectories, and
the source of a 6-part interactive explainer site set.

> 中文说明穿插在下方。术语保留英文。

---

## TL;DR results (measured this night, bf16, HF transformers, H100 80GB)

| Benchmark | Score | n | Official ref |
|---|---|---|---|
| GSM8K | **96.6%** | 500 | 94.3% (FP8 card) |
| MATH-500 | **88.0%** | 500 | — |
| MBPP | **87.9%** | 257 | — |
| Sudoku 9×9 | **0.0%** | 40 | untrained capability (needs SFT) |

**FP8 ablation (vLLM 0.24.0):** GSM8K **97.5%** (n=40) · weights **25.86 GB** (vs bf16 51.6 GB, −50%) ·
**433 tok/s** · recovery **100.9%**. → FP8 is essentially lossless and halves weight memory.

---

## The model in one paragraph

DiffusionGemma generates a **256-token canvas** (block) by **iterative denoising** rather than
left-to-right. Each step: run the denoiser → temperature-scale logits (anneal 0.8→0.4) → sample →
**accept the lowest-entropy tokens** whose joint mutual-information bound ≤ `entropy_bound=0.1`
(`EntropyBoundSampler.accept_canvas`) → **renoise** the rest with *fresh random vocab tokens*
(uniform/random-token diffusion, **not** `[MASK]`) → self-condition on this step's logits →
**adaptive stop** when the canvas is *stable* (argmax unchanged) **and** *confident* (low entropy).
Simple prompts converge in 2 steps; hard ones use more (cap 48). Architecture: 26B total / 4B active
MoE (128 experts), 30 layers, vocab 262144, ctx 262144.

---

## Environment / setup

Two inference paths, both verified:

- **HF transformers** (primary — benchmarks + instrumentation). No vLLM needed; the class
  `DiffusionGemmaForBlockDiffusion` ships in stock transformers.
  ```bash
  pip install -U transformers accelerate torchvision   # torch 2.13/cu130; torchvision is required by Gemma4Processor
  ```
  Load bf16 = 51.6 GB, ~11 s. Generate: `model.generate(**inp, max_new_tokens=N, max_denoising_steps=48)` → `.sequences`.

- **vLLM 0.24.0** (FP8 + throughput) in an **isolated venv** (a fresh venv dodges the debian PyJWT
  RECORD conflict that breaks a system-wide `pip install vllm`). Stock 0.24.0 *does* register the arch.
  ```bash
  python3 -m venv vllm-venv && vllm-venv/bin/pip install vllm==0.24.0 datasets math_verify
  VLLM_USE_V2_MODEL_RUNNER=1 vllm serve RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic \
    --trust-remote-code --max-num-seqs 4 \
    --hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1}'
  ```

**Gotchas (hard-won):**
- Vocab is **262144** → a single logits tensor is `bs×256×262144`; use **bs=4 / `--max-num-seqs 4`**
  or you OOM even on 80 GB. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **GSM8K needs `think=off`** to match the official 94.3% (that number is from the FP8 card's
  *non-thinking* section). Thinking → verbose over-reasoning → truncation. MATH/MBPP also `think=off`.
- Output structure is `<|channel>thought\n…<channel|>[final answer]`; extract answer = text after the
  **last** `<channel|>`.
- **FP8 via plain transformers FAILS** (`BFloat16 != Float8_e4m3fn` matmul) — the compressed-tensors
  FP8 kernel isn't wired into the custom diffusion forward. FP8 must go through vLLM.
- Capture per-step denoising state by monkeypatching **`EntropyBoundSampler.accept_canvas`** (see
  `scripts/capture_traj.py`); set `model.generation_config.disable_compile=True` first.

---

## How to run

```bash
# from the machine, with HF_HOME set to a big disk and the models downloaded:
export HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# single benchmark
python3 scripts/bench.py --task gsm8k --n 500 --bs 4 --max_new_tokens 640 --think off

# full suite (gsm8k, mbpp, math500, sudoku) -> results/SUMMARY.json
python3 scripts/run_all.py

# capture denoising trajectories -> trajectories/*.json (feeds the websites)
python3 scripts/capture_traj.py

# FP8 vs bf16 ablation (transformers side; FP8 fails here by design — see gotchas)
bash scripts/run_ablation.sh

# FP8 accuracy/throughput via vLLM (the working FP8 path)
EVAL_MODEL=RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic EVAL_TASK=gsm8k EVAL_N=40 \
  vllm-venv/bin/python scripts/vllm_eval.py
```

---

## Layout

```
scripts/        bench.py (batched HF eval: gsm8k/math500/mbpp/sudoku), run_all.py, run_ablation.sh,
                vllm_eval.py (vLLM FP8/bf16 eval), capture_traj.py (denoising-state instrumentation),
                smoke_test.py / smoke2.py / validate.py, install.sh / download.sh / vllm_setup.sh
results/        *.json — per-benchmark results (accuracy + per-sample preds + raw); SUMMARY.json
trajectories/   sky/mult/count/math_think.json (raw per-step denoising states) + traj_viz.json (compact, for the sites)
websites/       site0_hub … site6_compare — source of the 6-part explainer set (published as Artifacts)
logs/           bench_all / ablation / vllm_* — run logs (provenance)
```

## Explainer sites (published Claude Artifacts, Chinese + English terms)

- Hub: https://claude.ai/code/artifact/1f1ea486-fb7d-4e6f-8508-20bcd88510d5
- P1 Overview · P2 Trajectory player · P3 Entropy-bound sampler · P4 Adaptive compute · P5 Benchmarks & FP8 · P6 vs LLaDA/Dream
  (URLs in the hub; `websites/` holds the source HTML — data is inlined so each file is standalone).

---

## Relation to this repo (Fast-dLLM)

DiffusionGemma is a **random-token** block-diffusion model; the Fast-dLLM lines here (LLaDA / Dream /
v2) are **mask** diffusion. The transfer map: `[MASK]` ↔ random-noise token; confidence-threshold
unmask ↔ entropy-bound accept; `get_transfer_index` ↔ `accept_canvas`. See website Part 6.
