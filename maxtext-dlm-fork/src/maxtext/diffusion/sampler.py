"""MDLM low-confidence remasking sampler.

Port of `dllm/core/samplers/mdlm.py:35-238` (the `generate` method) plus the
linear-schedule case of `dllm/core/samplers/utils.py:get_num_transfer_tokens`.

Algorithm:
    1. Build a canvas [B, T] = [prompt | MASK*max_new_tokens], right-padded if
       needed.
    2. Walk over `num_blocks = ceil(max_new_tokens / block_size)` blocks.
    3. For each block, run `steps_per_block` forward passes; each pass
       commits `transfer[i]` predictions chosen by **highest predicted
       probability** (low-confidence remasking — keep the most confident
       picks, leave the rest masked for next step).
    4. Commits within a step are restricted to the current block's masked
       positions (other masks stay masked until their block's turn).

For the linear schedule, the per-step transfer count is exactly
`block_size // steps_per_block` (+1 for the first `remainder` steps).

This v1 supports: argmax / Gumbel noise / low-confidence remasking. Skipped
for now: classifier-free guidance, suppress_tokens, infill, stochastic
transfer (binomial), right_shift_logits. Each is a small extension; spec
calls those Waymo-side / Phase 9-onwards tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp

from maxtext.diffusion.schedulers import (
    BaseScheduler,
    LinearScheduler,
    make_scheduler,
    transfer_counts as _transfer_counts,
)


@dataclass
class SamplerConfig:
    max_new_tokens: int = 128
    block_size: int = 32
    steps: int = 32
    temperature: float = 0.0          # 0 → pure argmax; >0 → Gumbel-Max
    remasking: str = "low_confidence"  # or "random"
    scheduler: str | BaseScheduler = "linear"


def transfer_counts_linear(block_size: int, steps_per_block: int) -> jnp.ndarray:
    """Backward-compat helper. Equivalent to
    `transfer_counts(LinearScheduler(), block_size, steps_per_block)`."""
    return _transfer_counts(LinearScheduler(), block_size, steps_per_block)


def _topk_mask(scores: jnp.ndarray, k: int) -> jnp.ndarray:
    """[B, T] scores → [B, T] bool mask with the top-k positions per row True."""
    B, T = scores.shape
    if k <= 0:
        return jnp.zeros((B, T), dtype=bool)
    _, idx = jax.lax.top_k(scores, k)                                # [B, k]
    one_hot = jax.nn.one_hot(idx, T, dtype=jnp.int32)                # [B, k, T]
    return one_hot.sum(axis=1).astype(bool)                          # [B, T]


def generate(
    model,
    prompt: jnp.ndarray,                 # [B, L_p] int32
    *,
    mask_token_id: int,
    eos_token_id: int,
    cfg: SamplerConfig,
    rng_key: jax.Array,
) -> jnp.ndarray:                        # [B, L_p + max_new_tokens]
    """Generate `cfg.max_new_tokens` tokens conditioned on `prompt`.

    The model must be a callable matching Bonsai LLaDA's signature:
        logits, _, _ = model(input_ids, attention_mask, rng_key)
    where logits is [B, T, V].

    No jit is applied here; the inner forward through `model` is already
    jit-compiled by the caller (or by nnx).
    """
    B, L_p = prompt.shape
    T = L_p + cfg.max_new_tokens

    # ----- canvas -----
    x = jnp.full((B, T), eos_token_id, dtype=jnp.int32)
    x = x.at[:, :L_p].set(prompt)
    x = x.at[:, L_p:].set(mask_token_id)
    attention_mask = jnp.ones((B, T), dtype=jnp.int32)

    num_blocks = (cfg.max_new_tokens + cfg.block_size - 1) // cfg.block_size
    steps_per_block = max((cfg.steps + num_blocks - 1) // num_blocks, 1)
    scheduler = make_scheduler(cfg.scheduler)

    for b in range(num_blocks):
        start = L_p + b * cfg.block_size
        end = min(start + cfg.block_size, T)
        block_w = end - start
        transfer = _transfer_counts(scheduler, block_w, steps_per_block)   # [S]

        # Range mask for "is this position inside block b" — [1, T] bool.
        idx_T = jnp.arange(T)
        in_block = ((idx_T >= start) & (idx_T < end))[None, :]

        for s in range(steps_per_block):
            k = int(transfer[s])
            if k == 0:
                continue

            rng_key, kf = jax.random.split(rng_key)
            logits, _, _ = model(x, attention_mask, kf)                # [B, T, V]
            logits_f32 = logits.astype(jnp.float32)
            # Never predict the mask token.
            logits_f32 = logits_f32.at[..., mask_token_id].set(-jnp.inf)

            # Greedy / Gumbel-Max prediction.
            if cfg.temperature > 0.0:
                rng_key, kg = jax.random.split(rng_key)
                gumbel = jax.random.gumbel(kg, logits_f32.shape, dtype=jnp.float32)
                x0 = jnp.argmax(logits_f32 + cfg.temperature * gumbel, axis=-1)
            else:
                x0 = jnp.argmax(logits_f32, axis=-1)

            # Per-position confidence used to choose which masks to commit.
            if cfg.remasking == "low_confidence":
                p = jax.nn.softmax(logits_f32, axis=-1)
                conf = jnp.take_along_axis(p, x0[..., None], axis=-1).squeeze(-1)  # [B, T]
            elif cfg.remasking == "random":
                rng_key, kr = jax.random.split(rng_key)
                conf = jax.random.uniform(kr, x0.shape, dtype=jnp.float32)
            else:
                raise ValueError(f"Unknown remasking={cfg.remasking}")

            # Restrict to currently masked positions inside the current block.
            is_mask = (x == mask_token_id)
            scores = jnp.where(is_mask & in_block, conf, -jnp.inf)     # [B, T]
            transfer_idx = _topk_mask(scores, k)                       # [B, T]

            # Commit chosen predictions.
            x = jnp.where(transfer_idx, x0, x)

    return x


# Keep the old short name from the Phase 5 stub for backward-compat in case
# external code already imported it.
def unmask_step(*args, **kwargs):
    raise NotImplementedError(
        "unmask_step (single-step) is not used by the Phase 11 sampler. "
        "Use `generate(...)` instead."
    )
