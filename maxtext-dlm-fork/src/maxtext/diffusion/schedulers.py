"""Noise schedulers for masked discrete diffusion.

A scheduler defines `α(t) ∈ [0, 1]` (the proportion of unmasked tokens at
diffusion time t), with `α(0) = 1` (clean) and `α(1) → 0` (fully noised).
The MDLM loss weight is `w(t) = -α'(t) / (1 - α(t))`, and the reverse-step
mask probability for going from time `t` back to `s < t` is
`(1 - α(s)) / (1 - α(t))`.

References:
    - dllm/core/schedulers/alpha.py:101-117 (Linear, Cosine in PyTorch)
    - md4/models/diffusion/md4.py:34-74 (MaskingSchedule with eps)
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import jax.numpy as jnp


@dataclasses.dataclass
class BaseScheduler:
    """Subclass and override `_alpha`, `_alpha_derivative`."""
    eps: float = 1e-6

    __registry__: ClassVar[dict[str, type["BaseScheduler"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        BaseScheduler.__registry__[cls.__name__] = cls
        BaseScheduler.__registry__[cls.__name__.lower()] = cls
        # Also register a short tag (e.g. "linear" for LinearScheduler).
        short = cls.__name__.replace("Scheduler", "").lower()
        BaseScheduler.__registry__[short] = cls

    def _alpha(self, t):
        raise NotImplementedError
    def _alpha_derivative(self, t):
        raise NotImplementedError

    def alpha(self, t):
        return self._alpha(jnp.asarray(t, dtype=jnp.float32))
    def alpha_derivative(self, t):
        return self._alpha_derivative(jnp.asarray(t, dtype=jnp.float32))

    def weight(self, t):
        """`w(t) = -α'(t) / (1 - α(t) + eps)`. For linear: 1/t."""
        a = self.alpha(t)
        da = self.alpha_derivative(t)
        return -da / (1.0 - a + self.eps)

    def reverse_mask_prob(self, s, t):
        """`(1 - α(s)) / (1 - α(t) + eps)` for s < t. Used by the sampler
        to compute per-step reveal counts."""
        return (1.0 - self.alpha(s)) / (1.0 - self.alpha(t) + self.eps)


@dataclasses.dataclass
class LinearScheduler(BaseScheduler):
    """`α(t) = 1 - t`, `α'(t) = -1`. Weight w(t) = 1/t."""
    def _alpha(self, t):  return 1.0 - t
    def _alpha_derivative(self, t):  return -jnp.ones_like(t)


@dataclasses.dataclass
class CosineScheduler(BaseScheduler):
    """`α(t) = cos(πt/2)`, `α'(t) = -(π/2) sin(πt/2)`. Used by md4 by default."""
    def _alpha(self, t):
        return jnp.cos(jnp.pi * 0.5 * t)
    def _alpha_derivative(self, t):
        return -(jnp.pi * 0.5) * jnp.sin(jnp.pi * 0.5 * t)


@dataclasses.dataclass
class PolyScheduler(BaseScheduler):
    """`α(t) = 1 - t^p`. p=1 is linear; p>1 stays cleaner longer; p<1 noises faster."""
    p: float = 2.0
    def _alpha(self, t):
        return 1.0 - jnp.power(t, self.p)
    def _alpha_derivative(self, t):
        return -self.p * jnp.power(t, self.p - 1.0)


def make_scheduler(spec: str | BaseScheduler | None) -> BaseScheduler:
    """Look up by tag (e.g. "linear", "cosine"), polymorphic spec like
    "poly2.5", or pass through if already a scheduler. None → LinearScheduler()."""
    if spec is None:
        return LinearScheduler()
    if isinstance(spec, BaseScheduler):
        return spec
    if not isinstance(spec, str):
        raise TypeError(f"unknown scheduler spec: {spec!r}")

    if spec.startswith("poly"):
        try:
            p = float(spec[4:])
        except ValueError:
            raise ValueError(f"invalid poly spec {spec!r}; expected e.g. 'poly2'")
        return PolyScheduler(p=p)

    cls = BaseScheduler.__registry__.get(spec)
    if cls is None:
        raise ValueError(
            f"unknown scheduler {spec!r}; "
            f"known: linear, cosine, poly<float>"
        )
    return cls()


def transfer_counts(
    scheduler: BaseScheduler, block_size: int, steps_per_block: int
) -> jnp.ndarray:
    """Per-step number of tokens to reveal in a block of `block_size` masks
    when running `steps_per_block` reverse steps under `scheduler`.

    For linear schedule this reduces to the simple uniform split (with
    remainder distributed to the first few steps). For non-linear schedules
    we compute the cumulative un-mask fraction and round, then take diffs.
    """
    if isinstance(scheduler, LinearScheduler):
        base = block_size // steps_per_block
        rem = block_size % steps_per_block
        return jnp.asarray(
            [base + (1 if i < rem else 0) for i in range(steps_per_block)],
            dtype=jnp.int32,
        )

    # Non-linear: walk t from 1 → 0 across the block.
    # Cumulative *revealed* fraction after step i is α(t_{i+1}), where
    # t_i = (steps - i) / steps,  so t_0=1 (all masked) and t_steps=0 (all revealed).
    # Reveal count at step i = round(block_size * (α(t_{i+1}) - α(t_i))).
    import numpy as _np
    cum = []
    for i in range(steps_per_block + 1):
        t_i = (steps_per_block - i) / steps_per_block
        a_i = float(scheduler.alpha(jnp.asarray(t_i, jnp.float32)))
        cum.append(int(round(block_size * a_i)))      # cumulative *revealed* count
    cum_arr = _np.asarray(cum, dtype=_np.int32)
    cum_arr = _np.clip(cum_arr, 0, block_size)
    cum_arr = _np.maximum.accumulate(cum_arr)         # non-decreasing
    cum_arr[-1] = block_size                          # ensure final state reveals everything
    diffs = _np.diff(cum_arr)                         # [steps_per_block]
    return jnp.asarray(diffs, dtype=jnp.int32)
