"""LoRA for the Qwen2.5 text backbone (nnx 0.12.7) — adapted from
jax-mdlm-handoff/code/lora.py. Base kernel/bias demoted to frozen nnx.Variable;
lora_A/lora_B are the only nnx.Param after freeze_non_lora_params, so
nnx.Optimizer(model, tx, wrt=nnx.Param) trains adapters only. Lets us fine-tune
the 3.086B model on a single 5090 sharing GPU memory with other jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj")


class LoRALinear(nnx.Module):
    def __init__(self, base, rank: int, alpha: float, *, rngs: nnx.Rngs):
        kernel = jnp.asarray(base.kernel[...])
        in_dim, out_dim = kernel.shape
        self.base_kernel = nnx.Variable(kernel)
        self.base_bias = nnx.Variable(jnp.asarray(base.bias[...])) if base.bias is not None else None
        self.lora_A = nnx.Param(jax.nn.initializers.lecun_normal()(rngs.params(), (in_dim, rank), dtype=kernel.dtype))
        self.lora_B = nnx.Param(jnp.zeros((rank, out_dim), dtype=kernel.dtype))
        self.scale = alpha / float(rank)

    def __call__(self, x):
        y = jnp.matmul(x, jnp.asarray(self.base_kernel[...]))
        if self.base_bias is not None:
            y = y + jnp.asarray(self.base_bias[...])
        delta = jnp.matmul(jnp.matmul(x, self.lora_A), self.lora_B) * self.scale
        return y + delta


def apply_lora(model, cfg: LoRAConfig, *, rngs: nnx.Rngs) -> int:
    count = [0]

    def visit(obj):
        if isinstance(obj, nnx.List):
            for sub in obj:
                visit(sub)
            return
        if not isinstance(obj, nnx.Module):
            return
        for name in list(vars(obj).keys()):
            child = getattr(obj, name, None)
            if name in cfg.target_modules and hasattr(child, "kernel"):
                setattr(obj, name, LoRALinear(child, cfg.rank, cfg.alpha, rngs=rngs))
                count[0] += 1
            else:
                visit(child)

    visit(model)
    return count[0]


def freeze_non_lora_params(model) -> int:
    n = [0]

    def visit(obj, in_lora=False):
        if isinstance(obj, LoRALinear):
            for name in list(vars(obj).keys()):
                visit(getattr(obj, name), in_lora=True)
            return
        if isinstance(obj, nnx.List):
            for sub in obj:
                visit(sub, in_lora=in_lora)
            return
        if isinstance(obj, nnx.Module):
            for name in list(vars(obj).keys()):
                child = getattr(obj, name, None)
                if isinstance(child, nnx.Param) and not in_lora:
                    setattr(obj, name, nnx.Variable(jnp.asarray(child[...])))
                    n[0] += 1
                else:
                    visit(child, in_lora=in_lora)

    visit(model)
    return n[0]
