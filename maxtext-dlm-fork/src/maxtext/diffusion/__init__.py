"""Discrete-diffusion training surface.

Public API (Phase 5 = stubs, Phase 6 = real):
- forward_mask(x_0, t, mask_token_id, rng_key) -> (x_t, mask)
- mdlm_loss(logits, x_0, x_t, t, mask_token_id, pad_token_id) -> scalar
- unmask_step(model, x_t, t_curr, t_next, mask_token_id, rng_key) -> x_next
"""
from maxtext.diffusion.mdlm import forward_mask, mdlm_loss, mdlm_loss_inner
from maxtext.diffusion.sampler import (
    SamplerConfig,
    generate,
    transfer_counts_linear,
    unmask_step,
)
from maxtext.diffusion.schedulers import (
    BaseScheduler,
    CosineScheduler,
    LinearScheduler,
    PolyScheduler,
    make_scheduler,
    transfer_counts,
)

__all__ = [
    "forward_mask", "mdlm_loss", "mdlm_loss_inner",
    "SamplerConfig", "generate", "transfer_counts_linear", "unmask_step",
    "BaseScheduler", "LinearScheduler", "CosineScheduler", "PolyScheduler",
    "make_scheduler", "transfer_counts",
]
