# maxtext.diffusion — MDLM in MaxText

Lifted from `~/jax-dlm-baseline/maxtext-dlm/MaxText/diffusion/` (Phase 6, 11, 13, 14 of the JAX MDLM baseline). Backbone-agnostic; works against MaxText's Qwen3 / Llama / Gemma decoders without architectural change.

## Public API

```python
from maxtext.diffusion import (
    forward_mask, mdlm_loss, mdlm_loss_inner,        # mdlm.py
    SamplerConfig, generate, transfer_counts_linear,  # sampler.py
    BaseScheduler, LinearScheduler, CosineScheduler,
    PolyScheduler, make_scheduler, transfer_counts,   # schedulers.py
)
```

## Status (all upstream changes squashed)
- `mdlm.py` — `forward_mask` + `mdlm_loss(..., loss_valid=None)`. The optional `loss_valid` lets SFT exclude prompt tokens.
- `sampler.py` — `generate(model, prompt, *, mask_token_id, eos_token_id, cfg, rng_key)`. Block-by-block low-confidence remasking. Argmax + Gumbel-Max paths, plug-in scheduler.
- `schedulers.py` — Linear / Cosine / Poly + factory. `transfer_counts(scheduler, block_size, steps_per_block)` returns the per-step reveal count.

13 unit tests (`tests/{mdlm,schedulers}_test.py`) pass inside the MaxText namespace:

```
PYTHONPATH=src python -m pytest src/maxtext/diffusion/tests/ \
    --override-ini="testpaths=src/maxtext/diffusion/tests" \
    --override-ini="python_files=*_test.py"
```

The sampler test (which depends on a Bonsai LLaDA reference backbone) is omitted from this fork; it stays in the `maxtext-dlm/` project and Waymo-side will rewrite it against MaxText's actual decoder.

## To wire MDLM into MaxText's training loop (Waymo-side)

In `src/maxtext/trainers/pre_train/train.py`, around line 90's `loss_fn`, add an `objective == "mdlm"` branch. The minimal diff:

```python
# Top of file (with existing imports)
from maxtext.diffusion import forward_mask, mdlm_loss

# Inside loss_fn, *before* the model.apply call (for Linen) or model() call (for NNX),
# replace the input tokens with the diffusion-noised version:
if config.objective == "mdlm":
    rng1, rng_mask = jax.random.split(rng1)
    rng_mask, rng_t = jax.random.split(rng_mask)
    B = data["inputs"].shape[0]
    t = jax.random.uniform(rng_t, (B,), minval=1e-3, maxval=1.0)
    x_t, _ = forward_mask(data["inputs"], t, config.mask_token_id, rng_mask)
    data["inputs"] = x_t
    # The targets stay the *clean* tokens (data["targets"]), unchanged.
    # Attention should be bidirectional; set config.attention_mode='bidirectional'
    # in your YAML and patch layers/attentions.py to honor it.

# After the forward returns `logits`, replace the cross-entropy block (lines 160-183)
# with the MDLM loss when objective is mdlm:
if config.objective == "mdlm":
    loss = mdlm_loss(
        logits, x_0=data["targets"], x_t=data["inputs"],
        t=t, mask_token_id=config.mask_token_id,
        pad_token_id=config.pad_token_id,
    )
    xent_sum = loss * total_weights   # so the existing /total_weights divides cleanly
    total_z_loss = jnp.zeros(())       # no z-loss for MDLM
    # skip MTP / MoE LB / vocab_tiling — those are AR-only features
else:
    # ...existing AR cross-entropy code...
```

Add to YAML config:
```yaml
objective: mdlm
mask_token_id: 151669       # depends on tokenizer; len(tokenizer)
attention_mode: bidirectional
```

**No `attentions.py` patch needed** — MaxText already supports bidirectional attention via `AttentionType.FULL` (defined in `common/common_types.py:120`). At `layers/attention_op.py:692`, the causal mask is only built when `attention_type != AttentionType.FULL`, so setting `attention_type: "full"` in the YAML disables it cleanly. Just add `attention_type: "full"` to your MDLM config alongside `objective: "mdlm"`.

For the sampler, add a `decode_mdlm.py` mirroring our Bonsai version (`maxtext-dlm/MaxText/sample.py`) but using MaxText's model/tokenizer setup instead of Bonsai's.
