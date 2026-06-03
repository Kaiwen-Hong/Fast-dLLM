"""Orbax checkpointing for the Flax-NNX model state (TPU/multi-host-ready,
unlike the reference project's single-host pickle). Saves the full nnx.state
(Params + frozen Variables), so LoRA + base both round-trip.

    save_state(model, "/abs/dir/step_000100")
    restore_into(model, "/abs/dir/step_000100")   # in-place
"""
from __future__ import annotations

import jax
import orbax.checkpoint as ocp
from flax import nnx


def save_state(model, path: str) -> None:
    state = nnx.state(model)
    with ocp.StandardCheckpointer() as ckptr:
        ckptr.save(path, state)
        ckptr.wait_until_finished()


def restore_into(model, path: str):
    """Restore a checkpoint into `model` in place (shapes/dtypes from the live model
    act as the target/abstract pytree). Returns the model."""
    abstract = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, nnx.state(model))
    with ocp.StandardCheckpointer() as ckptr:
        restored = ckptr.restore(path, target=abstract)
    nnx.update(model, restored)
    return model
