"""Orbax ``CheckpointManager`` wrapper for the FSDP harness.

Composite layout (4 items), saved every ``save_interval_steps`` with ``max_to_keep`` GC:

  params     -> args.StandardSave(params)      sharded NNX Param pytree (nnx.state(model, nnx.Param))
  opt_state  -> args.StandardSave(opt_state)   the optax pytree (NEW vs checkpoint.py)
  meta       -> args.JsonSave({"step": int, ...})
  grain      -> args.JsonSave(loader.state())  -> {"grain": {"next_index": N}}

Restore builds **sharded abstract** targets (ShapeDtypeStruct + NamedSharding) so Orbax
restores arrays already partitioned across the mesh -- never materialising a full
unsharded copy on one host. This is the multi-host-correct upgrade over checkpoint.py.

Local abs path OR ``gs://bucket/...`` both work unchanged (Orbax handles GCS on a pod).
"""
from __future__ import annotations

import os

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import args as ocp_args


def build_manager(ckpt_dir: str, *, save_interval_steps: int, max_to_keep: int):
    """Construct an async CheckpointManager over a 4-item composite."""
    opts = ocp.CheckpointManagerOptions(
        save_interval_steps=int(save_interval_steps),
        max_to_keep=int(max_to_keep),
        create=True,
        enable_async_checkpointing=True,
    )
    directory = ckpt_dir if "://" in ckpt_dir else os.path.abspath(ckpt_dir)
    return ocp.CheckpointManager(
        directory,
        options=opts,
        item_names=("params", "opt_state", "meta", "grain"),
    )


def save_step(mgr, step: int, params, opt_state, grain_state: dict, *, extra_meta: dict | None = None):
    """Interval-gated composite save (async). `grain_state` is `loader.state()` (JSON-able)."""
    meta = {"step": int(step)}
    if extra_meta:
        meta.update(extra_meta)
    return mgr.save(
        int(step),
        args=ocp_args.Composite(
            params=ocp_args.StandardSave(params),
            opt_state=ocp_args.StandardSave(opt_state),
            meta=ocp_args.JsonSave(meta),
            grain=ocp_args.JsonSave(grain_state),
        ),
    )


def _to_sharded_abstract(pytree, sharding_tree):
    """Pair each leaf's (shape,dtype) with its NamedSharding -> ShapeDtypeStruct target."""
    return jax.tree.map(
        lambda leaf, sh: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sh),
        pytree,
        sharding_tree,
    )


def restore_latest(mgr, abstract_params, abstract_opt, params_sharding, opt_sharding):
    """Restore the latest step into sharded abstract targets.

    Returns ``(step:int, grain_state:dict, params, opt_state)`` or ``None`` for a fresh run.
    `abstract_params` / `abstract_opt` are pytrees of `ShapeDtypeStruct` (from `jax.eval_shape`),
    leaf-aligned with `params_sharding` / `opt_sharding`.
    """
    step = mgr.latest_step()
    if step is None:
        return None
    tgt_params = _to_sharded_abstract(abstract_params, params_sharding)
    tgt_opt = _to_sharded_abstract(abstract_opt, opt_sharding)
    restored = mgr.restore(
        step,
        args=ocp_args.Composite(
            params=ocp_args.StandardRestore(tgt_params),
            opt_state=ocp_args.StandardRestore(tgt_opt),
            meta=ocp_args.JsonRestore(),
            grain=ocp_args.JsonRestore(),
        ),
    )
    return (
        int(restored["meta"]["step"]),
        restored["grain"],
        restored["params"],
        restored["opt_state"],
    )


def wait(mgr) -> None:
    mgr.wait_until_finished()
