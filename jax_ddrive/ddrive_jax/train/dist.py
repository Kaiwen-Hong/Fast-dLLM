"""Distributed bootstrap + process-0 guards for the FSDP harness.

`init_distributed()` calls `jax.distributed.initialize()` ONLY in a real multi-host
setting (env `DDRIVE_MULTIHOST=1`, or a TPU-pod env, or JAX already sees >1 process).
On the CPU 8-device emulation (single process, 8 local devices) it is a no-op -- the 8
devices are the multi-host *proxy* but JAX sees one controller process.

`set_mesh(mesh)` resolves the JAX-0.10 mesh-context helper (`jax.sharding.use_mesh`
when present, else `jax.sharding.set_mesh`) so the rest of the harness imports one name.
"""
from __future__ import annotations

import os

import jax

from ddrive_jax import sharding

# JAX 0.10: the in-types mesh context is `use_mesh` on newer builds, `set_mesh` here.
mesh_context = getattr(jax.sharding, "use_mesh", None) or getattr(jax.sharding, "set_mesh")


def _already_multiprocess() -> bool:
    try:
        return jax.process_count() > 1
    except Exception:
        return False


def _tpu_pod_env_present() -> bool:
    """GKE / QueuedResource style env that implies a real pod (not CPU emulation)."""
    return bool(os.environ.get("TPU_WORKER_ID")) or bool(os.environ.get("MEGASCALE_NUM_SLICES"))


def init_distributed() -> None:
    """Init JAX distributed ONLY in a real multi-host setting; else no-op."""
    if _already_multiprocess():
        return  # a previous call / launcher already brought up >1 process
    if os.environ.get("DDRIVE_MULTIHOST", "0") == "1" or _tpu_pod_env_present():
        jax.distributed.initialize()  # reads coordinator/process_count/process_id from env
    # else: single controller (local GPU / CPU-emulation / single TPU host) -> no-op


def build_mesh(n_fsdp: int, n_tp: int = 1):
    """2D ('fsdp','tp') mesh via sharding.make_mesh. Pure-FSDP uses n_tp=1."""
    return sharding.make_mesh(n_fsdp, n_tp)


def is_primary() -> bool:
    return jax.process_index() == 0


def pprint(*a, **k) -> None:
    if is_primary():
        k.setdefault("flush", True)
        print(*a, **k)
