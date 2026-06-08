"""Path-A self-contained Flax-NNX FSDP training harness for Fast-dDrive.

Modules:
  dist            -> guarded jax.distributed.initialize(), mesh build, process-0 guards
  checkpoint_mgr  -> Orbax CheckpointManager wrapper (composite: params/opt/meta/grain)
  train_tpu       -> the driver (proxy on CPU 8-device emulation; real on GPU/TPU)

The CPU 8-device emulation is the multi-host proxy: same code, same shardings, run on
8 local CPU devices (single process). Set DDRIVE_MULTIHOST=1 on a real pod.
"""
