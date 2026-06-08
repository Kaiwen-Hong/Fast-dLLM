"""Verify the REAL pretrained 3.75B model's params SHARD correctly across multiple devices under
the harness FSDP rule (CPU emulation; build-only, NO forward/backward -> memory-safe). This closes
the 'real model x multi-device sharding' gap. Forward/backward NUMERICS at multi-device scale are
the TPU pod's job (proxy verifies the math; the GPU smoke verifies real-model forward+backward on 1
device).

Run (CPU emulation, >=2 devices):
  export PYTHONPATH=/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive JAX_PLATFORMS=cpu
  export XLA_FLAGS="--xla_force_host_platform_device_count=4"
  /home/kaiwen/jax-dlm-baseline/.venv/bin/python jax_ddrive/scripts/check_real_fsdp_shard.py
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.train import train_tpu as T, dist


def main():
    cfg = T.HarnessConfig(proxy=False, dtype=jnp.bfloat16, n_fsdp=2, n_tp=1,
                          opt="adafactor", mrope_section=(16, 24, 24), seed=0)
    dist.init_distributed()
    print(f"devices={jax.device_count()}; building REAL harness sharded over n_fsdp=2 (no fwd)...", flush=True)
    h = T.build_harness(cfg)
    flat = dict(nnx.to_flat_state(h.params))
    kernels = [k for k in flat if k and k[-1] == "kernel" and "embed" not in str(k)]
    sharded = []
    for k in kernels:
        spec = flat[k][...].sharding.spec
        if any(("fsdp" in (str(s) if s is not None else "")) for s in spec):
            sharded.append(k)
    q = flat[("layers", 0, "self_attn", "q_proj", "kernel")][...]
    ndev = len({d for d in q.sharding.device_set})
    emb = tuple(flat[("embed_tokens", "embedding")][...].sharding.spec)
    nparam = sum(int(np.prod(v[...].shape)) for v in flat.values())
    # optimizer state also sharded?
    oflat = dict(nnx.to_flat_state(h.opt_state)) if hasattr(h.opt_state, "__iter__") else {}
    print(f"real params total = {nparam/1e9:.2f}B")
    print(f"sharded kernels on 'fsdp' = {len(sharded)}/{len(kernels)}; q_proj across {ndev} devices; "
          f"embedding spec = {emb}")
    ok = len(sharded) >= 20 and ndev == 2 and (emb == () or all(s is None for s in emb))
    print("REAL_MODEL_SHARDS_OK" if ok else "REAL_MODEL_SHARDS_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
