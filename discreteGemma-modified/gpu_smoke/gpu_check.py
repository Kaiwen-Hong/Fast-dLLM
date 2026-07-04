"""Step 1: prove JAX sees the RTX 5090 (sm_120 / Blackwell) and actually
compiles+runs a kernel on it. No gemma code involved yet."""
import os
import jax
import jax.numpy as jnp

print("jax        :", jax.__version__)
import jaxlib
print("jaxlib     :", jaxlib.__version__)
print("backend    :", jax.default_backend())
print("devices    :", jax.devices())

# Force a real kernel to compile & run for this GPU's SASS (sm_120).
x = jnp.ones((1024, 1024), jnp.float32)
y = (x @ x).block_until_ready()
print("matmul[0,0]:", float(y[0, 0]), "(expect 1024.0)")

# bf16 too (the model's compute dtype).
xb = jnp.ones((512, 512), jnp.bfloat16)
yb = (xb @ xb).block_until_ready()
print("bf16 mm[0,0]:", float(yb[0, 0]), "(expect 512.0)")

d = jax.devices()[0]
print("device_kind:", d.device_kind, "| platform:", d.platform)
print("PASS: JAX runs on GPU" if d.platform == "gpu" else "FAIL: not on GPU")
