#!/usr/bin/env python3
"""TPU smoke test for the refactored differentiable ViT body (trainable-ViT Step 1).

Validates that ``VisionTransformer.body`` (the pure-jax half produced by the
precompute_structural / body split) COMPILES and runs forward AND backward on a
TPU, with finite, nonzero gradients to every ViT param. This de-risks the
TPU-specific concerns (XLA-TPU compilation of the window-reorder gathers, the
per-segment boolean attention mask, fp32 softmax, and reverse-permutation) BEFORE
the full trainable-ViT integration is wired into the MaxText train graph.

Random init by default — needs ONLY the ddrive_jax code + jax[tpu] (no weights, no
data). Pass --snapshot <release HF dir> + --ar <arrayrecord shard> to additionally
check the in-graph forward reproduces the pre-baked embeds (cosine, goal-2 sanity).

Run on any TPU VM:
  pip install -U "jax[tpu]" flax jaxtyping numpy \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  PYTHONPATH=<...>/jax_ddrive python tpu_vit_body_smoke.py            # compile+grad smoke
  PYTHONPATH=<...>/jax_ddrive python tpu_vit_body_smoke.py \
    --snapshot <release_snapshot> --ar <val-00000-of-00008.arrayrecord>   # + cosine
"""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from ddrive_jax.models.vision_qwen25vl import VisionConfig, VisionTransformer


def _cos(a, b):
    a = np.asarray(a, np.float32).ravel(); b = np.asarray(b, np.float32).ravel()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _real_sample(ar_path):
    import tensorflow as tf
    from array_record.python.array_record_module import ArrayRecordReader
    f = tf.train.Example.FromString(ArrayRecordReader(ar_path).read([0])[0]).features.feature
    g = lambda n, dt: tf.io.parse_tensor(f[n].bytes_list.value[0], dt).numpy()
    return g("pixel_values", tf.float16), g("image_grid_thw", tf.int64), g("image_embeds", tf.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--n_imgs", type=int, default=3, help="random-mode: imgs at 16x14 grid")
    ap.add_argument("--snapshot", default=None, help="optional release HF dir -> load real ViT")
    ap.add_argument("--ar", default=None, help="optional arrayrecord shard -> real pixels + cosine")
    args = ap.parse_args()
    dt = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32

    print("jax devices:", jax.devices(), flush=True)
    if not any(d.platform == "tpu" for d in jax.devices()):
        print("WARNING: no TPU visible — this is the CPU/GPU fallback, not a TPU validation.", flush=True)

    cfg = VisionConfig(dtype=dt)

    # weights: real (release) if --snapshot given, else random init.
    if args.snapshot:
        from ddrive_jax.convert.hf_to_jax import load_fast_ddrive_vit
        out = load_fast_ddrive_vit(args.snapshot, cfg, dtype=dt)
        vit = out[0] if isinstance(out, tuple) else out
        print("[init] loaded release ViT weights", flush=True)
    else:
        vit = VisionTransformer(cfg, rngs=nnx.Rngs(0))
        print("[init] random ViT weights", flush=True)

    # inputs: real sample if --ar given, else random pixels at a fixed grid.
    ie_stored = None
    if args.ar:
        pv, grid, ie_stored = _real_sample(args.ar)
        pv = jnp.asarray(pv, dt)
    else:
        grid = np.array([[1, 16, 14]] * args.n_imgs, np.int64)
        N = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
        patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        pv = jax.random.normal(jax.random.PRNGKey(0), (N, patch_dim), dt)

    struct = vit.precompute_structural(grid)   # HOST (numpy geometry -> jnp arrays)

    @nnx.jit
    def fwd(m, pv, struct):
        return m.body(pv, struct)

    e = fwd(vit, pv, struct)
    e.block_until_ready()
    e_finite = bool(jnp.all(jnp.isfinite(e.astype(jnp.float32))))
    print(f"[fwd] TPU forward OK -> embeds {tuple(e.shape)} {e.dtype}, finite={e_finite}", flush=True)
    if ie_stored is not None:
        c = _cos(e, ie_stored)
        print(f"[fwd] cosine(in-graph body, pre-baked embeds) = {c:.5f}  (expect >=0.999)", flush=True)

    @nnx.jit
    def grads(m, pv, struct):
        def loss(mm):
            return jnp.sum(mm.body(pv, struct).astype(jnp.float32) ** 2)
        return nnx.grad(loss)(m)

    g = grads(vit, pv, struct)
    leaves = jax.tree_util.tree_leaves(nnx.state(g, nnx.Param))
    norms = [float(jnp.linalg.norm(x.astype(jnp.float32))) for x in leaves]
    nz = sum(n > 0 for n in norms)
    fin = all(np.isfinite(norms))
    print(f"[bwd] TPU grad OK -> {len(norms)} leaves, {nz} nonzero, finite={fin}, "
          f"norm min/med/max = {min(norms):.2e}/{np.median(norms):.2e}/{max(norms):.2e}", flush=True)

    cosine_ok = (ie_stored is None) or (_cos(e, ie_stored) >= 0.999)
    ok = e_finite and fin and nz == len(norms) and cosine_ok
    print("TPU_VIT_BODY_SMOKE:", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
