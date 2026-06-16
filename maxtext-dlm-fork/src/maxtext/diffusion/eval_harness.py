"""SASD eval harness entry point.

B2 fills this in: the self-contained, internal-TPU-ready inference lives in `eval_sasd/`
(vendored validated NNX sampler + bf16-hardened loaders + driver). This module just
re-exports the public entry points so existing callers / an internal eval framework can
`from maxtext.diffusion.eval_harness import run_eval, run_parity`.

  run_eval   — generate one sample from a B1-exported snapshot + T2 scalar metrics
  run_parity — frozen-ViT embedding parity (fp32 vs bf16 vs reference)

See eval_sasd/__init__.py and the docs (jax_ddrive/docs) for the full pipeline.
"""
from maxtext.diffusion.eval_sasd.driver import run_eval
from maxtext.diffusion.eval_sasd.embedding_parity import run_parity

__all__ = ["run_eval", "run_parity"]
