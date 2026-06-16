"""Self-containment guard for B2 (internal-TPU inference).

The internal side runs with PYTHONPATH=fork/src ONLY — no ddrive_jax. This test asserts that
importing the eval_sasd inference package pulls in ZERO ddrive_jax modules, that flax.nnx is
importable (the sampler is NNX), and that the public API resolves. Run:

  PYTHONPATH=/home/kaiwen/jax-dlm-baseline/maxtext-dlm-fork/src \
    python -m maxtext.diffusion.tests.eval_sasd_import_test
  (or: pytest src/maxtext/diffusion/tests/eval_sasd_import_test.py)
"""
import sys


def test_eval_sasd_is_self_contained():
    pre = [m for m in sys.modules if m == "ddrive_jax" or m.startswith("ddrive_jax.")]
    assert not pre, f"ddrive_jax was already imported before the test: {pre}"

    import flax.nnx  # noqa: F401 — the validated sampler is NNX; must be available in the fork env

    import maxtext.diffusion.eval_sasd as E

    leaked = [m for m in sys.modules if m == "ddrive_jax" or m.startswith("ddrive_jax.")]
    assert not leaked, f"eval_sasd leaked ddrive_jax imports (not self-contained): {leaked}"

    for name in ("run_eval", "run_parity", "mm_section_diffusion_sample", "decode_generation",
                 "block_ranges_from_rbi", "load_fast_ddrive_text", "load_fast_ddrive_vit",
                 "Qwen25TextConfig", "Qwen25TextModel", "VisionConfig", "VisionTransformer"):
        assert hasattr(E, name), f"missing public symbol: {name}"

    print("EVAL_SASD_SELFCONTAINED_PASS")


if __name__ == "__main__":
    test_eval_sasd_is_self_contained()
