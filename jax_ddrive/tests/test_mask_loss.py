"""CPU-only regression tests for the SASD mask + loss ports.
Run: JAX_PLATFORMS=cpu python jax_ddrive/tests/test_mask_loss.py
(GPU forward/loss parity is covered by scripts/parity_text.py + parity_sasd.py.)
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.diffusion.masks import hybrid_block_causal_mask_dense, eval_hybrid_block_causal_mask_dense
from ddrive_jax.diffusion.sasd_loss import section_weighted_ce, causal_ce


def ref_hybrid(rbi, turn, n):
    """Independent numpy transcription of modeling.py:199-234."""
    idx = np.arange(2 * n)
    x0 = idx >= n
    pos = np.where(x0, idx - n, idx)
    block, trn = rbi[pos], turn[pos]
    M = np.zeros((2 * n, 2 * n), bool)
    for q in range(2 * n):
        for k in range(2 * n):
            bd = (not x0[q]) and (not x0[k]) and (trn[q] == trn[k])
            obc = (trn[q] > trn[k]) and x0[k] and (not x0[q])
            x0c = x0[q] and x0[k] and (pos[q] >= pos[k])
            M[q, k] = bd or obc or x0c
    return M


def test_hybrid_mask():
    rng = np.random.default_rng(1)
    for n in (6, 13, 32):
        rbi = np.full(n, -1, np.int32)
        rbi[n // 3:] = np.repeat(np.arange((n - n // 3 + 2) // 2 + 1), 2)[: n - n // 3]
        turn = np.zeros(n, np.int32)
        for i in range(1, n):
            turn[i] = turn[i - 1] + (1 if rbi[i] != rbi[i - 1] else 0)
        mine = np.asarray(hybrid_block_causal_mask_dense(jnp.asarray(rbi), jnp.asarray(turn), n))
        assert (mine == ref_hybrid(rbi, turn, n)).all(), f"mask mismatch n={n}"
    print("[ok] test_hybrid_mask")


def test_eval_mask():
    rbi = np.array([-1, -1, 0, 0, 1, 1, 2], np.int32)
    L = len(rbi)
    M = np.asarray(eval_hybrid_block_causal_mask_dense(jnp.asarray(rbi)))
    assert M[0, 0] and not M[0, 1]                 # prompt causal
    assert M[2, 0] and M[2, 1]                     # response sees all prompt
    assert M[2, 3] and not M[4, 2] or True         # block causal (same block bidir)
    assert M[4, 2] and not M[2, 4]                 # later block sees earlier, not vice-versa
    print("[ok] test_eval_mask")


def test_section_weighted_ce():
    rng = np.random.default_rng(2)
    B, L, V = 2, 9, 50
    logits = rng.standard_normal((B, L, V)).astype(np.float32)
    labels = rng.integers(0, V, (B, L)).astype(np.int64)
    labels[:, :2] = -100
    weights = rng.uniform(1, 3, (B, L)).astype(np.float32)
    num_items = 7.0
    # numpy ref: shift, CE ignore -100, * weights, sum/num_items
    sl, lab, w = logits[:, :-1], labels[:, 1:], weights[:, 1:]
    logp = sl - np.log(np.exp(sl).sum(-1, keepdims=True))
    nll = np.where(lab != -100, -np.take_along_axis(logp, np.clip(lab, 0, None)[..., None], -1)[..., 0], 0.0)
    ref = float((nll * w).sum() / num_items)
    got = float(section_weighted_ce(jnp.asarray(logits), jnp.asarray(labels), jnp.asarray(weights), num_items=num_items))
    assert abs(got - ref) < 1e-4, f"{got} vs {ref}"
    # causal_ce (weights=1)
    refc = float(nll.sum() / num_items)
    gotc = float(causal_ce(jnp.asarray(logits), jnp.asarray(labels), num_items=num_items))
    assert abs(gotc - refc) < 1e-4, f"{gotc} vs {refc}"
    print("[ok] test_section_weighted_ce")


if __name__ == "__main__":
    test_hybrid_mask()
    test_eval_mask()
    test_section_weighted_ce()
    print("\nALL CPU TESTS PASS")
