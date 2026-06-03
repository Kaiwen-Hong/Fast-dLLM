"""CPU unit tests for the SASD noising (audit finding #2): scaffold freeze, im_end always
masked, complementary inverse (modulo im_end), Beta rate, edge cases.
Run: JAX_PLATFORMS=cpu python jax_ddrive/tests/test_noising.py"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
sys.path.insert(0, "/home/kaiwen/Desktop/research/Fast-dLLM/jax_ddrive")
from ddrive_jax.diffusion.noise import make_batch, MASK_ID, IM_END


def _sample(alpha=1.0, beta=1.0):
    L = 8
    ids = np.array([100, 101, 200, 201, IM_END, 203, 204, 205], np.int64)
    labels = ids.copy(); labels[:2] = -100                 # response = positions 2..7
    rbi = np.array([-1, -1, 0, 0, 0, 1, 1, 1], np.int32)
    scaff = np.array([0, 0, 1, 0, 0, 0, 1, 0], bool)       # positions 2 and 6 frozen
    return dict(input_ids=ids, labels=labels, rbi=rbi, scaffold=scaff,
                weight_vec=np.ones(L, np.float32),
                block_alpha=np.array([alpha, alpha], np.float32),
                block_beta=np.array([beta, beta], np.float32)), L


def test_scaffold_frozen_and_im_end():
    s, L = _sample()
    rng = np.random.default_rng(0)
    scaff_idx = np.where(s["scaffold"])[0]
    for _ in range(200):
        ifn, lfn, ol, w = make_batch(s, rng)
        noisy = ifn[0, :L]                                  # primary noisy half
        # scaffold tokens are never masked
        assert np.all(noisy[scaff_idx] == s["input_ids"][scaff_idx])
        # im_end (pos 4, response) is ALWAYS masked in both primary and complementary
        assert noisy[4] == MASK_ID and ifn[1, 4] == MASK_ID
    print("[ok] test_scaffold_frozen_and_im_end")


def test_complementary_inverse():
    s, L = _sample()
    rng = np.random.default_rng(1)
    resp = s["labels"] != -100
    free = resp & (~s["scaffold"]) & (s["input_ids"] != IM_END)   # non-im_end, non-scaffold response
    for _ in range(100):
        ifn, *_ = make_batch(s, rng)
        m1 = (ifn[0, :L] == MASK_ID)
        m2 = (ifn[1, :L] == MASK_ID)
        # on free positions, complementary is the exact inverse of primary
        assert np.all(m1[free] != m2[free])
    print("[ok] test_complementary_inverse")


def test_beta_rate_and_edges():
    s, L = _sample(alpha=1.0, beta=1.0)                     # Beta(1,1) = uniform -> ~0.5 mask rate
    rng = np.random.default_rng(2)
    free = (s["labels"] != -100) & (~s["scaffold"]) & (s["input_ids"] != IM_END)
    rates = []
    for _ in range(400):
        ifn, *_ = make_batch(s, rng)
        rates.append((ifn[0, :L][free] == MASK_ID).mean())
    assert 0.35 < float(np.mean(rates)) < 0.65, float(np.mean(rates))
    # edge cases must not crash
    for scf in (np.ones(8, bool), np.zeros(8, bool)):
        e = dict(s); e["scaffold"] = scf
        make_batch(e, rng)
    print("[ok] test_beta_rate_and_edges")


if __name__ == "__main__":
    test_scaffold_frozen_and_im_end()
    test_complementary_inverse()
    test_beta_rate_and_edges()
    print("\nALL NOISING TESTS PASS")
