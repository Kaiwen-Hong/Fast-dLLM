"""Phase 13: scheduler unit tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from maxtext.diffusion.schedulers import (
    BaseScheduler, CosineScheduler, LinearScheduler, PolyScheduler,
    make_scheduler, transfer_counts,
)


def test_linear_basics():
    s = LinearScheduler()
    assert float(s.alpha(0.0)) == pytest.approx(1.0, abs=1e-6)
    assert float(s.alpha(1.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(s.alpha_derivative(0.5)) == pytest.approx(-1.0, abs=1e-6)
    # weight(t) = 1/t (with eps in denom)
    w = float(s.weight(0.5))
    assert abs(w - 2.0) < 1e-3


def test_cosine_basics():
    s = CosineScheduler()
    assert float(s.alpha(0.0)) == pytest.approx(1.0, abs=1e-6)
    assert float(s.alpha(1.0)) == pytest.approx(0.0, abs=1e-6)
    # cos at π/4 = √2/2
    assert float(s.alpha(0.5)) == pytest.approx(jnp.cos(jnp.pi / 4), abs=1e-5)


def test_poly_basics():
    s = PolyScheduler(p=2.0)
    assert float(s.alpha(0.0)) == pytest.approx(1.0, abs=1e-6)
    assert float(s.alpha(1.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(s.alpha(0.5)) == pytest.approx(0.75, abs=1e-6)


def test_make_scheduler_dispatch():
    assert isinstance(make_scheduler(None), LinearScheduler)
    assert isinstance(make_scheduler("linear"), LinearScheduler)
    assert isinstance(make_scheduler("cosine"), CosineScheduler)
    s = make_scheduler("poly3")
    assert isinstance(s, PolyScheduler) and s.p == 3.0


def test_reverse_mask_prob_linear():
    s = LinearScheduler()
    # For linear: (1-α(s))/(1-α(t)) = s/t
    val = float(s.reverse_mask_prob(0.3, 0.6))
    assert val == pytest.approx(0.5, abs=1e-3)


def test_transfer_counts_linear_uniform():
    s = LinearScheduler()
    c = transfer_counts(s, block_size=32, steps_per_block=8)
    assert int(c.sum()) == 32
    # uniform 4-each
    assert all(int(v) == 4 for v in c)


def test_transfer_counts_linear_remainder():
    s = LinearScheduler()
    c = transfer_counts(s, block_size=32, steps_per_block=5)
    # 32 / 5 = 6 base + 2 remainder; expected [7,7,6,6,6]
    assert int(c.sum()) == 32
    assert list(int(v) for v in c) == [7, 7, 6, 6, 6]


def test_transfer_counts_cosine_sums():
    s = CosineScheduler()
    for bs, st in [(32, 8), (16, 5), (10, 10)]:
        c = transfer_counts(s, bs, st)
        assert int(c.sum()) == bs, (bs, st, c)
        assert all(int(v) >= 0 for v in c)


def test_transfer_counts_poly_sums():
    s = PolyScheduler(p=2.0)
    c = transfer_counts(s, block_size=32, steps_per_block=8)
    assert int(c.sum()) == 32
