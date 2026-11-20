"""Inequality constraints with a minimum separation (``CSet.iq`` margin)."""
import numpy as np
import pytest
import torch

import curlew
from curlew.core import CSet, HSet, iq_margin, iq_pair_loss


@pytest.fixture(autouse=True)
def _defaults():
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def test_margin_is_optional_and_non_negative():
    a, b = np.zeros((2, 3)), np.ones((2, 3))
    assert iq_margin((a, b, ">")) == 0.0
    assert iq_margin((a, b, ">", None)) == 0.0
    assert iq_margin((a, b, ">", 40)) == 40.0
    with pytest.raises(ValueError, match=">= 0"):
        iq_margin((a, b, ">", -1.0))


@pytest.mark.parametrize("rel, sign", [(">", 1.0), ("<", -1.0)])
def test_margin_moves_the_hinge(rel, sign):
    """`>` with margin m penalises start - end < m; `<` penalises start - end > -m."""
    m = 40.0
    low = torch.full((5,), -np.inf) if rel == ">" else torch.zeros(5)
    high = torch.zeros(5) if rel == ">" else torch.full((5,), np.inf)
    shift = torch.full((5,), sign * m)
    gaps = torch.tensor([0.0, 20.0, 40.0, 60.0, -10.0]) * sign      # start - end
    delta, _ = iq_pair_loss(gaps, torch.zeros(5), low, high, shift)
    expected = torch.clamp(gaps * sign - m, max=0.0) ** 2              # shortfall below m
    assert torch.allclose(delta, expected)
    # Without a margin, equal values cost nothing; with one they do.
    zero, _ = iq_pair_loss(torch.zeros(1), torch.zeros(1), low[:1], high[:1])
    assert float(zero) == 0.0
    assert float(delta[0]) == pytest.approx(m ** 2)


def test_cset_builds_signed_shifts_and_keeps_margin_through_transforms():
    rng = np.random.default_rng(0)
    p, q = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    C = CSet(iq=(4, [(p, q, ">", 25.0), (q, p, "<", 10.0), (p, q, ">")]))
    from curlew.fields.series import FSF
    f = FSF("f", H=HSet(), input_dim=3, rff_features=8, length_scale_range=(1.0, 2.0), seed=1)
    f.bind(C)                                                          # binds a copy
    shift = f.C._iq_shift.numpy()
    assert np.allclose(shift[:4], 25.0) and np.allclose(shift[4:8], -10.0) and np.allclose(shift[8:], 0.0)
    moved = C.transform(lambda x: x + 1.0)
    assert [len(t) for t in moved.iq[1]] == [4, 4, 3]
    assert moved.iq[1][0][3] == 25.0
