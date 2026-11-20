"""Deferred resolution of string loss weights (``HSet.resolve_after``)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import curlew
from curlew.core import (
    CSet,
    HSet,
    RESOLVE_FLOORS,
    resolve_floor,
    resolve_string_weights,
    take_weights_changed,
)
from curlew.kinematics.restoration import RestorationField


@pytest.fixture(autouse=True)
def _defaults():
    curlew.default_dim = 2
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def test_default_is_off_and_survives_zero():
    assert HSet().resolve_after == 0
    h = HSet().zero(grad_loss="1.")
    assert h.resolve_after == 0
    assert resolve_floor(h, "eq_loss") == RESOLVE_FLOORS["eq_loss"]   # zero() leaves floors usable
    h = HSet().zero(eq_loss="1.", resolve_after=5, resolve_floor={"eq_loss": 4.0})
    assert h.resolve_after == 5 and resolve_floor(h, "eq_loss") == 4.0


def test_small_term_waits_until_K_then_uses_its_value_or_floor():
    owner = SimpleNamespace()
    H = HSet().zero(grad_loss="1.", eq_loss="0.1", resolve_after=3)
    w = resolve_string_weights(owner, H, {"grad_loss": 2.0, "eq_loss": 1e-6})
    assert w == {"grad_loss": 0.5, "eq_loss": 0.0}
    assert H.grad_loss == 0.5 and H.eq_loss == "0.1"
    assert not take_weights_changed(owner)
    for _ in range(2):                                              # evaluations 1 and 2
        assert resolve_string_weights(owner, H, {"grad_loss": 1.0, "eq_loss": 9.0})["eq_loss"] == 0.0
    w = resolve_string_weights(owner, H, {"grad_loss": 1.0, "eq_loss": 50.0})   # evaluation K
    assert w["eq_loss"] == pytest.approx(0.1 / 50.0) and H.eq_loss == pytest.approx(0.1 / 50.0)
    assert take_weights_changed(owner) and not take_weights_changed(owner)

    H = HSet().zero(grad_loss="1.", eq_loss="0.1", resolve_after=1)
    resolve_string_weights(owner := SimpleNamespace(), H, {"grad_loss": 2.0, "eq_loss": 0.0})
    w = resolve_string_weights(owner, H, {"grad_loss": 1.0, "eq_loss": 0.2})
    assert w["eq_loss"] == pytest.approx(0.1 / RESOLVE_FLOORS["eq_loss"])   # floored


def test_numeric_term_drives_the_warm_up():
    H = HSet().zero(grad_loss=1.0, eq_loss="0.1", resolve_after=10)
    w = resolve_string_weights(SimpleNamespace(), H, {"grad_loss": 0.3, "eq_loss": 1e-9})
    assert w == {"grad_loss": 1.0, "eq_loss": 0.0}


def test_all_terms_small_resolve_against_floors_at_once():
    H = HSet().zero(grad_loss="1.", eq_loss="0.1", resolve_after=10)
    w = resolve_string_weights(SimpleNamespace(), H, {"grad_loss": 1e-7, "eq_loss": 1e-9})
    assert w["grad_loss"] == pytest.approx(1.0 / RESOLVE_FLOORS["grad_loss"])
    assert w["eq_loss"] == pytest.approx(0.1 / RESOLVE_FLOORS["eq_loss"])


def _flat_horizon_field(resolve_after, seed=3):
    """Identity restoration of a flat horizon with dipping bedding: eq is satisfied at the start."""
    from curlew.kinematics.euler import euler_velocity

    torch.manual_seed(seed)
    xs = np.linspace(0.0, 10.0, 25)
    horizon = np.stack([xs, np.zeros_like(xs)], 1)
    gp = np.stack([np.linspace(1.0, 9.0, 8), np.zeros(8)], 1)
    dip = math.radians(30.0)
    gv = np.tile([-math.sin(dip), math.cos(dip)], (8, 1))
    vel = euler_velocity(2, n_features=32, seed=7, length_scale_range=(4.0, 12.0))
    H = HSet().zero(grad_loss="1.", eq_loss="0.1", resolve_after=resolve_after)
    field = RestorationField("r0", H=H, input_dim=2, velocity=vel, n_steps=1, depth_axis=1,
                             learning_rate=1e-2, sigma_floor=0.0, init_std=1e-6)
    field.bind(CSet(gp=gp, gv=gv, eq=[horizon]))
    return field


def test_resolve_after_zero_keeps_the_old_weights():
    """Default: weights are w / (first value), so a satisfied eq gets an enormous weight."""
    field = _flat_horizon_field(0)
    peb = field.loss(transform=False)
    v_grad = float(peb.losses["r0"]["grad_loss"].detach())
    assert field.H.grad_loss == pytest.approx(1.0 / v_grad)
    if isinstance(field.H.eq_loss, str):                             # eq exactly 0: not resolved yet
        assert "eq_loss" not in peb.losses["r0"]
    else:
        assert field.H.eq_loss > 1e3


def test_restore_defers_a_satisfied_eq_and_fit_keeps_post_resolution_state():
    K = 20
    field = _flat_horizon_field(K)
    peb = field.loss(transform=False)
    assert "eq_loss" not in peb.losses["r0"] and field.H.eq_loss == "0.1"
    assert isinstance(field.H.grad_loss, float)
    grad0 = float(peb.losses["r0"]["grad_loss"].detach())
    _, best = field.fit(80, transform=False, vb=False, early_stop=None, best=True)
    assert isinstance(field.H.eq_loss, float) and field.H.eq_loss <= 0.1 / RESOLVE_FLOORS["eq_loss"]
    assert "eq_loss" in best.losses["r0"]            # best state chosen after eq switched on
    assert float(best.losses["r0"]["grad_loss"]) < grad0


def test_scalar_field_defers_too():
    from curlew.fields.series import FSF

    torch.manual_seed(0)
    xs = np.linspace(0.0, 10.0, 12)
    H = HSet().zero(grad_loss="1.", eq_loss="0.1", resolve_after=4)
    f = FSF("f", H=H, input_dim=2, rff_features=16, length_scale_range=(4.0, 12.0), seed=1)
    gp = np.stack([xs, np.full_like(xs, 1.0)], 1)
    f.bind(CSet(gp=gp, gv=np.tile([0.0, 1.0], (12, 1)), eq=[np.stack([xs, np.zeros_like(xs)], 1)]))
    f.H.resolve_floor = {"eq_loss": 1e30}                            # force eq below its floor
    f.loss()
    assert f.H.eq_loss == "0.1" and isinstance(f.H.grad_loss, float)
    for _ in range(4):
        f.loss()
    assert f.H.eq_loss == pytest.approx(0.1 / 1e30)
    assert take_weights_changed(f)
