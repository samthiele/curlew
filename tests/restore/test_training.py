"""Tests for spectral warm-up (curlew.kinematics.training)."""
import math

import numpy as np
import pytest
import torch

import curlew
from curlew.core import CSet, HSet
from curlew.kinematics.training import SpectralWarmup
from curlew.kinematics.euler import euler_velocity
from curlew.geology import restore

LSR = (481.0, 1926.0)


def make_event():
    curlew.default_dim = 2
    v = euler_velocity(2, n_features=64, n_directions=8, length_scale_range=LSR, stream="alpha",
                       activation="cubic", seed=1, energy_normalised=True)
    k = 2 * np.pi / 900.0
    x = np.linspace(0, 2000, 40)
    eq = [np.c_[x, c + 40 * np.sin(k * x)] for c in (0.0, 100.0)]
    slope = 40 * k * np.cos(k * x)
    gv = np.c_[-slope, np.ones_like(x)] / np.sqrt(1 + slope ** 2)[:, None]
    event = restore("t", C=CSet(gp=eq[0][::4], gv=gv[::4], eq=eq), H=HSet().zero(grad_loss="1.", eq_loss="0.1"),
                    velocity=v, n_steps=0, integrator="shear", depth_axis=1, init_std=0.0, sigma_floor=0.0,
                    signed_normals=True, pin=torch.zeros(2),
                    optimizer={"amplitude": (torch.optim.SGD, dict(lr=1e-2, momentum=0.9)),
                               "shape": (torch.optim.Adam, dict(lr=1e-2))})
    return event, v


def lengths(v):
    return 2 * np.pi / v.alpha.omega_spatial.norm(dim=1).numpy()


def test_schedule_opens_octaves_in_order():
    event, v = make_event()
    warmup = SpectralWarmup(event, ramp=2 / 3)
    assert warmup.n_octaves == pytest.approx(math.log2(lengths(v).max() / lengths(v).min()))
    window = warmup._entries[0][1]
    warmup(0, 5000)
    start = window.clone()
    assert 0 < int((start < 1e-6).sum()) < len(start)                  # some closed at the start
    assert lengths(v)[start.numpy() > 0.99].min() > lengths(v)[start.numpy() < 1e-6].max()   # long ones open
    warmup(int(5000 * warmup.n_octaves / (warmup.n_octaves + 1)) + 1, 5000)
    assert torch.all(window > 0.999)                                    # all open before the end
    warmup.remove()


def test_closed_modes_stay_at_zero_and_hooks_come_off():
    event, v = make_event()
    amp = v.alpha.C_mu
    model = curlew.GeoModel([event])
    with SpectralWarmup(event) as warmup:
        assert torch.all(warmup._entries[0][1] == 1)          # open until the schedule is called
        warmup(0, 5000)
        closed = (warmup._entries[0][1] < 1e-6).clone()
        model.prefit(epochs=20, vb=False, early_stop=None, best=False, schedule=lambda e, n: warmup(0, 5000))
    assert torch.all(amp.detach()[closed] == 0)
    assert torch.any(amp.detach()[~closed] != 0)
    # after the context, gradients are no longer windowed
    amp.grad = None
    (amp * torch.arange(len(amp), dtype=amp.dtype)).sum().backward()
    assert torch.equal(amp.grad, torch.arange(len(amp), dtype=amp.dtype))


def test_accepts_a_velocity():
    _, v = make_event()
    with SpectralWarmup(v) as warmup:
        assert len(warmup._entries) == 1


def test_ramp_sets_when_every_octave_is_open():
    event, v = make_event()
    with SpectralWarmup(event, ramp=1 / 3) as warmup:
        window = warmup._entries[0][1]
        warmup(int(5000 / 3) + 1, 5000)
        assert torch.all(window > 0.999)            # all open after the first third
        warmup(800, 5000)
        assert torch.any(window < 0.999)            # not yet halfway through the ramp
    with SpectralWarmup(event) as warmup:          # default: 1/6
        assert warmup.ramp == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        SpectralWarmup(event, ramp=0)
