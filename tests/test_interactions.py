"""Tests for kinematic offset helpers in curlew.geology.interactions."""

import numpy as np
import pytest
import torch

import curlew
from curlew import GeoField, _tensor
from curlew.fields import BaseAF
from curlew.fields.analytical import LinearField
from curlew.geology.interactions import FaultOffset, SheetOffset, VFieldOffset

class _ConstVecField(BaseAF):
    """Latent field returning the same displacement/velocity vector for every sample."""
    def initField(self, vec, **kwargs):
        v = np.asarray(vec, dtype=np.float64).ravel()
        self._vec = _tensor(v, dev=curlew.device, dt=curlew.dtype)
    def evaluate(self, x: torch.Tensor):
        return self._vec.unsqueeze(0).expand(x.shape[0], -1)

def _setup():
    curlew.device = "cpu"
    curlew.dtype = torch.float64

def _linear_geofield(name, origin, gradient, *, normalise=False, input_dim=None):
    """Minimal GeoField + LinearField."""
    o = np.asarray(origin, dtype=np.float64).ravel()
    g = np.asarray(gradient, dtype=np.float64).ravel()
    dim = input_dim if input_dim is not None else len(o)
    lf = LinearField(
        name,
        input_dim=dim,
        origin=o,
        gradient=g,
        normalise=normalise,
    )
    return GeoField(name, type=LinearField, field=lf)

def test_dispfieldoffset_matches_latent_field():
    _setup()
    curlew.default_dim = 2
    latent = _ConstVecField(
        "uv", input_dim=2, output_dim=2, vec=[0.25, -0.5]
    )
    off = VFieldOffset(latent, n_steps=1)
    X = torch.tensor([[0.0, 0.0], [1.0, 2.0], [-1.0, 3.0]], dtype=curlew.dtype)
    d = off.disp(X, None)
    direct = latent.forward(X, transform=False)
    # Default dt=-1 integrates backward (modern toward paleo): one step u = -v.
    assert torch.allclose(d, -direct)
    assert d.shape == X.shape

def test_velfieldoffset_constant_velocity_scales_with_n_steps_and_dt():
    _setup()
    curlew.default_dim = 2
    c = np.array([0.7, -0.2], dtype=np.float64)
    latent = _ConstVecField("uv", input_dim=2, output_dim=2, vec=c)
    X = torch.randn(4, 2, dtype=curlew.dtype)
    disp_once = VFieldOffset(latent, n_steps=1, dt=-0.1).disp(X, None)
    euler = VFieldOffset(latent, n_steps=10, dt=-0.1).disp(X, None)
    assert torch.allclose(disp_once, euler)

class _QuadPosField(BaseAF):
    """v(x) = scale * x**2 element-wise (nonlinear; path length changes total Euler displacement)."""
    def initField(self, scale=1.0, **kwargs):
        self.s = float(scale)
    def evaluate(self, x: torch.Tensor):
        return self.s * x * x

def test_velfieldoffset_position_dependent_differs_from_single_evaluation():
    _setup()
    curlew.default_dim = 2
    latent = _QuadPosField("qx", input_dim=2, output_dim=2, scale=0.25)
    X = torch.tensor([[0.25, -0.1], [1.0, 0.5]], dtype=curlew.dtype)
    d_short = VFieldOffset(latent, n_steps=1, dt=-0.02).disp(X, None)
    d_path = VFieldOffset(latent, n_steps=50, dt=-0.02).disp(X, None)
    assert (d_short - d_path).abs().max().item() > 1e-5
    assert d_path.shape == X.shape

def test_sheetoffset_multistep_matches_single_for_uniform_gradient():
    """Along a straight isofault, extra Euler steps should not change total sheet displacement."""
    _setup()
    curlew.default_dim = 2
    gfield = _linear_geofield(
        "dyke",
        origin=[0.0, 0.0],
        gradient=[0.0, 1.0],
        normalise=True,
    )
    s1 = SheetOffset(contact=(-1.0, 1.0), aperture=1.0, n_steps=1)
    s4 = SheetOffset(contact=(-1.0, 1.0), aperture=1.0, n_steps=4)
    X = torch.tensor([[0.0, 2.0], [0.0, -2.0]], device=curlew.device, dtype=curlew.dtype)
    u1 = s1.disp(X, gfield)
    u4 = s4.disp(X, gfield)
    assert torch.allclose(u1, u4, rtol=1e-10, atol=1e-10)

def test_faultoffset_slip_tangent_to_gradient():
    """Simplified from ``test_geology.test_michell`` / ``test_analyticImplicitField`` fault setup."""
    _setup()
    curlew.default_dim = 2
    gfield = _linear_geofield(
        "fault",
        origin=[600.0, 0.0],
        gradient=[-1.0, -1.0],
        normalise=True,
    )
    shortening = _tensor([0.0, 1.0], dev=curlew.device, dt=curlew.dtype)
    fault = FaultOffset(
        shortening=shortening,
        offset=60.0,
        contact=0.0,
        width=(1.0, 1.0 / 50.0, 0.4),
        n_steps=1,
    )
    X = torch.tensor([[600.0, 40.0], [620.0, -30.0]], device=curlew.device, dtype=curlew.dtype)
    u = fault.disp(X, gfield)
    assert u.shape == (2, 2)
    assert torch.isfinite(u).all()

    ds, _ = fault.dss(X, gfield, normalize=True)
    dot = (u * ds).sum(dim=-1).abs().max().item()
    assert dot < 1e-5

    fault2 = FaultOffset(
        shortening=shortening,
        offset=60.0,
        contact=0.0,
        width=(1.0, 1.0 / 50.0, 0.4),
        n_steps=2,
    )
    u2 = fault2.disp(X, gfield)
    assert u2.shape == u.shape
    assert torch.isfinite(u2).all()

