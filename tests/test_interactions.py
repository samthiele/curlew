"""
Tests for kinematic interaction classes in ``curlew.geology.interactions``:
velocity-field integration (``VFieldOffset``), sheet intrusions (``SheetOffset``),
and faults (``FaultOffset``).
"""
import numpy as np
import torch

import curlew
from curlew import GeoEvent, _tensor
from curlew.fields import BaseAF
from curlew.fields.analytical import LinearField
from curlew.geology.interactions import FaultOffset, SheetOffset, VFieldOffset

# run these tests on CPU with stable float64 arithmetic
curlew.device = "cpu"
curlew.dtype = torch.float64
curlew.default_dim = 2

class _ConstVecField(BaseAF):
    """Analytical velocity field returning the same vector at every position."""

    def initField(self, vec, **kwargs):
        v = np.asarray(vec, dtype=np.float64).ravel()
        self._vec = _tensor(v, dev=curlew.device, dt=curlew.dtype)

    def evaluate(self, x: torch.Tensor):
        return self._vec.unsqueeze(0).expand(x.shape[0], -1)

class _QuadPosField(BaseAF):
    """Nonlinear velocity v(x) = scale * x**2 (path length affects integrated displacement)."""

    def initField(self, scale=1.0, **kwargs):
        self.s = float(scale)

    def evaluate(self, x: torch.Tensor):
        return self.s * x * x

def _LinearField(name, origin, gradient, *, normalise=False, input_dim=None):
    """Build a minimal GeoEvent with an attached linear scalar field."""
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
    return GeoEvent(name, type=LinearField, field=lf)

def test_vfieldoffset():
    """
    Test the ``VFieldOffset`` class and check integration (approximation) is working.
    """
    
    #  ``VFieldOffset`` with an attached latent field should integrate that field's
    # forward pass. With default ``dt=-1`` and ``n_steps=1``, displacement is ``-v(x)``.
    latent = _ConstVecField("uv", input_dim=2, output_dim=2, vec=[0.25, -0.5])
    off = VFieldOffset(latent, n_steps=1)
    X = torch.tensor([[0.0, 0.0], [1.0, 2.0], [-1.0, 3.0]], dtype=curlew.dtype)

    d = off.disp(X, None)
    direct = latent.forward(X, transform=False)

    assert torch.allclose(d, -direct)
    assert d.shape == X.shape

    # For a spatially constant velocity field, total displacement should not depend
    # on the number of Euler substeps (only on ``dt`` and the velocity magnitude).
    c = np.array([0.7, -0.2], dtype=np.float64)
    latent = _ConstVecField("uv", input_dim=2, output_dim=2, vec=c)
    X = torch.randn(4, 2, dtype=curlew.dtype)

    disp_once = VFieldOffset(latent, n_steps=1, dt=-0.1).disp(X, None)
    disp_many = VFieldOffset(latent, n_steps=10, dt=-0.1).disp(X, None)

    assert torch.allclose(disp_once, disp_many)
    
    # When velocity varies with position, multi-step Euler integration should differ
    # from a single evaluation at the start point.
    latent = _QuadPosField("qx", input_dim=2, output_dim=2, scale=0.25)
    X = torch.tensor([[0.25, -0.1], [1.0, 0.5]], dtype=curlew.dtype)

    d_one = VFieldOffset(latent, n_steps=1, dt=-0.02).disp(X, None)
    d_path = VFieldOffset(latent, n_steps=50, dt=-0.02).disp(X, None)

    assert (d_one - d_path).abs().max().item() > 1e-5
    assert d_path.shape == X.shape
    
    # Along a straight, uniformly graded dyke wall, sheet opening is locally uniform so
    # extra Euler steps should not change the total displacement.
    
def test_SheetOffset():
    """
    Along a straight, uniformly graded dyke wall, sheet opening is locally uniform so
    extra Euler steps should not change the total displacement.
    """
    gfield = _LinearField(
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

def test_FaultOffset():
    """
    Fault slip should lie in the tangent plane of the host scalar field (orthogonal to
    the gradient). Setup follows the inclined fault used in ``test_analytic`` /
    ``test_geology`` michell-style examples.
    """
    gfield = _LinearField(
        "fault",
        origin=[600.0, 0.0],
        gradient=[-1.0, -1.0],
        normalise=True,
    )
    shortening = _tensor([0.0, 1.0], dev=curlew.device, dt=curlew.dtype)
    width = (1.0, 1.0 / 50.0, 0.4)
    X = torch.tensor([[600.0, 40.0], [620.0, -30.0]], device=curlew.device, dtype=curlew.dtype)

    # single-step fault offset
    fault1 = FaultOffset(
        shortening=shortening,
        offset=60.0,
        contact=0.0,
        width=width,
        n_steps=1,
    )
    u = fault1.disp(X, gfield)
    assert u.shape == (2, 2)
    assert torch.isfinite(u).all()

    ds, _ = fault1.dss(X, gfield, normalize=True)
    dot = (u * ds).sum(dim=-1).abs().max().item()
    assert dot < 1e-5  # slip perpendicular to gradient (in tangent plane)

    # multi-step integration should remain finite with the same geometry
    fault2 = FaultOffset(
        shortening=shortening,
        offset=60.0,
        contact=0.0,
        width=width,
        n_steps=2,
    )
    u2 = fault2.disp(X, gfield)
    assert u2.shape == u.shape
    assert torch.isfinite(u2).all()
