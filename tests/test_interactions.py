"""
Tests for kinematic interaction classes in ``curlew.geology.interactions``:
RK2 flow integration (``FlowOffset``), sheet intrusions (``SheetOffset``),
and faults (``FaultOffset``) — ``SheetOffset``/``FaultOffset`` are ``FlowOffset``
subclasses that derive velocity from the owning ``GeoEvent`` on each sub-step.
"""
import numpy as np
import torch
from torch import nn

import curlew
from curlew import GeoEvent, _tensor
from curlew.fields.analytical import LinearField
from curlew.geology.interactions import FaultOffset, FlowOffset, SheetOffset

# run these tests on CPU with stable float64 arithmetic
curlew.device = "cpu"
curlew.dtype = torch.float64
curlew.default_dim = 2

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


class _ConstVelocity(nn.Module):
    """Constant velocity for FlowOffset integration checks."""

    def __init__(self, vec, dim=2):
        super().__init__()
        self.dim = dim
        self.register_buffer("v", _tensor(vec, dev=curlew.device, dt=curlew.dtype).reshape(dim))

    def forward(self, x, w=None, t=None):
        return self.v.unsqueeze(0).expand(x.shape[0], -1)

    def forward_and_jacobian(self, x, w=None, t=None):
        n = x.shape[0]
        v = self.forward(x, w, t)
        jv = torch.zeros(n, self.dim, self.dim, dtype=x.dtype, device=x.device)
        return v, jv


def test_flow_offset_constant_velocity():
    """FlowOffset restoration displacement matches RK2 integration of a uniform field."""
    vel = _ConstVelocity([0.25, -0.5])
    off = FlowOffset(vel, n_steps=4, direction=-1.0)
    X = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=curlew.dtype)

    d = off.disp(X, None)
    expected = off.inverse_map(X) - X
    assert torch.allclose(d, expected)
    assert torch.allclose(d, -vel.v.unsqueeze(0).expand_as(X), atol=1e-10)

    x_ref, J = off.inverse_map_with_jacobian(X)
    assert torch.allclose(x_ref, X + d)
    det = torch.linalg.det(J)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-8)


def test_flow_offset_geoevent_undeform():
    """FlowOffset plugs into GeoEvent.undeform like other OffsetBase subclasses."""
    vel = _ConstVelocity([0.1, 0.2])
    event = _LinearField("host", origin=[0.0, 0.0], gradient=[1.0, 0.0])
    event.deformation = FlowOffset(vel, n_steps=2)
    X = torch.tensor([[0.0, 0.0], [2.0, -1.0]], dtype=curlew.dtype)
    x_paleo = event.undeform(X.clone())
    disp = -vel.v.unsqueeze(0).expand_as(X)
    assert torch.allclose(x_paleo, X + disp, atol=1e-10)


class _QuadVelocity(nn.Module):
    """Nonlinear velocity v(x) = scale * x**2 (path shape affects integrated displacement)."""

    def __init__(self, scale=1.0, dim=2):
        super().__init__()
        self.dim = dim
        self.scale = float(scale)

    def forward(self, x, w=None, t=None):
        return self.scale * x * x


def test_flow_offset_duration_scales_constant_velocity():
    """
    For a spatially constant velocity, total displacement should not depend on the
    number of RK2 substeps (only on ``direction``'s magnitude and the velocity) —
    the RK2 analogue of the old ``VFieldOffset`` Euler-substep invariance.
    """
    vel = _ConstVelocity([0.7, -0.2])
    X = torch.randn(4, 2, dtype=curlew.dtype)

    disp_once = FlowOffset(vel, n_steps=1, direction=-0.1).disp(X, None)
    disp_many = FlowOffset(vel, n_steps=10, direction=-0.1).disp(X, None)

    assert torch.allclose(disp_once, disp_many)


def test_flow_offset_position_dependent_velocity_needs_substeps():
    """
    When velocity varies with position, multi-step RK2 integration should differ
    from a single (coarse) step, since the path shape now matters.
    """
    vel = _QuadVelocity(scale=0.25)
    X = torch.tensor([[1.0, 0.8], [1.2, -1.0]], dtype=curlew.dtype)

    d_one = FlowOffset(vel, n_steps=1, direction=-0.3).disp(X, None)
    d_path = FlowOffset(vel, n_steps=50, direction=-0.3).disp(X, None)

    assert (d_one - d_path).abs().max().item() > 1e-5
    assert d_path.shape == X.shape


def test_sheet_and_fault_offset_are_flow_offsets():
    """SheetOffset/FaultOffset are FlowOffset subclasses; VFieldOffset has been removed."""
    assert issubclass(SheetOffset, FlowOffset)
    assert issubclass(FaultOffset, FlowOffset)
    import curlew.geology.interactions as interactions
    assert not hasattr(interactions, "VFieldOffset")


def test_flow_offset_compile_only_for_spatial_velocity():
    """
    ``curlew.compile`` wraps ``FlowOffset._integrate`` with ``torch.compile`` only
    for the spatial-velocity-module path (used by restoration): its ``_velocity_at``
    is closed-form tensor math with no ``G``-dependence, so it's safe to compile.
    ``SheetOffset``/``FaultOffset`` (``velocity=None``) derive velocity from ``G``
    via a nested ``autograd.grad`` call on every sub-step, so they must stay eager
    regardless of the flag.
    """
    prev = curlew.compile
    try:
        vel = _ConstVelocity([0.3, -0.1])
        X = torch.tensor([[0.0, 0.0], [1.5, -2.0]], dtype=curlew.dtype)

        curlew.compile = True
        off_compiled = FlowOffset(vel, n_steps=3, direction=-1.0)
        assert "_integrate" in vars(off_compiled)  # instance-level compiled override

        curlew.compile = False
        off_eager = FlowOffset(vel, n_steps=3, direction=-1.0)
        assert "_integrate" not in vars(off_eager)  # plain class method, uncompiled

        assert torch.allclose(off_compiled.disp(X, None), off_eager.disp(X, None), atol=1e-8)

        curlew.compile = True
        sheet = SheetOffset(contact=(-1.0, 1.0), aperture=1.0, n_steps=1)
        assert "_integrate" not in vars(sheet)  # G-dependent path stays eager
    finally:
        curlew.compile = prev
