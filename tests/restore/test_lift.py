"""Tests for fault lift / GWN primitives (``curlew.kinematics.lift``)."""
import curlew
import pytest
import torch
from torch import nn

from curlew.kinematics.lift import FaultLift
from curlew.geometry.gwn import gwn_mesh, gwn_polyline, gwn_ribbon_mesh
from curlew.geology.interactions import FlowOffset


@pytest.fixture(autouse=True)
def _lift_cpu():
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def test_gwn_polyline_rejects_3d_trace():
    trace = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    x = torch.tensor([[0.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="gwn_mesh"):
        gwn_polyline(x, trace)


def test_gwn_mesh_planar_fault_matches_polyline_sign():
    trace = torch.tensor([[1.0, 0.0], [1.0, 2.0]])
    verts = torch.tensor(
        [[1.0, 0.0, -40.0], [1.0, 2.0, -40.0], [1.0, 2.0, 40.0], [1.0, 0.0, 40.0]]
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    x2 = torch.tensor([[1.5, 1.0], [0.5, 1.0]])
    x3 = torch.tensor([[1.5, 1.0, 0.0], [0.5, 1.0, 0.0]])

    w2 = gwn_polyline(x2, trace)
    w3 = gwn_mesh(x3, verts, faces)

    assert torch.all(torch.sign(w2) == torch.sign(w3))
    assert torch.allclose(w2.abs(), w3.abs() / 2.0, rtol=1e-3, atol=1e-3)


def test_gwn_ribbon_mesh_builds_finite_gwn():
    poly = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    normals = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    verts, faces = gwn_ribbon_mesh(poly, normals, half_width=0.5)
    x = torch.tensor([[1.0, 1.0, 0.5]])
    w = gwn_mesh(x, verts, faces)
    assert torch.isfinite(w).all()
    assert w.abs() > 0
    w_flip = gwn_mesh(x, verts, faces[:, [0, 2, 1]])
    assert w * w_flip < 0


def _planar_fault_mesh():
    verts = torch.tensor(
        [[1.0, 0.0, -40.0], [1.0, 2.0, -40.0], [1.0, 2.0, 40.0], [1.0, 0.0, 40.0]]
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return verts, faces


def test_fault_lift3d_state_shapes():
    mesh = _planar_fault_mesh()
    lift = FaultLift([mesh])
    assert lift.dim == 3
    x = torch.tensor([[1.5, 1.0, 0.0], [0.5, 1.0, 0.0]])
    st = lift.state(x)
    assert st.w_raw.shape == (2, 1)
    assert st.w_active.shape == (2, 1)
    assert st.normal.shape == (2, 1, 3)
    assert st.w_raw[0] * st.w_raw[1] < 0


class _SheetVelocity(nn.Module):
    def __init__(self, dim=2, lift_dim=1, dtype=torch.float64):
        super().__init__()
        self.dim = dim
        self.lift_dim = lift_dim
        self._dtype = dtype

    def forward(self, x, w=None, t=None):
        v = torch.zeros(x.shape[0], self.dim, dtype=self._dtype, device=x.device)
        if w is not None:
            v[:, 0] = 0.05 * w.reshape(x.shape[0], -1)[:, 0]
        return v

    def forward_and_jacobian(self, x, w=None, t=None):
        v = self.forward(x, w, t)
        jv = torch.zeros(x.shape[0], self.dim, self.dim, dtype=self._dtype, device=x.device)
        if w is not None:
            jv[:, 0, 0] = 0.05
        return v, jv


def test_flow_offset_fault_lift2d_integration():
    curlew.device = "cpu"
    curlew.dtype = torch.float64
    trace = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    lift = FaultLift([trace], tip_taper=0.0)
    vel = _SheetVelocity(dim=2)
    off = FlowOffset(vel, n_steps=4, direction=-1.0, fault_lift=lift)
    x = torch.tensor([[1.0, 0.5], [1.0, -0.5]])
    y = off.inverse_map(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert (y[:, 0] - x[:, 0]).abs().max() > 0


def test_flow_offset_fault_lift3d_integration():
    curlew.device = "cpu"
    curlew.dtype = torch.float64
    lift = FaultLift([_planar_fault_mesh()], tip_taper=0.0)
    vel = _SheetVelocity(dim=3)
    off = FlowOffset(vel, n_steps=4, direction=-1.0, fault_lift=lift)
    x = torch.tensor([[1.5, 1.0, 0.0], [0.5, 1.0, 0.0]])
    y = off.inverse_map(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


class _ShearSheetVelocity(nn.Module):
    """v_y = k*x + 0.05*w — has genuine x-dependence, so a wrong confinement
    Jacobian doesn't cancel out against a trivially-zero base Jacobian."""

    def __init__(self, dim=2, k=0.3, dtype=torch.float64):
        super().__init__()
        self.dim = dim
        self.k = float(k)
        self._dtype = dtype

    def forward(self, x, w=None, t=None):
        v = torch.zeros(x.shape[0], self.dim, dtype=self._dtype, device=x.device)
        v[:, 1] = self.k * x[:, 0]
        if w is not None:
            v[:, 0] = 0.05 * w.reshape(x.shape[0], -1)[:, 0]
        return v

    def forward_and_jacobian(self, x, w=None, t=None):
        v = self.forward(x, w, t)
        jv = torch.zeros(x.shape[0], self.dim, self.dim, dtype=self._dtype, device=x.device)
        jv[:, 1, 0] = self.k
        return v, jv


def _per_point_jacobian(f, x):
    n, d = x.shape
    J = torch.zeros(n, d, d, dtype=x.dtype)
    for i in range(d):
        grad_outputs = torch.zeros_like(f)
        grad_outputs[:, i] = 1.0
        g = torch.autograd.grad(f, x, grad_outputs=grad_outputs, retain_graph=True)[0]
        J[:, i, :] = g
    return J


def test_flow_offset_fault_confinement_jacobian_matches_autograd():
    """
    With ``confine=True`` (the default whenever ``fault_lift`` is set), the
    reported Jacobian from ``inverse_map_with_jacobian`` must match autograd
    on ``inverse_map`` — i.e. the same confinement transform applied to the
    velocity must also be applied to its Jacobian, not just to ``v``.
    """
    curlew.device = "cpu"
    curlew.dtype = torch.float64
    trace = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=curlew.dtype)
    lift = FaultLift([trace], tip_taper=0.0, confine_width=0.6)
    vel = _ShearSheetVelocity(dim=2)
    off = FlowOffset(vel, n_steps=8, direction=-1.0, fault_lift=lift)
    assert off.confine is True

    x = torch.tensor(
        [[1.0, 0.15], [1.0, -0.2], [1.0, 0.05]], dtype=curlew.dtype,
    )
    x_grad = x.clone().requires_grad_(True)
    y = off.inverse_map(x_grad)
    J_auto = _per_point_jacobian(y, x_grad)

    with torch.no_grad():
        _, J_reported = off.inverse_map_with_jacobian(x)

    assert torch.allclose(J_auto, J_reported, atol=1e-8)


def test_flow_offset_fault_mobility_jacobian_matches_autograd():
    """Same consistency check with asymmetric sheet_mobility (scales v by mu(x))."""
    curlew.device = "cpu"
    curlew.dtype = torch.float64
    trace = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=curlew.dtype)
    lift = FaultLift(
        [trace], tip_taper=0.0, confine_width=0.6, sheet_mobility="positive",
    )
    vel = _ShearSheetVelocity(dim=2)
    off = FlowOffset(vel, n_steps=8, direction=-1.0, fault_lift=lift)

    x = torch.tensor(
        [[1.0, 0.15], [1.0, -0.2], [1.0, 0.05]], dtype=curlew.dtype,
    )
    x_grad = x.clone().requires_grad_(True)
    y = off.inverse_map(x_grad)
    J_auto = _per_point_jacobian(y, x_grad)

    with torch.no_grad():
        _, J_reported = off.inverse_map_with_jacobian(x)

    assert torch.allclose(J_auto, J_reported, atol=1e-8)


def test_sheet_mobility_scalar_expands_and_preserves_velocity_fault():
    t0 = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    t1 = torch.tensor([[0.5, -1.0], [0.5, 1.0]])
    lift = FaultLift(
        [t0, t1], tip_taper=0.0, sheet_mobility="positive", mobility_fault=1,
    )
    assert lift.sheet_mobility == ("positive", "positive")
    assert lift._sheet_mobility_scalar is True

    x = torch.tensor([[1.0, 0.5]])
    st = lift.state(x)
    assert st.mobility is not None
    mu1 = lift._mobility_weight(st.w_raw[:, 1], "positive")
    assert torch.allclose(st.mobility, mu1)


def test_sheet_mobility_per_fault_independent_lift_and_velocity_product():
    t0 = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    t1 = torch.tensor([[0.5, -1.0], [0.5, 1.0]])
    lift = FaultLift(
        [t0, t1],
        tip_taper=0.0,
        sheet_mobility=("positive", "both"),
    )
    assert lift.sheet_mobility == ("positive", "both")
    assert lift._sheet_mobility_scalar is False

    x = torch.tensor([[1.0, 0.5], [1.0, -0.5]])
    st = lift.state(x)
    mu0 = lift._mobility_weight(st.w_raw[:, 0], "positive")
    assert torch.allclose(st.w_active[:, 0], st.w_raw[:, 0] * mu0)
    assert torch.allclose(st.w_active[:, 1], st.w_raw[:, 1])
    assert torch.allclose(st.mobility, mu0)

    lift2 = FaultLift(
        [t0, t1],
        tip_taper=0.0,
        sheet_mobility=("positive", "negative"),
    )
    st2 = lift2.state(x)
    mu_a = lift2._mobility_weight(st2.w_raw[:, 0], "positive")
    mu_b = lift2._mobility_weight(st2.w_raw[:, 1], "negative")
    assert torch.allclose(st2.mobility, mu_a * mu_b)


def test_sheet_mobility_sequence_length_validated():
    t0 = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="sheet_mobility length"):
        FaultLift([t0], sheet_mobility=("positive", "both"))


def test_restore_wires_fault_lift_from_traces():
    from curlew.core import CSet, HSet
    from curlew.kinematics.euler import euler_velocity
    from curlew.geology import restore

    curlew.default_dim = 2
    trace = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    fault_lift = FaultLift([trace])
    velocity = euler_velocity(2, n_features=8)
    event = restore(
        "fold", C=CSet(), H=HSet(), velocity=velocity, fault_lift=fault_lift,
    )
    assert event.field.fault_lift is fault_lift
    assert event.field.fault_lift.dim == 2
    assert event.field.integrator.fault_lift is fault_lift
    assert event.field.velocity.dim == fault_lift.dim


def test_restore_wires_fault_lift_from_meshes():
    from curlew.core import CSet, HSet
    from curlew.kinematics.euler import euler_velocity
    from curlew.geology import restore

    curlew.default_dim = 3
    verts = torch.tensor(
        [[1.0, 0.0, -10.0], [1.0, 2.0, -10.0], [1.0, 2.0, 10.0], [1.0, 0.0, 10.0]]
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    fault_lift = FaultLift([(verts, faces)])
    velocity = euler_velocity(3, n_features=8)
    event = restore(
        "fold3d",
        C=CSet(),
        H=HSet(),
        velocity=velocity,
        fault_lift=fault_lift,
        input_dim=3,
    )
    assert event.field.fault_lift is fault_lift
    assert event.field.fault_lift.dim == 3
    assert event.field.velocity.dim == fault_lift.dim
