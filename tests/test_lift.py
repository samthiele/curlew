"""Tests for fault lift / GWN primitives (``curlew.fields.lift``)."""
import curlew
import pytest
import torch
from torch import nn

from curlew.fields.lift import FaultLift
from curlew.utils.gwn import gwn_mesh, gwn_polyline, gwn_ribbon_mesh
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
