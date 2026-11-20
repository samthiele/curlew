"""Tests for Euler velocity fields (``curlew.kinematics.euler``)."""
import curlew
import pytest
import torch

from curlew.core import HSet
from curlew.kinematics.euler import (
    EulerVelocity,
    euler_velocity,
)
from curlew.fields.series import FSF


@pytest.fixture(autouse=True)
def _euler_defaults():
    curlew.default_dim = 2
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def _fsf(name: str, dim: int, *, seed: int = 0, lift_dim: int = 0) -> FSF:
    return FSF(
        name,
        H=HSet(),
        input_dim=dim,
        lift_dim=lift_dim,
        rff_features=32,
        length_scale_range=(0.5, 2.0),
        seed=seed,
    )


def _autograd_velocity_jacobian(velocity, x: torch.Tensor) -> tuple:
    """Autograd ∂v/∂x for v = velocity(x)."""
    x = x.detach().clone().requires_grad_(True)
    v = velocity(x)
    dim = x.shape[-1]
    rows = [
        torch.autograd.grad(v[:, i].sum(), x, retain_graph=(i < dim - 1))[0]
        for i in range(dim)
    ]
    return v.detach(), torch.stack(rows, dim=1)


def test_euler_velocity_2d_divergence_free():
    beta = _fsf("beta", 2, seed=1)
    beta.A_mu.data.normal_(0, 0.05)
    beta.B_mu.data.normal_(0, 0.05)
    vel = EulerVelocity(beta)
    x = torch.randn(8, 2, dtype=curlew.dtype)
    _, jv = vel.forward_and_jacobian(x)
    trace = jv[:, 0, 0] + jv[:, 1, 1]
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-10, rtol=1e-8)


def test_euler_velocity_jacobian_autograd_2d():
    beta = _fsf("beta", 2, seed=2)
    beta.A_mu.data.normal_(0, 0.08)
    beta.B_mu.data.normal_(0, 0.08)
    vel = EulerVelocity(beta)
    x = torch.randn(6, 2, dtype=curlew.dtype)
    v, jv = vel.forward_and_jacobian(x)
    v_auto, jv_auto = _autograd_velocity_jacobian(vel, x)
    assert torch.allclose(v, v_auto, atol=1e-10, rtol=1e-7)
    assert torch.allclose(jv, jv_auto, atol=1e-9, rtol=1e-6)


def test_euler_velocity_3d_divergence_free():
    beta = _fsf("beta", 3, seed=3)
    alpha = _fsf("alpha", 3, seed=4)
    beta.A_mu.data.normal_(0, 0.05)
    alpha.A_mu.data.normal_(0, 0.05)
    beta.B_mu.data.normal_(0, 0.05)
    alpha.B_mu.data.normal_(0, 0.05)
    vel = EulerVelocity(beta, alpha)
    x = torch.randn(5, 3, dtype=curlew.dtype)
    _, jv = vel.forward_and_jacobian(x)
    trace = jv.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-9, rtol=1e-7)


def test_euler_velocity_factory():
    vel = euler_velocity(2, n_features=16, length_scale_range=(1.0, 2.0), seed=5)
    vel.stream.A_mu.data.normal_(0, 0.1)
    vel.stream.B_mu.data.normal_(0, 0.1)
    x = torch.randn(4, 2, dtype=curlew.dtype)
    v = vel(x)
    assert v.shape == (4, 2)


def test_euler_velocity_requires_alpha_in_3d():
    beta = _fsf("beta", 3)
    with pytest.raises(ValueError, match="alpha"):
        EulerVelocity(beta)


def test_euler_velocity_hinge_rejected_in_2d():
    with pytest.raises(ValueError, match="3-D only"):
        euler_velocity(2, hinge=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="3-D only"):
        euler_velocity(2, section_axis=0)


def test_euler_velocity_hinge_unit_vector_3d():
    """Fixed hinge direction â and FSF streamfunction give div-free v = ∇ψ × â."""
    axis = torch.tensor([1.0, 2.0, 0.0])
    axis = axis / axis.norm()
    vel = euler_velocity(
        3,
        n_features=16,
        length_scale_range=(1.0, 2.0),
        seed=6,
        stream="beta",
        hinge=axis.tolist(),
    )
    assert vel.hinge is not None
    assert torch.allclose(vel.hinge.direction, axis.to(curlew.dtype), atol=1e-12)
    vel.beta.A_mu.data.normal_(0, 0.05)
    vel.beta.B_mu.data.normal_(0, 0.05)
    x = torch.randn(6, 3, dtype=curlew.dtype)
    _, jv = vel.forward_and_jacobian(x)
    trace = jv.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-9, rtol=1e-7)


def test_euler_velocity_matches_legacy_2d_perp_grad():
    """Unified cross product matches the old ∇⊥β formula on the same FSF."""
    beta = _fsf("beta", 2, seed=8)
    beta.A_mu.data.normal_(0, 0.08)
    beta.B_mu.data.normal_(0, 0.08)
    x = torch.randn(10, 2, dtype=curlew.dtype)
    g, h = beta.gradient_and_hessian(x)
    v_legacy = torch.stack([g[:, 1], -g[:, 0]], dim=1)
    vel = EulerVelocity(beta)
    v, jv = vel.forward_and_jacobian(x)
    assert torch.allclose(v, v_legacy, atol=1e-12)
    j_legacy = torch.stack([h[:, :, 1], -h[:, :, 0]], dim=1)
    assert torch.allclose(jv, j_legacy, atol=1e-12)


def test_euler_velocity_paired_frequencies_orthogonal():
    """Paired mode k uses orthogonal spatial directions with equal |Ω|."""
    vel = euler_velocity(3, n_features=32, seed=11, paired_frequencies=True)
    oa = vel.alpha._omega_spatial()
    ob = vel.beta._omega_spatial()
    dots = (oa * ob).sum(1)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-10)
    assert torch.allclose(oa.norm(dim=1), ob.norm(dim=1), atol=1e-12)
    crosses = torch.linalg.cross(oa, ob, dim=1)
    assert crosses.norm(dim=1).gt(1e-8).all()


def test_euler_velocity_paired_frequencies_divergence_free():
    vel = euler_velocity(3, n_features=24, seed=12, paired_frequencies=True)
    vel.beta.A_mu.data.normal_(0, 0.05)
    vel.alpha.A_mu.data.normal_(0, 0.05)
    vel.beta.B_mu.data.normal_(0, 0.05)
    vel.alpha.B_mu.data.normal_(0, 0.05)
    x = torch.randn(7, 3, dtype=curlew.dtype)
    _, jv = vel.forward_and_jacobian(x)
    trace = jv.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-9, rtol=1e-7)


def test_euler_velocity_paired_frequencies_rejects_hinge():
    with pytest.raises(ValueError, match="hinge"):
        euler_velocity(3, paired_frequencies=True, hinge=[0.0, 0.0, 1.0])


# ── velocity fast path (no Hessian) ──────────────────────────────────────────

@pytest.mark.parametrize(
    "dim, kw",
    [
        (2, {}),
        (2, {"stream": "alpha"}),
        (3, {}),
        (3, {"paired_frequencies": True}),
        (3, {"section_axis": 1}),
    ],
)
def test_velocity_only_path_matches_the_jacobian_path(dim, kw):
    """
    ``forward`` skips the Hessian entirely; it must still agree bit-for-bit
    with the velocity returned by ``forward_and_jacobian``.

    The Hessian is an (N, F, d, d) einsum that the velocity never uses, and
    every RK2 sub-step calls this path twice — computing it was ~60% of the
    cost of a plain ``inverse_map``. This test is what makes skipping it safe:
    the slot-swap convention has to stay in lockstep between the two branches.
    """
    from curlew.kinematics.euler import euler_velocity

    torch.manual_seed(0)
    vel = euler_velocity(dim, n_features=24, seed=5, **kw)
    for m in vel.modules():
        if hasattr(m, "A_mu"):
            with torch.no_grad():
                m.A_mu.normal_(0, 0.4)
                if hasattr(m, "B_mu"):
                    m.B_mu.normal_(0, 0.4)

    x = torch.rand(256, dim, dtype=curlew.dtype) * 4
    v_fast = vel(x)
    v_ref, _ = vel.forward_and_jacobian(x)
    assert (v_fast - v_ref).abs().max() == 0.0


def test_gradient_raw_matches_gradient_and_hessian():
    """``FSF.gradient_raw`` is the same gradient, just without the Hessian."""
    from curlew.fields.series import FSF

    field = FSF(
        "f", H=HSet(), input_dim=2, rff_features=32,
        length_scale_range=(0.5, 3.0), seed=1,
    )
    with torch.no_grad():
        field.A_mu.normal_(0, 0.3)
        field.B_mu.normal_(0, 0.3)
    x = torch.rand(64, 2, dtype=curlew.dtype) * 3
    assert (field.gradient_raw(x) - field.gradient_and_hessian(x)[0]).abs().max() == 0.0


@pytest.mark.parametrize("activation", ["tanh", "cubic", "softplus"])
def test_gradient_raw_matches_for_activated_modes(activation):
    from curlew.fields.series import FSF

    field = FSF(
        "f", H=HSet(), input_dim=2, rff_features=16, activation=activation,
        length_scale_range=(0.5, 3.0), seed=2,
    )
    with torch.no_grad():
        field.phi_mu.uniform_(0, 6.0)
        field.bias.normal_(0, 0.2)
    x = torch.rand(64, 2, dtype=curlew.dtype) * 3
    assert (field.gradient_raw(x) - field.gradient_and_hessian(x)[0]).abs().max() == 0.0


def test_euler_velocity_per_slot_kwargs():
    vel = euler_velocity(
        3,
        n_features=16,
        length_scale_range=(1.0, 4.0),
        seed=9,
        alpha={"n_features": 8},
        beta={"n_features": 32},
    )
    assert vel.alpha.A_mu.numel() == 8
    assert vel.beta.A_mu.numel() == 32


def test_cylindrical_requires_a_3d_hinge():
    with pytest.raises(ValueError, match="3-D only"):
        euler_velocity(2, cylindrical=True, hinge=[0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="hinge or section_axis"):
        euler_velocity(3, cylindrical=True)


def test_cylindrical_draws_features_and_directions_in_the_hinge_plane():
    axis = torch.tensor([1.0, 2.0, -0.5], dtype=torch.float64)
    axis = axis / axis.norm()
    n_dir = 8
    vel = euler_velocity(
        3,
        n_features=24,
        n_directions=n_dir,
        length_scale_range=(1.0, 4.0),
        seed=6,
        stream="alpha",
        hinge=axis.tolist(),
        cylindrical=True,
    )
    ox = vel.stream.omega_spatial
    assert ox.shape == (24, 3)
    assert torch.allclose(ox @ axis.to(ox.dtype), torch.zeros(24, dtype=ox.dtype), atol=1e-12)
    u = torch.nn.functional.normalize(ox, dim=-1)
    flip = u[:, int(axis.abs().argmax())] < 0
    u = torch.where(flip.unsqueeze(-1), -u, u)
    nuniq = int(torch.unique(torch.round(u * 1e6), dim=0).shape[0])
    assert nuniq == n_dir
    vel.stream.A_mu.data.normal_(0, 0.05)
    vel.stream.B_mu.data.normal_(0, 0.05)
    x = torch.randn(6, 3, dtype=curlew.dtype)
    _, jv = vel.forward_and_jacobian(x)
    trace = jv.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-9, rtol=1e-7)


def test_hinge_without_cylindrical_keeps_a_3d_frequency_bank():
    axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    vel = euler_velocity(
        3, n_features=32, length_scale_range=(1.0, 2.0), seed=6,
        stream="alpha", hinge=axis.tolist(),
    )
    dots = (vel.stream.omega_spatial @ axis.to(curlew.dtype)).abs()
    assert float(dots.max()) > 0.1


def test_slow_hinge_wanders_while_the_fourier_potential_stays_in_plane():
    from curlew.kinematics.euler import SlowHingePotential
    from curlew.kinematics.flow import FlowOffset

    axis = torch.tensor([0.2, 0.9, -0.3], dtype=torch.float64)
    axis = axis / axis.norm()
    vel = euler_velocity(
        3,
        n_features=12,
        n_directions=4,
        length_scale_range=(1.0, 4.0),
        seed=3,
        stream="alpha",
        activation="cubic",
        hinge=axis.tolist(),
        cylindrical=True,
        slow_hinge=(8.0, 32.0),
        slow_hinge_features=6,
        slow_hinge_directions=2,
    )
    assert isinstance(vel.hinge, SlowHingePotential)
    dots = vel.stream.omega_spatial @ axis.to(vel.stream.omega_spatial.dtype)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-12)
    lam = 2.0 * torch.pi / vel.hinge.slow.omega_spatial.norm(dim=-1)
    assert float(lam.min()) >= 8.0 - 1e-6
    assert float(lam.max()) <= 32.0 + 1e-6

    vel.stream.C_mu.data.normal_(0.0, 0.05)
    x = torch.randn(8, 3, dtype=curlew.dtype)
    g = vel.stream.gradient_raw(x)
    v_fixed = torch.linalg.cross(g, vel.hinge.direction.expand_as(g), dim=-1)
    v = vel(x)
    assert torch.allclose(v, v_fixed, atol=1e-10, rtol=1e-8)

    vel.hinge.slow.A_mu.data.normal_(0.0, 1.0)
    vel.hinge.slow.B_mu.data.normal_(0.0, 1.0)
    g_h = vel.hinge.gradient_raw(x)
    wander = (g_h - vel.hinge.direction).norm(dim=-1)
    assert float(wander.detach().max()) > 1e-3
    _, jv = vel.forward_and_jacobian(x)
    trace = jv.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-8, rtol=1e-6)

    flow = FlowOffset(vel, n_steps=1, integrator="shear")
    _, jac = flow.inverse_map_with_jacobian(x)
    det = torch.linalg.det(jac)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize(
    "dim, gauge",
    [(2, {}), (3, {"hinge": [0.0, 1.0, 0.0]}), (3, {"hinge": [0.0, 1.0, 0.0], "cylindrical": True})],
)
def test_sole_learnable_slot_shares_the_fsf_basis(dim, gauge):
    """With one learnable slot, euler_velocity(seed=s) draws FSF(seed=s)'s wavevectors."""
    curlew.default_dim = dim
    common = dict(length_scale_range=(0.5, 2.0), n_directions=8, activation="cubic")
    vel = euler_velocity(dim, n_features=32, seed=11, **common, **gauge)
    assert vel.stream is vel.alpha  # the default stream is alpha
    if gauge.get("cylindrical"):
        return  # the cylindrical bank is drawn in the hinge plane, not by FSF itself
    fsf = FSF("phi", H=HSet(), input_dim=dim, rff_features=32, seed=11, **common)
    assert torch.equal(vel.alpha.omega_spatial, fsf.omega_spatial)


def test_full_3d_slots_keep_distinct_seeds_and_legacy_stream():
    """Two learnable slots still differ, and the default keeps the old cross-product order."""
    curlew.default_dim = 3
    vel = euler_velocity(3, n_features=16, length_scale_range=(0.5, 2.0), seed=11)
    assert not torch.equal(vel.alpha.omega_spatial, vel.beta.omega_spatial)
    assert vel._stream == "beta"
