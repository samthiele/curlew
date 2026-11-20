"""Tests for :class:`~curlew.kinematics.restoration.RestorationField` and ``restore()``."""
import math

import curlew
import numpy as np
import pytest
import torch
from torch import nn

from curlew.core import CSet, HSet
from curlew.kinematics.restoration import RestorationField
from curlew.geology import restore
from curlew.geology.geomodel import GeoModel


@pytest.fixture(autouse=True)
def _defaults():
    curlew.default_dim = 2
    curlew.device = "cpu"
    curlew.dtype = torch.float64


class _ConstVelocity(nn.Module):
    """Uniform velocity for integration checks."""

    def __init__(self, vec, dim=2):
        super().__init__()
        self.dim = dim
        self.register_buffer(
            "v", torch.tensor(vec, dtype=curlew.dtype, device=curlew.device).reshape(dim)
        )

    def forward(self, x, w=None, t=None):
        return self.v.unsqueeze(0).expand(x.shape[0], -1)

    def forward_and_jacobian(self, x, w=None, t=None):
        n = x.shape[0]
        v = self.forward(x, w, t)
        return v, torch.zeros(n, self.dim, self.dim, dtype=x.dtype, device=x.device)


class _XShearVelocity(nn.Module):
    """``v_y = -k x`` so restored depth at fixed modern y depends on x."""

    def __init__(self, k=0.4, dim=2):
        super().__init__()
        self.dim = dim
        self.k = float(k)

    def forward(self, x, w=None, t=None):
        n = x.shape[0]
        v = torch.zeros(n, self.dim, dtype=x.dtype, device=x.device)
        v[:, 1] = -self.k * x[:, 0]
        return v

    def forward_and_jacobian(self, x, w=None, t=None):
        n = x.shape[0]
        v = self.forward(x, w, t)
        jac = torch.zeros(n, self.dim, self.dim, dtype=x.dtype, device=x.device)
        jac[:, 1, 0] = -self.k
        return v, jac


def _restoration_field(velocity, **kwargs) -> RestorationField:
    h = HSet(
        value_loss="1.0",
        grad_loss="1.0",
        eq_loss="1.0",
        mono_loss=0,
        thick_loss=0,
    )
    return RestorationField(
        "r0",
        H=h,
        input_dim=2,
        velocity=velocity,
        n_steps=kwargs.pop("n_steps", 4),
        depth_axis=1,
        learning_rate=1e-2,
        **kwargs,
    )


def test_restoration_field_evaluate_is_restored_depth():
    vel = _ConstVelocity([0.0, -0.5])
    field = _restoration_field(vel)
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=curlew.dtype)
    depth = field.evaluate(x)
    expected = (x + field.retro_displacement(x))[:, 1]
    assert torch.allclose(depth, expected)
    assert torch.allclose(depth, x[:, 1] + 0.5, atol=1e-10)


def test_restoration_field_value_loss():
    vel = _ConstVelocity([0.0, -0.2])
    field = _restoration_field(vel)
    x = torch.randn(6, 2, dtype=curlew.dtype)
    target_depth = field.restored_depth(x).detach()
    c = CSet(
        vp=x.numpy(),
        vv=target_depth.numpy(),
    )
    field.bind(c)
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-12


def test_strain_ellipse_loss_matches_forward_gradient():
    F = torch.tensor([[2.0, 0.4], [0.1, 0.5]], dtype=curlew.dtype)
    J = torch.linalg.inv(F).expand(3, 2, 2).contiguous()
    B = F @ F.T
    evals, evecs = torch.linalg.eigh(B)
    ratio = (evals[-1] / evals[0]).expand(3).contiguous()
    evecs = evecs.expand(3, 2, 2).contiguous()
    loss = RestorationField._strain_ellipse_loss(J, evecs, ratio)
    assert loss.item() < 1e-8

    wrong = torch.ones(3, dtype=curlew.dtype)
    assert RestorationField._strain_ellipse_loss(J, evecs, wrong).item() > 0.1

    Jg = torch.linalg.inv(F).detach().requires_grad_(True)
    graded = RestorationField._strain_ellipse_loss(
        Jg.expand(3, 2, 2), evecs, wrong,
    )
    graded.backward()
    assert Jg.grad is not None and Jg.grad.abs().sum().item() > 0

    # Equal singular values (a circle) make an SVD backward undefined.
    eye = torch.eye(2, dtype=curlew.dtype, requires_grad=True)
    circle = RestorationField._strain_ellipse_loss(
        eye.expand(1, 2, 2),
        torch.eye(2, dtype=curlew.dtype).expand(1, 2, 2),
        torch.tensor([4.0], dtype=curlew.dtype),
    )
    circle.backward()
    assert torch.isfinite(circle).item()
    assert torch.isfinite(eye.grad).all()
    assert eye.grad.abs().sum().item() > 0


def test_restoration_field_strain_loss():
    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel)
    x = torch.tensor([[0.0, 0.0], [1.0, 0.2]], dtype=curlew.dtype)
    eye = torch.eye(2, dtype=curlew.dtype).expand(2, 2, 2).contiguous()
    field.H = HSet.restore(
        value_loss=0, grad_loss=0, eq_loss=0, strain_loss=1.0,
    )
    field.bind(CSet(sp=x.numpy(), se=eye.numpy(), sr=np.ones(2)))
    matched = field.loss(transform=False)
    assert matched.total().item() == 0.0

    field.bind(CSet(sp=x.numpy(), se=eye.numpy(), sr=np.full(2, 4.0)))
    mismatched = field.loss(transform=False)
    assert mismatched.losses["r0"]["strain_loss"].item() > 0.5

    field.H = HSet.restore(value_loss=0, grad_loss=0, eq_loss=0, strain_loss=0)
    field.bind(CSet(sp=x.numpy(), se=eye.numpy(), sr=np.full(2, 4.0)))
    assert field.loss(transform=False).total().item() == 0.0


def test_strain_ellipse_loss_compares_only_what_is_given():
    """A ratio alone ignores the axes; eigenvectors alone ignore the ratio."""
    F = torch.diag(torch.tensor([2.0, 0.5], dtype=curlew.dtype))
    J = torch.linalg.inv(F).expand(2, 2, 2).contiguous()
    # Columns are minor then major: the short axis is y, the long axis is x.
    axes = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=curlew.dtype)
    evecs = axes.expand(2, 2, 2).contiguous()
    ratio = torch.full((2,), 16.0, dtype=curlew.dtype)

    assert RestorationField._strain_ellipse_loss(J, evecs, ratio).item() < 1e-8
    assert RestorationField._strain_ellipse_loss(J, None, ratio).item() < 1e-8
    assert RestorationField._strain_ellipse_loss(J, evecs, None).item() < 1e-8

    wrong_ratio = torch.ones(2, dtype=curlew.dtype)
    assert RestorationField._strain_ellipse_loss(J, None, wrong_ratio).item() > 0.1

    swapped = torch.eye(2, dtype=curlew.dtype).expand(2, 2, 2).contiguous()
    assert RestorationField._strain_ellipse_loss(J, swapped, None).item() > 0.1

    # A ratio is rotation-invariant, so its gradient vanishes on a circle.
    # It is finite there, and nonzero once the ellipse is already stretched.
    eye = torch.eye(2, dtype=curlew.dtype, requires_grad=True)
    graded = RestorationField._strain_ellipse_loss(
        eye.expand(1, 2, 2), None, torch.tensor([4.0], dtype=curlew.dtype),
    )
    graded.backward()
    assert graded.item() > 0.1
    assert torch.isfinite(eye.grad).all()

    Jg = torch.diag(
        torch.tensor([0.5, 2.0], dtype=curlew.dtype)
    ).detach().requires_grad_(True)
    stretched = RestorationField._strain_ellipse_loss(
        Jg.expand(1, 2, 2), None, torch.ones(1, dtype=curlew.dtype),
    )
    stretched.backward()
    assert torch.isfinite(Jg.grad).all()
    assert Jg.grad.abs().sum().item() > 0


def test_restoration_field_strain_loss_partial():
    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel)
    x = torch.tensor([[0.0, 0.0]], dtype=curlew.dtype)
    field.H = HSet.restore(
        value_loss=0, grad_loss=0, eq_loss=0, strain_loss=1.0,
    )
    field.bind(CSet(sp=x.numpy(), sr=np.array([4.0])))
    ratio_only = field.loss(transform=False)
    assert ratio_only.losses["r0"]["strain_loss"].item() > 0.1

    eye = torch.eye(2, dtype=curlew.dtype)
    field.bind(CSet(sp=x.numpy(), se=eye.numpy()))
    assert field.loss(transform=False).total().item() == 0.0

    kept = CSet(
        sp=np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
        sr=np.array([1.0, 2.0, 4.0]),
    ).filter(lambda pts: pts[:, 0] < 1.5)
    assert kept.se is None
    assert len(kept.sp) == 2
    assert np.allclose(kept.sr, [1.0, 2.0])


def test_restoration_field_eq_loss_flat_horizon():
    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel, sigma_floor=0.01)
    trace = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=curlew.dtype)
    c = CSet(eq=[trace.numpy()])
    field.bind(c)
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-8


def test_line_variance_is_scatter_about_the_best_fit_line():
    straight = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=curlew.dtype,
    )
    assert RestorationField._line_variance(straight).item() < 1e-10
    shifted = straight + torch.tensor([0.4, -1.0, 2.0], dtype=curlew.dtype)
    assert RestorationField._line_variance(shifted).item() < 1e-10
    bent = straight.clone()
    bent[1, 1] = 1.0
    bent[2, 1] = -0.5
    excess = torch.relu(
        RestorationField._line_variance(bent) - RestorationField._line_variance(straight)
    )
    assert excess.item() > 0.05
    y = bent.detach().requires_grad_(True)
    torch.relu(
        RestorationField._line_variance(y) - RestorationField._line_variance(straight).detach()
    ).backward()
    assert torch.isfinite(y.grad).all()
    assert y.grad[1, 1].abs() > 0


def test_eq_loss_charges_bending_a_straight_trace_in_plan():
    """A flat straight trace that bends horizontally pays the off-line variance, and nothing while it stays straight."""
    vel = _ConstVelocity([0.0, 0.0, 0.0], dim=3)
    h = HSet(value_loss=0, grad_loss=0, eq_loss="1.0", mono_loss=0, thick_loss=0)
    field = RestorationField(
        "r0", H=h, input_dim=3, velocity=vel, n_steps=1, depth_axis=2, learning_rate=1e-2,
    )
    trace = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    field.bind(CSet(eq=[trace]))
    assert field.loss(transform=False).total().item() < 1e-8

    def bend(x):
        y = x.clone()
        y[:, 1] = y[:, 1] + torch.sin(y[:, 0])
        return y

    field.inverse_map = bend
    # Off by default: a bend that leaves restored depths unchanged is free.
    assert field.loss(transform=False).total().item() < 1e-8
    field.collinear = True
    raw = field.loss(transform=False).losses["r0"]["eq_loss"].item()
    restored = bend(torch.tensor(trace, dtype=curlew.dtype))
    expect = float(torch.relu(RestorationField._line_variance(restored)).item())
    assert raw == pytest.approx(expect, rel=1e-6)
    assert raw > 1e-3


def test_restoration_field_eq_sample_subsamples_before_inverse_map():
    """eq_sample draws a random subset of each eq trace before inverse_map."""
    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel, sigma_floor=0.0, eq_sample=10)
    xs = torch.linspace(0, 5, 100, dtype=curlew.dtype)
    trace = torch.stack([xs, torch.ones_like(xs)], dim=1)
    field.bind(CSet(eq=[trace.numpy()]))

    sizes = []
    orig = field.inverse_map

    def _spy(x):
        sizes.append(int(x.shape[0]))
        return orig(x)

    field.inverse_map = _spy
    field.loss(transform=False)
    assert sizes == [10]

    field.eq_sample = 0.2
    sizes.clear()
    field.loss(transform=False)
    assert sizes == [20]

    field.eq_sample = None
    sizes.clear()
    field.loss(transform=False)
    assert sizes == [100]


@pytest.mark.parametrize("shape", ["surface", "line", "duplicates"])
def test_eq_strata_partition_any_unordered_horizon(shape):
    """
    Horizons are unordered points on a line or a surface. The strata must be a
    partition — every point in exactly one of ``k`` non-empty strata — whatever
    the shape, including repeated points.
    """
    from curlew.kinematics.restoration import RestorationField

    torch.manual_seed(0)
    if shape == "surface":
        pts = torch.rand(500, 3, dtype=curlew.dtype)
    elif shape == "line":
        t = torch.rand(300, dtype=curlew.dtype)
        pts = torch.stack([t, 2 * t, torch.zeros_like(t)], 1)[torch.randperm(300)]
    else:
        pts = torch.rand(20, 3, dtype=curlew.dtype).repeat(10, 1)
    k = 25
    members, offsets, counts = RestorationField._eq_strata(pts, k)
    assert counts.shape == (k,) and bool((counts > 0).all())
    assert torch.equal(torch.sort(members).values, torch.arange(len(pts)))
    assert int(offsets[-1] + counts[-1]) == len(pts)


def test_stratified_eq_sample_covers_the_whole_horizon():
    """
    One point per stratum per iteration: the sample has ``k`` distinct points
    and follows the horizon's extent, not its digitisation density. A random
    subset of a horizon digitised mostly in one corner sees the rest of it a
    fraction of the time; the stratified one sees it in proportion to its area.
    """
    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel, sigma_floor=0.0, eq_sample=20)
    torch.manual_seed(0)
    xy = torch.cat([torch.rand(450, 2) * 0.2, torch.rand(50, 2)]).to(curlew.dtype)
    outside = lambda p: ((p[:, 0] > 0.2) | (p[:, 1] > 0.2)).to(curlew.dtype).mean()

    share = {}
    for mode in ("random", "stratified"):
        field.eq_sampling = mode
        draws = [field._subsample_eq_trace(xy) for _ in range(200)]
        assert all(d.shape == (20, 2) for d in draws)
        assert all(len(torch.unique(d, dim=0)) == 20 for d in draws)
        share[mode] = float(torch.stack([outside(d) for d in draws]).mean())
    assert share["random"] < 0.2              # ~10 %: the dense corner dominates
    assert share["stratified"] > 0.6          # ~ the 96 % of the area it covers


def test_eq_strata_are_cached_and_rebuilt_when_k_changes():
    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel, sigma_floor=0.0, eq_sample=10)
    pts = torch.rand(200, 2, dtype=curlew.dtype)
    field._subsample_eq_trace(pts)
    first = field._eq_strata_cache[id(pts)][2]
    field._subsample_eq_trace(pts)
    assert field._eq_strata_cache[id(pts)][2] is first
    field.eq_sample = 30
    assert field._subsample_eq_trace(pts).shape == (30, 2)
    assert field._eq_strata_cache[id(pts)][1] == 30


def test_eq_sampling_is_validated():
    vel = _ConstVelocity([0.0, 0.0])
    with pytest.raises(ValueError, match="eq_sampling"):
        _restoration_field(vel, eq_sampling="nope")


def test_restoration_field_eq_loss_flattens_curved_horizon():
    """eq_loss must reduce restored-depth spread via the flow, not a side-channel parameter."""
    from curlew.kinematics.euler import euler_velocity

    xs = torch.linspace(0, 10, 40, dtype=curlew.dtype)
    ys = torch.sin(xs) * 2 + 5
    trace = torch.stack([xs, ys], dim=1)

    vel = euler_velocity(2, n_features=32, seed=7, length_scale_range=(2.0, 5.0))
    field = _restoration_field(vel, sigma_floor=0.0, init_std=0.01)
    field.bind(CSet(eq=[trace.numpy()]))

    var0 = field.restored_depth(trace).var().item()
    for _ in range(250):
        field.optim.zero_grad()
        field.loss(transform=False).total().backward()
        field.optim.step()
    var1 = field.restored_depth(trace).var().item()

    assert var1 < 0.5 * var0
    assert getattr(field, "trace_targets", None) is None


def test_restoration_field_eq_loss_no_cheat_with_frozen_flow():
    """With zero velocity, eq_loss equals trace variance (no side-channel DOF)."""
    vel = _ConstVelocity([0.0, 0.0])
    xs = torch.linspace(0, 5, 20, dtype=curlew.dtype)
    trace = torch.stack([xs, torch.sin(xs) + 3.0], dim=1)
    field = _restoration_field(vel, sigma_floor=0.0)
    field.bind(CSet(eq=[trace.numpy()]))

    depths = field.restored_depth(trace)
    var = depths.var(unbiased=False).item()
    pebble = field.loss(transform=False)
    eq = pebble.losses["r0"]["eq_loss"].item()
    assert abs(eq - var) < 1e-6


def test_restore_requires_velocity():
    with pytest.raises(TypeError):
        restore("fold")


def test_restore_geoevent_undeform():
    vel = _ConstVelocity([0.1, -0.2])
    h = HSet(value_loss=0, grad_loss=0, eq_loss=0, mono_loss=0, thick_loss=0)
    field = RestorationField("fold", H=h, input_dim=2, velocity=vel, n_steps=2)
    event = restore("fold", C=None, H=h, field=field, velocity=vel)
    assert event.deformation is field.integrator
    x = torch.tensor([[0.0, 0.0], [1.0, 0.5]], dtype=curlew.dtype)
    x_paleo = event.undeform(x.clone())
    assert torch.allclose(x_paleo, x + field.retro_displacement(x), atol=1e-10)


def test_restore_factory_binds_cset():
    vel = _ConstVelocity([0.0, 0.0])
    x = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=curlew.dtype)
    c = CSet(vp=x.numpy(), vv=torch.tensor([0.0, 2.0], dtype=curlew.dtype).numpy())
    event = restore(
        "rest",
        C=c,
        velocity=vel,
        n_steps=2,
    )
    assert isinstance(event.field, RestorationField)
    pebble = event.loss()
    assert pebble.total().item() >= 0


def test_restore_is_generative_like_strati():
    """
    `restore` is a `strati`-style generative event: it carries an Overprint, so
    isosurfaces added to it produce lithoID (via GeoModel) exactly like any other
    stratigraphic package, rather than being silently ignored.
    """
    vel = _ConstVelocity([0.0, 0.0])  # identity flow -> restored depth == y exactly
    h = HSet(value_loss=0, grad_loss=0, eq_loss=0, mono_loss=0, thick_loss=0)
    field = RestorationField("fold", H=h, input_dim=2, velocity=vel, n_steps=2, depth_axis=1)
    event = restore("fold", C=None, H=h, field=field, velocity=vel)
    assert event.overprint is not None

    event.addIsosurface("mid", value=0.5)

    M = GeoModel([event])
    pred = M.predict(np.array([[0.0, 0.0], [0.0, 1.0]]), coords="model")

    litho_names = set(pred.lithoLookup.values())
    assert "fold" in litho_names
    assert "fold_mid" in litho_names


def test_restoration_field_energy_loss_term():
    """`H.energy_loss` adds the analytic path-energy term; absent when 0 (default)."""
    from curlew.kinematics.euler import euler_velocity

    vel = euler_velocity(2, n_features=8, seed=13)
    vel.stream.A_mu.data.normal_(0, 0.2)
    gp = np.random.RandomState(0).uniform(-1, 1, size=(6, 2))
    gv = np.random.RandomState(1).normal(size=(6, 2))
    gv /= np.linalg.norm(gv, axis=1, keepdims=True)
    c = CSet(gp=gp, gv=gv)

    h_off = HSet(value_loss=0, grad_loss="1.0", eq_loss=0, mono_loss=0, thick_loss=0)
    f_off = RestorationField("r_off", H=h_off, input_dim=2, velocity=vel, n_steps=2, depth_axis=1)
    f_off.bind(c)
    assert "energy_loss" not in f_off.loss(transform=False).losses.get("r_off", {})

    h_on = HSet(
        value_loss=0, grad_loss="1.0", eq_loss=0, mono_loss=0, thick_loss=0,
        energy_loss="0.5",
    )
    f_on = RestorationField("r_on", H=h_on, input_dim=2, velocity=vel, n_steps=2, depth_axis=1)
    f_on.bind(c)
    pebble = f_on.loss(transform=False)
    assert "energy_loss" in pebble.losses.get("r_on", {})
    pebble.total().backward()
    assert vel.stream.A_mu.grad is not None
    assert (vel.stream.A_mu.grad.abs() > 0).any()


def test_init_std_zero_builds_identity_flow():
    from curlew.kinematics.euler import euler_velocity

    linear = euler_velocity(2, n_features=8, seed=14)
    activated = euler_velocity(2, n_features=8, seed=15, activation="cubic")
    f_linear = _restoration_field(linear, init_std=0.0)
    f_activated = _restoration_field(activated, init_std=0.0)

    assert torch.count_nonzero(linear.stream.A_mu) == 0
    assert torch.count_nonzero(linear.stream.B_mu) == 0
    assert torch.count_nonzero(activated.stream.C_mu) == 0
    assert torch.count_nonzero(activated.stream.phi_mu) == 0
    assert torch.count_nonzero(activated.stream.bias) == 0

    x = torch.randn(6, 2, dtype=curlew.dtype)
    assert torch.equal(f_linear.inverse_map(x), x)
    assert torch.equal(f_activated.inverse_map(x), x)


def test_init_std_none_preserves_preconfigured_velocity():
    from curlew.kinematics.euler import euler_velocity

    vel = euler_velocity(2, n_features=8, seed=16)
    before_a = vel.stream.A_mu.detach().clone()
    before_b = vel.stream.B_mu.detach().clone()
    _restoration_field(vel, init_std=None)
    assert torch.equal(vel.stream.A_mu, before_a)
    assert torch.equal(vel.stream.B_mu, before_b)


def test_restoration_field_orthogonal_thickness_identity():
    """Identity flow has |J^T e_d| = 1 everywhere."""
    from curlew.kinematics.constraints import orthogonal_thickness_loss

    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel)
    x = torch.randn(12, 2, dtype=curlew.dtype)
    x0, J = field.integrator.inverse_map_with_jacobian(x)
    assert orthogonal_thickness_loss(x0, J, field.depth_axis).item() < 1e-12


def test_restoration_field_orthogonal_thickness_constraint():
    from curlew.kinematics.euler import euler_velocity
    from curlew.kinematics.constraints import OrthogonalThickness

    vel = euler_velocity(2, n_features=16, seed=3)
    vel.stream.A_mu.data.normal_(0, 0.05)
    pts = torch.randn(8, 2, dtype=curlew.dtype)
    field = _restoration_field(vel, init_std=None)
    field.constraints = [OrthogonalThickness(weight=1.0, points=pts)]
    field.bind(CSet())
    pebble = field.loss(transform=False)
    assert pebble.losses["r0"]["thick_loss"].item() > 0.0


def test_restoration_field_layer_thickness_constraint():
    """Layer gaps should match relative thickness ratios."""
    from curlew.kinematics.constraints import LayerThickness, layer_thickness_loss

    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel)
    t0 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=curlew.dtype)
    t1 = torch.tensor([[0.0, 2.0], [1.0, 2.0], [2.0, 2.0]], dtype=curlew.dtype)
    t2 = torch.tensor([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]], dtype=curlew.dtype)
    depths = [field.restored_depth(t) for t in (t0, t1, t2)]
    assert (
        layer_thickness_loss(
            depths, torch.tensor([2.0, 3.0]), mode="relative"
        ).item()
        < 1e-10
    )

    field.constraints = [
        LayerThickness(
            [t0.numpy(), t1.numpy(), t2.numpy()],
            thicknesses=[1.0, 1.0],
            weight=1.0,
            mode="relative",
        )
    ]
    field.bind(CSet())
    pebble = field.loss(transform=False)
    assert pebble.losses["r0"]["layer_thick_loss"].item() > 0.01


def test_restoration_field_orthogonal_thickness_restored_layer():
    """Restored-layer sampling uses inverse-mapped horizons, not depth targets."""
    from curlew.kinematics.constraints import OrthogonalThickness

    vel = _ConstVelocity([0.0, 0.0])
    h3 = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    h4 = np.array([[-2.0, 2.0], [0.0, 2.0], [2.0, 2.0]])
    field = _restoration_field(vel)
    field.constraints = [
        OrthogonalThickness(
            weight=1.0,
            restored_layer=(0, 1),
            horizons=[h3, h4],
            n_along=8,
            n_between=3,
        )
    ]
    field.bind(CSet())
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-10


def test_restoration_field_orthogonal_thickness_restored_layer_nonzero():
    from curlew.kinematics.euler import euler_velocity
    from curlew.kinematics.constraints import OrthogonalThickness

    vel = euler_velocity(2, n_features=16, seed=11)
    vel.stream.A_mu.data.normal_(0, 0.05)
    h3 = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    h4 = np.array([[-2.0, 2.0], [0.0, 2.0], [2.0, 2.0]])
    field = _restoration_field(vel, init_std=None)
    field.constraints = [
        OrthogonalThickness(
            weight=1.0,
            restored_layer=(0, 1),
            horizons=[h3, h4],
            n_along=8,
            n_between=3,
        )
    ]
    field.bind(CSet())
    pebble = field.loss(transform=False)
    assert pebble.losses["r0"]["thick_loss"].item() > 0.0


def test_points_between_horizons_graph_uses_union_of_x_spans():
    from curlew.kinematics.constraints import points_between_horizons

    h0 = np.array([[-2.0, 0.0], [0.0, 0.0]])
    h1 = np.array([[0.0, 2.0], [3.0, 2.0]])
    x0 = points_between_horizons(
        h0, h1, n_along=5, n_between=1, t_min=0.5, t_max=0.5,
        parameterization="graph", depth_axis=1,
    ).detach().cpu().numpy()
    assert x0[:, 0] == pytest.approx(np.linspace(-2.0, 3.0, 5))


def test_orthogonal_thickness_horizon_indices_from_cset():
    """Integer horizon pair samples the unit between bound CSet.eq traces."""
    from curlew.kinematics.constraints import OrthogonalThickness

    vel = _ConstVelocity([0.0, 0.0])
    rng = np.random.default_rng(0)
    h0 = np.column_stack([rng.uniform(-2.0, 2.0, 9), np.zeros(9)])
    h1 = np.column_stack([rng.uniform(-2.0, 2.0, 9), np.full(9, 2.0)])
    field = _restoration_field(vel)
    constraint = OrthogonalThickness(
        weight=1.0,
        horizons=(0, 1),
        n_along=8,
        n_between=3,
        t_min=0.1,
        t_max=0.9,
    )
    field.constraints = [constraint]
    field.bind(CSet(eq=[h0, h1]))
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-10
    x0 = constraint._restored_layer_points(field, None).detach().cpu().numpy()
    assert x0.shape == (24, 2)
    assert x0[:, 1].min() >= 0.2 - 1e-8
    assert x0[:, 1].max() <= 1.8 + 1e-8
    assert x0[:, 0].min() == pytest.approx(
        min(h0[:, 0].min(), h1[:, 0].min())
    )
    assert x0[:, 0].max() == pytest.approx(
        max(h0[:, 0].max(), h1[:, 0].max())
    )


def test_orthogonal_thickness_restored_layer_looks_up_eq_without_polylines():
    from curlew.kinematics.constraints import OrthogonalThickness

    vel = _ConstVelocity([0.0, 0.0])
    h0 = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    h1 = np.array([[-2.0, 2.0], [0.0, 2.0], [2.0, 2.0]])
    field = _restoration_field(vel)
    field.constraints = [
        OrthogonalThickness(weight=1.0, restored_layer=(0, 1), n_along=8, n_between=3)
    ]
    field.bind(CSet(eq=[h0, h1]))
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-10


def test_orthogonal_thickness_keeps_present_day_points_and_rejects_mixed_modes():
    from curlew.kinematics.constraints import OrthogonalThickness

    pts = [[0.0, 0.0], [1.0, 0.5]]
    OrthogonalThickness(weight=1.0, points=pts)
    with pytest.raises(ValueError, match="exactly one"):
        OrthogonalThickness(weight=1.0, points=pts, horizons=(0, 1))
    with pytest.raises(ValueError, match="disagree"):
        OrthogonalThickness(weight=1.0, restored_layer=(0, 1), horizons=(2, 3))


def test_hencky_strain_loss_invariants():
    from curlew.kinematics.constraints import hencky_strain_loss

    eye = torch.eye(2, dtype=curlew.dtype).expand(5, 2, 2).clone()
    assert hencky_strain_loss(eye).item() < 1e-14

    theta = 0.4
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s], [s, c]], dtype=curlew.dtype)
    assert hencky_strain_loss(rot.expand(3, 2, 2).clone()).item() < 1e-14

    stretch = torch.diag(torch.tensor([2.0, 0.5], dtype=curlew.dtype))
    got = hencky_strain_loss(stretch.unsqueeze(0))
    expected = 2.0 * math.log(2.0) ** 2
    assert abs(float(got) - expected) < 1e-12
    assert abs(float(hencky_strain_loss(rot @ stretch)) - expected) < 1e-12

    # volume-preserving inverse: Hencky is the same
    inv = torch.diag(torch.tensor([0.5, 2.0], dtype=curlew.dtype))
    assert abs(float(hencky_strain_loss(inv.unsqueeze(0))) - expected) < 1e-12

    # pure dilation is stripped by the isochoric projection
    dil = torch.diag(torch.tensor([3.0, 3.0], dtype=curlew.dtype))
    assert hencky_strain_loss(dil.unsqueeze(0)).item() < 1e-14


def test_points_between_horizons_single_row_is_midpoint():
    from curlew.kinematics.constraints import points_between_horizons

    h0 = np.array([[0.0, 0.0], [10.0, 0.0]])
    h1 = np.array([[0.0, 10.0], [10.0, 10.0]])
    pts = points_between_horizons(h0, h1, n_along=3, n_between=1, t_min=0.4, t_max=0.6)
    assert np.allclose(pts.cpu().numpy()[:, 1], 5.0)


def test_layer_parallel_stretch_loss():
    from curlew.kinematics.constraints import layer_parallel_stretch_loss

    # rotation, layer-parallel shear and stretch across the layer are free
    theta = 0.4
    c, s = math.cos(theta), math.sin(theta)
    rot = torch.tensor([[c, -s], [s, c]], dtype=curlew.dtype)
    assert layer_parallel_stretch_loss(rot.unsqueeze(0), 1).item() < 1e-14
    shear = torch.tensor([[1.0, 0.7], [0.0, 1.0]], dtype=curlew.dtype)
    assert layer_parallel_stretch_loss(shear.unsqueeze(0), 1).item() < 1e-14
    across = torch.diag(torch.tensor([1.0, 3.0], dtype=curlew.dtype))
    assert layer_parallel_stretch_loss(across.unsqueeze(0), 1).item() < 1e-14

    # shortening along the layer is charged, rotated or not; depth_axis picks the layer
    short = torch.diag(torch.tensor([0.5, 2.0], dtype=curlew.dtype))
    expected = math.log(2.0) ** 2
    assert abs(float(layer_parallel_stretch_loss(short.unsqueeze(0), 1)) - expected) < 1e-12
    assert abs(float(layer_parallel_stretch_loss((rot @ short).unsqueeze(0), 1)) - expected) < 1e-12
    assert abs(float(layer_parallel_stretch_loss(short.unsqueeze(0), 0)) - expected) < 1e-12

    # 3D: both in-plane stretches count
    F3 = torch.diag(torch.tensor([2.0, 0.5, 1.0], dtype=curlew.dtype))
    assert abs(float(layer_parallel_stretch_loss(F3.unsqueeze(0), 2)) - 2 * expected) < 1e-12


def test_hencky_strain_loss_3d_stretch():
    from curlew.kinematics.constraints import hencky_strain_loss

    J = torch.diag(torch.tensor([2.0, 0.5, 1.0], dtype=curlew.dtype)).unsqueeze(0)
    expected = 2.0 * math.log(2.0) ** 2
    assert abs(float(hencky_strain_loss(J)) - expected) < 1e-12


def test_restoration_field_hencky_identity():
    from curlew.kinematics.constraints import HenckyStrain, hencky_strain_loss

    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel)
    x = torch.randn(12, 2, dtype=curlew.dtype)
    _, J = field.integrator.inverse_map_with_jacobian(x)
    assert hencky_strain_loss(J).item() < 1e-12

    field.constraints = [HenckyStrain(weight=1.0, points=x)]
    field.bind(CSet())
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-10


def test_restoration_field_hencky_constraint():
    from curlew.kinematics.euler import euler_velocity
    from curlew.kinematics.constraints import HenckyStrain

    vel = euler_velocity(2, n_features=16, seed=3)
    vel.stream.A_mu.data.normal_(0, 0.05)
    vel.stream.B_mu.data.normal_(0, 0.05)
    pts = torch.randn(8, 2, dtype=curlew.dtype)
    field = _restoration_field(vel, init_std=None)
    field.constraints = [HenckyStrain(weight=1.0, points=pts)]
    field.bind(CSet())
    pebble = field.loss(transform=False)
    assert pebble.losses["r0"]["hencky_loss"].item() > 0.0
    pebble.total().backward()
    assert vel.stream.A_mu.grad is not None
    assert (vel.stream.A_mu.grad.abs() > 0).any()


def test_restoration_field_hencky_restored_layer():
    from curlew.kinematics.constraints import HenckyStrain

    vel = _ConstVelocity([0.0, 0.0])
    h3 = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    h4 = np.array([[-2.0, 2.0], [0.0, 2.0], [2.0, 2.0]])
    field = _restoration_field(vel)
    field.constraints = [
        HenckyStrain(
            weight=1.0,
            restored_layer=(0, 1),
            horizons=[h3, h4],
            n_along=8,
            n_between=3,
        )
    ]
    field.bind(CSet())
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-10


def test_hencky_strain_requires_one_sample_mode():
    from curlew.kinematics.constraints import HenckyStrain

    with pytest.raises(ValueError, match="exactly one"):
        HenckyStrain(weight=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        HenckyStrain(weight=1.0, points=[[0.0, 0.0]], grid=object())


def test_restore_accepts_constraints():
    from curlew.kinematics.constraints import LayerThickness

    vel = _ConstVelocity([0.0, 0.0])
    t0 = np.array([[0.0, 0.0], [1.0, 0.0]])
    t1 = np.array([[0.0, 3.0], [1.0, 3.0]])
    t2 = np.array([[0.0, 7.0], [1.0, 7.0]])
    event = restore(
        "fold",
        C=CSet(eq=[t0, t1, t2]),
        velocity=vel,
        n_steps=2,
        constraints=[
            LayerThickness([t0, t1, t2], [1.0, 1.0], weight=1.0),
        ],
    )
    assert len(event.field.constraints) == 1


def test_restoration_field_iq_loss_satisfied():
    """Restored-depth ordering: upper pool shallower than lower → zero hinge loss."""
    vel = _ConstVelocity([0.0, 0.0])
    h = HSet(
        value_loss=0, grad_loss=0, eq_loss=0, iq_loss="1.0",
        mono_loss=0, thick_loss=0,
    )
    field = RestorationField(
        "r0", H=h, input_dim=2, velocity=vel, n_steps=2, depth_axis=1,
    )
    upper = np.array([[0.0, 5.0], [1.0, 5.5], [2.0, 4.8]])
    lower = np.array([[0.0, 1.0], [1.0, 1.2], [2.0, 0.9]])
    field.bind(CSet(iq=(8, [(upper, lower, ">")])))
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-10


def test_restoration_field_iq_loss_violated():
    """Swapped pools violate restored-depth ordering → positive hinge loss."""
    vel = _ConstVelocity([0.0, 0.0])
    h = HSet(
        value_loss=0, grad_loss=0, eq_loss=0, iq_loss=1.0,
        mono_loss=0, thick_loss=0,
    )
    field = RestorationField(
        "r0", H=h, input_dim=2, velocity=vel, n_steps=2, depth_axis=1,
    )
    upper = np.array([[0.0, 5.0], [1.0, 5.5]])
    lower = np.array([[0.0, 1.0], [1.0, 1.2]])
    field.bind(CSet(iq=(16, [(lower, upper, ">")])))
    pebble = field.loss(transform=False)
    assert pebble.losses["r0"]["iq_loss"].item() > 0.5


def test_restoration_field_iq_loss_uses_restored_depth():
    """Inequality compares Φ⁻¹ depth, not modern y when flow couples x into depth."""
    vel = _XShearVelocity(k=0.5)
    h = HSet(
        value_loss=0, grad_loss=0, eq_loss=0, iq_loss=1.0,
        mono_loss=0, thick_loss=0,
    )
    field = RestorationField(
        "r0", H=h, input_dim=2, velocity=vel, n_steps=4, depth_axis=1,
    )
    left = torch.tensor([[0.0, 2.0], [1.0, 2.0]], dtype=curlew.dtype)
    right = torch.tensor([[6.0, 2.0], [7.0, 2.0]], dtype=curlew.dtype)
    assert left[:, 1].mean() == right[:, 1].mean()
    assert field.restored_depth(right).mean() > field.restored_depth(left).mean()

    field.bind(CSet(iq=(8, [(right.numpy(), left.numpy(), ">")])))
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-8

    field.bind(CSet(iq=(8, [(left.numpy(), right.numpy(), ">")])))
    pebble = field.loss(transform=False)
    assert pebble.losses["r0"]["iq_loss"].item() > 0.1


def test_restoration_field_iq_loss_training_reduces_violation():
    """Optimising the flow should enforce restored-depth ordering."""
    from curlew.kinematics.euler import euler_velocity

    xs = torch.linspace(0, 6, 12, dtype=curlew.dtype)
    # Crossed horizons in modern coords: require upper restored > lower.
    upper = torch.stack([xs, torch.sin(xs) + 2.0], dim=1)
    lower = torch.stack([xs, torch.sin(xs) + 5.0], dim=1)
    c = CSet(iq=(32, [(upper.numpy(), lower.numpy(), ">")]))

    vel = euler_velocity(2, n_features=32, seed=19, length_scale_range=(2.0, 5.0))
    h = HSet(
        value_loss=0, grad_loss=0, eq_loss=0, iq_loss="1.0",
        mono_loss=0, thick_loss=0,
    )
    field = RestorationField(
        "r0", H=h, input_dim=2, velocity=vel, n_steps=4, depth_axis=1,
        learning_rate=5e-3, init_std=0.0,
    )
    field.bind(c)

    def _violation():
        gap = field.restored_depth(lower) - field.restored_depth(upper)
        return torch.relu(gap).mean().item()

    v0 = _violation()
    assert v0 > 0.01
    for _ in range(600):
        field.optim.zero_grad()
        field.loss(transform=False).total().backward()
        field.optim.step()
    v1 = _violation()
    assert v1 < 0.75 * v0


def test_predict_normals_uses_reference_field_gradient():
    """With identity local flow, eq normals match the reference shape event."""
    v_shape = _ConstVelocity([0.0, -0.15])
    v_eq = _ConstVelocity([0.0, 0.0])
    shape = restore(
        "shape",
        velocity=v_shape,
        C=CSet(),
        n_steps=3,
        learning_rate=1e-2,
    )
    eq = restore(
        "equality",
        velocity=v_eq,
        C=CSet(),
        reference=shape,
        n_steps=3,
        learning_rate=1e-2,
    )
    GeoModel([shape, eq])
    x = torch.tensor([[0.0, 1.0], [1.5, 2.0], [-0.5, 0.5]], dtype=curlew.dtype)
    n_shape = shape.field.predict_normals(x)
    n_eq = eq.field.predict_normals(x)
    assert torch.allclose(n_eq, n_shape, atol=1e-5)


def test_default_flat_reference_stratigraphy():
    field = restore(
        "solo",
        velocity=_ConstVelocity([0.0, -0.1]),
        C=CSet(),
        n_steps=2,
        learning_rate=1e-2,
    ).field
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=curlew.dtype)
    g_flat = field.reference.gradient(x)
    assert field.reference_event is None
    assert torch.allclose(g_flat[..., 1], torch.ones(2, dtype=curlew.dtype))
    assert torch.allclose(g_flat[..., 0], torch.zeros(2, dtype=curlew.dtype))


def test_reference_is_queried_not_composed():
    """
    The reference supplies the *stratigraphy*, the local map supplies the
    pullback: phi_B(x) = phi_A(Phi_B_local^-1 x). Nothing is composed, and the
    child keeps its own (local) integrator.
    """
    from curlew.geology.interactions import FlowOffset

    v1 = _ConstVelocity([0.0, -0.1])
    v2 = _ConstVelocity([0.0, -0.2])
    shape = restore(
        "shape", velocity=v1, C=CSet(), H=HSet().zero(grad_loss="1."),
        n_steps=2, learning_rate=1e-2,
    )
    eq = restore(
        "equality", velocity=v2, C=CSet(), H=HSet().zero(eq_loss="1."),
        reference=shape, n_steps=2, learning_rate=1e-2,
    )
    GeoModel([shape, eq])

    # one integrator, and it is the child's own local flow
    assert isinstance(eq.field.integrator, FlowOffset)
    assert eq.field.reference_event is shape

    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=curlew.dtype)
    # phi_eq(x) == phi_shape(Phi_eq_local^-1 x), evaluated explicitly
    x0 = eq.field.integrator.inverse_map(x)
    expected = shape.field.restored_depth(x0)
    assert torch.allclose(eq.field.restored_depth(x), expected)

    # each event contributes only its own displacement to the undeform chain
    assert torch.allclose(shape.displacement(x), shape.field.retro_displacement(x))
    assert torch.allclose(eq.displacement(x), eq.field.retro_displacement(x))


def test_reference_normal_is_the_reference_gradient_transported():
    """
    n_B = normalize(J_B_local^T . grad phi_A(x0)) — the reference's own bedding
    orientation, transported by the child's Jacobian. With a *curved* reference
    this is not the depth row of any composed Jacobian.
    """
    from curlew.kinematics.euler import euler_velocity

    v1 = euler_velocity(2, n_features=16, seed=11)
    v1.stream.A_mu.data.normal_(0, 0.3)
    v2 = euler_velocity(2, n_features=16, seed=12)
    v2.stream.A_mu.data.normal_(0, 0.3)
    a = restore("a", velocity=v1, C=CSet(), n_steps=6, learning_rate=1e-2)
    b = restore("b", velocity=v2, C=CSet(), reference=a, n_steps=6, learning_rate=1e-2)
    GeoModel([a, b])

    x = torch.tensor([[0.3, 0.4], [-0.5, 0.8]], dtype=curlew.dtype)
    with torch.no_grad():
        x0, j = b.field.integrator.inverse_map_with_jacobian(x)
        g_ref = a.field.stratigraphy_gradient(x0)
        expected = torch.nn.functional.normalize(
            torch.bmm(j.mT, g_ref.unsqueeze(-1)).squeeze(-1), dim=-1
        )
        assert torch.allclose(b.field.predict_normals(x), expected, atol=1e-12)


def test_reference_stratigraphy_can_be_any_scalar_field():
    """A restoration can be fitted relative to an existing implicit model."""
    from curlew.fields.analytical import LinearField

    ref = LinearField("ref", input_dim=2, gradient=[0.0, 1.0])
    ev = restore(
        "r", velocity=_ConstVelocity([0.0, -0.25]), C=CSet(),
        reference=ref, n_steps=4, learning_rate=1e-2,
    )
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=curlew.dtype)
    # this LinearField *is* the flat layer cake, so it must agree exactly
    flat = restore(
        "f", velocity=_ConstVelocity([0.0, -0.25]), C=CSet(),
        n_steps=4, learning_rate=1e-2,
    )
    assert torch.allclose(ev.field.restored_depth(x), flat.field.restored_depth(x))
    assert torch.allclose(ev.field.predict_normals(x), flat.field.predict_normals(x))


def test_restore_reference_parent_resolves_in_geomodel():
    v1 = _ConstVelocity([0.0, -0.05])
    v2 = _ConstVelocity([0.0, -0.05])
    shape = restore("shape", velocity=v1, C=CSet(), n_steps=2, learning_rate=1e-2)
    eq = restore(
        "equality",
        velocity=v2,
        C=CSet(),
        reference="parent",
        n_steps=2,
        learning_rate=1e-2,
    )
    model = GeoModel([shape, eq])
    assert eq.field.reference_event is shape


def test_predict_normals_requires_parent_reference_wired():
    from curlew.kinematics.euler import euler_velocity

    v_ref = euler_velocity(2, n_features=24, seed=21)
    v_ref.stream.A_mu.data.normal_(0, 0.1)
    v_loc = euler_velocity(2, n_features=24, seed=22)
    v_loc.stream.A_mu.data.normal_(0, 0.05)
    shape = restore("shape", velocity=v_ref, C=CSet(), n_steps=4, learning_rate=1e-2)
    eq = restore(
        "equality",
        velocity=v_loc,
        C=CSet(),
        reference="parent",
        n_steps=4,
        learning_rate=1e-2,
    )
    x = torch.randn(6, 2, dtype=curlew.dtype)
    with pytest.raises(RuntimeError, match="reference='parent'"):
        eq.field.predict_normals(x)

    GeoModel([shape, eq])
    n_ref = eq.field.predict_normals(x)
    n_bare = restore(
        "bare", velocity=v_loc, C=CSet(), n_steps=4, learning_rate=1e-2,
    ).field.predict_normals(x)
    assert not torch.allclose(n_ref, n_bare, atol=1e-4)


def test_prefit_staged_two_stage_restoration():
    from curlew.kinematics.euler import euler_velocity

    xs = torch.linspace(0, 4, 20, dtype=curlew.dtype)
    trace = torch.stack([xs, torch.sin(xs) + 2.0], dim=1)
    gp = trace[::2].numpy()
    gv = np.column_stack([-np.sin(gp[:, 0]), np.ones(len(gp))])
    gv /= np.linalg.norm(gv, axis=1, keepdims=True)

    c = CSet(
        gp=gp,
        gv=gv,
        eq=[trace.numpy()],
        iq=(8, [(trace[:6].numpy(), trace[6:12].numpy(), "<")]),
    )

    v1 = euler_velocity(2, n_features=24, seed=3, length_scale_range=(2.0, 2.0))
    v2 = euler_velocity(2, n_features=24, seed=4, length_scale_range=(2.0, 2.0))
    shape = restore(
        "shape",
        velocity=v1,
        C=c,
        H=HSet().zero(grad_loss="1."),
        n_steps=3,
        learning_rate=5e-3,
        init_std=0.05,
        sigma_floor=0.0,
    )
    eq = restore(
        "equality",
        velocity=v2,
        C=c,
        H=HSet().zero(eq_loss="1.", iq_loss="1."),
        reference=shape,
        n_steps=3,
        learning_rate=5e-3,
        init_std=0.05,
        sigma_floor=0.0,
    )
    model = GeoModel([shape, eq])

    var0 = eq.field.restored_depth(trace).var().item()
    model.prefit_staged([("shape", 80)])
    grad_pebble = shape.field.loss(transform=False)
    model.freeze("shape")
    model.prefit_staged([("equality", 120)])
    var1 = eq.field.restored_depth(trace).var().item()

    assert grad_pebble.losses["shape"]["grad_loss"].item() < 0.95
    # Small tolerance: var0 is the untrained baseline, var1 a short (80+120
    # epoch) two-stage fit — not an exact inequality, just "training doesn't
    # make it substantially worse".
    assert var1 <= var0 * 1.05
    eq_pebble = eq.field.loss(transform=False)
    assert eq_pebble.losses["equality"]["eq_loss"].item() < 0.95


def test_predict_normals_matches_autograd_three_level_chain():
    """
    predict_normals/stratigraphy_gradient must stay consistent with
    restored_depth (verified via autograd) at any reference-chain depth, not
    just two levels.
    """
    from curlew.kinematics.euler import euler_velocity

    v1 = euler_velocity(2, n_features=16, seed=1)
    v1.stream.A_mu.data.normal_(0, 0.2)
    v2 = euler_velocity(2, n_features=16, seed=2)
    v2.stream.A_mu.data.normal_(0, 0.2)
    v3 = euler_velocity(2, n_features=16, seed=3)
    v3.stream.A_mu.data.normal_(0, 0.2)
    base = restore("base", velocity=v1, C=CSet(), n_steps=8, learning_rate=1e-2)
    mid = restore(
        "mid", velocity=v2, C=CSet(), reference="parent", n_steps=8, learning_rate=1e-2,
    )
    top = restore(
        "top", velocity=v3, C=CSet(), reference="parent", n_steps=8, learning_rate=1e-2,
    )
    GeoModel([base, mid, top])

    x = torch.tensor(
        [[0.2, 0.5], [-0.6, 1.0], [0.9, -0.2]], dtype=curlew.dtype, requires_grad=True,
    )
    depth = top.field.restored_depth(x)
    grad_true = torch.autograd.grad(depth.sum(), x)[0]
    grad_true_dir = torch.nn.functional.normalize(grad_true, dim=-1)

    with torch.no_grad():
        n_pred = top.field.predict_normals(x.detach())
        g_pred = top.field.stratigraphy_gradient(x.detach())

    assert torch.allclose(grad_true_dir, n_pred, atol=1e-3)
    assert torch.allclose(
        grad_true_dir, torch.nn.functional.normalize(g_pred, dim=-1), atol=1e-3,
    )


def test_set_reference_event_rejects_cycle():
    a = restore("a", velocity=_ConstVelocity([0.0, -0.1]), C=CSet(), n_steps=2, learning_rate=1e-2)
    b = restore(
        "b", velocity=_ConstVelocity([0.0, -0.1]), C=CSet(), reference=a,
        n_steps=2, learning_rate=1e-2,
    )
    with pytest.raises(ValueError, match="cycle"):
        a.field.set_reference(b)


def test_unresolved_parent_reference_raises_on_evaluation():
    """
    `reference='parent'` is only resolved by GeoModel. Until then, evaluating
    would silently fall back to the flat reference, so it raises instead.
    """
    mid = restore(
        "mid", velocity=_ConstVelocity([0.0, -0.1]), C=CSet(), reference="parent",
        n_steps=2, learning_rate=1e-2,
    )
    x = torch.tensor([[0.0, 1.0]], dtype=curlew.dtype)
    with pytest.raises(RuntimeError, match="not.*wired yet"):
        mid.field.restored_depth(x)


# ── invariants that must survive the reference-stratigraphy refactor ─────────

def _curved_field(seed=7, n_steps=8):
    from curlew.kinematics.euler import euler_velocity

    v = euler_velocity(2, n_features=24, seed=seed, length_scale_range=(2.0, 8.0))
    with torch.no_grad():
        v.stream.A_mu.normal_(0.0, 0.4)
    return restore("r", velocity=v, C=CSet(), n_steps=n_steps, learning_rate=1e-2).field


def test_flat_reference_is_bit_identical_to_the_depth_component():
    """
    The single-event path must be *exactly* the historic model. With a flat
    reference the relinearisation bracket is exactly zero in floating point, so
    `evaluate` is the depth row of the pullback, bit for bit — and the normal is
    the depth row of the Jacobian.
    """
    field = _curved_field()
    x = _grid = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 8, 9, dtype=curlew.dtype),
            torch.linspace(0, 6, 7, dtype=curlew.dtype),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)

    with torch.no_grad():
        x0, j = field.integrator.inverse_map_with_jacobian(x)
        assert (field.evaluate(x) - x0[:, field.depth_axis]).abs().max() == 0.0
        assert (field.stratigraphy_gradient(x) - j[:, field.depth_axis, :]).abs().max() == 0.0


def test_volume_is_preserved_and_the_round_trip_closes():
    """
    Volume preservation of the *map*, and a closing round trip.

    Under the default RK4 the reported Jacobian is the chain-rule derivative
    of the discrete map, so `det J` is `1 + O(dt^4)` rather than exactly 1.
    The RK2 map itself carries an O(dt^2) volume error of ~1e-2.
    """
    field = _curved_field(n_steps=12)
    x = torch.rand(64, 2, dtype=curlew.dtype) * 6.0
    with torch.no_grad():
        _, j = field.integrator.inverse_map_with_jacobian(x)
        assert (torch.linalg.det(j) - 1.0).abs().max() < 1e-6
        rt = field.integrator.forward_map(field.integrator.inverse_map(x)) - x
        assert rt.abs().max() < 1e-5


def test_reference_gradient_matches_autograd_of_the_value():
    """
    `stratigraphy_gradient` must agree with autograd of the value even against a
    curved reference — i.e. the frozen reference query is exact for the value
    path, and the covector transport is consistent with it.

    Both the value path and `stratigraphy_gradient` differentiate the same
    discrete map (chain-rule Jacobian of the integrator), so they must agree.
    """
    from curlew.kinematics.euler import euler_velocity

    v1 = euler_velocity(2, n_features=16, seed=21)
    v1.stream.A_mu.data.normal_(0, 0.3)
    v2 = euler_velocity(2, n_features=16, seed=22)
    v2.stream.A_mu.data.normal_(0, 0.3)
    a = restore("a", velocity=v1, C=CSet(), n_steps=8, learning_rate=1e-2)
    b = restore("b", velocity=v2, C=CSet(), reference=a, n_steps=8, learning_rate=1e-2)
    GeoModel([a, b])

    x = torch.tensor([[0.3, 0.4], [-0.5, 0.8]], dtype=curlew.dtype, requires_grad=True)
    val = b.field.restored_depth(x)
    g_auto = torch.autograd.grad(val.sum(), x)[0]
    with torch.no_grad():
        g_pred = b.field.stratigraphy_gradient(x.detach())
    assert torch.allclose(g_auto, g_pred, atol=1e-4)


def test_total_map_chains_and_matches_autograd():
    """
    `total_inverse_map_with_jacobian` must equal the true chained pullback and
    its Jacobian — this is what replaces the old composed integrator for
    finite-strain work.
    """
    from curlew.kinematics.euler import euler_velocity

    vs = []
    for s in (31, 32, 33):
        v = euler_velocity(2, n_features=16, seed=s)
        v.stream.A_mu.data.normal_(0, 0.25)
        vs.append(v)
    base = restore("base", velocity=vs[0], C=CSet(), n_steps=6, learning_rate=1e-2)
    mid = restore("mid", velocity=vs[1], C=CSet(), reference="parent", n_steps=6, learning_rate=1e-2)
    top = restore("top", velocity=vs[2], C=CSet(), reference="parent", n_steps=6, learning_rate=1e-2)
    GeoModel([base, mid, top])

    x = torch.tensor([[0.2, 0.5], [-0.6, 1.0]], dtype=curlew.dtype)
    with torch.no_grad():
        # explicit chain: own map first (youngest), then each reference
        a = top.field.integrator.inverse_map(x)
        b = mid.field.integrator.inverse_map(a)
        c = base.field.integrator.inverse_map(b)
        x0, j = top.field.total_inverse_map_with_jacobian(x)
        assert (x0 - c).abs().max() == 0.0
        assert (top.field.total_inverse_map(x) - c).abs().max() == 0.0

    # Jacobian of the chain vs autograd of the chained map
    xg = x.clone().requires_grad_(True)
    x0g, jg = top.field.total_inverse_map_with_jacobian(xg)
    for k in range(x.shape[0]):
        for i in range(2):
            g = torch.autograd.grad(x0g[k, i], xg, retain_graph=True)[0][k]
            assert torch.allclose(g, jg[k, i], atol=1e-4)


def test_total_map_is_the_local_map_without_a_reference():
    field = _curved_field()
    x = torch.rand(16, 2, dtype=curlew.dtype) * 5.0
    with torch.no_grad():
        x0a, ja = field.total_inverse_map_with_jacobian(x)
        x0b, jb = field.integrator.inverse_map_with_jacobian(x)
        assert (x0a - x0b).abs().max() == 0.0 and (ja - jb).abs().max() == 0.0


def test_reference_does_not_leak_into_the_module_tree():
    """
    A frozen reference is not this field's parameter. Registering it as a
    submodule would put its amplitudes into `parameters()`, `state_dict()` and
    the optimiser — breaking checkpointing, param counts and any future
    ensemble/vmap stacking.
    """
    v1 = _ConstVelocity([0.0, -0.1])
    v2 = _ConstVelocity([0.0, -0.2])
    a = restore("a", velocity=v1, C=CSet(), n_steps=2, learning_rate=1e-2)
    b = restore("b", velocity=v2, C=CSet(), reference=a, n_steps=2, learning_rate=1e-2)
    GeoModel([a, b])

    ap = {id(p) for p in a.field.parameters()}
    bp = {id(p) for p in b.field.parameters()}
    assert not (ap & bp), "reference parameters leaked into the referring field"
    assert not any(k.startswith("_reference") for k in b.field.state_dict())
    if b.field.optim is not None:
        opt = {id(p) for g in b.field.optim.param_groups for p in g["params"]}
        assert not (ap & opt), "reference parameters leaked into the optimiser"


def test_normal_path_differentiates_through_a_curved_reference():
    """
    Regression guard. The bedding-normal term consumes `grad phi_ref` itself, so
    freezing it drops `grad^2 phi_ref . dx0/dtheta`. Against a curved reference
    that is NOT a small correction — measured at 59 degrees off the exact
    descent direction with 31% of its magnitude on a fitted fold. The tempting
    "it vanishes at a perfect fit" argument fails because the kept term vanishes
    there too, so the *relative* error never shrinks.

    So `_reference_gradient_at` must differentiate through the reference by
    default, and must differ from the frozen variant on a curved reference.
    """
    from curlew.kinematics.euler import euler_velocity

    v1 = euler_velocity(2, n_features=16, seed=41)
    v1.stream.A_mu.data.normal_(0, 0.4)
    v2 = euler_velocity(2, n_features=16, seed=42)
    v2.stream.A_mu.data.normal_(0, 0.4)
    a = restore("a", velocity=v1, C=CSet(), n_steps=6, learning_rate=1e-2)
    b = restore("b", velocity=v2, C=CSet(), reference=a, n_steps=6, learning_rate=1e-2)
    GeoModel([a, b])

    x = torch.tensor([[0.3, 0.4], [-0.5, 0.8], [0.1, -0.3]], dtype=curlew.dtype)
    p = b.field.velocity.stream.A_mu

    def grad_of_normal_sum(frozen):
        if p.grad is not None:
            p.grad = None
        b.field.stratigraphy_gradient(x, frozen_reference=frozen).sum().backward()
        return p.grad.detach().clone()

    g_exact = grad_of_normal_sum(False)
    g_frozen = grad_of_normal_sum(True)
    assert not torch.allclose(g_exact, g_frozen, atol=1e-9), (
        "default normal path is still using the frozen reference gradient"
    )

    # and against a FLAT reference the two must coincide exactly (grad^2 phi0 = 0)
    solo = restore(
        "solo", velocity=euler_velocity(2, n_features=16, seed=43),
        C=CSet(), n_steps=6, learning_rate=1e-2,
    ).field
    with torch.no_grad():
        assert (
            solo.stratigraphy_gradient(x, frozen_reference=True)
            - solo.stratigraphy_gradient(x, frozen_reference=False)
        ).abs().max() == 0.0


# ── angular normal residual ─────────────────────────────────────────────────

def _n_at(deg, requires_grad=False):
    a = math.radians(deg)
    return torch.tensor(
        [[math.sin(a), math.cos(a)]], dtype=curlew.dtype, requires_grad=requires_grad
    )


_UP = torch.tensor([[0.0, 1.0]], dtype=torch.float64)


def _loss_and_pull(deg, fn, **kw):
    v = _n_at(deg, requires_grad=True)
    L = fn(torch.nn.functional.normalize(v, dim=-1), _UP.to(v.dtype), **kw)
    L.backward()
    return float(L), float(v.grad.norm())


def test_normal_loss_is_the_squared_angle():
    for deg in (0.0, 10.0, 37.5, 90.0, 170.0):
        L, _ = _loss_and_pull(deg, RestorationField._normal_loss, signed=True)
        assert L == pytest.approx(math.radians(deg) ** 2, rel=1e-9), deg


def test_normal_loss_pull_grows_with_the_error():
    """
    The point of the change. `1 - cos` has derivative `sin`, so a 170 deg error
    pulls no harder than a 10 deg one and a flipped normal has no restoring
    force. `theta^2` has derivative `2 theta`: monotonic in the error.
    """
    pulls = [
        _loss_and_pull(d, RestorationField._normal_loss, signed=True)[1]
        for d in (10.0, 45.0, 90.0, 135.0, 170.0, 179.0)
    ]
    assert all(b > a for a, b in zip(pulls, pulls[1:])), pulls
    assert pulls[-1] / pulls[0] > 15                       # ~17x at 170 vs 10


def test_unsigned_normal_loss_folds_and_does_not_drive_to_180():
    """
    `gop` and unoriented `gv` must treat +n and -n alike. The folded angle has
    strong, correctly-signed pull on both sides of 90 deg.
    """
    f = RestorationField._normal_loss
    for deg in (0.0, 180.0):
        assert _loss_and_pull(deg, f, signed=False)[0] == pytest.approx(0.0, abs=1e-12)
    for deg in (30.0, 60.0, 89.0):
        a = _loss_and_pull(deg, f, signed=False)
        b = _loss_and_pull(180.0 - deg, f, signed=False)
        assert a[0] == pytest.approx(b[0], rel=1e-9)       # symmetric about 90
        assert a[1] == pytest.approx(b[1], rel=1e-9)

    assert _loss_and_pull(91.0, f, signed=False)[1] > 3.0


def test_normal_loss_is_stable_at_the_endpoints():
    """atan2 rather than arccos: no singular derivative at a perfect fit or flip."""
    for deg in (0.0, 1e-6, 180.0 - 1e-6, 180.0):
        L, g = _loss_and_pull(deg, RestorationField._normal_loss, signed=True)
        assert math.isfinite(L) and math.isfinite(g), deg


def test_restoration_field_ori_loss_uses_gop():
    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel)
    field.H = HSet.restore(grad_loss=0, value_loss=0, eq_loss=0, ori_loss=1.0)
    gop = np.array([[0.0, 0.0], [1.0, 0.0]])
    gov = np.array([[1.0, 0.0], [1.0, 0.0]])
    field.bind(CSet(gop=gop, gov=gov))
    pebble = field.loss(transform=False)
    assert pebble.losses["r0"]["ori_loss"].item() > 0.5


def test_restore_queries_a_strati_parent():
    from curlew.fields.analytical import LinearField
    from curlew.geology import strati

    s = strati("s", C=LinearField("s", input_dim=2, gradient=[0.0, 1.0]))
    r = restore(
        "r", velocity=_ConstVelocity([0.0, -0.25]), C=CSet(),
        reference=s, n_steps=4, learning_rate=1e-2,
    )
    GeoModel([s, r])
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=curlew.dtype)
    x0 = r.field.integrator.inverse_map(x)
    expected = s.field.forward(x0, transform=False).squeeze()
    assert torch.allclose(r.field.evaluate(x), expected)


def test_restore_3d_fit_reduces_normal_loss():
    curlew.default_dim = 3
    from curlew.kinematics.euler import euler_velocity

    rng = np.random.default_rng(0)
    gp = rng.uniform(-1.0, 1.0, (16, 3))
    gv = np.tile([0.0, 0.0, 1.0], (16, 1))
    v = euler_velocity(3, n_features=16, seed=3, length_scale_range=(1.0, 4.0))
    with torch.no_grad():
        v.alpha.A_mu.normal_(0.0, 0.3)
        v.stream.A_mu.normal_(0.0, 0.3)
    event = restore(
        "r3",
        velocity=v,
        C=CSet(gp=gp, gv=gv),
        H=HSet.restore(),
        n_steps=4,
        input_dim=3,
        depth_axis=2,
        init_std=None,
        learning_rate=5e-2,
    )
    model = GeoModel([event])
    l0 = float(event.field.loss(transform=False).total().detach())
    model.fit(25, vb=False, early_stop=None)
    l1 = float(event.field.loss(transform=False).total().detach())
    assert l1 < l0



def test_strain_ellipse_loss_rejects_3d():
    J = torch.eye(3, dtype=curlew.dtype).expand(4, 3, 3)
    with pytest.raises(NotImplementedError):
        RestorationField._strain_ellipse_loss(J, None, torch.ones(4, dtype=curlew.dtype))
