"""Tests for :class:`~curlew.fields.restoration.RestorationField` and ``restore()``."""
import curlew
import numpy as np
import pytest
import torch
from torch import nn

from curlew.core import CSet, HSet
from curlew.fields.restoration import RestorationField
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
        n_steps=4,
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


def test_restoration_field_eq_loss_flat_horizon():
    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel, sigma_floor=0.01)
    trace = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=curlew.dtype)
    c = CSet(eq=[trace.numpy()])
    field.bind(c)
    pebble = field.loss(transform=False)
    assert pebble.total().item() < 1e-8


def test_restoration_field_eq_loss_flattens_curved_horizon():
    """eq_loss must reduce restored-depth spread via the flow, not a side-channel parameter."""
    from curlew.fields.clebsch import clebsch_velocity

    xs = torch.linspace(0, 10, 40, dtype=curlew.dtype)
    ys = torch.sin(xs) * 2 + 5
    trace = torch.stack([xs, ys], dim=1)

    vel = clebsch_velocity(2, n_features=32, seed=7, length_scale_range=(2.0, 5.0))
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
    from curlew.fields.clebsch import clebsch_velocity

    vel = clebsch_velocity(2, n_features=8, seed=13)
    vel.beta.A_mu.data.normal_(0, 0.2)
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
    assert vel.beta.A_mu.grad is not None
    assert (vel.beta.A_mu.grad.abs() > 0).any()


def test_restoration_field_orthogonal_thickness_identity():
    """Identity flow has |J^T e_d| = 1 everywhere."""
    from curlew.fields.restoration_constraints import orthogonal_thickness_loss

    vel = _ConstVelocity([0.0, 0.0])
    field = _restoration_field(vel)
    x = torch.randn(12, 2, dtype=curlew.dtype)
    x0, J = field.integrator.inverse_map_with_jacobian(x)
    assert orthogonal_thickness_loss(x0, J, field.depth_axis).item() < 1e-12


def test_restoration_field_orthogonal_thickness_constraint():
    from curlew.fields.clebsch import clebsch_velocity
    from curlew.fields.restoration_constraints import OrthogonalThickness

    vel = clebsch_velocity(2, n_features=16, seed=3)
    vel.beta.A_mu.data.normal_(0, 0.05)
    pts = torch.randn(8, 2, dtype=curlew.dtype)
    field = _restoration_field(vel, init_std=0.0)
    field.constraints = [OrthogonalThickness(weight=1.0, points=pts)]
    field.bind(CSet())
    pebble = field.loss(transform=False)
    assert pebble.losses["r0"]["thick_loss"].item() > 0.0


def test_restoration_field_layer_thickness_constraint():
    """Layer gaps should match relative thickness ratios."""
    from curlew.fields.restoration_constraints import LayerThickness, layer_thickness_loss

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
    from curlew.fields.restoration_constraints import OrthogonalThickness

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
    from curlew.fields.clebsch import clebsch_velocity
    from curlew.fields.restoration_constraints import OrthogonalThickness

    vel = clebsch_velocity(2, n_features=16, seed=11)
    vel.beta.A_mu.data.normal_(0, 0.05)
    h3 = np.array([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    h4 = np.array([[-2.0, 2.0], [0.0, 2.0], [2.0, 2.0]])
    field = _restoration_field(vel, init_std=0.0)
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


def test_restore_accepts_constraints():
    from curlew.fields.restoration_constraints import LayerThickness

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
    from curlew.fields.clebsch import clebsch_velocity

    xs = torch.linspace(0, 6, 12, dtype=curlew.dtype)
    # Crossed horizons in modern coords: require upper restored > lower.
    upper = torch.stack([xs, torch.sin(xs) + 2.0], dim=1)
    lower = torch.stack([xs, torch.sin(xs) + 5.0], dim=1)
    c = CSet(iq=(32, [(upper.numpy(), lower.numpy(), ">")]))

    vel = clebsch_velocity(2, n_features=32, seed=19, length_scale_range=(2.0, 5.0))
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
