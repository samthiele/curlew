"""Post-hoc pin of the restored frame (:meth:`~curlew.kinematics.flow.FlowOffset.pin`).

A constant ``frame_offset``. Its Jacobian is the identity, so strain, bedding
normals, and the losses are unchanged.
"""
import curlew
import numpy as np
import pytest
import torch

from curlew.core import CSet, HSet
from curlew.kinematics.euler import euler_velocity
from curlew.kinematics.restoration import (
    RestorationField,
    pin_anchor,
    pin_mean,
)


@pytest.fixture(autouse=True)
def _defaults():
    curlew.default_dim = 2
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def _field(n_steps=6, seed=5, amp=0.4):
    torch.manual_seed(0)
    vel = euler_velocity(2, n_features=24, seed=seed, length_scale_range=(2.0, 8.0))
    with torch.no_grad():
        vel.stream.A_mu.normal_(0.0, amp)
    h = HSet(
        value_loss=0, grad_loss=1.0, ori_loss=0, eq_loss=1.0,
        mono_loss=0, thick_loss=0, energy_loss=0,
    )
    return RestorationField(
        "r", H=h, input_dim=2, velocity=vel, n_steps=n_steps, depth_axis=1,
        sigma_floor=0.0,
    )


def _grid(nx=9, ny=7):
    return torch.stack(
        torch.meshgrid(
            torch.linspace(0, 8, nx, dtype=curlew.dtype),
            torch.linspace(0, 6, ny, dtype=curlew.dtype),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)


def _fold_constraints(n=21):
    xs = torch.linspace(0, 8, n, dtype=curlew.dtype)
    traces = [torch.stack([xs, y0 + 1.2 * torch.sin(xs / 4.0 * np.pi)], 1)
              for y0 in (2.0, 4.0)]
    gp, gv = [], []
    for t in traces:
        slope = 1.2 * (np.pi / 4.0) * torch.cos(t[:, 0] / 4.0 * np.pi)
        nrm = torch.stack([-slope, torch.ones_like(slope)], dim=1)
        gp.append(t[::4])
        gv.append((nrm / nrm.norm(dim=1, keepdim=True))[::4])
    return traces, torch.cat(gp), torch.cat(gv)


# ── the pin does what it says ────────────────────────────────────────────────

def test_pin_anchor_fixes_the_point_exactly():
    field = _field()
    a = torch.tensor([[4.0, 3.0]], dtype=curlew.dtype)
    pinned = pin_anchor(field.integrator, a)
    assert (pinned.inverse_map(a) - a).abs().max() == 0.0


def test_pin_mean_removes_the_mean_displacement():
    field = _field()
    pts = _grid()
    pinned = pin_mean(field.integrator, pts)
    disp = pinned.inverse_map(pts) - pts
    assert disp.mean(0).abs().max() < 1e-14


def test_shift_is_a_pure_translation_of_the_map():
    field = _field()
    pts = _grid()
    raw = field.integrator.inverse_map(pts).clone()
    pinned = pin_anchor(field.integrator, [[4.0, 3.0]])
    delta = pinned.inverse_map(pts) - raw
    # constant to machine precision — this is what the in-loop gauge could not do
    assert float(delta.std(0).max().detach()) < 1e-15
    assert (delta[0] + pinned.frame_offset).abs().max() < 1e-15


# ── and, crucially, what it does not touch ───────────────────────────────────

def test_jacobian_strain_and_volume_are_bit_identical():
    field = _field()
    pts = _grid()
    _, j_ref = field.integrator.inverse_map_with_jacobian(pts)
    j_ref = j_ref.clone()
    pinned = pin_anchor(field.integrator, [[1.0, 5.0]])
    _, j_pin = pinned.inverse_map_with_jacobian(pts)
    assert (j_pin - j_ref).abs().max() == 0.0
    assert (torch.linalg.det(j_pin) - 1.0).abs().max() < 1e-6


def test_losses_are_unchanged_by_pinning():
    """
    Translation is exactly a symmetry of the eq and normal residuals, so every
    loss term must be numerically identical after the shift.
    """
    traces, gp, gv = _fold_constraints()
    field = _field()
    field.bind(CSet(eq=[t.numpy() for t in traces], gp=gp.numpy(), gv=gv.numpy()))
    before = {k: float(v.detach()) for k, v in field.loss(transform=False).losses["r"].items()}

    pin_anchor(field.integrator, [[6.0, 1.0]])
    after = {k: float(v.detach()) for k, v in field.loss(transform=False).losses["r"].items()}

    assert set(before) == set(after) and before
    for k in before:
        assert after[k] == pytest.approx(before[k], rel=1e-12), k


def test_bedding_normals_are_unchanged():
    traces, gp, _ = _fold_constraints()
    field = _field()
    n_ref = field.predict_normals(gp).detach().clone()
    pin_mean(field.integrator, gp)
    assert (field.predict_normals(gp) - n_ref).abs().max() == 0.0


# ── round trips and plumbing ─────────────────────────────────────────────────

@pytest.mark.parametrize("n_steps", [6, 24])
def test_pinning_does_not_degrade_the_round_trip(n_steps):
    """
    ``Φ̃(Φ̃⁻¹(x)) = x`` requires ``Φ̃(y) = Φ(y + c)`` — with the sign the other
    way the shift is applied twice and the round trip drifts by ``2c``.

    The residual is pure RK2 discretisation error (O(h²): 2.8e-06 at 6 steps,
    4.4e-08 at 24), so the meaningful assertion is that pinning leaves it
    *exactly* unchanged rather than that it clears some absolute threshold.
    """
    field = _field(n_steps=n_steps)
    pts = _grid(5, 4)
    raw = (
        field.integrator.forward_map(field.integrator.inverse_map(pts)) - pts
    ).abs().max().detach().clone()
    pinned = pin_anchor(field.integrator, [[4.0, 3.0]])

    pin = (pinned.forward_map(pinned.inverse_map(pts)) - pts).abs().max()
    assert float(pin.detach()) == pytest.approx(float(raw.detach()), rel=1e-12)
    assert float(pin.detach()) < 1e-5

    y = pinned.inverse_map(pts)
    assert (pinned.inverse_map(pinned.forward_map(y)) - y).abs().max() < 1e-5


def test_disp_and_restored_depth_shift_consistently():
    field = _field()
    pts = _grid()
    d_ref = field.integrator.disp(pts, None).clone()
    pinned = pin_anchor(field.integrator, [[2.0, 4.0]])
    assert (pinned.disp(pts, None) - (d_ref - pinned.frame_offset)).abs().max() < 1e-15
    assert (field.restored_depth(pts) - (pinned.inverse_map(pts)[:, 1])).abs().max() == 0.0


def test_axes_restricts_the_shift():
    """With vp/vv value constraints the depth component is fixed by data."""
    field = _field()
    pinned = pin_anchor(field.integrator, [[4.0, 3.0]], axes=[0])
    assert float(pinned.frame_offset[1]) == 0.0
    assert float(pinned.frame_offset[0]) != 0.0


def test_refresh_tracks_parameter_changes():
    field = _field()
    a = torch.tensor([[4.0, 3.0]], dtype=curlew.dtype)
    pinned = pin_anchor(field.integrator, a)
    with torch.no_grad():
        field.velocity.stream.A_mu.mul_(1.5)
    assert (pinned.inverse_map(a) - a).abs().max() > 1e-6   # stale
    pin_anchor(pinned, a)                                   # re-pin = refresh
    assert (pinned.inverse_map(a) - a).abs().max() == 0.0   # current


def test_metadata_passes_through():
    field = _field(n_steps=9)
    pinned = pin_anchor(field.integrator, [[0.0, 0.0]])
    assert pinned.n_steps == 9
    assert pinned.velocity is field.velocity
    assert pinned.learnable() is field.integrator.learnable()
    assert pinned is field.integrator          # pinning mutates, never wraps


def test_pin_anchor_rejects_multiple_points():
    field = _field()
    with pytest.raises(ValueError, match="single point"):
        pin_anchor(field.integrator, _grid())


# ── restore(pin=...) integration ─────────────────────────────────────────────

def _restore_event(**kw):
    from curlew.geology import restore

    traces, gp, gv = _fold_constraints()
    c = CSet(eq=[t.numpy() for t in traces], gp=gp.numpy(), gv=gv.numpy())
    h = HSet(
        value_loss=0, grad_loss=1.0, ori_loss=0, eq_loss=1.0,
        mono_loss=0, thick_loss=0, energy_loss=1e-3,
    )
    torch.manual_seed(0)
    vel = euler_velocity(2, n_features=16, seed=5, length_scale_range=(2.0, 8.0))
    return restore(
        "fold", C=c, H=h, velocity=vel, input_dim=2, n_steps=4, depth_axis=1,
        sigma_floor=0.0, learning_rate=2e-2, **kw
    )


ANCHOR = np.array([[4.0, 3.0]])


def test_restore_pin_does_not_change_the_fit():
    """
    Fitting runs unpinned by construction, so every pin setting must reach the
    *same* loss. If this ever drifts, the pin has leaked into training.
    """
    losses = {}
    for tag, kw in (
        ("none", {}),
        ("point", dict(pin=ANCHOR)),
        ("mean", dict(pin="mean")),
        ("axes", dict(pin=ANCHOR, pin_axes=[0])),
    ):
        ev = _restore_event(**kw)
        loss, _ = ev.fit(40, vb=False)
        losses[tag] = float(loss)
    ref = losses["none"]
    for tag, value in losses.items():
        assert value == pytest.approx(ref, rel=1e-12), tag


def test_restore_pin_point_is_applied_after_fitting():
    ev = _restore_event(pin=ANCHOR)
    assert not ev.field.pinned                      # not yet — nothing fitted
    ev.fit(40, vb=False)
    assert ev.field.pinned
    a = torch.tensor(ANCHOR, dtype=curlew.dtype)
    assert (ev.field.inverse_map(a) - a).abs().max() == 0.0


def test_restore_pin_applies_when_fit_is_interrupted():
    ev = _restore_event(pin=ANCHOR)
    real_loss = ev.field.loss
    calls = {"n": 0}

    def boom(transform=True):
        calls["n"] += 1
        if calls["n"] > 2:
            raise KeyboardInterrupt
        return real_loss(transform=transform)

    ev.field.loss = boom
    with pytest.raises(KeyboardInterrupt):
        ev.fit(40, vb=False)
    a = torch.tensor(ANCHOR, dtype=curlew.dtype)
    assert (ev.field.inverse_map(a) - a).abs().max() == 0.0


def test_restore_pin_applies_on_the_next_map_call():
    ev = _restore_event(pin=ANCHOR)
    assert not ev.field.pinned
    a = torch.tensor(ANCHOR, dtype=curlew.dtype)
    assert (ev.field.inverse_map(a) - a).abs().max() == 0.0
    assert ev.field._pin_applied


def test_restore_pin_mean_uses_the_constraint_points():
    ev = _restore_event(pin="mean")
    ev.fit(40, vb=False)
    pts = ev.field._pin_default_points()
    disp = ev.field.inverse_map(pts) - pts
    assert disp.mean(0).abs().max() < 1e-13


def test_restore_pin_axes_leaves_the_other_component_alone():
    ev = _restore_event(pin=ANCHOR, pin_axes=[0])
    ev.fit(40, vb=False)
    assert float(ev.field.integrator.frame_offset[1]) == 0.0
    assert float(ev.field.integrator.frame_offset[0]) != 0.0


def test_restore_pin_reuses_a_named_anchor():
    """``pin="centre"`` resolves against anchors added with ``addAnchor``."""
    ev = _restore_event(pin="centre")
    ev.addAnchor("centre", position=[4.0, 3.0])
    ev.fit(40, vb=False)
    a = torch.tensor(ANCHOR, dtype=curlew.dtype)
    assert (ev.field.inverse_map(a) - a).abs().max() == 0.0
    # the existing anchor API agrees: the restored position is now the present one
    pos, _ = ev.getAnchor("centre", to_numpy=False)
    assert (pos.reshape(-1) - a.reshape(-1)).abs().max() < 1e-12


def test_restore_pin_unknown_anchor_name_raises():
    ev = _restore_event(pin="not_an_anchor")
    with pytest.raises(KeyError, match="not an anchor"):
        ev.fit(40, vb=False)


def test_deformation_follows_the_pin():
    """Younger events must undeform into the *same* restored frame."""
    ev = _restore_event(pin=ANCHOR)
    ev.fit(40, vb=False)
    assert ev.field.pinned
    assert ev.deformation is ev.field.integrator


def test_refitting_refreshes_rather_than_stacking_pins():
    ev = _restore_event(pin=ANCHOR)
    ev.fit(30, vb=False)
    first = ev.field.integrator.frame_offset.clone()
    ev.fit(30, vb=False)
    a = torch.tensor(ANCHOR, dtype=curlew.dtype)
    assert (ev.field.inverse_map(a) - a).abs().max() == 0.0
    # refreshed in place — no stacking, because there is nothing to stack
    assert not torch.equal(first, ev.field.integrator.frame_offset)


def test_euler_velocity_no_longer_takes_a_gauge():
    """The Killing machinery is gone; pinning is post-hoc only."""
    with pytest.raises(TypeError):
        euler_velocity(2, n_features=8, seed=1, anchor=[0.0, 0.0])
    vel = euler_velocity(2, n_features=8, seed=1)
    assert not hasattr(vel, "anchor") and not hasattr(vel, "killing")
