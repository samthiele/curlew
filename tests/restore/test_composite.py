"""Banded flows: ``FlowOffset`` composes a list of velocities, coarse to fine.

One factor matches a single field. The prior variance does not grow with the
number of bands.
"""
import math

import curlew
import numpy as np
import pytest
import torch
import torch.nn as nn

from curlew.core import CSet, HSet
from curlew.kinematics.euler import _log_split_bands, euler_velocity
from curlew.kinematics.lift import FaultLift
from curlew.geology import restore
from curlew.geology.interactions import FlowOffset

LSR = (200.0, 5000.0)


@pytest.fixture(autouse=True)
def _defaults():
    curlew.device = "cpu"
    curlew.dtype = torch.float64
    curlew.default_dim = 2


def _velocity(n_features, lsr=LSR, seed=1, excite=True):
    """A 2-D Euler velocity with non-zero amplitudes (so the flow moves)."""
    v = euler_velocity(
        2, n_features=n_features, length_scale_range=lsr, stream="alpha", seed=seed,
    )
    if excite:
        mods = v if isinstance(v, list) else [v]
        for i, m in enumerate(mods):
            torch.manual_seed(100 + i)
            with torch.no_grad():
                # Match the old one-amplitude variance: E[A² + B²] = 1.
                amp = 1.0 / math.sqrt(2.0)
                m.stream.A_mu.normal_(0.0, amp)
                m.stream.B_mu.normal_(0.0, amp)
    return v


def _points(n=64, spread=800.0):
    torch.manual_seed(0)
    return torch.rand(n, 2, dtype=curlew.dtype) * spread - spread / 2


# ── band construction ───────────────────────────────────────────────────────

def test_bands_are_log_equal_disjoint_and_coarsest_first():
    bands = _log_split_bands(LSR, 3)
    assert len(bands) == 3
    # coarsest first: the list is in deformation order
    assert bands[0][1] == pytest.approx(LSR[1])
    assert bands[-1][0] == pytest.approx(LSR[0])
    # disjoint and tiling: each band starts where the next one ends
    for lo, hi in zip(bands[:-1], bands[1:]):
        assert lo[0] == pytest.approx(hi[1])
    # equal ratio per band (log-equal spacing)
    ratios = [hi / lo for lo, hi in bands]
    assert ratios == pytest.approx([ratios[0]] * 3)


def test_explicit_bands_may_overlap():
    bands = _log_split_bands([(1000.0, 5000.0), (200.0, 1500.0)], 2)
    assert bands == [(1000.0, 5000.0), (200.0, 1500.0)]


def test_band_count_must_match_n_features():
    with pytest.raises(ValueError, match="bands but n_features"):
        _log_split_bands([(1.0, 2.0)], 3)


def test_euler_velocity_returns_a_list_for_a_list_of_features():
    vs = _velocity([64, 64, 64], excite=False)
    assert isinstance(vs, list) and len(vs) == 3
    lams = [2 * math.pi / v.stream._omega_spatial().norm(dim=1) for v in vs]
    # coarsest factor first, and the bands do not overlap
    assert float(lams[0].min()) > float(lams[1].max())
    assert float(lams[1].min()) > float(lams[2].max())


def test_out_scale_is_invariant_to_band_count():
    """
    ``out_scale = scale / sqrt(F)`` is per field, so without correction a K-band
    bank would carry K times the prior variance of the monolithic model. The
    factory rescales by ``sqrt(F_i / F_total)`` so the prior — and hence the
    gradient scale — depends only on the total mode count.
    """
    mono = _velocity(512, excite=False)
    for counts in ([256, 256], [171, 170, 171], [128] * 4):
        for v in _velocity(counts, excite=False):
            assert float(v.stream.out_scale) == pytest.approx(
                float(mono.stream.out_scale), rel=1e-12
            )


# ── composition semantics ───────────────────────────────────────────────────

def test_single_factor_matches_a_bare_module_exactly():
    """A one-element list and a bare module must be the same flow, bit for bit."""
    v = _velocity(64)
    x = _points()
    bare = FlowOffset(v, n_steps=6)
    listed = FlowOffset([v], n_steps=[6])
    assert (bare.inverse_map(x) - listed.inverse_map(x)).abs().max() == 0.0
    assert bare.n_steps == 6 and listed.n_steps == 6
    assert bare.n_factors == listed.n_factors == 1


def test_inverse_walks_the_list_backwards():
    """
    The list is in deformation order, so restoration undoes the *last* factor
    first. Checked against an explicit hand-composed traversal.
    """
    vs = _velocity([48, 48, 48])
    steps = [4, 6, 8]
    x = _points()
    composite = FlowOffset(vs, n_steps=steps).inverse_map(x)

    manual = x.clone()
    for v, n in reversed(list(zip(vs, steps))):
        manual = FlowOffset(v, n_steps=n).inverse_map(manual)
    assert float((composite - manual).abs().max()) == pytest.approx(0.0, abs=1e-14)


def test_forward_walks_the_list_in_order():
    vs = _velocity([48, 48])
    steps = [5, 7]
    x = _points()
    composite = FlowOffset(vs, n_steps=steps).forward_map(x)

    manual = x.clone()
    for v, n in zip(vs, steps):
        manual = FlowOffset(v, n_steps=n).forward_map(manual)
    assert float((composite - manual).abs().max()) == pytest.approx(0.0, abs=1e-14)


def test_forward_and_inverse_are_inverses():
    off = FlowOffset(_velocity([48, 48, 48]), n_steps=[16, 16, 16])
    x = _points(spread=400.0)
    back = off.forward_map(off.inverse_map(x))
    # integration error only; the maps are exact inverses analytically
    assert float((back - x).norm(dim=1).median()) < 1e-2


def test_composition_preserves_incompressibility():
    """
    Composition must not *degrade* volume preservation, whatever the factor count.

    Under the default RK4 the Jacobian is the chain-rule derivative of the map,
    so each factor contributes `1 + O(dt^4)` rather than an exact 1; the product
    stays at that level instead of compounding. Under `integrator="shear"` each
    factor is exactly unit-determinant and the product is too.
    """
    x = _points()
    for counts in ([64], [48, 48], [32, 32, 32, 32]):
        off = FlowOffset(_velocity(counts), n_steps=8)
        _, j = off.inverse_map_with_jacobian(x)
        # these fields run at dt*||grad v|| ~ 0.1-0.2, where the fourth-order
        # term lands around 1e-6; more factors does not make it worse
        assert float((torch.linalg.det(j) - 1.0).abs().max()) < 1e-5, off.stability(x)
        _, js = FlowOffset(
            _velocity(counts), n_steps=2, integrator="shear"
        ).inverse_map_with_jacobian(x)
        assert float((torch.linalg.det(js) - 1.0).abs().max()) < 1e-10


def _jac_gap(off, n_pts=8):
    """max |reported J - autograd J of the same map|."""
    x = _points(n=n_pts, spread=400.0).requires_grad_(True)
    x0, j = off.inverse_map_with_jacobian(x)
    rows = [torch.autograd.grad(x0[:, i].sum(), x, retain_graph=True)[0] for i in range(2)]
    return float((j - torch.stack(rows, dim=1)).abs().max())


def test_default_jacobian_is_the_maps_own_derivative():
    """
    RK4 reports the chain-rule Jacobian, so it equals the map's derivative
    exactly — at every step count, not in the limit. Composing factors must not
    break that.
    """
    for n in (2, 8, 32):
        assert _jac_gap(FlowOffset(_velocity([48, 48]), n_steps=[n, n])) < 1e-12, n


def test_rk2_jacobian_is_the_maps_own_derivative():
    """RK2 reports the chain-rule Jacobian of the midpoint map, same as RK4."""
    for n in (2, 8, 32):
        assert _jac_gap(
            FlowOffset(_velocity([48, 48]), n_steps=[n, n], integrator="rk2")
        ) < 1e-12, n


# ── per-factor controls ─────────────────────────────────────────────────────

def test_n_steps_broadcasts_and_validates():
    vs = _velocity([32, 32, 32], excite=False)
    assert FlowOffset(vs, n_steps=5).n_steps == [5, 5, 5]
    assert FlowOffset(vs, n_steps=[1, 2, 3]).n_steps == [1, 2, 3]
    with pytest.raises(ValueError, match="entries but there are"):
        FlowOffset(vs, n_steps=[1, 2])
    with pytest.raises(ValueError, match=">= 0"):
        FlowOffset(vs, n_steps=[1, -1, 3])
    with pytest.raises(ValueError, match="integrator='shear'"):    # 0 is the one-way shear sweep
        FlowOffset(vs, n_steps=[1, 0, 3])


def test_n_steps_setter_round_trips():
    off = FlowOffset(_velocity([32, 32], excite=False), n_steps=4)
    off.n_steps = [3, 9]
    assert off.steps == [3, 9]
    off.n_steps = 7
    assert off.steps == [7, 7]


def test_velocity_property_rejects_ambiguity():
    single = FlowOffset(_velocity(32, excite=False), n_steps=2)
    assert single.velocity is not None
    composite = FlowOffset(_velocity([32, 32], excite=False), n_steps=2)
    with pytest.raises(AttributeError):
        _ = composite.velocity
    assert len(composite.velocities) == 2


def test_all_factors_are_trainable_and_reachable():
    """
    Unlike the reference-event chain — which hides the referenced field to keep
    it frozen — band factors must train jointly.
    """
    off = FlowOffset(_velocity([32, 64], excite=False), n_steps=2)
    assert sum(p.numel() for p in off.parameters()) == 192
    assert off.learnable()


def test_freeze_factor():
    off = FlowOffset(_velocity([32, 64], excite=False), n_steps=2)
    off.freeze_factor(0)
    assert not any(p.requires_grad for p in off.velocities[0].parameters())
    assert all(p.requires_grad for p in off.velocities[1].parameters())
    off.freeze_factor(0, frozen=False)
    assert all(p.requires_grad for p in off.velocities[0].parameters())


def test_factors_must_share_dim():
    v2 = euler_velocity(2, n_features=16, length_scale_range=LSR, seed=1)
    v3 = euler_velocity(3, n_features=16, length_scale_range=LSR, seed=2)
    with pytest.raises(ValueError, match="share `dim`"):
        FlowOffset([v2, v3], n_steps=2)


def test_empty_velocity_list_rejected():
    with pytest.raises(ValueError, match="empty"):
        FlowOffset([], n_steps=2)


# ── global time ─────────────────────────────────────────────────────────────

def test_map_to_time_spans_factors():
    """
    ``tau`` runs over the whole composition: factor ``i`` owns ``[i/K, (i+1)/K]``.
    At ``tau = 1`` the partial map must equal the full forward map.
    """
    off = FlowOffset(_velocity([48, 48, 48]), n_steps=[8, 8, 8])
    x = _points(n=32, spread=400.0)
    assert (off.map_to_time(x, 0.0) - x).abs().max() == 0.0
    full = off._run(x, sign=+1.0)                # forward_map without the pin
    assert float((off.map_to_time(x, 1.0) - full).abs().max()) < 1e-12
    # one third of the way through == the first factor alone
    first = FlowOffset(off.velocities[0], n_steps=8)._run(x, sign=+1.0)
    assert float((off.map_to_time(x, 1.0 / 3.0) - first).abs().max()) < 1e-12


def test_time_labels_span_the_unit_interval_once():
    """
    Each factor integrates a full unit flow but is *labelled* with its slice of
    global time, so a time-dependent velocity sees tau sweeping [0, 1] exactly
    once across the whole composition.
    """
    off = FlowOffset(_velocity([16, 16, 16], excite=False), n_steps=[2, 2, 2])
    spans_fwd = [span for _, _, span in off._plan(+1.0)]
    assert spans_fwd == [(0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)]
    spans_inv = [span for _, _, span in off._plan(-1.0)]
    assert spans_inv == [(1.0, 2 / 3), (2 / 3, 1 / 3), (1 / 3, 0.0)]


def test_amplitudes_are_invariant_to_factor_count():
    """
    The deformation must not shrink as bands are added. (Flowing ``v`` for
    ``1/K`` equals flowing ``v/K`` for 1 — the naive "split the duration"
    convention would divide the displacement by K.)
    """
    x = _points()
    disp = []
    for counts in ([512], [256, 256], [171, 170, 171]):
        # Averaged over banks: one draw alone can differ by a factor of five.
        per_seed = []
        for seed in range(6):
            off = FlowOffset(_velocity(counts, seed=seed), n_steps=16)
            with torch.no_grad():
                per_seed.append(float((x - off.inverse_map(x)).norm(dim=1).median()))
        disp.append(sum(per_seed) / len(per_seed))
    # different banks give different maps, but all of the same magnitude —
    # not disp[k] ~ disp[0] / k
    assert min(disp) > 0.25 * max(disp)


# ── the pin ─────────────────────────────────────────────────────────────────

def test_single_pin_applies_to_the_composed_map():
    """One translation of the composed restored frame, not one per factor."""
    off = FlowOffset(_velocity([48, 48, 48]), n_steps=[6, 6, 6])
    anchor = torch.zeros(1, 2, dtype=curlew.dtype)
    off.pin(anchor, reduce="anchor")
    assert off.pinned
    assert float((off.inverse_map(anchor) - anchor).abs().max()) < 1e-10
    # a pin is a translation: the Jacobian is untouched
    x = _points(n=16, spread=400.0)
    _, j_pinned = off.inverse_map_with_jacobian(x)
    _, j_bare = off.unpin().inverse_map_with_jacobian(x)
    assert (j_pinned - j_bare).abs().max() == 0.0


# ── through the restore event ───────────────────────────────────────────────

def test_restore_accepts_a_banded_velocity():
    rng = np.random.default_rng(0)
    gp = rng.uniform(-200, 200, (24, 2))
    gv = np.tile([0.0, 1.0], (24, 1))
    C = CSet(gp=gp, gv=gv)
    event = restore(
        "banded", C=C, H=HSet().zero(grad_loss="100."),
        velocity=_velocity([64, 64, 64], excite=False),
        n_steps=[4, 6, 8], depth_axis=1, init_std=0.0, pin=[0.0, 0.0],
    )
    field = event.field
    assert isinstance(field.velocity, nn.ModuleList)
    assert field.n_steps == [4, 6, 8]
    assert sum(p.numel() for p in field.parameters()) == 384
    x = torch.tensor(gp, dtype=curlew.dtype)
    _, j = field.total_inverse_map_with_jacobian(x)
    assert float((torch.linalg.det(j) - 1.0).abs().max()) < 1e-12
    assert torch.isfinite(event.loss().total())


def test_restore_validates_n_steps_against_factor_count():
    with pytest.raises(ValueError, match="entries but there are"):
        restore(
            "bad", H=HSet().zero(grad_loss="1."),
            velocity=_velocity([32, 32], excite=False),
            n_steps=[4, 6, 8], depth_axis=1, init_std=0.0,
        )


def test_one_band_is_bit_identical_to_the_monolithic_bank():
    """
    ``n_features=[F]`` and ``n_features=F`` must be the *same model*, not merely
    a close one — otherwise the composite path silently changes results for
    anyone who writes a one-element list. This is the acceptance test for the
    whole feature.
    """
    mono = _velocity(256, excite=False)
    (listed,) = _velocity([256], excite=False)
    assert (mono.stream.fourier_projection - listed.stream.fourier_projection).abs().max() == 0.0
    assert (mono.stream.B_mu - listed.stream.B_mu).abs().max() == 0.0
    assert float(mono.stream.out_scale) == float(listed.stream.out_scale)

    torch.manual_seed(3)
    with torch.no_grad():
        a = torch.randn_like(mono.stream.A_mu)
        b = torch.randn_like(mono.stream.B_mu)
        mono.stream.A_mu.copy_(a)
        listed.stream.A_mu.copy_(a)
        mono.stream.B_mu.copy_(b)
        listed.stream.B_mu.copy_(b)
    x = _points()
    m = FlowOffset(mono, n_steps=7).inverse_map_with_jacobian(x)
    l = FlowOffset([listed], n_steps=[7]).inverse_map_with_jacobian(x)
    assert (m[0] - l[0]).abs().max() == 0.0
    assert (m[1] - l[1]).abs().max() == 0.0


def test_velocity_probe_degrades_to_the_getattr_default():
    """
    `getattr(obj, "velocity", default)` is used to ask "does this thing have a
    single velocity?" — notably in `GeoEvent.pin_frames`. A composite flow must
    answer "no" by falling back to the default, not by raising through it.
    """
    composite = FlowOffset(_velocity([32, 32, 32], excite=False), n_steps=2)
    assert getattr(composite, "velocity", None) is None
    assert getattr(composite, "velocity", "sentinel") == "sentinel"


def test_prefit_and_pin_survive_a_composite_flow():
    """Regression: pinning after a fit used to probe `.velocity` and explode."""
    rng = np.random.default_rng(1)
    gp = rng.uniform(-200, 200, (24, 2))
    gv = np.tile([0.0, 1.0], (24, 1))
    C = CSet(gp=gp, gv=gv)
    event = restore(
        "banded", C=C, H=HSet().zero(grad_loss="100."),
        velocity=_velocity([48, 48, 48]),      # excited: a real, non-identity flow
        n_steps=[4, 4, 4], depth_axis=1, init_std=None, pin=[0.0, 0.0],
    )
    model = curlew.GeoModel([event])
    pebble = model.prefit(3, vb=False)          # this is what used to raise
    assert event.name in pebble
    model.fit(3, vb=False, early_stop=None)
    event._apply_frame_pins()
    assert event.field.pinned
    anchor = torch.zeros(1, 2, dtype=curlew.dtype)
    assert float((event.field.total_inverse_map(anchor) - anchor).abs().max()) < 1e-9


def test_composite_fault_lift_opens_one_session():
    """One snapshot/restore wraps every band; traces are not rewound mid-map."""
    vs = _velocity([16, 16, 16], excite=False)
    trace = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=curlew.dtype)
    lift = FaultLift([trace])
    n_snap = n_restore = 0
    orig_snap = lift.snapshot_traces
    orig_restore = lift.restore_traces

    def snap():
        nonlocal n_snap
        n_snap += 1
        return orig_snap()

    def restore_tr():
        nonlocal n_restore
        n_restore += 1
        return orig_restore()

    lift.snapshot_traces = snap
    lift.restore_traces = restore_tr
    off = FlowOffset(vs, n_steps=2, fault_lift=lift)
    off.inverse_map(_points(8, spread=2.0))
    assert n_snap == 1
    assert n_restore == 1


def test_composite_fault_activation_uses_global_tau():
    """activation_windows see τ across the whole composition, not [0, 1] per band."""
    vs = _velocity([16, 16], excite=False)
    trace = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=curlew.dtype)
    lift = FaultLift([trace], activation_windows=[(0.0, 1.0)])
    times = []
    orig = lift.activation

    def spy(t):
        if t is not None:
            times.append(float(t))
        return orig(t)

    lift.activation = spy
    FlowOffset(vs, n_steps=2, fault_lift=lift).inverse_map(_points(4, spread=2.0))
    assert times, "activation was never called"
    assert sum(1 for t in times if abs(t - 1.0) < 1e-12) == 1
    assert min(times) < 0.5 < max(times)
