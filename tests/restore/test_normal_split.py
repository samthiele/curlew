"""Normal split of a full 3-D Euler velocity.

Rays ``u_p``, ``v_q`` give shears along ``n = u_p × v_q``. Depth is the
number of distinct normals, at most ``R_a R_b``, not the mode count.
``det J = 1`` at any step count.
"""
import math

import curlew
import pytest
import torch

from curlew.kinematics.euler import euler_velocity
from curlew.kinematics.flow import FlowOffset


@pytest.fixture(autouse=True)
def _defaults():
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def _velocity(n_features=32, n_directions=8, scale=0.1, both=False, seed=3, **kw):
    """Full 3-D Euler (two FSF slots). ``both`` excites the A and B channels."""
    curlew.default_dim = 3
    v = euler_velocity(
        3, n_features=n_features, length_scale_range=(200.0, 5000.0),
        seed=seed, n_directions=n_directions, **kw,
    )
    torch.manual_seed(11)
    for m in v.modules():
        for name in (("C_mu",) if hasattr(m, "C_mu") else ("A_mu", "B_mu") if both else ("A_mu",)):
            if hasattr(m, name):
                with torch.no_grad():
                    getattr(m, name).normal_(0.0, scale)
    return v


def _points(n=40, box=4000.0):
    torch.manual_seed(0)
    return torch.rand(n, 3, dtype=curlew.dtype) * box


def _flow(v, n_steps=4, **kw):
    return FlowOffset(v, n_steps=n_steps, integrator="shear", **kw)


def _group_field(g, x):
    """``psi_g(x)`` for one plan group, evaluated directly from its pieces."""
    F = torch.cos(torch.addmm(g["oa"], x, g["Wa"])) @ g["Ea"]
    G = torch.cos(torch.addmm(g["ob"], x, g["Wb"])) @ g["Eb"]
    return (F * G) @ g["c"]


def _det_error(J):
    """
    ``|det J - 1|`` in units of what LU can resolve. Computing a determinant
    loses ``~eps ||J||^2`` to rounding in 3-D, so a strongly sheared ``J``
    (``||J|| ~ 1e3``) cannot show ``det = 1`` below ~1e-10 *however* exact
    the map is. Normalising by that floor keeps the check equally sharp for
    gentle and violent flows: an inexact scheme misses by orders of magnitude.
    """
    eps = torch.finfo(J.dtype).eps
    floor = eps * torch.linalg.matrix_norm(J, ord=2).pow(2).clamp_min(1.0)
    return float(((torch.linalg.det(J) - 1.0).abs() / floor).max())


def _autograd_jacobian(fn, x):
    xg = x.clone().requires_grad_(True)
    y = fn(xg)
    return torch.stack(
        [torch.autograd.grad(y[:, k].sum(), xg, retain_graph=True)[0]
         for k in range(3)], dim=1,
    )


# ── 1. the ridge form ───────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    {}, {"both": True}, {"stream": "alpha"}, {"activation": "cubic"},
    {"n_directions": None},
])
def test_ridge_form_reconstructs_each_gradient_and_the_velocity(kw):
    """
    ``grad P = sum_i amp_i dir_i cos(kap_i dir_i.x + off_i)`` must equal the slot's
    own gradient, and ``grad P x grad Q`` the velocity — with the slot order
    the forward path uses (``stream`` swaps it, and swapping negates ``v``).
    """
    v = _velocity(**kw)
    x = _points()
    bank_a, bank_b = v.ridge_decomposition()

    def grad(bank):
        d, amp, kap, off = bank
        assert float((d.norm(dim=-1) - 1).abs().max()) < 1e-15
        ph = (x @ d.T) * kap + off
        return (torch.cos(ph) * amp) @ d

    first, second = (v.beta, v.alpha) if v._swap_slots() else (v.alpha, v.beta)
    for bank, slot in ((bank_a, first), (bank_b, second)):
        ref = slot.gradient_raw(x)
        assert float((grad(bank) - ref).abs().max()) < 1e-12 * float(ref.abs().max())

    vel = torch.linalg.cross(grad(bank_a), grad(bank_b), dim=-1)
    ref = v(x)
    assert float((vel - ref).abs().max()) < 1e-12 * float(ref.abs().max())


def test_hinged_and_2d_fields_have_no_ridge_form():
    """Those already group by ray in the phase split; the normal split stays out."""
    curlew.default_dim = 3
    hinged = euler_velocity(3, n_features=8, hinge=[0.0, 1.0, 0.0], seed=3)
    curlew.default_dim = 2
    planar = euler_velocity(2, n_features=8, seed=3)
    assert hinged.ridge_decomposition() is None
    assert planar.ridge_decomposition() is None


# ── 2. the grouped form ─────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [
    {}, {"both": True}, {"activation": "cubic"}, {"stream": "alpha"},
    {"n_directions": None, "n_features": 6},
    {"alpha": {"n_features": 12}, "beta": {"n_features": 20}},
])
def test_grouped_form_reconstructs_the_velocity(kw):
    """``sum_g psi_g(x) n_g`` is the velocity, to rounding."""
    v = _velocity(**kw)
    x = _points()
    plan = _flow(v)._ridge_plan(None, x, None)
    recon = sum(_group_field(g, x).unsqueeze(-1) * g["nhat"] for g in plan["groups"])
    ref = v(x)
    assert float((recon - ref).abs().max()) < 1e-12 * float(ref.abs().max())


@pytest.mark.parametrize("kw", [{}, {"activation": "cubic"}, {"shared": True}])
def test_every_phase_direction_is_perpendicular_to_its_normal(kw):
    """
    The *only* property the exact step and ``det = 1`` rest on. Asserted at
    machine precision, including merged groups (``shared``), where each pair's
    directions are re-projected onto the group normal's complement.
    """
    v = _shared_velocity() if kw.pop("shared", False) else _velocity(**kw)
    plan = _flow(v)._ridge_plan(None, _points(), None)
    for g in plan["groups"]:
        n = g["nhat"]
        assert abs(float(n.norm()) - 1.0) < 1e-15
        assert float((g["S"] @ n).abs().max()) < 1e-15
        assert float((g["T"] @ n).abs().max()) < 1e-15
        # Phase weights are built from S and T, so they inherit it.
        assert float((n @ g["Wa"]).abs().max()) < 1e-15 * float(g["Wa"].abs().max())
        assert float((n @ g["Wb"]).abs().max()) < 1e-15 * float(g["Wb"].abs().max())


# ── 3. one group is integrated exactly ──────────────────────────────────────

def test_one_group_substep_is_its_exact_flow():
    """
    ``psi_g`` is constant along ``psi_g n_g``, so ``x + h psi_g(x) n_g`` is the
    exact time-``h`` flow. Check both halves: ``psi_g`` is unchanged at the
    image point, and a fine RK4 of the group field lands on the same point.
    """
    v = _velocity(scale=0.3)
    x = _points()
    plan = _flow(v)._ridge_plan(None, x, None)
    g = max(plan["groups"], key=lambda g: float(_group_field(g, x).abs().max()))
    field = lambda y: _group_field(g, y).unsqueeze(-1) * g["nhat"]

    h = 3.0e3 / float(_group_field(g, x).abs().max())   # ~3 km displacement
    y = x + h * field(x)
    assert float((_group_field(g, y) - _group_field(g, x)).abs().max()) < 1e-12 * float(
        _group_field(g, x).abs().max())

    z, k = x.clone(), 200
    for _ in range(k):
        dt = h / k
        a = field(z); b = field(z + dt / 2 * a); c = field(z + dt / 2 * b); d = field(z + dt * c)
        z = z + dt / 6 * (a + 2 * b + 2 * c + d)
    assert float((z - y).abs().max()) < 1e-8


# ── 4. the composed map ─────────────────────────────────────────────────────

@pytest.mark.parametrize("n_steps", [1, 2, 8, 32])
def test_determinant_is_exactly_one_at_any_step_count(n_steps):
    v = _velocity(scale=0.2, both=True)
    _, J = _flow(v, n_steps)._run(_points(), sign=-1.0, jac=True)
    assert float(torch.linalg.matrix_norm(J, ord=2).max()) > 10.0   # not a trivial map
    # Composing ~2mn rank-one factors adds ~eps ||J|| to J per step, so the
    # global det sits a few-to-hundreds of LU floors out, depending on the
    # sweep order (4-182x here; the phase split measures 29-170x on this same
    # flow). An inexact scheme misses by orders of magnitude more. The
    # per-factor identity is pinned exactly in
    # `test_every_factor_is_exactly_unimodular`.
    assert _det_error(J) < 250.0


@pytest.mark.parametrize("n_steps", [1, 8, 32])
def test_every_factor_is_exactly_unimodular(n_steps):
    """
    ``det(I + n grad(psi)^T) = 1 + n . grad(psi)``. Read every factor of the
    actual run off the pullback stack and check ``n . grad(psi)`` vanishes at
    a few ulps — the identity itself, free of the conditioning that blurs a
    global ``det``.

    The yardstick is the size of the two *terms* of
    ``grad(psi) = c (f' g S + f g' T)``, not of their sum: each term is
    perpendicular to ``n`` to rounding, but where they nearly cancel
    ``|grad(psi)|`` is far smaller than the rounding they carry, and a bound
    relative to it would fail on an exact identity.
    """
    v = _velocity(scale=0.2, both=True)
    x = _points()
    off = _flow(v, n_steps)
    stack, record = [], []
    off._integrate(x, sign=-1.0, stack=stack, record=record)
    groups = off._ridge_plan(None, x, None)["groups"]
    sched = FlowOffset._shear_schedule(n_steps, len(groups))
    assert len(stack) == len(sched) == len(record)
    h = 1.0 / n_steps / 2.0
    eps = torch.finfo(curlew.dtype).eps
    with torch.no_grad():
        for k, ((gi, mult), (n, grad)) in enumerate(zip(sched, stack)):
            g, xs = groups[gi], (x if k == 0 else record[k - 1])
            pa = torch.addmm(g["oa"], xs, g["Wa"])
            pb = torch.addmm(g["ob"], xs, g["Wb"])
            F, dF = torch.cos(pa) @ g["Ea"], -(torch.sin(pa) @ g["Eka"])
            G, dG = torch.cos(pb) @ g["Eb"], -(torch.sin(pb) @ g["Ekb"])
            terms = g["c"].abs() * (
                (dF * G).abs() * g["S"].norm(dim=-1)
                + (F * dG).abs() * g["T"].norm(dim=-1)
            )
            bound = 8 * eps * h * mult * terms.sum(-1)
            assert bool(((grad @ n).abs() <= bound + 1e-300).all())


@pytest.mark.parametrize("kw", [{}, {"activation": "cubic"}])
def test_jacobian_is_the_maps_own_derivative(kw):
    v = _velocity(**kw)
    x = _points()
    off = _flow(v, 3)
    _, J = off._run(x, sign=-1.0, jac=True)
    A = _autograd_jacobian(lambda y: off._run(y, sign=-1.0), x)
    assert float((J - A).abs().max()) < 1e-9 * max(1.0, float(A.abs().max()))


def test_jacobian_and_plain_paths_move_points_identically():
    v = _velocity()
    x = _points()
    off = _flow(v, 5)
    y, _ = off._run(x, sign=-1.0, jac=True)
    assert float((y - off._run(x, sign=-1.0)).abs().max()) < 1e-10


@pytest.mark.parametrize("kw", [{}, {"activation": "cubic"}, {"n_directions": None, "n_features": 6}])
def test_converges_second_order_to_the_true_flow(kw):
    """
    Strang over the normal groups: halving ``dt`` quarters the error, and the
    limit is the trajectory RK4 computes — and the one the phase split
    computes, since both cut up the same field.
    """
    v = _velocity(**kw)
    x = _points()
    ref = FlowOffset(v, n_steps=1024, integrator="rk4")._run(x, sign=-1.0)
    assert float((ref - x).norm(dim=-1).max()) > 300.0        # a real deformation

    # Coarse sweeps of a km-scale flow are pre-asymptotic (ratios of 6-8 at
    # n = 8-16 for some draws), so measure the rate where it has settled.
    errs = []
    for n in (32, 64, 128):
        y = _flow(v, n)._run(x, sign=-1.0)
        errs.append(float((y - ref).norm(dim=-1).max()))
    for a, b in zip(errs, errs[1:]):
        assert 3.5 < a / b < 4.5
    assert errs[-1] < 0.5

    phase = _flow(v, 64, shear_split="phase")._run(x, sign=-1.0)
    assert float((phase - ref).norm(dim=-1).max()) < 0.5


def test_forward_and_inverse_are_reciprocal():
    """Each sub-step inverts exactly, and the Strang sweep is symmetric."""
    v = _velocity()
    x = _points()
    off = _flow(v, 6)
    assert float((off.forward_map(off.inverse_map(x)) - x).abs().max()) < 1e-6


def test_gradients_reach_both_potentials_through_position_and_jacobian():
    v = _velocity()
    y, J = _flow(v, 3)._run(_points(), sign=-1.0, jac=True)
    (y.pow(2).sum() + J.pow(2).sum()).backward()
    for slot in (v.alpha, v.beta):
        g = slot.A_mu.grad
        assert g is not None and torch.isfinite(g).all() and float(g.abs().max()) > 0.0


# ── 5. sequential depth ─────────────────────────────────────────────────────

@pytest.mark.parametrize("F", [16, 32, 64])
def test_depth_is_rays_squared_independent_of_the_bank_size(F):
    """
    The point of the split: ``n_directions=8`` on both slots gives at most 64
    groups whatever ``F`` is, where the phase split's group count grows as F^2.
    """
    v = _velocity(n_features=F, n_directions=8)
    off = _flow(v)
    plan = off._ridge_plan(None, _points(), None)
    assert len(plan["groups"]) == 64
    assert sum(len(g["c"]) for g in plan["groups"]) == 64
    C, om, _ = v.shear_decomposition()
    assert len(FlowOffset._shear_groups(om.detach())) >= F * F / 2


@pytest.mark.parametrize("seed", [3, 4])
def test_rays_and_normals_sweep_from_the_coarsest(seed):
    """
    Each slot's rays run longest wavelength first — ray ``j`` carries the
    ``j``-th longest wavelength, so this is the Kronecker order whatever the
    seed — and the normals follow ``p``-major, so the sweep opens on the
    coarsest ray pair. The ridge banks hold *unit* directions, so the order has
    to come from ``kap``; ordering by ``|dir|`` would silently fall back to row
    order.
    """
    v = _velocity(n_features=32, n_directions=8, seed=seed)
    (dir_a, _, kap_a, _), (dir_b, _, kap_b, _) = v.ridge_decomposition()
    geo = FlowOffset._ridge_geometry(dir_a, dir_b, kap_a, kap_b)
    for ray, kap, R in ((geo["ray_a"], kap_a, geo["Ra"]), (geo["ray_b"], kap_b, geo["Rb"])):
        lo = torch.stack([kap.abs()[ray == r].min() for r in range(R)])
        assert torch.allclose(lo, torch.unique(kap.abs())[:R], rtol=1e-12)
    pq = [(int(g["p"][0]), int(g["q"][0])) for g in geo["groups"]]
    assert pq[0] == (0, 0)
    assert pq == sorted(pq)


def _shared_velocity(F=32, D=6, scale=0.1):
    """Both slots on the *same* direction set (same seed): normals pair up."""
    v = _velocity(n_features=F, n_directions=D, scale=scale,
                  alpha={"seed": 5}, beta={"seed": 5})
    torch.manual_seed(12)
    with torch.no_grad():
        v.alpha.A_mu.normal_(0.0, scale)
    return v


def test_shared_direction_sets_merge_mirror_pairs():
    """
    ``u_p x u_q = -(u_q x u_p)``, so with one direction set the ``(p, q)`` and
    ``(q, p)`` terms share a normal and merge, and ``p == q`` vanishes:
    ``D (D - 1) / 2`` groups of two pairs each.
    """
    v = _shared_velocity(D=6)
    assert torch.equal(v.alpha.omega_spatial, v.beta.omega_spatial)
    plan = _flow(v)._ridge_plan(None, _points(), None)
    assert len(plan["groups"]) == 6 * 5 // 2
    assert all(len(g["c"]) == 2 for g in plan["groups"])


def test_merged_groups_are_still_exact_and_reconstruct():
    v = _shared_velocity()
    x = _points()
    off = _flow(v, 4)
    plan = off._ridge_plan(None, x, None)
    recon = sum(_group_field(g, x).unsqueeze(-1) * g["nhat"] for g in plan["groups"])
    ref = v(x)
    assert float((recon - ref).abs().max()) < 1e-12 * float(ref.abs().max())
    _, J = off._run(x, sign=-1.0, jac=True)
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-11
    A = _autograd_jacobian(lambda y: off._run(y, sign=-1.0), x)
    assert float((J - A).abs().max()) < 1e-9 * max(1.0, float(A.abs().max()))


# ── 6. the machinery around it ──────────────────────────────────────────────

@pytest.mark.parametrize("n_steps", [1, 4])
def test_pullback_matches_the_full_jacobian(n_steps):
    v = _velocity()
    x = _points()
    off = _flow(v, n_steps)
    _, J = off.inverse_map_with_jacobian(x)
    x0, pull = off.inverse_map_with_pullback(x)
    g = torch.randn(x.shape[0], 3, dtype=x.dtype)
    ref = torch.bmm(J.mT, g.unsqueeze(-1)).squeeze(-1)
    assert float((pull(g) - ref).abs().max()) < 1e-10 * max(1.0, float(ref.abs().max()))
    assert float((x0 - off.inverse_map(x)).abs().max()) < 1e-12


@pytest.mark.parametrize("k", [1, 7, 40])
def test_jac_rows_restricts_the_pullback_without_changing_it(k):
    v = _velocity()
    x = _points()
    off = _flow(v, 3)
    x_full, pull_full = off.inverse_map_with_pullback(x)
    x_pre, pull_pre = off.inverse_map_with_pullback(x, jac_rows=k)
    g = torch.randn(k, 3, dtype=x.dtype)
    pad = torch.cat([g, torch.zeros(x.shape[0] - k, 3, dtype=x.dtype)])
    assert torch.equal(x_full, x_pre)
    assert float((pull_pre(g) - pull_full(pad)[:k]).abs().max()) < 1e-12


def test_composite_factors_chain_through_the_pullback():
    vs = [_velocity(n_features=16, n_directions=4), _velocity(n_features=16, n_directions=5)]
    x = _points()
    off = FlowOffset(vs, n_steps=[2, 3], integrator="shear")
    _, J = off.inverse_map_with_jacobian(x)
    _, pull = off.inverse_map_with_pullback(x)
    g = torch.randn(x.shape[0], 3, dtype=x.dtype)
    ref = torch.bmm(J.mT, g.unsqueeze(-1)).squeeze(-1)
    assert float((pull(g) - ref).abs().max()) < 1e-10 * max(1.0, float(ref.abs().max()))


def test_tables_converge_to_the_direct_map():
    v = _velocity()
    x = _points()
    exact = _flow(v, 4)._run(x, sign=-1.0)
    errs = [float((_flow(v, 4, shear_table=T)._run(x, sign=-1.0) - exact).norm(dim=-1).max())
            for T in (2048, 8192, 32768)]       # T <= 1024 can be pre-asymptotic
    for a, b in zip(errs, errs[1:]):
        assert 10.0 < a / b < 22.0        # linear interpolation: O(ds^2), 4x finer
    assert errs[-1] < 1.0


@pytest.mark.parametrize("T", [128, 1024])
def test_tables_keep_the_determinant_and_their_own_jacobian(T):
    v = _velocity(scale=0.2)
    x = _points()
    off = _flow(v, 3, shear_table=T)
    _, J = off._run(x, sign=-1.0, jac=True)
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-10
    A = _autograd_jacobian(lambda y: off._run(y, sign=-1.0), x)
    assert float((J - A).abs().max()) < 1e-8 * max(1.0, float(A.abs().max()))
    _, pull = off.inverse_map_with_pullback(x)
    g = torch.randn(x.shape[0], 3, dtype=x.dtype)
    _, Jinv = off.inverse_map_with_jacobian(x)
    ref = torch.bmm(Jinv.mT, g.unsqueeze(-1)).squeeze(-1)
    assert float((pull(g) - ref).abs().max()) < 1e-9 * max(1.0, float(ref.abs().max()))


def _hinged_velocity(scale=2.0, n_features=32):
    curlew.default_dim = 3
    v = euler_velocity(3, n_features=n_features, hinge=[0.0, 1.0, 0.0], seed=3,
                       length_scale_range=(200.0, 5000.0), n_directions=4)
    torch.manual_seed(11)
    with torch.no_grad():
        v.stream.A_mu.normal_(0.0, scale)
    return v


@pytest.mark.parametrize("split", ["normal", "phase"])
def test_tables_never_read_off_the_end(split, monkeypatch):
    """
    The table must cover every point for the *whole* traversal. Each group
    runs for the full duration, not one step, so many steps of a strong flow
    is the case that would expose an under-padded grid. (The phase-split
    grid used to be padded for ``duration / n_steps`` only.)
    """
    x = _points()
    seen = []
    if split == "normal":
        v, name = _velocity(scale=0.2), "_ridge_lookup"
        orig = FlowOffset._ridge_lookup

        def spy(table, rays, s):
            tab, s0, ds = table
            u = (s - s0[rays]) / ds[rays]
            seen.append((float(u.min()), float(u.max()), tab.shape[1]))
            return orig(table, rays, s)
    else:
        v, name = _hinged_velocity(), "_table_lookup"
        orig = FlowOffset._table_lookup

        def spy(tab, s0, ds, s):
            u = (s - s0) / ds
            seen.append((float(u.min()), float(u.max()), tab.shape[0]))
            return orig(tab, s0, ds, s)

    monkeypatch.setattr(FlowOffset, name, staticmethod(spy))
    off = _flow(v, 16, shear_table=256, shear_split=split)
    y = off._run(x, sign=-1.0)
    assert float((y - x).norm(dim=-1).max()) > 300.0
    assert seen
    assert all(lo >= 0.0 and hi <= T - 1 for lo, hi, T in seen)


def test_cache_follows_the_frequency_buffers():
    """
    The grouping is cached per velocity. If the frequency buffers change (a
    re-draw, a rotation), it must be rebuilt rather than reused with stale
    rays: the cached flow must equal a fresh one, and differ from before.
    """
    v = _velocity()
    x = _points()
    off = _flow(v, 3)
    before = off._run(x, sign=-1.0)
    c, s = math.cos(0.4), math.sin(0.4)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=x.dtype)
    with torch.no_grad():
        v.alpha.omega_spatial.copy_(v.alpha.omega_spatial @ R.T)
    after = off._run(x, sign=-1.0)
    assert torch.equal(after, _flow(v, 3)._run(x, sign=-1.0))
    assert not torch.allclose(after, before)


def test_phase_split_cache_survives_a_channel_waking_up():
    """
    The phase split prunes zero-amplitude terms, so the row set of
    ``shear_decomposition`` changes when a channel leaves zero. The cached
    grouping must follow it rather than index the new rows with old groups.
    """
    curlew.default_dim = 3
    v = euler_velocity(3, n_features=16, hinge=[0.0, 1.0, 0.0], seed=3,
                       length_scale_range=(200.0, 5000.0), n_directions=4)
    torch.manual_seed(11)
    with torch.no_grad():
        v.stream.A_mu.normal_(0.0, 0.5)
        v.stream.B_mu.zero_()
    x = _points()
    off = _flow(v, 3)
    off._run(x, sign=-1.0)                         # caches the A-only grouping
    with torch.no_grad():
        v.stream.B_mu.normal_(0.0, 0.5)
    assert torch.equal(off._run(x, sign=-1.0), _flow(v, 3)._run(x, sign=-1.0))


def test_phase_split_is_still_available_and_unknown_splits_are_refused():
    v = _velocity(n_features=8, n_directions=4)
    x = _points()
    a = _flow(v, 2, shear_split="phase")._run(x, sign=-1.0)
    b = _flow(v, 2, shear_split="normal")._run(x, sign=-1.0)
    assert not torch.equal(a, b)                   # genuinely different splittings
    with pytest.raises(ValueError, match="shear_split"):
        _flow(v, shear_split="nope")
