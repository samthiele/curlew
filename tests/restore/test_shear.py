"""Exact shear integrator (Quispel and McLaren 2003).

``det J = 1`` at any step count. The default integrator remains RK4.
"""
import curlew
import pytest
import torch

from curlew.kinematics.euler import euler_velocity
from curlew.kinematics.flow import FlowOffset


@pytest.fixture(autouse=True)
def _defaults():
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def _velocity(dim=2, n_features=48, scale=0.5, seed=3, **kw):
    curlew.default_dim = dim
    v = euler_velocity(
        dim, n_features=n_features, length_scale_range=(200.0, 5000.0),
        seed=seed, **kw,
    )
    torch.manual_seed(11)
    for m in v.modules():
        if hasattr(m, "C_mu"):
            with torch.no_grad():
                m.C_mu.normal_(0.0, scale)
        elif hasattr(m, "A_mu"):
            with torch.no_grad():
                m.A_mu.normal_(0.0, scale)
    return v


def _points(dim, n=40):
    torch.manual_seed(0)
    return torch.rand(n, dim, dtype=curlew.dtype) * 4000.0


# ── the decomposition itself ────────────────────────────────────────────────

@pytest.mark.parametrize("dim,kw", [
    (2, {}),
    (2, {"stream": "beta"}),
    (3, {"hinge": [0.0, 1.0, 0.0]}),
    (3, {"n_features": 8}),
    (3, {"n_features": 8, "stream": "alpha"}),
])
def test_decomposition_reconstructs_the_field_and_is_a_shear(dim, kw):
    """
    ``sum_k C_k cos(Om_k.x + off_k)`` must equal the velocity exactly, and each
    ``C_k`` must be orthogonal to its own ``Om_k`` — the shear condition is what
    makes the closed-form step and the unit determinant true rather than
    approximate, so it is asserted at machine precision, not loosely.
    """
    v = _velocity(dim, **kw)
    x = _points(dim)
    C, om, off = v.shear_decomposition()

    assert float((C * om).sum(-1).abs().max()) < 1e-14
    recon = (torch.cos(x @ om.T + off).unsqueeze(-1) * C).sum(1)
    ref = v(x)
    assert float((recon - ref).abs().max()) < 1e-12 * float(ref.abs().max())


def test_cartesian_channels_reconstruct_as_shears():
    """Both ``A`` and ``B`` channels must appear in the cosine bank."""
    v = _velocity(2, n_features=24)
    fsf = v.stream
    with torch.no_grad():
        fsf.A_mu.fill_(0.3)
        fsf.B_mu.fill_(0.4)
    x = _points(2)
    C, om, off = v.shear_decomposition()
    assert C.shape[0] == 2 * fsf.A_mu.numel()
    assert float((C * om).sum(-1).abs().max()) < 1e-14
    recon = (torch.cos(x @ om.T + off).unsqueeze(-1) * C).sum(1)
    ref = v(x)
    assert float((recon - ref).abs().max()) < 1e-12 * float(ref.abs().max())


def test_full_3d_product_has_two_shears_per_mode_pair():
    """Two learnable slots expand to 2 F_α F_β shears (minus a handful of zeros)."""
    v = _velocity(3, n_features=6)
    C, om, _ = v.shear_decomposition()
    n_expect = 2 * v.alpha.A_mu.numel() * v.beta.A_mu.numel()
    assert C.shape[0] == om.shape[0]
    assert n_expect - 4 <= C.shape[0] <= n_expect


def test_full_3d_asymmetric_slots():
    v = euler_velocity(
        3, n_features=8, length_scale_range=(200.0, 5000.0), seed=3,
        alpha={"n_features": 4}, beta={"n_features": 7},
    )
    torch.manual_seed(11)
    for m in v.modules():
        if hasattr(m, "A_mu"):
            with torch.no_grad():
                m.A_mu.normal_(0.0, 0.5)
    x = _points(3)
    C, om, off = v.shear_decomposition()
    recon = (torch.cos(x @ om.T + off).unsqueeze(-1) * C).sum(1)
    ref = v(x)
    assert float((recon - ref).abs().max()) < 1e-12 * float(ref.abs().max())
    assert C.shape[0] <= 2 * 4 * 7


def test_full_3d_cubic_reconstructs():
    v = _velocity(3, n_features=4, activation="cubic")
    x = _points(3)
    C, om, off = v.shear_decomposition()
    assert C.shape[0] <= 2 * (3 * 4) * (3 * 4)
    recon = (torch.cos(x @ om.T + off).unsqueeze(-1) * C).sum(1)
    ref = v(x)
    assert float((C * om).sum(-1).abs().max()) < 1e-13
    assert float((recon - ref).abs().max()) < 1e-11 * max(float(ref.abs().max()), 1e-12)


def test_full_3d_determinant_is_exactly_one():
    v = _velocity(3, n_features=6, scale=0.05)
    x = torch.rand(16, 3, dtype=curlew.dtype) * 200.0
    _, J = FlowOffset(v, n_steps=2, integrator="shear")._run(
        x, sign=-1.0, jac=True,
    )
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-10


# ── what the splitting is for ───────────────────────────────────────────────

@pytest.mark.parametrize("n_steps", [1, 2, 8, 32])
def test_determinant_is_exactly_one_at_any_step_count(n_steps):
    """
    The point of the scheme. ``rk4`` gets ``1 + O(dt^4)`` and degrades as steps
    are removed; here the rank-one nilpotent factor makes ``det = 1`` an
    identity, so a single step is as exact as thirty-two.
    """
    v = _velocity(scale=2.0)
    _, J = FlowOffset(v, n_steps=n_steps, integrator="shear")._run(
        _points(2), sign=-1.0, jac=True,
    )
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-11


def test_determinant_beats_rk4_where_rk4_is_stretched():
    """
    At a step count coarse enough for RK4's fourth-order volume error to show,
    the shear map is still exact. This is the trade the integrator exists to
    make; if RK4 ever matched it here the test would be vacuous, so assert the
    gap rather than just the shear number.
    """
    v = _velocity(scale=4.0, n_features=128)
    x = _points(2)
    _, Js = FlowOffset(v, n_steps=2, integrator="shear")._run(x, sign=-1.0, jac=True)
    _, Jr = FlowOffset(v, n_steps=2, integrator="rk4")._run(x, sign=-1.0, jac=True)
    shear = float((torch.linalg.det(Js) - 1.0).abs().max())
    rk4 = float((torch.linalg.det(Jr) - 1.0).abs().max())
    assert shear < 1e-11
    assert rk4 > 1e-3
    assert shear < rk4


@pytest.mark.parametrize("dim,kw", [(2, {}), (3, {"hinge": [0.0, 1.0, 0.0]})])
def test_converges_second_order_to_the_same_flow(dim, kw):
    """
    Strang over non-commuting shears is second order, and it must converge to
    the *same* trajectory the other integrators compute — exactness of the
    determinant would be worthless if the map were a different flow.
    """
    v = _velocity(dim, **kw)
    x = _points(dim)
    ref = FlowOffset(v, n_steps=2048, integrator="rk4")._run(x, sign=-1.0)

    errs = []
    for n in (8, 16, 32, 64):
        y = FlowOffset(v, n_steps=n, integrator="shear")._run(x, sign=-1.0)
        errs.append(float((y - ref).norm(dim=-1).max()))

    assert errs[-1] < 1e-2
    for a, b in zip(errs, errs[1:]):
        assert 3.5 < a / b < 4.5          # halving dt quarters the error


def test_jacobian_is_the_maps_own_derivative():
    """
    Nothing is accumulated on the side: ``J`` is the chain-rule derivative of
    the composed shears, so it must agree with autograd through the same map.
    """
    v = _velocity()
    x = _points(2)
    off = FlowOffset(v, n_steps=6, integrator="shear")
    _, J = off._run(x, sign=-1.0, jac=True)

    xg = x.clone().requires_grad_(True)
    y = off._run(xg, sign=-1.0)
    A = torch.stack(
        [torch.autograd.grad(y[:, k].sum(), xg, retain_graph=True)[0]
         for k in range(2)], dim=1,
    )
    assert float((J - A).abs().max()) < 1e-9


def test_gradients_reach_the_potential_through_both_outputs():
    """Training needs both the position and the Jacobian to be differentiable."""
    v = _velocity()
    y, J = FlowOffset(v, n_steps=4, integrator="shear")._run(
        _points(2), sign=-1.0, jac=True,
    )
    (y.pow(2).sum() + J.pow(2).sum()).backward()
    grads = [p.grad for p in v.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert max(float(g.abs().max()) for g in grads) > 0.0


# ── grouping parallel modes ─────────────────────────────────────────────────

def test_parallel_and_antiparallel_modes_share_a_group():
    """
    Modes sharing a direction commute, so they may be applied as one shear.
    ``+d`` and ``-d`` define the same invariant and must not be split apart.
    """
    om = torch.tensor(
        [[1.0, 0.0], [2.0, 0.0], [-3.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    groups = sorted(sorted(g.tolist()) for g in FlowOffset._shear_groups(om))
    assert groups == [[0, 1, 2], [3], [4]]


def test_grouping_keeps_the_determinant_exact_and_the_answer_unchanged():
    """
    The merged perturbation ``sum_k m_k |Om_k| C_k d^T`` is still rank one with
    every ``C_k`` orthogonal to ``d``, so grouping must change neither the
    determinant nor the trajectory — only the sequential depth.
    """
    v = _velocity()
    x = _points(2)
    C, om, offs = v.shear_decomposition()
    n = om.shape[0]

    class _Stub(torch.nn.Module):
        """Same field, but every mode forced to share one wave direction."""
        dim = 2

        def shear_decomposition(self, x=None, w=None):
            d = om[:1] / om[:1].norm()
            k = om.norm(dim=-1, keepdim=True) * d          # all parallel
            perp = torch.stack([-k[:, 1], k[:, 0]], -1)
            perp = perp / perp.norm(dim=-1, keepdim=True) * C.norm(dim=-1, keepdim=True)
            return perp, k, offs

    stub = _Stub()
    grouped = FlowOffset._shear_groups(stub.shear_decomposition()[1])
    assert len(grouped) == 1 and len(grouped[0]) == n

    flow = FlowOffset(stub, n_steps=5, integrator="shear")
    y, J = flow._run(x, sign=-1.0, jac=True)
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-11

    # One commuting group is integrated exactly, so the step count is irrelevant.
    y2 = FlowOffset(stub, n_steps=37, integrator="shear")._run(x, sign=-1.0)
    assert float((y - y2).abs().max()) < 1e-9


# ── the polar grid: sharing directions to make the splitting cheap ──────────

@pytest.mark.parametrize("n_dir", [4, 8, 32])
def test_polar_grid_gives_exactly_n_directions_groups(n_dir):
    """
    The whole point of the polar bank: sequential depth becomes ``n_directions``
    instead of ``rff_features``. Assert the group count exactly, since an
    off-by-one in the cycling would quietly double it.
    """
    v = _velocity(n_features=64, n_directions=n_dir)
    groups = FlowOffset._shear_groups(v.shear_decomposition()[1])
    assert len(groups) == n_dir
    # Both Cartesian channels are kept, including a B that is still zero:
    # dropping it would freeze that amplitude for the rest of training.
    assert sum(len(g) for g in groups) == 2 * 64


def test_sweep_runs_the_rays_longest_wavelength_first():
    """
    Grouping alone would order the sweep by Sobol row — a seed-dependent
    shuffle. Ray ``j`` carries the ``j``-th longest wavelength, so sorting the
    groups by scale runs the rays in the order they were drawn. The rays are an
    even set (exactly ``pi / R`` apart) visited in golden-ratio order, so for
    *every* seed the sweep takes the same steps, and a new seed only rotates
    the whole sequence.
    """
    n_dir = 8
    angles = []
    for seed in (3, 4, 17):
        om = _velocity(n_features=64, n_directions=n_dir, seed=seed).shear_decomposition()[1]
        groups = FlowOffset._shear_groups(om)
        k = om.norm(dim=-1)
        lo = torch.stack([k[g].min() for g in groups])
        assert bool((lo[1:] > lo[:-1]).all())
        # Ray j's coarsest mode is the j-th longest wavelength in the bank.
        assert torch.allclose(lo, torch.unique(k)[:n_dir], rtol=1e-12)

        axis = torch.stack([om[g[0]] for g in groups])
        ang = torch.atan2(axis[:, 1], axis[:, 0]) % torch.pi
        # An even set: sorted axes are exactly pi / R apart, wrap included.
        a = torch.sort(ang).values
        gaps = torch.diff(torch.cat([a, a[:1] + torch.pi]))
        assert torch.allclose(gaps, torch.full_like(gaps, torch.pi / n_dir), atol=1e-12)
        angles.append(ang)
    # Same steps for every seed: the sequences differ by one rotation only...
    for ang in angles[1:]:
        shift = (ang - angles[0]) % torch.pi
        assert float(((shift - shift[0] + torch.pi / 2) % torch.pi - torch.pi / 2).abs().max()) < 1e-12
    # ...and that rotation is genuine, not zero.
    starts = [float(a[0]) for a in angles]
    assert min(abs(a - b) for i, a in enumerate(starts) for b in starts[i + 1:]) > 1e-3


def test_polar_grid_leaves_the_radial_spectrum_alone():
    """
    Only the *angular* draw changes. The set of wavelengths — and hence the
    spectral measure the amplitude prior is built from — must be identical to
    the bank we would have drawn anyway.
    """
    plain = _velocity(n_features=64)
    polar = _velocity(n_features=64, n_directions=8)
    fsf = lambda v: v.alpha if hasattr(v.alpha, "omega_spatial") else v.beta
    a = torch.sort(fsf(plain).omega_spatial.norm(dim=-1)).values
    b = torch.sort(fsf(polar).omega_spatial.norm(dim=-1)).values
    assert float((a - b).abs().max()) < 1e-12


def test_polar_grid_is_still_exactly_volume_preserving():
    """Merging commuting modes must not cost the determinant identity."""
    v = _velocity(n_features=64, n_directions=8, scale=2.0)
    _, J = FlowOffset(v, n_steps=3, integrator="shear")._run(
        _points(2), sign=-1.0, jac=True,
    )
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-11


def test_polar_grid_converges_to_the_same_flow():
    """Cheaper splitting is worthless if it integrates a different field."""
    v = _velocity(n_features=64, n_directions=8)
    x = _points(2)
    ref = FlowOffset(v, n_steps=1024, integrator="rk4")._run(x, sign=-1.0)
    errs = [
        float((FlowOffset(v, n_steps=n, integrator="shear")._run(x, sign=-1.0) - ref)
              .norm(dim=-1).max())
        for n in (8, 16, 32)
    ]
    assert errs[-1] < 1e-2
    for a, b in zip(errs, errs[1:]):
        assert 3.5 < a / b < 4.5


def test_n_directions_equal_to_the_bank_size_is_the_stratified_bank():
    """
    ``n_directions=F`` must reproduce ``stratify_directions=True`` bit for bit —
    the cycling index ``arange(F) % F`` is the identity, so anything else would
    mean the polar path had introduced a second change by accident.
    """
    fsf = lambda v: v.alpha if hasattr(v.alpha, "omega_spatial") else v.beta
    a = _velocity(n_features=64, stratify_directions=True)
    b = _velocity(n_features=64, n_directions=64)
    assert torch.equal(fsf(a).omega_spatial, fsf(b).omega_spatial)


def test_n_directions_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="n_directions"):
        _velocity(n_features=16, n_directions=17)


# ── polynomial activations ──────────────────────────────────────────────────

def test_cubic_activation_expands_to_three_parallel_harmonics():
    """
    A cubic activation keeps the mode a trigonometric polynomial: it becomes
    harmonics at 1, 2, 3 x Om_k. Crucially those are *parallel*, so they share a
    direction group and cost no sequential depth — only the term count grows.
    """
    v = _velocity(n_features=32, activation="cubic", n_directions=8)
    C, om, off = v.shear_decomposition()
    assert C.shape[0] == 3 * 32
    assert len(FlowOffset._shear_groups(om)) == 8      # unchanged by the harmonics

    x = _points(2)
    recon = (torch.cos(x @ om.T + off).unsqueeze(-1) * C).sum(1)
    ref = v(x)
    assert float((C * om).sum(-1).abs().max()) < 1e-14
    assert float((recon - ref).abs().max()) < 1e-12 * float(ref.abs().max())


def test_cubic_activation_still_integrates_exactly():
    v = _velocity(n_features=32, activation="cubic", n_directions=8)
    _, J = FlowOffset(v, n_steps=3, integrator="shear")._run(
        _points(2), sign=-1.0, jac=True,
    )
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-11


@pytest.mark.parametrize("act", ["tanh", "softplus"])
def test_non_polynomial_activations_are_refused(act):
    """These need infinitely many harmonics; say so rather than truncate."""
    v = _velocity(n_features=16, activation=act)
    with pytest.raises(NotImplementedError, match="polynomial"):
        v.shear_decomposition()


# ── covector pullback instead of the full matrix ────────────────────────────

@pytest.mark.parametrize("n_dir,n_steps", [(8, 4), (16, 8), (1, 3)])
def test_pullback_matches_the_full_jacobian(n_dir, n_steps):
    """
    ``pull(g)`` must equal ``J^T g`` exactly — it is the same product of the
    same rank-one factors, just contracted in the other order.
    """
    v = _velocity(n_features=64, n_directions=n_dir)
    x = _points(2)
    off = FlowOffset(v, n_steps=n_steps, integrator="shear")
    _, J = off.inverse_map_with_jacobian(x)
    x0, pull = off.inverse_map_with_pullback(x)

    g = torch.nn.functional.normalize(
        torch.randn(x.shape[0], 2, dtype=x.dtype), dim=-1,
    )
    ref = torch.bmm(J.mT, g.unsqueeze(-1)).squeeze(-1)
    assert float((pull(g) - ref).abs().max()) < 1e-10
    assert float((x0 - off.inverse_map(x)).abs().max()) < 1e-12


def test_pullback_falls_back_for_other_integrators():
    """The call site should not have to know which integrator is in use."""
    v = _velocity(n_features=32)
    x = _points(2)
    off = FlowOffset(v, n_steps=4, integrator="rk4")
    _, J = off.inverse_map_with_jacobian(x)
    _, pull = off.inverse_map_with_pullback(x)
    g = torch.randn(x.shape[0], 2, dtype=x.dtype)
    ref = torch.bmm(J.mT, g.unsqueeze(-1)).squeeze(-1)
    assert float((pull(g) - ref).abs().max()) < 1e-12


def test_pullback_carries_gradients_to_the_potential():
    v = _velocity(n_features=32, n_directions=8)
    _, pull = FlowOffset(v, n_steps=4, integrator="shear").inverse_map_with_pullback(
        _points(2)
    )
    g = torch.randn(40, 2, dtype=curlew.dtype)
    pull(g).pow(2).sum().backward()
    grads = [p.grad for p in v.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g_).all() for g_ in grads)
    assert max(float(g_.abs().max()) for g_ in grads) > 0.0


def test_pullback_chains_composite_factors_in_the_right_order():
    """
    ``J = J_K ... J_1`` so ``J^T = J_1^T ... J_K^T``. Factors are appended in
    execution order, so a single reversed replay must handle them — get the
    order wrong and this silently returns a different covector.
    """
    vs = [_velocity(n_features=16, n_directions=4),
          _velocity(n_features=16, n_directions=4)]
    x = _points(2)
    off = FlowOffset(vs, n_steps=[3, 5], integrator="shear")
    assert off.n_factors == 2
    _, J = off.inverse_map_with_jacobian(x)
    _, pull = off.inverse_map_with_pullback(x)
    g = torch.randn(x.shape[0], 2, dtype=x.dtype)
    ref = torch.bmm(J.mT, g.unsqueeze(-1)).squeeze(-1)
    assert float((pull(g) - ref).abs().max()) < 1e-10


@pytest.mark.parametrize("k", [1, 7, 40])
def test_jac_rows_restricts_the_pullback_without_changing_it(k):
    """
    Narrowing the pullback to a prefix must be *bit identical* to computing it
    for everything and slicing — it is the same arithmetic on fewer rows.
    """
    v = _velocity(n_features=64, n_directions=8)
    x = _points(2, n=40 if k <= 40 else k)
    off = FlowOffset(v, n_steps=5, integrator="shear")

    x_full, pull_full = off.inverse_map_with_pullback(x)
    x_pre, pull_pre = off.inverse_map_with_pullback(x, jac_rows=k)
    g = torch.randn(k, 2, dtype=x.dtype)
    pad = torch.cat([g, torch.zeros(x.shape[0] - k, 2, dtype=x.dtype)])

    assert torch.equal(x_full, x_pre)          # positions untouched
    assert float((pull_pre(g) - pull_full(pad)[:k]).abs().max()) == 0.0


def test_jac_rows_also_narrows_the_fallback_integrator():
    v = _velocity(n_features=32)
    x = _points(2)
    off = FlowOffset(v, n_steps=4, integrator="rk4")
    _, pull = off.inverse_map_with_pullback(x, jac_rows=9)
    assert pull(torch.randn(9, 2, dtype=x.dtype)).shape == (9, 2)


# ── fused Strang schedule ───────────────────────────────────────────────────

@pytest.mark.parametrize("n,m", [(1, 2), (2, 2), (1, 3), (8, 16), (4, 5)])
def test_schedule_is_the_naive_sweep_with_neighbours_fused(n, m):
    """
    Expanding the fused schedule back into half-steps must reproduce the plain
    ``g0..g_{m-1} g_{m-1}..g0`` sweep exactly — same groups, same order, same
    total time in each.
    """
    sched = FlowOffset._shear_schedule(n, m)
    expanded = [g for g, mult in sched for _ in range(int(mult))]
    order = list(range(m))
    naive = (order + order[::-1]) * n
    assert expanded == naive
    assert len(sched) == 2 * m * n - 2 * n + 1


def test_single_group_collapses_to_one_substep():
    """One group is exactly integrable, so n_steps cannot matter."""
    assert FlowOffset._shear_schedule(37, 1) == [(0, 1.0)]


# ── through the restore event ───────────────────────────────────────────────

def _mixed_event(constraints=None):
    """gp + vp + eq together, so the orientation prefix is a strict subset."""
    import numpy as np
    from curlew.core import CSet, HSet
    from curlew.geology import restore

    curlew.default_dim = 2
    rng = np.random.default_rng(0)
    gp = rng.uniform(-200, 200, (12, 2))
    gv = np.tile([0.0, 1.0], (12, 1))
    vp = rng.uniform(-200, 200, (9, 2))
    vv = np.zeros(9)
    eq = [rng.uniform(-200, 200, (11, 2)), rng.uniform(-200, 200, (7, 2))]
    C = CSet(gp=gp, gv=gv, vp=vp, vv=vv, eq=eq)
    v = euler_velocity(
        2, n_features=32, length_scale_range=(200.0, 5000.0), seed=3,
        n_directions=8,
    )
    return restore(
        "mixed", C=C, H=HSet().zero(grad_loss="1.", value_loss="1.", eq_loss="1."),
        velocity=v, n_steps=4, integrator="shear", depth_axis=1,
        init_std=0.0, pin=[0.0, 0.0], constraints=constraints,
    )


def test_orientation_prefix_does_not_disturb_the_other_loss_terms():
    """
    The Jacobian now stops after the gp/gop rows, and ``n_pred`` is split over
    those chunks alone. If that slicing were misaligned the *value* and *eq*
    terms would silently be scored against the wrong points, so pin the whole
    loss breakdown, not just the total.
    """
    event = _mixed_event()
    pebble = event.loss()
    total = pebble.total()
    assert torch.isfinite(total)
    for name in ("grad_loss", "value_loss", "eq_loss"):
        term = getattr(pebble, name, None)
        if term is not None:
            assert torch.isfinite(torch.as_tensor(term)).all()


def test_matrix_and_pullback_branches_agree():
    """
    Adding a constraint flips the code onto the full-matrix branch. Both must
    produce the same orientation loss — that is the check that the prefix
    bookkeeping is identical on either side.
    """
    from curlew.kinematics.constraints import RestorationConstraint

    class _Null(RestorationConstraint):
        """Contributes nothing, but its presence forces the matrix branch."""

        def raw(self, field, cache):
            return torch.zeros((), dtype=curlew.dtype, device=curlew.device)

    a = _mixed_event().loss().total()                      # pullback branch
    b = _mixed_event(constraints=[_Null()]).loss().total()  # full-matrix branch

    assert torch.isfinite(a) and torch.isfinite(b)
    assert float((a - b).abs()) < 1e-12


# ── tabulated profiles ──────────────────────────────────────────────────────

def test_table_converges_to_the_exact_shear():
    """
    Within a group the displacement is a function of one scalar, so it can be
    tabulated. Refining the table must recover the directly-evaluated map.
    """
    v = _velocity(n_features=64, n_directions=8)
    x = _points(2)
    exact = FlowOffset(v, n_steps=4, integrator="shear")._run(x, sign=-1.0)
    errs = []
    for T in (128, 512, 2048):
        y = FlowOffset(
            v, n_steps=4, integrator="shear", shear_table=T
        )._run(x, sign=-1.0)
        errs.append(float((y - exact).norm(dim=-1).max()))
    # points span ~4000 units, so this is ~1e-6 relative
    assert errs[-1] < 1e-2
    for a, b in zip(errs, errs[1:]):
        assert b < a                       # monotone in table resolution


@pytest.mark.parametrize("T", [128, 1024])
def test_table_keeps_volume_preservation_exact(T):
    """
    A tabulated profile is still a shear: every sample lies in dhat's orthogonal
    complement and interpolation is a convex combination, so it stays there.
    """
    v = _velocity(n_features=64, n_directions=8, scale=2.0)
    _, J = FlowOffset(
        v, n_steps=3, integrator="shear", shear_table=T
    )._run(_points(2), sign=-1.0, jac=True)
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-10


def test_table_jacobian_is_the_maps_own_derivative():
    """
    The reported slope must be the derivative of the *interpolant*, not of the
    underlying series — otherwise x and J describe different maps again.
    """
    v = _velocity(n_features=64, n_directions=8)
    x = _points(2)
    off = FlowOffset(v, n_steps=4, integrator="shear", shear_table=512)
    _, J = off._run(x, sign=-1.0, jac=True)
    xg = x.clone().requires_grad_(True)
    y = off._run(xg, sign=-1.0)
    A = torch.stack(
        [torch.autograd.grad(y[:, k].sum(), xg, retain_graph=True)[0]
         for k in range(2)], dim=1,
    )
    assert float((J - A).abs().max()) < 1e-9


def test_table_carries_gradients_and_supports_the_pullback():
    v = _velocity(n_features=64, n_directions=8)
    off = FlowOffset(v, n_steps=3, integrator="shear", shear_table=512)
    x = _points(2)
    _, J = off.inverse_map_with_jacobian(x)
    _, pull = off.inverse_map_with_pullback(x)
    g = torch.randn(x.shape[0], 2, dtype=x.dtype)
    ref = torch.bmm(J.mT, g.unsqueeze(-1)).squeeze(-1)
    assert float((pull(g) - ref).abs().max()) < 1e-9

    y, Jj = off._run(x, sign=-1.0, jac=True)
    (y.pow(2).sum() + Jj.pow(2).sum()).backward()
    grads = [p.grad for p in v.parameters() if p.grad is not None]
    assert grads and max(float(g_.abs().max()) for g_ in grads) > 0.0


def test_table_is_ignored_by_other_integrators():
    """`shear_table` is a shear-only knob; it must not silently alter rk4."""
    v = _velocity(n_features=32)
    x = _points(2)
    a = FlowOffset(v, n_steps=4, integrator="rk4")._run(x, sign=-1.0)
    b = FlowOffset(v, n_steps=4, integrator="rk4", shear_table=256)._run(x, sign=-1.0)
    assert torch.equal(a, b)


# ── guard rails ─────────────────────────────────────────────────────────────

def test_default_integrator_is_unchanged():
    """Existing notebooks must keep the flow they were fitted with."""
    assert FlowOffset(_velocity(), n_steps=4).integrator == "rk4"


def test_unknown_integrator_is_rejected_by_name():
    with pytest.raises(ValueError, match="shear"):
        FlowOffset(_velocity(), integrator="nope")


def test_velocity_without_a_decomposition_says_so():
    class _Plain(torch.nn.Module):
        dim = 2

        def forward(self, x, w=None, t=None):
            return torch.zeros_like(x)

    with pytest.raises(TypeError, match="shear_decomposition"):
        FlowOffset(_Plain(), n_steps=2, integrator="shear")._run(
            _points(2), sign=-1.0,
        )


def test_forward_and_inverse_are_reciprocal():
    """Volume exactness is no use if the pair drifts apart."""
    v = _velocity()
    x = _points(2)
    off = FlowOffset(v, n_steps=64, integrator="shear")
    assert float((off.forward_map(off.inverse_map(x)) - x).abs().max()) < 1e-2


# ── one-way sweep (n_steps = 0) ─────────────────────────────────────────────

@pytest.mark.parametrize("m", [1, 2, 5])
def test_one_way_schedule_runs_each_group_once_reversed_for_the_inverse(m):
    """``n_steps=0``: every group once at full step, in order; backwards for the inverse."""
    assert FlowOffset._shear_schedule(0, m, +1.0) == [(i, 1.0) for i in range(m)]
    assert FlowOffset._shear_schedule(0, m, -1.0) == [(i, 1.0) for i in range(m)][::-1]
    assert FlowOffset._shear_step(0, m, -1.0, 0.5) == -0.5


@pytest.mark.parametrize("dim,kw,scale", [
    (2, {}, 2.0),
    (3, {"hinge": [0.0, 1.0, 0.0]}, 2.0),
    # full 3-D (the normal / ridge split): at strong amplitudes its round trip is
    # not exact for the Strang sweep either, so test it where both are
    (3, {}, 0.05),
])
def test_one_way_sweep_is_exactly_invertible_and_volume_preserving(dim, kw, scale):
    """A product of exact shears, undone in reverse order: the round trip is the
    identity and ``det J = 1``."""
    v = _velocity(dim=dim, scale=scale, **kw)
    x = _points(dim)
    off = FlowOffset(v, n_steps=0, integrator="shear")
    back = off.forward_map(off.inverse_map(x))
    assert float((back - x).abs().max()) < 1e-8
    _, J = off._run(x, sign=-1.0, jac=True)
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-10


def test_one_way_sweep_is_not_the_strang_sweep():
    """The one-way product is a different map from the palindromic one."""
    v = _velocity(scale=2.0)
    x = _points(2)
    one_way = FlowOffset(v, n_steps=0, integrator="shear").inverse_map(x)
    strang = FlowOffset(v, n_steps=1, integrator="shear").inverse_map(x)
    assert float((one_way - strang).abs().max()) > 1.0


def test_one_way_sweep_jacobian_and_pullback_match_autograd():
    """The accumulated Jacobian and the covector pullback are the map's own derivative."""
    v = _velocity(n_features=64, n_directions=8)
    x = _points(2)
    off = FlowOffset(v, n_steps=0, integrator="shear")
    _, J = off.inverse_map_with_jacobian(x)
    xg = x.clone().requires_grad_(True)
    y = off.inverse_map(xg)
    A = torch.stack([torch.autograd.grad(y[:, k].sum(), xg, retain_graph=True)[0]
                     for k in range(2)], dim=1)
    assert float((J - A).abs().max()) < 1e-9
    _, pull = off.inverse_map_with_pullback(x)
    g = torch.randn(x.shape[0], 2, dtype=x.dtype)
    assert float((pull(g) - torch.bmm(J.mT, g.unsqueeze(-1)).squeeze(-1)).abs().max()) < 1e-10


def test_one_way_sweep_mixes_with_strang_factors_and_has_a_stability():
    """Per-factor ``n_steps`` may mix 0 and Strang; ``stability`` treats a one-way
    factor as a single step."""
    v1, v2 = _velocity(seed=3), _velocity(seed=4)
    x = _points(2)
    off = FlowOffset([v1, v2], n_steps=[0, 4], integrator="shear")
    assert float((off.forward_map(off.inverse_map(x)) - x).abs().max()) < 1e-8
    assert off.stability(x) >= 0.0


@pytest.mark.parametrize("integrator", ["rk2", "rk4"])
def test_one_way_sweep_needs_the_shear_integrator(integrator):
    with pytest.raises(ValueError, match="integrator='shear'"):
        FlowOffset(_velocity(), n_steps=0, integrator=integrator)
    off = FlowOffset(_velocity(), n_steps=2, integrator=integrator)
    with pytest.raises(ValueError, match="integrator='shear'"):
        off.n_steps = 0


def test_negative_n_steps_is_rejected():
    with pytest.raises(ValueError, match=">= 0"):
        FlowOffset(_velocity(), n_steps=-1, integrator="shear")


# ── repeated sweeps: Lie, Strang and Yoshida compositions ───────────────────

@pytest.mark.parametrize("n,m", [(1, 2), (3, 4), (2, 16)])
def test_lie_schedule_repeats_the_one_way_sweep(n, m):
    """``lie``: the one-way sweep ``n`` times at ``1/n``; reversed for the inverse."""
    assert FlowOffset._shear_schedule(n, m, +1.0, "lie") == [(i, 1.0) for i in range(m)] * n
    assert FlowOffset._shear_schedule(n, m, -1.0, "lie") == [(i, 1.0) for i in range(m)][::-1] * n
    assert FlowOffset._shear_step(n, m, -1.0, 0.5, "lie") == -0.5 / n


@pytest.mark.parametrize("n,m", [(1, 2), (2, 3), (4, 16)])
def test_yoshida_schedule_is_three_fused_palindromes_per_step(n, m):
    """
    Each group gets the full duration per step (``w1 + w0 + w1 = 1``, two half-step
    units), the schedule reads the same backwards (so the inverse is exact), and no
    two neighbours are the same group (they would have been fused).
    """
    sched = FlowOffset._shear_schedule(n, m, +1.0, "yoshida")
    for g in range(m):
        assert abs(sum(mult for i, mult in sched if i == g) - 2.0 * n) < 1e-12
    assert [i for i, _ in sched] == [i for i, _ in sched][::-1]
    assert all(a[0] != b[0] for a, b in zip(sched, sched[1:]))
    assert any(mult < 0 for _, mult in sched)                # the backward jump


def test_unknown_shear_scheme_is_rejected_by_name():
    with pytest.raises(ValueError, match="shear_scheme"):
        FlowOffset(_velocity(), n_steps=2, integrator="shear", shear_scheme="ruth")


@pytest.mark.parametrize("n_steps", [1, 4])
def test_default_scheme_is_strang_and_leaves_the_map_unchanged(n_steps):
    """Adding the option must not move a single bit of the existing behaviour."""
    v = _velocity(scale=2.0)
    x = _points(2)
    default = FlowOffset(v, n_steps=n_steps, integrator="shear")
    assert default.shear_scheme == "strang"
    explicit = FlowOffset(v, n_steps=n_steps, integrator="shear", shear_scheme="strang")
    assert torch.equal(default.inverse_map(x), explicit.inverse_map(x))


@pytest.mark.parametrize("scheme", ["strang", "lie", "yoshida"])
def test_one_way_sweep_ignores_the_scheme(scheme):
    """``n_steps=0`` is the one-way sweep whatever the scheme, and Lie at ``N = 1`` is it too."""
    v = _velocity(scale=2.0)
    x = _points(2)
    ref = FlowOffset(v, n_steps=0, integrator="shear").inverse_map(x)
    assert torch.equal(FlowOffset(v, n_steps=0, integrator="shear", shear_scheme=scheme).inverse_map(x), ref)
    assert torch.equal(FlowOffset(v, n_steps=1, integrator="shear", shear_scheme="lie").inverse_map(x), ref)


def test_old_pickles_without_a_scheme_still_run():
    """A FlowOffset saved before ``shear_scheme`` existed falls back to Strang."""
    v = _velocity()
    x = _points(2)
    off = FlowOffset(v, n_steps=3, integrator="shear")
    ref = off.inverse_map(x)
    del off.shear_scheme
    assert torch.equal(off.inverse_map(x), ref)


@pytest.mark.parametrize("scheme", ["lie", "yoshida"])
@pytest.mark.parametrize("dim,kw,scale", [
    (2, {}, 2.0),
    (3, {"hinge": [0.0, 1.0, 0.0]}, 2.0),
    (3, {}, 0.05),
])
@pytest.mark.parametrize("n_steps", [1, 3])
def test_repeated_sweeps_are_exactly_invertible_and_volume_preserving(scheme, dim, kw, scale, n_steps):
    """Every scheme is a product of exact shears: unit determinant, exact round trip."""
    v = _velocity(dim=dim, scale=scale, **kw)
    x = _points(dim)
    off = FlowOffset(v, n_steps=n_steps, integrator="shear", shear_scheme=scheme)
    assert float((off.forward_map(off.inverse_map(x)) - x).abs().max()) < 1e-8
    _, J = off._run(x, sign=-1.0, jac=True)
    assert float((torch.linalg.det(J) - 1.0).abs().max()) < 1e-10


@pytest.mark.parametrize("scheme", ["lie", "yoshida"])
def test_repeated_sweeps_jacobian_and_pullback_match_autograd(scheme):
    v = _velocity(n_features=64, n_directions=8)
    x = _points(2)
    off = FlowOffset(v, n_steps=3, integrator="shear", shear_scheme=scheme)
    _, J = off.inverse_map_with_jacobian(x)
    xg = x.clone().requires_grad_(True)
    y = off.inverse_map(xg)
    A = torch.stack([torch.autograd.grad(y[:, k].sum(), xg, retain_graph=True)[0]
                     for k in range(2)], dim=1)
    assert float((J - A).abs().max()) < 1e-9
    _, pull = off.inverse_map_with_pullback(x)
    g = torch.randn(x.shape[0], 2, dtype=x.dtype)
    assert float((pull(g) - torch.bmm(J.mT, g.unsqueeze(-1)).squeeze(-1)).abs().max()) < 1e-10


@pytest.mark.parametrize("scheme,order,ns", [
    ("lie", 1, (16, 32, 64, 128)),
    ("strang", 2, (8, 16, 32, 64)),
    ("yoshida", 4, (2, 4, 8)),
])
def test_each_scheme_converges_at_its_order_to_the_same_flow(scheme, order, ns):
    """Halving the duration divides the distance to the flow by ``2^order``."""
    v = _velocity()
    x = _points(2)
    ref = FlowOffset(v, n_steps=2048, integrator="rk4")._run(x, sign=-1.0)
    errs = []
    for n in ns:
        y = FlowOffset(v, n_steps=n, integrator="shear", shear_scheme=scheme)._run(x, sign=-1.0)
        errs.append(float((y - ref).norm(dim=-1).max()))
    for a, b in zip(errs, errs[1:]):
        assert 0.8 * 2 ** order < a / b < 1.25 * 2 ** order


def test_restore_passes_the_scheme_to_the_integrator():
    event = _mixed_event()
    assert event.field.integrator.shear_scheme == "strang"
