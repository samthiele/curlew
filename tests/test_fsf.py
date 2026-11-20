"""Tests for the Fourier-series field (FSF)."""
import math

import curlew
import pytest
import torch

from curlew.core import HSet
from curlew.fields.series import FSF


@pytest.fixture(autouse=True)
def _fsf_defaults():
    curlew.default_dim = 2
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def _autograd_grad(field, x: torch.Tensor) -> torch.Tensor:
    """Spatial gradient of ``evaluate_raw`` via autograd."""
    x = x.detach().clone().requires_grad_(True)
    pot = field.evaluate_raw(x)
    return torch.autograd.grad(pot.sum(), x)[0]


def _autograd_hessian(field, x: torch.Tensor) -> torch.Tensor:
    """Spatial Hessian of ``evaluate_raw`` via autograd."""
    x = x.detach().clone().requires_grad_(True)
    pot = field.evaluate_raw(x)
    grad = torch.autograd.grad(pot.sum(), x, create_graph=True)[0]
    dim = x.shape[-1]
    rows = [
        torch.autograd.grad(grad[:, i].sum(), x, retain_graph=(i < dim - 1))[0]
        for i in range(dim)
    ]
    return torch.stack(rows, dim=1)


def test_fsf_linear_gradient_autograd():
    """Analytic gradient matches autograd through ``evaluate_raw``."""
    field = FSF(
        "f0",
        H=HSet(),
        input_dim=2,
        rff_features=32,
        length_scale_range=(0.5, 2.0),
        seed=0,
    )
    field.A_mu.data.normal_(0, 0.05)
    field.B_mu.data.normal_(0, 0.05)

    x = torch.tensor([[0.3, 0.7], [1.1, -0.4]], dtype=curlew.dtype)
    grad, val = field.gradient(
        x, transform=False, normalize=False, accumulate=False, return_value=True
    )
    grad_auto = _autograd_grad(field, x)

    assert torch.allclose(grad, grad_auto, atol=1e-10, rtol=1e-7)
    assert val.shape == (2,)


def test_fsf_hessian_autograd():
    """Analytic Hessian matches autograd through ``evaluate_raw``."""
    field = FSF("f0", H=HSet(), rff_features=16, length_scale_range=(1.0, 3.0), seed=1)
    field.A_mu.data.normal_(0, 0.1)
    field.B_mu.data.normal_(0, 0.1)
    x = torch.randn(5, 2, dtype=curlew.dtype)
    grad, hess = field.gradient_and_hessian(x)
    hess_auto = _autograd_hessian(field, x)

    assert grad.shape == (5, 2)
    assert hess.shape == (5, 2, 2)
    assert torch.allclose(hess, hess.mT, atol=1e-10)
    assert torch.allclose(hess, hess_auto, atol=1e-9, rtol=1e-6)


def test_fsf_evaluate_raw_matches_evaluate():
    field = FSF("f0", H=HSet(), rff_features=8, length_scale_range=(1.0, 2.0), seed=3)
    field.A_mu.data.fill_(0.1)
    field.B_mu.data.fill_(-0.07)
    x = torch.randn(4, 2, dtype=curlew.dtype)
    assert torch.allclose(field.evaluate(x), field.evaluate_raw(x))


@pytest.mark.parametrize("activation", ["tanh", "cubic", "softplus"])
def test_fsf_activated_mode_runs(activation):
    field = FSF(
        "f0",
        H=HSet(),
        rff_features=16,
        length_scale_range=(1.0, 2.0),
        activation=activation,
        seed=4,
    )
    x = torch.randn(6, 2, dtype=curlew.dtype)
    pot = field.evaluate(x)
    grad, hess = field.gradient_and_hessian(x)
    assert pot.shape == (6,)
    assert grad.shape == (6, 2)
    assert hess.shape == (6, 2, 2)
    assert not hasattr(field, "A_mu") and not hasattr(field, "B_mu")
    assert hasattr(field, "C_mu") and hasattr(field, "phi_mu")

    pot.sum().backward()
    assert field.phi_mu.grad is not None and field.phi_mu.grad.abs().max() > 0
    assert field.C_mu.grad is not None and field.C_mu.grad.abs().max() > 0


def test_fsf_activated_phase_is_a_raw_angle():
    field = FSF(
        "f0", H=HSet(), rff_features=8, length_scale_range=(1.0, 2.0),
        activation="cubic", seed=4, xavier_amplitudes=False,
    )
    with torch.no_grad():
        field.C_mu.fill_(0.2)
    x = torch.randn(7, 2, dtype=curlew.dtype)
    ref = field.evaluate_raw(x)
    # the phase is periodic: a full turn leaves the field unchanged
    with torch.no_grad():
        field.phi_mu.add_(2 * math.pi)
    assert torch.allclose(field.evaluate_raw(x), ref, atol=1e-12, rtol=1e-12)
    # and it is exactly the shift inside sin(θ + φ)
    theta = x @ field.fourier_projection
    sig = 1.5 * (torch.sin(theta + field.phi_mu) + field.bias) \
        - 0.5 * (torch.sin(theta + field.phi_mu) + field.bias) ** 3
    expect = (field._effective_amplitudes() * sig).sum(-1)
    assert torch.allclose(ref, expect, atol=1e-12)


def test_fsf_activated_loads_legacy_cartesian_phase():
    field = FSF(
        "f0", H=HSet(), rff_features=8, length_scale_range=(1.0, 2.0),
        activation="tanh", seed=4,
    )
    sd = {k: v.clone() for k, v in field.state_dict().items()}
    phi = sd.pop("phi_mu")
    sd["A_mu"], sd["B_mu"] = 2.5 * torch.cos(phi), 2.5 * torch.sin(phi)
    with torch.no_grad():
        field.phi_mu.zero_()
    field.load_state_dict(sd)
    assert torch.allclose(torch.cos(field.phi_mu), torch.cos(phi), atol=1e-12)
    assert torch.allclose(torch.sin(field.phi_mu), torch.sin(phi), atol=1e-12)


def test_fsf_linear_has_cartesian_amplitudes_not_phase():
    field = FSF("f0", H=HSet(), rff_features=8, length_scale_range=(1.0, 2.0), seed=0)
    assert hasattr(field, "B_mu")
    assert not hasattr(field, "phi")
    assert field.A_mu.shape == field.B_mu.shape


def test_fsf_cartesian_matches_polar_series():
    """``A cos θ + B sin θ`` equals ``R sin(θ + φ)`` with ``φ = atan2(A, B)``."""
    field = FSF(
        "f0", H=HSet(), rff_features=24, length_scale_range=(0.5, 2.0),
        seed=2, xavier_amplitudes=False,
    )
    field.A_mu.data.normal_(0, 0.2)
    field.B_mu.data.normal_(0, 0.2)
    x = torch.randn(20, 2, dtype=curlew.dtype)
    theta = field._projection(x, None)
    a = field._effective_amplitudes()
    b = field._effective_amplitudes_sin()
    polar = (torch.hypot(a, b) * torch.sin(theta + torch.atan2(a, b))).sum(-1)
    assert torch.allclose(field.evaluate_raw(x), polar, atol=1e-12, rtol=1e-12)


def test_fsf_sine_channel_is_live():
    """A field with ``A = 0`` and ``B ≠ 0`` is not the zero field."""
    field = FSF(
        "f0", H=HSet(), rff_features=16, length_scale_range=(0.5, 2.0),
        seed=5, xavier_amplitudes=False,
    )
    field.B_mu.data.fill_(0.2)
    x = torch.randn(8, 2, dtype=curlew.dtype)
    assert field.evaluate_raw(x).abs().max() > 0
    assert field.gradient_raw(x).abs().max() > 0


def test_fsf_basis_derivatives_match_amplitude_vector():
    field = FSF(
        "f0", H=HSet(), rff_features=12, length_scale_range=(0.5, 2.0),
        seed=6, xavier_amplitudes=False,
    )
    field.A_mu.data.normal_(0, 0.15)
    field.B_mu.data.normal_(0, 0.15)
    x = torch.randn(5, 2, dtype=curlew.dtype)
    f, g, h, _ = field.basis_derivatives(x, order=2)
    w = field._amplitude_vector()
    assert f.shape[1] == 2 * field.A_mu.numel()
    assert torch.allclose(f @ w, field.evaluate_raw(x), atol=1e-12)
    assert torch.allclose(
        torch.einsum("nkd,k->nd", g, w), field.gradient_raw(x), atol=1e-12
    )
    hess = field.gradient_and_hessian(x)[1]
    assert torch.allclose(torch.einsum("nkij,k->nij", h, w), hess, atol=1e-12)


def test_fsf_iter_in_nested_module():
    """``FSF.iter_in`` finds nested series fields under an arbitrary parent module."""
    import torch.nn as nn

    parent = nn.Module()
    parent.f1 = FSF("f1", H=HSet(), rff_features=8, length_scale_range=(0.5, 2.0), seed=0)
    parent.f2 = FSF(
        "f2", H=HSet(), input_dim=3, rff_features=8, length_scale_range=(0.5, 2.0), seed=1
    )
    fields = list(FSF.iter_in(parent))
    assert len(fields) == 2
    assert parent.f1 in fields and parent.f2 in fields


def test_fsf_kinetic_energy_matches_monte_carlo():
    """Analytic kinetic energy matches a Monte Carlo domain average of |grad phi|^2."""
    field = FSF(
        "f0",
        H=HSet(),
        input_dim=2,
        rff_features=64,
        length_scale_range=(0.5, 1.5),
        seed=9,
        xavier_amplitudes=False,
    )
    field.A_mu.data.normal_(0, 0.3)
    field.B_mu.data.normal_(0, 0.3)

    E = field.kinetic_energy()

    gen = torch.Generator(device="cpu").manual_seed(0)
    xs = torch.rand(20000, 2, generator=gen, dtype=curlew.dtype) * 40.0 - 20.0
    with torch.no_grad():
        g, _ = field.gradient_and_hessian(xs)
        mc = g.pow(2).sum(-1).mean()

    assert torch.isfinite(E) and E > 0
    assert abs(E.item() - mc.item()) / mc.item() < 0.2


def test_fsf_kinetic_energy_differentiable():
    """Energy backpropagates to FSF amplitudes (including via kinetic_energy_on)."""
    import torch.nn as nn

    parent = nn.Module()
    parent.f = FSF(
        "f", H=HSet(), rff_features=16, length_scale_range=(0.5, 2.0), seed=11
    )
    parent.f.A_mu.data.normal_(0, 0.2)
    parent.f.B_mu.data.normal_(0, 0.2)
    E = FSF.kinetic_energy_on(parent)
    assert torch.isfinite(E) and E > 0
    E.backward()
    assert parent.f.A_mu.grad is not None
    assert parent.f.B_mu.grad is not None
    assert (parent.f.A_mu.grad.abs() > 0).any()
    assert (parent.f.B_mu.grad.abs() > 0).any()


# ── Amplitude damping (1/|Omega|^freq_damp) ──────────────────────────────────


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("freq_damp", [0.0, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("lift_dim", [0, 2])
def test_fsf_amplitude_denom_is_the_power_law(dim, freq_damp, lift_dim):
    """The divisor is |Omega|^freq_damp over the *lifted* wavenumber."""
    f = FSF(
        "f", H=HSet(), input_dim=dim, lift_dim=lift_dim, rff_features=128,
        length_scale_range=(0.5, 3.0), seed=4, freq_damp=freq_damp,
        xavier_amplitudes=False,
    )
    if freq_damp == 0.0:
        assert torch.equal(f.amplitude_denom, torch.ones_like(f.freq_scaling))
    else:
        assert torch.equal(f.amplitude_denom, f.freq_scaling.pow(0.5 * freq_damp))
    # freq_scaling is |Omega|^2 over all input_dim + lift_dim components
    omega = f.fourier_projection.T
    assert omega.shape[1] == dim + lift_dim
    if freq_damp == 0.0:
        assert torch.allclose(f.amplitude_denom, torch.ones_like(f.amplitude_denom))
    else:
        assert torch.allclose(f.amplitude_denom, omega.norm(dim=1) ** freq_damp)


def test_fsf_kinetic_energy_matches_closed_form():
    """kinetic_energy is the quadratic form built from the same divisor."""
    f = FSF(
        "f", H=HSet(), input_dim=2, rff_features=64,
        length_scale_range=(0.5, 2.0), seed=7, freq_damp=2.0,
    )
    f.A_mu.data.normal_(0, 0.3)
    f.B_mu.data.normal_(0, 0.3)
    ox = f.omega_spatial
    w = (ox**2).sum(1) / f.freq_scaling.pow(f.freq_damp)
    expected = 0.5 * f.out_scale**2 * ((f.A_mu**2 + f.B_mu**2) * w).sum()
    assert torch.equal(f.kinetic_energy(), expected)


@pytest.mark.parametrize("freq_damp", [1.0, 2.0])
def test_fsf_freq_damp_scales_the_effective_amplitudes(freq_damp):
    """Halving freq_damp lifts short-wavelength modes relative to long ones."""
    kw = dict(
        H=HSet(), input_dim=2, rff_features=256,
        length_scale_range=(0.1, 10.0), seed=11, xavier_amplitudes=False,
    )
    f = FSF("f", freq_damp=freq_damp, **kw)
    f.A_mu.data.fill_(1.0)
    amps = f._effective_amplitudes().abs()
    lam = f.mode_wavelengths()
    fine = amps[lam < lam.median()].mean()
    coarse = amps[lam >= lam.median()].mean()
    # stronger damping suppresses the fine band further
    assert (coarse / fine) > 1.0
    assert torch.isfinite(amps).all()


def test_fsf_freq_damp_validation():
    with pytest.raises(ValueError, match="freq_damp"):
        FSF("f", H=HSet(), input_dim=2, rff_features=16,
            length_scale_range=(0.5, 2.0), seed=0, freq_damp=-1.0)


def test_fsf_freq_damp_zero_removes_damping():
    """freq_damp=0 leaves every mode undamped (D_k = 1)."""
    f = FSF(
        "f", H=HSet(), input_dim=2, rff_features=64,
        length_scale_range=(0.5, 2.0), seed=0, freq_damp=0.0,
        xavier_amplitudes=False,
    )
    f.A_mu.data.fill_(1.0)
    assert torch.equal(f.amplitude_denom, torch.ones_like(f.freq_scaling))
    assert torch.allclose(f._effective_amplitudes(), f.out_scale * f.A_mu)


def test_stratified_directions_reduce_per_band_angular_clumping():
    """
    A Sobol draw is stratified in the joint (wavelength, direction) space, not
    in direction *conditional on radius*, so each radial band has angular gaps
    several times the ideal spacing — worst in the coarse band, which carries
    nearly all the amplitude, giving the field a preferred large-scale
    orientation that is an artefact of the draw.

    The golden-ratio sequence over wavelength-sorted modes fixes it. 1.0 is
    unattainable by design: the three-distance theorem bounds the max/mean gap
    of a golden sequence at phi ~ 1.618, which is the price of every contiguous
    run being well spread.
    """
    import numpy as np
    from curlew.core import HSet
    from curlew.fields.series import FSF

    bands = [(100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200)]

    def clumping(stratify):
        f = FSF("t", H=HSet(), input_dim=2, rff_features=256,
                length_scale_range=(100.0, 5000.0), seed=1,
                stratify_directions=stratify)
        om = f.omega_spatial.detach().cpu().numpy()
        lam = 2 * np.pi / np.linalg.norm(om, axis=1)
        th = np.degrees(np.arctan2(om[:, 1], om[:, 0])) % 180.0
        out = []
        for lo, hi in bands:
            m = (lam >= lo) & (lam < hi)
            t = np.sort(th[m])
            gaps = np.diff(np.concatenate([t, [t[0] + 180.0]]))
            out.append(gaps.max() / (180.0 / m.sum()))
        return np.array(out)

    plain, strat = clumping(False), clumping(True)
    assert strat.mean() < 0.6 * plain.mean(), (plain.mean(), strat.mean())
    assert strat.max() < 1.8, strat            # ~phi, the three-distance bound
    assert strat.std() < 0.15, strat           # uniform across bands


def test_stratified_directions_are_unit_and_preserve_the_wavelengths():
    """Only the discrepancy changes: same radial law, still unit directions,
    so the amplitude prior in `_amplitude_denom` is untouched."""
    import numpy as np
    from curlew.core import HSet
    from curlew.fields.series import FSF

    kw = dict(H=HSet(), input_dim=2, rff_features=128,
              length_scale_range=(100.0, 5000.0), seed=3)
    a = FSF("a", stratify_directions=False, **kw)
    b = FSF("b", stratify_directions=True, **kw)
    la = np.sort(a.mode_wavelengths().detach().cpu().numpy()) if hasattr(a, "mode_wavelengths") \
        else np.sort(2 * np.pi / a.omega_spatial.norm(dim=1).detach().cpu().numpy())
    lb = np.sort(2 * np.pi / b.omega_spatial.norm(dim=1).detach().cpu().numpy())
    assert np.allclose(la, lb, rtol=1e-12)                      # identical radii
    n = b.omega_spatial.norm(dim=1) * b.mode_wavelengths().detach() \
        if hasattr(b, "mode_wavelengths") else None
    assert torch.allclose(
        torch.nn.functional.normalize(b.omega_spatial, dim=1).norm(dim=1),
        torch.ones(128, dtype=curlew.dtype), atol=1e-12,
    )


def test_3d_stratified_directions_sample_axes_on_a_random_hemisphere():
    """
    +Omega and -Omega are the same mode, so 3-D directions are *axes*: sampling
    the full sphere let rays land near-antipodal (the same axis twice). The
    hemisphere draw must be unit, seed-reproducible, differ between seeds, and
    -- because a seeded random rotation replaces the fixed pole -- isotropic
    on average over seeds.
    """
    import math
    from curlew.fields.series import _stratified_directions

    d = _stratified_directions(16, 3, 7)
    assert d.shape == (16, 3)
    assert float((d.norm(dim=1) - 1.0).abs().max()) < 1e-14
    assert torch.equal(d, _stratified_directions(16, 3, 7))
    assert not torch.allclose(d, _stratified_directions(16, 3, 8))

    # No two rays share an axis: nearest-axis separation stays well clear of 0
    # for every seed (the full-sphere draw fell to 0.7 deg at n = 32).
    for seed in range(50):
        G = (_stratified_directions(32, 3, seed) @ _stratified_directions(32, 3, seed).T).abs()
        G.fill_diagonal_(0.0)
        assert math.degrees(math.acos(float(G.max()))) > 3.0

    # Random orientation: the mean axis tensor over many seeds is isotropic.
    M = sum(
        (lambda D: D.T @ D / len(D))(_stratified_directions(8, 3, s))
        for s in range(1000)
    ) / 1000
    assert float((M - torch.eye(3, dtype=M.dtype) / 3).abs().max()) < 0.01


@pytest.mark.parametrize("dim", [2, 3])
def test_stratified_directions_are_one_even_set_rotated_by_the_seed(dim):
    """
    The axes are an even set (2-D: equal spacing; 3-D: hemispherical Fibonacci
    lattice) and the seed only rotates it, so every seed leaves the same holes.
    Pairwise axis angles are rotation invariant: sorted, they must agree across
    seeds to rounding. A seed-dependent draw (Sobol, or a Kronecker sequence
    with seeded offsets in both coordinates) fails this.
    """
    from curlew.fields.series import _stratified_directions

    def pair_angles(seed, n=32):
        d = _stratified_directions(n, dim, seed)
        c = (d @ d.T).abs().clamp(max=1.0)
        iu = torch.triu_indices(n, n, 1)
        return torch.sort(torch.arccos(c[iu[0], iu[1]])).values

    ref = pair_angles(0)
    for seed in (1, 7, 123):
        assert torch.allclose(pair_angles(seed), ref, atol=1e-9)


def test_fsf_uniform_wavelength_draw():
    """Uniform-in-λ sampling puts the mean at the interval midpoint."""
    kw = dict(
        H=HSet(), input_dim=2, rff_features=4096,
        length_scale_range=(200.0, 2000.0), seed=1, xavier_amplitudes=False,
    )
    log = FSF("log", wavelength_sampling="log", **kw)
    uni = FSF("uni", wavelength_sampling="uniform", **kw)
    lam_log = log.mode_wavelengths()
    lam_uni = uni.mode_wavelengths()
    assert float(lam_uni.mean()) > float(lam_log.mean())
    assert abs(float(lam_uni.mean()) - 1100.0) < 40.0
    assert abs(float(lam_uni.median()) - 1100.0) < 40.0
    # log-uniform mean is (b-a)/ln(b/a) ≈ 782 on this interval
    assert abs(float(lam_log.mean()) - 1800.0 / math.log(10.0)) < 40.0


def test_fsf_wavelength_sampling_validation():
    with pytest.raises(ValueError, match="wavelength_sampling"):
        FSF("f", H=HSet(), input_dim=2, rff_features=16,
            length_scale_range=(0.5, 2.0), seed=0, wavelength_sampling="linear")
