"""Tests for TemporalFSF and temporal Clebsch velocities."""
import curlew
import pytest
import torch

from curlew.core import HSet
from curlew.fields.clebsch import temporal_clebsch_velocity
from curlew.fields.series import TemporalFSF


@pytest.fixture(autouse=True)
def _defaults():
    curlew.default_dim = 2
    curlew.device = "cpu"
    curlew.dtype = torch.float64


def _autograd_grad(field: TemporalFSF, x: torch.Tensor, t: float) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    pot = field.evaluate_raw(x, t=t)
    return torch.autograd.grad(pot.sum(), x)[0]


def _autograd_hessian(field: TemporalFSF, x: torch.Tensor, t: float) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    pot = field.evaluate_raw(x, t=t)
    grad = torch.autograd.grad(pot.sum(), x, create_graph=True)[0]
    dim = x.shape[-1]
    rows = [
        torch.autograd.grad(grad[:, i].sum(), x, retain_graph=(i < dim - 1))[0]
        for i in range(dim)
    ]
    return torch.stack(rows, dim=1)


def test_temporal_fsf_cutoff_sweep():
    field = TemporalFSF(
        "t0",
        H=HSet(),
        input_dim=2,
        rff_features=32,
        length_scale_range=(0.5, 4.0),
        freq_sampling="quasi",
        seed=0,
    )
    assert float(field.cutoff_wavelength(0.0)) == pytest.approx(4.0)
    assert float(field.cutoff_wavelength(1.0)) == pytest.approx(0.5)
    g0 = field.spectral_gates(0.0)
    g1 = field.spectral_gates(1.0)
    assert g1.mean() > g0.mean()
    assert g1.min() >= g0.min()


def test_temporal_fsf_reverse_cutoff():
    field = TemporalFSF(
        "rev",
        H=HSet(),
        input_dim=2,
        rff_features=32,
        length_scale_range=(0.5, 4.0),
        freq_sampling="quasi",
        seed=1,
        spectral_gate="reverse",
    )
    assert float(field.cutoff_wavelength(0.0)) == pytest.approx(0.5)
    assert float(field.cutoff_wavelength(1.0)) == pytest.approx(4.0)
    g0 = field.spectral_gates(0.0)
    g1 = field.spectral_gates(1.0)
    assert g0.mean() > g1.mean()


def test_temporal_fsf_bandpass_sweep_closes_long_modes():
    field = TemporalFSF(
        "bp",
        H=HSet(),
        input_dim=2,
        rff_features=128,
        length_scale_range=(0.5, 4.0),
        freq_sampling="quasi",
        seed=2,
        spectral_gate="bandpass",
        spectral_band_width=0.4,
    )
    ox = field._omega_spatial()
    long_modes = ox.norm(dim=1) < ox.norm(dim=1).median()
    g0 = field.spectral_gates(0.0)
    g1 = field.spectral_gates(1.0)
    assert float(g0[long_modes].mean()) > float(g1[long_modes].mean())
    short_modes = ~long_modes
    assert float(g1[short_modes].mean()) > float(g0[short_modes].mean())


def test_temporal_fsf_expected_energy_ramps_with_t():
    """With open gates only (no c(t)), expected gradient energy grows toward t=1."""
    field = TemporalFSF(
        "t0",
        H=HSet(),
        input_dim=2,
        rff_features=64,
        length_scale_range=(0.3, 3.0),
        freq_sampling="quasi",
        seed=2,
    )
    e0 = float((field.spectral_gates(0.0) ** 2 * field.energy_w).sum())
    e1 = float((field.spectral_gates(1.0) ** 2 * field.energy_w).sum())
    assert e1 > e0


def test_temporal_fsf_velocity_increases_with_t():
    """Mean |v| should increase from t=0 toward t=1 as spectral gates open."""
    vel = temporal_clebsch_velocity(
        2,
        n_features=512,
        length_scale_range=(0.3, 3.0),
        seed=5,
        init_std=0.1,
    )
    torch.manual_seed(0)
    vel.beta.set_amplitudes(torch.abs(torch.randn_like(vel.beta.A_rho)) * 0.1)
    gx, gy = torch.meshgrid(
        torch.linspace(-2, 2, 25, dtype=curlew.dtype),
        torch.linspace(-2, 2, 25, dtype=curlew.dtype),
        indexing="ij",
    )
    x = torch.stack([gx.ravel(), gy.ravel()], dim=-1)
    with torch.no_grad():
        m0, m05, m1 = [
            float(vel(x, t=t).norm(dim=-1).mean()) for t in (0.0, 0.5, 1.0)
        ]
    assert m1 > m0
    assert m05 > m0


def test_temporal_fsf_lower_freq_damp_boosts_high_modes():
    """freq_damp=1 weakens |Ω|^{-2} damping relative to the default freq_damp=2."""
    common = dict(
        H=HSet(),
        input_dim=2,
        rff_features=128,
        length_scale_range=(0.3, 3.0),
        freq_sampling="quasi",
        seed=3,
        init_std=0.0,
    )
    field_strong = TemporalFSF("s", **common, freq_damp=2.0)
    field_weak = TemporalFSF("w", **common, freq_damp=1.0)
    ox = field_strong._omega_spatial()
    high = ox.norm(dim=1) > ox.norm(dim=1).median()
    field_strong.set_amplitudes(torch.empty_like(field_strong.A_rho).uniform_(0.05, 0.15))
    field_weak.set_amplitudes(field_strong.A.detach())
    x = torch.randn(20, 2, dtype=curlew.dtype)
    with torch.no_grad():
        g_strong = field_strong.gradient_and_hessian(x, t=1.0)[0].norm(dim=-1).mean()
        g_weak = field_weak.gradient_and_hessian(x, t=1.0)[0].norm(dim=-1).mean()
    assert float(g_weak) > float(g_strong)
    w2 = field_strong.energy_w[high].mean()
    w1 = field_weak.energy_w[high].mean()
    assert float(w1) > float(w2)


def test_temporal_fsf_kinetic_energy():
    """Path-averaged energy integrates E(t); gradients flow to A."""
    from curlew.fields.series import FSF

    field = TemporalFSF(
        "t0",
        H=HSet(),
        input_dim=2,
        rff_features=48,
        length_scale_range=(0.3, 3.0),
        freq_sampling="quasi",
        seed=6,
    )
    field.set_amplitudes(0.7)

    e_path = field.kinetic_energy()
    e0 = field.kinetic_energy(t=0.0)
    e1 = field.kinetic_energy(t=1.0)
    assert float(e1) > float(e0)
    assert float(e0) <= float(e_path) <= float(e1)

    e = field.kinetic_energy()
    (grad,) = torch.autograd.grad(e, field.A_rho)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0

    vel = temporal_clebsch_velocity(
        2, n_features=16, length_scale_range=(0.5, 2.0), seed=7, init_std=0.1
    )
    total = FSF.kinetic_energy_on(vel)
    assert total is not None and float(total) > 0


def test_temporal_fsf_gradient_hessian_autograd():
    field = TemporalFSF(
        "t0",
        H=HSet(),
        input_dim=2,
        rff_features=24,
        length_scale_range=(0.4, 3.0),
        freq_sampling="quasi",
        activation="cubic",
        seed=1,
    )
    field.set_amplitudes(torch.abs(torch.randn_like(field.A_rho)) * 0.05)
    field.phi.data.uniform_(-0.2, 0.2)
    field.bias.data.uniform_(-0.1, 0.1)

    x = torch.randn(12, 2, dtype=curlew.dtype)
    for t in (0.0, 0.35, 1.0):
        grad, hess = field.gradient_and_hessian(x, t=t)
        grad_auto = _autograd_grad(field, x, t)
        hess_auto = _autograd_hessian(field, x, t)
        assert torch.allclose(grad, grad_auto, atol=1e-9, rtol=1e-6)
        assert torch.allclose(hess, hess_auto, atol=1e-8, rtol=1e-6)


def test_temporal_clebsch_divergence_free():
    vel = temporal_clebsch_velocity(
        2,
        n_features=20,
        length_scale_range=(0.5, 2.5),
        seed=3,
        init_std=0.05,
        activation="cubic",
    )
    vel.beta.set_amplitudes(torch.abs(torch.randn_like(vel.beta.A_rho)) * 0.08)
    x = torch.randn(15, 2, dtype=curlew.dtype)
    for t in (0.0, 0.5, 1.0):
        _, jv = vel.forward_and_jacobian(x, t=t)
        div = jv[:, 0, 0] + jv[:, 1, 1]
        assert torch.allclose(div, torch.zeros_like(div), atol=1e-9, rtol=1e-6)


def test_temporal_clebsch_time_varying():
    vel = temporal_clebsch_velocity(
        2,
        n_features=32,
        length_scale_range=(0.3, 3.0),
        seed=4,
        init_std=0.08,
    )
    vel.beta.set_amplitudes(torch.abs(torch.randn_like(vel.beta.A_rho)) * 0.1)
    x = torch.randn(8, 2, dtype=curlew.dtype)
    v0 = vel(x, t=0.0)
    v1 = vel(x, t=1.0)
    assert not torch.allclose(v0, v1, atol=1e-12)


def test_temporal_fsf_learnable_directions_default():
    """Directions are learnable by default; |ω| stays frozen under θ updates."""
    field = TemporalFSF(
        "dirs",
        H=HSet(),
        input_dim=2,
        rff_features=16,
        length_scale_range=(0.5, 2.0),
        freq_sampling="quasi",
        seed=9,
    )
    assert field.learnable_directions is True
    assert isinstance(field.theta, torch.nn.Parameter)
    assert field.theta in set(field.parameters())

    mag0 = field._omega_spatial().norm(dim=1).detach().clone()
    gates0 = field.spectral_gates(0.5).detach().clone()
    field.theta.data += 0.35
    mag1 = field._omega_spatial().norm(dim=1)
    gates1 = field.spectral_gates(0.5)
    assert torch.allclose(mag0, mag1, atol=1e-12, rtol=1e-10)
    assert torch.allclose(gates0, gates1, atol=1e-12, rtol=1e-10)

    # Gradients reach θ through the analytic series.
    field.set_amplitudes(0.2)
    x = torch.randn(6, 2, dtype=curlew.dtype)
    pot = field.evaluate_raw(x, t=1.0).sum()
    (g_theta,) = torch.autograd.grad(pot, field.theta)
    assert torch.isfinite(g_theta).all() and g_theta.abs().sum() > 0


def test_temporal_fsf_learnable_directions_off():
    field = TemporalFSF(
        "fixed",
        H=HSet(),
        input_dim=2,
        rff_features=12,
        length_scale_range=(0.5, 2.0),
        freq_sampling="quasi",
        seed=10,
        learnable_directions=False,
    )
    assert field.learnable_directions is False
    assert not hasattr(field, "theta")
    names = {n for n, _ in field.named_parameters()}
    assert "theta" not in names
    assert "dir_raw" not in names


def test_temporal_fsf_learnable_directions_3d():
    field = TemporalFSF(
        "d3",
        H=HSet(),
        input_dim=3,
        rff_features=10,
        length_scale_range=(0.5, 2.0),
        freq_sampling="quasi",
        seed=11,
    )
    assert isinstance(field.dir_raw, torch.nn.Parameter)
    ox = field._omega_spatial()
    assert ox.shape == (10, 3)
    assert torch.allclose(
        ox.norm(dim=1), field.spatial_omega_mag, atol=1e-12, rtol=1e-10
    )


def test_temporal_fsf_positive_amplitudes():
    """A = softplus(A_rho) ≥ 0; signed sets fold polarity into φ."""
    field = TemporalFSF(
        "pos",
        H=HSet(),
        input_dim=2,
        rff_features=8,
        length_scale_range=(0.5, 2.0),
        freq_sampling="quasi",
        seed=12,
        learnable_directions=False,
    )
    assert isinstance(field.A_rho, torch.nn.Parameter)
    assert "A" not in {n for n, _ in field.named_parameters()}
    assert torch.all(field.A >= 0)

    field.set_amplitudes(0.3)
    assert torch.allclose(field.A, torch.full_like(field.A, 0.3), atol=1e-6)

    with pytest.raises(ValueError, match="non-negative"):
        field.set_amplitudes(-0.1)

    # Negative magnitude ≡ phase shift by π for a linear sinusoid
    field2 = TemporalFSF(
        "pos2",
        H=HSet(),
        input_dim=2,
        rff_features=8,
        length_scale_range=(0.5, 2.0),
        freq_sampling="quasi",
        seed=12,
        learnable_directions=False,
        activation=None,
    )
    field2.set_amplitudes(0.25)
    x = torch.randn(16, 2, dtype=curlew.dtype)
    y_pos = field2.evaluate_raw(x, t=1.0).detach().clone()
    field2.set_amplitudes(-0.25, signed=True)
    y_neg = field2.evaluate_raw(x, t=1.0)
    assert torch.allclose(y_neg, -y_pos, atol=1e-10, rtol=1e-8)
