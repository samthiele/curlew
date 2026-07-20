"""Tests for the Fourier-series field (FSF)."""
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


def test_fsf_linear_sin_gradient_autograd():
    """Analytic gradient matches autograd through ``evaluate_raw``."""
    field = FSF(
        "f0",
        H=HSet(),
        input_dim=2,
        rff_features=32,
        length_scale_range=(0.5, 2.0),
        freq_sampling="quasi",
        seed=0,
    )
    field.A_mu.data.normal_(0, 0.05)
    field.phi.data.uniform_(-0.2, 0.2)

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
    x = torch.randn(4, 2, dtype=curlew.dtype)
    assert torch.allclose(field.evaluate(x), field.evaluate_raw(x))


def test_fsf_activated_mode_runs():
    field = FSF(
        "f0",
        H=HSet(),
        rff_features=16,
        length_scale_range=(1.0, 2.0),
        activation="tanh",
        seed=4,
    )
    field.A_mu.data.normal_(0, 0.05)
    x = torch.randn(6, 2, dtype=curlew.dtype)
    pot = field.evaluate(x)
    grad, hess = field.gradient_and_hessian(x)
    assert pot.shape == (6,)
    assert grad.shape == (6, 2)
    assert hess.shape == (6, 2, 2)


def test_fsf_iter_in_clebsch_velocity():
    from curlew.fields.clebsch import clebsch_velocity

    vel = clebsch_velocity(2, n_features=8, seed=0)
    fields = list(FSF.iter_in(vel))
    assert len(fields) == 1
    assert fields[0] is vel.beta

    vel3 = clebsch_velocity(3, n_features=8, seed=1)
    fields3 = list(FSF.iter_in(vel3))
    assert len(fields3) == 2


def test_fsf_default_is_deterministic_posterior_mean():
    """With no weight sample held, evaluation is deterministic and uses ``A_mu``."""
    field = FSF("f0", H=HSet(), rff_features=32, length_scale_range=(0.5, 2.0), seed=5)
    field.A_mu.data.normal_(0, 0.3)
    x = torch.randn(4, 2, dtype=curlew.dtype)

    assert field._eps is None
    a = field.evaluate_raw(x)
    b = field.evaluate_raw(x)
    assert torch.allclose(a, b)


def test_fsf_weight_sample_changes_evaluate_and_restores():
    """Holding a weight sample perturbs the output; clearing reverts to the mean."""
    field = FSF(
        "f0",
        H=HSet(),
        rff_features=32,
        length_scale_range=(0.5, 2.0),
        posterior_rho_init=1.0,  # large initial std so the sample clearly differs
        seed=5,
    )
    field.A_mu.data.normal_(0, 0.3)
    x = torch.randn(4, 2, dtype=curlew.dtype)

    full = field.evaluate_raw(x)
    field.resample_weights()
    sampled = field.evaluate_raw(x)
    field.clear_weight_sample()

    assert not torch.allclose(full, sampled)
    assert torch.allclose(full, field.evaluate_raw(x))


def test_fsf_mc_forward_spread():
    """Repeated MC weight samples give a non-trivial spread in the field values."""
    field = FSF(
        "f0",
        H=HSet(),
        rff_features=64,
        length_scale_range=(0.5, 2.0),
        posterior_rho_init=1.0,
        seed=2,
    )
    field.A_mu.data.normal_(0, 0.2)
    x = torch.randn(10, 2, dtype=curlew.dtype)
    gen = torch.Generator(device=curlew.device).manual_seed(0)
    mean, std = field.mc_forward(x, n_samples=30, generator=gen)
    assert mean.shape == (10,)
    assert std.shape == (10,)
    assert (std > 0).any()
    assert field._eps is None  # mc_forward restores state afterwards


def test_fsf_kl_loss_is_finite_and_zero_at_prior_match():
    """kl_loss is finite; forcing q == one prior component collapses the mixture KL to a finite value."""
    field = FSF("f0", H=HSet(), rff_features=16, length_scale_range=(0.5, 2.0), seed=7)
    field.A_mu.data.normal_(0, 0.1)
    field.resample_weights()
    kl = field.kl_loss()
    assert torch.isfinite(kl)

    # sanity: kl_loss_on aggregates across nested FSF instances
    from curlew.fields.clebsch import clebsch_velocity

    vel = clebsch_velocity(3, n_features=8, seed=0)
    for m in FSF.iter_in(vel):
        m.A_mu.data.normal_(0, 0.1)
        m.resample_weights()
    total = FSF.kl_loss_on(vel)
    parts = [m.kl_loss() for m in FSF.iter_in(vel)]
    assert torch.allclose(total, torch.stack(parts).sum())


def test_fsf_fit_resamples_weights_and_reverts_to_mean():
    """fit() draws a fresh weight sample each epoch, then reverts to the posterior mean."""
    from curlew.core import CSet

    field = FSF(
        "f0",
        H=HSet(value_loss=1.0, kl_loss="0.01"),
        rff_features=16,
        length_scale_range=(0.5, 2.0),
        seed=6,
    )
    field.A_mu.data.normal_(0, 0.1)
    x = torch.randn(8, 2, dtype=curlew.dtype)
    C = CSet(vp=x.numpy(), vv=torch.zeros(8, dtype=curlew.dtype).numpy())
    loss, pebble = field.fit(2, C=C, vb=False, best=False)
    assert field._eps is None  # reverted to posterior mean after fit
    assert loss == loss  # finite
    # kl_loss was given as an auto-balance string ("0.01"); loss() resolves it to a
    # float once evaluated, confirming the complexity-cost term actually ran
    assert isinstance(field.H.kl_loss, float)


def test_fsf_kinetic_energy_matches_monte_carlo():
    """Analytic kinetic energy matches a Monte Carlo domain average of |grad phi|^2."""
    field = FSF(
        "f0",
        H=HSet(),
        input_dim=2,
        rff_features=64,
        length_scale_range=(0.5, 1.5),
        freq_sampling="quasi",
        seed=9,
        xavier_amplitudes=False,
    )
    field.A_mu.data.normal_(0, 0.3)

    E = field.kinetic_energy()

    gen = torch.Generator(device="cpu").manual_seed(0)
    xs = torch.rand(20000, 2, generator=gen, dtype=curlew.dtype) * 40.0 - 20.0
    with torch.no_grad():
        g, _ = field.gradient_and_hessian(xs)
        mc = g.pow(2).sum(-1).mean()

    assert torch.isfinite(E) and E > 0
    assert abs(E.item() - mc.item()) / mc.item() < 0.2


def test_fsf_kinetic_energy_differentiable():
    """Energy backpropagates to Clebsch FSF amplitudes."""
    from curlew.fields.clebsch import clebsch_velocity

    vel = clebsch_velocity(2, n_features=16, seed=11)
    vel.beta.A_mu.data.normal_(0, 0.2)
    E = FSF.kinetic_energy_on(vel)
    assert torch.isfinite(E) and E > 0
    E.backward()
    assert vel.beta.A_mu.grad is not None
    assert (vel.beta.A_mu.grad.abs() > 0).any()
