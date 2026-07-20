"""Tests for Clebsch velocity fields (``curlew.fields.clebsch``)."""
import curlew
import pytest
import torch

from curlew.core import HSet
from curlew.fields.clebsch import (
    ClebschVelocity,
    clebsch_velocity,
)
from curlew.fields.series import FSF


@pytest.fixture(autouse=True)
def _clebsch_defaults():
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
        freq_sampling="quasi",
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


def test_clebsch_velocity_2d_divergence_free():
    beta = _fsf("beta", 2, seed=1)
    beta.A_mu.data.normal_(0, 0.05)
    beta.phi.data.uniform_(-0.2, 0.2)
    vel = ClebschVelocity(beta)
    x = torch.randn(8, 2, dtype=curlew.dtype)
    _, jv = vel.forward_and_jacobian(x)
    trace = jv[:, 0, 0] + jv[:, 1, 1]
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-10, rtol=1e-8)


def test_clebsch_velocity_jacobian_autograd_2d():
    beta = _fsf("beta", 2, seed=2)
    beta.A_mu.data.normal_(0, 0.08)
    vel = ClebschVelocity(beta)
    x = torch.randn(6, 2, dtype=curlew.dtype)
    v, jv = vel.forward_and_jacobian(x)
    v_auto, jv_auto = _autograd_velocity_jacobian(vel, x)
    assert torch.allclose(v, v_auto, atol=1e-10, rtol=1e-7)
    assert torch.allclose(jv, jv_auto, atol=1e-9, rtol=1e-6)


def test_clebsch_velocity_3d_divergence_free():
    beta = _fsf("beta", 3, seed=3)
    alpha = _fsf("alpha", 3, seed=4)
    beta.A_mu.data.normal_(0, 0.05)
    alpha.A_mu.data.normal_(0, 0.05)
    vel = ClebschVelocity(beta, alpha)
    x = torch.randn(5, 3, dtype=curlew.dtype)
    _, jv = vel.forward_and_jacobian(x)
    trace = jv.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-9, rtol=1e-7)


def test_clebsch_velocity_factory():
    vel = clebsch_velocity(2, n_features=16, length_scale_range=(1.0, 2.0), seed=5)
    vel.beta.A_mu.data.normal_(0, 0.1)
    x = torch.randn(4, 2, dtype=curlew.dtype)
    v = vel(x)
    assert v.shape == (4, 2)


def test_clebsch_velocity_requires_alpha_in_3d():
    beta = _fsf("beta", 3)
    with pytest.raises(ValueError, match="alpha"):
        ClebschVelocity(beta)
