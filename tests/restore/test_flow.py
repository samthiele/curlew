"""RK2 / RK4 contract for :class:`~curlew.kinematics.flow.FlowOffset`.

The default integrator is RK4. Its reported Jacobian is the derivative of
the discrete step, and ``stability`` reads the injectivity margin of that
step (``||D Phi_h - I||_2``). Exact volume preservation is the shear
integrator's job; these tests only pin the Runge–Kutta path.
"""
import math

import curlew
import pytest
import torch

from curlew.kinematics.euler import euler_velocity
from curlew.kinematics.flow import FlowOffset


def _points(dim, n=40):
    torch.manual_seed(0)
    return torch.rand(n, dim, dtype=curlew.dtype) * 400 - 200


def _velocity(dim, stream="beta", n_features=48, target=0.3, n_steps=8):
    """A Euler velocity scaled so the discrete-step residual Lipschitz matches."""
    curlew.default_dim = dim
    v = euler_velocity(
        dim, n_features=n_features, length_scale_range=(200.0, 5000.0),
        seed=3, stream=stream,
    )
    torch.manual_seed(11)
    for m in v.modules():
        if hasattr(m, "A_mu"):
            with torch.no_grad():
                m.A_mu.normal_(0.0, 1.0)
                if hasattr(m, "B_mu"):
                    m.B_mu.normal_(0.0, 1.0)
    x = _points(dim)
    s = FlowOffset(v, n_steps=n_steps).stability(x)
    for m in v.modules():
        if hasattr(m, "A_mu"):
            with torch.no_grad():
                scale = max(s / target, 1e-12)
                m.A_mu.div_(scale)
                if hasattr(m, "B_mu"):
                    m.B_mu.div_(scale)
    return v


def test_default_is_rk4():
    """RK4 is the default, and it is not the same map as one RK2 step."""
    v = _velocity(2)
    x = _points(2)
    assert FlowOffset(v, n_steps=5).integrator == "rk4"
    plain = FlowOffset(v, n_steps=5)
    explicit = FlowOffset(v, n_steps=5, integrator="rk4")
    assert (plain.inverse_map(x) - explicit.inverse_map(x)).abs().max() == 0.0
    rk2 = FlowOffset(v, n_steps=5, integrator="rk2")
    assert float((plain.inverse_map(x) - rk2.inverse_map(x)).abs().max()) > 1e-6


def test_unknown_integrator_rejected():
    with pytest.raises(ValueError, match="integrator must be"):
        FlowOffset(_velocity(2), n_steps=2, integrator="nambu")


def test_stability_reports_the_injectivity_margin():
    """``||D Phi_h - I||_2 < 1`` is a sufficient injectivity certificate."""
    v = _velocity(2, target=0.3, n_steps=8)
    s = FlowOffset(v, n_steps=8).stability(_points(2))
    assert 0.0 < s < 1.0
    assert FlowOffset(v, n_steps=32).stability(_points(2)) < s


class _Rotation(torch.nn.Module):
    dim = 2

    def forward(self, x, w=None, t=None):
        v = torch.empty_like(x)
        v[:, 0] = -x[:, 1]
        v[:, 1] = x[:, 0]
        return v

    def forward_and_jacobian(self, x, w=None, t=None):
        L = x.new_zeros(x.shape[0], 2, 2)
        L[:, 0, 1] = -1.0
        L[:, 1, 0] = 1.0
        return self.forward(x), L


class _Zero(torch.nn.Module):
    dim = 2

    def forward(self, x, w=None, t=None):
        return torch.zeros_like(x)

    def forward_and_jacobian(self, x, w=None, t=None):
        n, d = x.shape
        return torch.zeros_like(x), x.new_zeros(n, d, d)


def test_stability_uses_the_discrete_step_jacobian():
    """Constant rotation: rho is ||J_h - I|| of RK2 / RK4, not dt ||L||."""
    x = _points(2)
    n_steps = 4
    h = -1.0 / n_steps
    rho2 = FlowOffset(_Rotation(), n_steps=n_steps, integrator="rk2").stability(x)
    assert rho2 == pytest.approx(abs(h) * math.sqrt(1.0 + 0.25 * h * h), rel=1e-10)
    a = -0.5 * h * h + (h ** 4) / 24.0
    b = h - (h ** 3) / 6.0
    rho4 = FlowOffset(_Rotation(), n_steps=n_steps, integrator="rk4").stability(x)
    assert rho4 == pytest.approx(math.sqrt(a * a + b * b), rel=1e-10)
    assert rho4 < rho2
    assert FlowOffset(_Zero(), n_steps=n_steps).stability(x) == 0.0
