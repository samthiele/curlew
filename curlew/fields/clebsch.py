"""
Clebsch (streamfunction) velocity fields built from :class:`~curlew.fields.series.FSF` potentials.

At fixed lift coordinate ``w``, the velocity is divergence-free by construction:

    2-D:  v = ∇⊥β = (∂β/∂x₂, −∂β/∂x₁)
    3-D:  v = ∇α × ∇β

Jacobians are analytic (via ``FSF.gradient_and_hessian``) for use by diffeomorphic
integrators (forthcoming).
"""
from typing import Optional, Tuple

import curlew
import torch
import torch.nn as nn

from curlew.core import HSet, _tensor
from curlew.fields.series import FSF, TemporalFSF


def _pot_gradient_and_hessian(
    pot: nn.Module,
    x: torch.Tensor,
    w: Optional[torch.Tensor],
    t: Optional[float],
):
    """Call ``gradient_and_hessian``, forwarding ``t`` for temporal potentials."""
    if isinstance(pot, TemporalFSF):
        return pot.gradient_and_hessian(x, w, t=t)
    return pot.gradient_and_hessian(x, w)


def _killing_frame(
    v0: torch.Tensor,
    J0: torch.Tensor,
    x: torch.Tensor,
    anchor: torch.Tensor,
    mode: str,
):
    """
    Rigid-body velocity k(x) matching translation (and optionally rotation) of
    the field at ``anchor``. The Jacobian Jk is trace-free, so subtracting k
    preserves div(v) = 0 away from gauge singularities.
    """
    n, dim = x.shape
    k = v0.expand(n, -1)
    if mode == "translation":
        return k, x.new_zeros(n, dim, dim)
    dx = x - anchor
    if dim == 2:
        omega = 0.5 * (J0[1, 0] - J0[0, 1])
        k = k + omega * torch.stack([-dx[:, 1], dx[:, 0]], dim=-1)
        rot = x.new_tensor([[0.0, -1.0], [1.0, 0.0]])
        jk = (omega * rot).expand(n, -1, -1)
    else:
        w = 0.5 * (J0 - J0.T)
        omega = torch.stack([w[2, 1], w[0, 2], w[1, 0]])
        k = k + torch.linalg.cross(omega.expand_as(dx), dx)
        jk = w.expand(n, -1, -1)
    return k, jk


class ClebschVelocity(nn.Module):
    """
    Divergence-free velocity from one (2-D) or two (3-D) :class:`FSF` potentials.

    Parameters
    ----------
    beta : FSF
        Streamfunction potential β. In 2-D, v = ∇⊥β; in 3-D, v = ∇α × ∇β.
    alpha : FSF, optional
        Second potential for 3-D; required when ``beta.dim == 3``.
    anchor : array-like, optional
        Point where rigid modes are removed (see ``killing``).
    killing : ``"full"`` | ``"translation"``
        ``"full"`` removes translation and rotation; ``"translation"`` only
        removes uniform translation.

    Notes
    -----
    Potentials are evaluated on the **raw** FSF path (``gradient_and_hessian``),
    without geomodelling ``Transform`` or drift. For lifted potentials (fault
    sheet labels), pass ``w`` to ``forward``; derivatives are w.r.t. spatial
    coordinates only. Pass integration time ``t`` when the underlying
    potentials are :class:`~curlew.fields.series.TemporalFSF` instances.
    """

    def __init__(
        self,
        beta: FSF,
        alpha: Optional[FSF] = None,
        anchor=None,
        killing: str = "full",
    ) -> None:
        super().__init__()
        if beta.dim not in (2, 3):
            raise ValueError(f"beta.dim must be 2 or 3, got {beta.dim}")
        if beta.dim == 3 and alpha is None:
            raise ValueError("alpha potential is required for dim=3")
        if alpha is not None and alpha.dim != beta.dim:
            raise ValueError("alpha and beta must have the same spatial dimension")
        if killing not in ("full", "translation"):
            raise ValueError(f"unknown killing mode {killing!r}")

        self.dim = beta.dim
        self.beta = beta
        self.alpha = alpha
        self.killing = killing

        if anchor is not None:
            a = _tensor(anchor, dev=curlew.device, dt=curlew.dtype).reshape(1, self.dim)
        else:
            if hasattr(beta, "A_mu"):
                amp = beta.A_mu
            elif hasattr(beta, "A_rho"):
                amp = beta.A_rho
            else:
                amp = beta.A
            a = torch.zeros(0, self.dim, dtype=amp.dtype, device=amp.device)
        self.register_buffer("anchor", a)

    @property
    def anchored(self) -> bool:
        return self.anchor.numel() > 0

    def _raw(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor],
        jac: bool,
        t: Optional[float] = None,
    ):
        g_b, h_b = _pot_gradient_and_hessian(self.beta, x, w, t)
        if self.dim == 2:
            v = torch.stack([g_b[:, 1], -g_b[:, 0]], dim=1)
            if not jac:
                return v, None
            jv = torch.stack([h_b[:, :, 1], -h_b[:, :, 0]], dim=1)
            return v, jv

        g_a, h_a = _pot_gradient_and_hessian(self.alpha, x, w, t)
        v = torch.linalg.cross(g_a, g_b, dim=-1)
        if not jac:
            return v, None
        jv = (
            torch.linalg.cross(h_a, g_b.unsqueeze(1).expand_as(h_a), dim=-1).mT
            + torch.linalg.cross(g_a.unsqueeze(1).expand_as(h_b), h_b, dim=-1).mT
        )
        return v, jv

    def forward(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        t: Optional[float] = None,
    ) -> torch.Tensor:
        """Velocity v(x[, w], t) with shape (N, dim)."""
        v, _ = self._raw(x, w, jac=False, t=t)
        if self.anchored:
            k, _ = self._gauge(x, t=t)
            v = v - k
        return v

    def forward_and_jacobian(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        t: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return velocity (N, dim) and Jacobian ∂v/∂x (N, dim, dim)."""
        v, jv = self._raw(x, w, jac=True, t=t)
        if self.anchored:
            k, jk = self._gauge(x, t=t)
            v, jv = v - k, jv - jk
        return v, jv

    def _gauge(self, x: torch.Tensor, t: Optional[float] = None):
        v0, j0 = self._raw(self.anchor, None, jac=self.killing == "full", t=t)
        j0 = j0[0] if j0 is not None else None
        return _killing_frame(v0, j0, x, self.anchor, self.killing)


def clebsch_velocity(
    dim: int,
    *,
    n_features: int = 128,
    length_scale_range: Tuple[float, float] = (0.5, 3.0),
    lift_dim: int = 0,
    anchor=None,
    killing: str = "full",
    seed: int = 42,
    **fsf_kwargs,
) -> ClebschVelocity:
    """
    Build :class:`FSF` potentials and wrap in :class:`ClebschVelocity`.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    n_features, length_scale_range, lift_dim, seed
        Forwarded to each underlying :class:`FSF`.
    anchor, killing
        Passed to :class:`ClebschVelocity`.
    **fsf_kwargs
        Additional keywords for :class:`FSF` (e.g. ``freq_sampling``, ``scale``).

    Returns
    -------
    ClebschVelocity
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    xavier = fsf_kwargs.pop("xavier_amplitudes", False)
    h = HSet()
    beta = FSF(
        "beta",
        H=h,
        input_dim=dim,
        lift_dim=lift_dim,
        rff_features=n_features,
        length_scale_range=length_scale_range,
        seed=seed,
        xavier_amplitudes=xavier,
        **fsf_kwargs,
    )
    alpha = None
    if dim == 3:
        alpha = FSF(
            "alpha",
            H=h.copy(),
            input_dim=dim,
            lift_dim=lift_dim,
            rff_features=n_features,
            length_scale_range=length_scale_range,
            seed=seed + 1,
            xavier_amplitudes=xavier,
            **fsf_kwargs,
        )
    return ClebschVelocity(beta, alpha, anchor=anchor, killing=killing)


def temporal_clebsch_velocity(
    dim: int,
    *,
    n_features: int = 128,
    length_scale_range: Tuple[float, float] = (0.5, 3.0),
    lift_dim: int = 0,
    anchor=None,
    killing: str = "full",
    seed: int = 42,
    init_std: float = 0.01,
    **temporal_fsf_kwargs,
) -> ClebschVelocity:
    """
    Build :class:`ClebschVelocity` from :class:`~curlew.fields.series.TemporalFSF`
    potentials with a coarse-to-fine spectral schedule over ``t ∈ [0, 1]``.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    n_features, length_scale_range, lift_dim, seed, init_std
        Forwarded to each :class:`TemporalFSF` (``length_scale_range`` is
        ``(λ_fine, λ_coarse)`` for the spectral cutoff sweep).
    anchor, killing
        Passed to :class:`ClebschVelocity`.
    **temporal_fsf_kwargs
        Additional keywords for :class:`TemporalFSF` (e.g. ``freq_sampling``,
        ``activation``, ``scale``, ``freq_damp``, ``spectral_gate``,
        ``spectral_band_width``, ``learnable_directions``).
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    h = HSet()
    beta = TemporalFSF(
        "beta",
        H=h,
        input_dim=dim,
        lift_dim=lift_dim,
        rff_features=n_features,
        length_scale_range=length_scale_range,
        seed=seed,
        init_std=init_std,
        **temporal_fsf_kwargs,
    )
    alpha = None
    if dim == 3:
        alpha = TemporalFSF(
            "alpha",
            H=h.copy(),
            input_dim=dim,
            lift_dim=lift_dim,
            rff_features=n_features,
            length_scale_range=length_scale_range,
            seed=seed + 1,
            init_std=init_std,
            **temporal_fsf_kwargs,
        )
    return ClebschVelocity(beta, alpha, anchor=anchor, killing=killing)
