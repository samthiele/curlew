"""
Euler velocity fields: v = ∇α × ∇β (single code path in all dimensions).

Each slot is either a learnable :class:`~curlew.fields.series.FSF` (optionally
lifted with sheet labels ``w``) or a :class:`FixedDirectionPotential` with
constant unit gradient (hinge / section normal / ignorable coordinate).

In 2-D, spatial gradients are embedded in R³ for the cross product; the fixed
slot is always the section normal (``e_z``), then velocity is projected back to
the section. Prefer ``stream="alpha"`` so the hinge sits on ``β``
(``∇β = e_z``): then ``A = α ∇β`` and ``v = ∇⊥α``. Custom ``hinge`` directions
apply in 3-D only (fold axis / ignorable coordinate).
"""
from __future__ import annotations

import math
from typing import Literal, Mapping, Optional, Sequence, Tuple, Union

import curlew
import torch
import torch.nn as nn

from curlew.core import HSet, _tensor
from curlew.fields.series import (
    FSF,
    _build_cylindrical_fourier_modes,
    _build_paired_fourier_modes,
)


class FixedDirectionPotential(nn.Module):
    """
    Constant-gradient Euler potential: ∇φ = ``direction`` (unit vector).

    Used for the hinge / section / ignorable coordinate in a Euler pair.
    Independent of fault-lift labels ``w``.
    """

    def __init__(
        self,
        direction=None,
        *,
        axis: Optional[int] = None,
        dim: Optional[int] = None,
        spatial_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if direction is not None:
            d = _tensor(direction, dev="cpu", dt=torch.float64).reshape(-1)
        elif axis is not None:
            if dim is None:
                raise ValueError("dim is required when axis is set")
            d = torch.zeros(int(dim), dtype=torch.float64)
            d[int(axis) % int(dim)] = 1.0
            spatial_dim = int(dim)
        else:
            raise ValueError("FixedDirectionPotential requires direction or (axis and dim)")

        n = int(d.numel())
        if n not in (2, 3):
            raise ValueError(f"hinge direction must have length 2 or 3, got {n}")
        norm = float(d.norm())
        if norm < 1e-12:
            raise ValueError("hinge direction must be nonzero")
        d = (d / norm).to(dtype=curlew.dtype, device=curlew.device)
        if spatial_dim is None:
            spatial_dim = 2 if n == 2 else 3
        if spatial_dim not in (2, 3):
            raise ValueError(f"spatial_dim must be 2 or 3, got {spatial_dim}")
        self.register_buffer("direction", d)
        self.spatial_dim = int(spatial_dim)
        # Cross products use R³ when the physical section is 2-D.
        self.embed_dim = 3 if self.spatial_dim == 2 else self.spatial_dim

    @property
    def dim(self) -> int:
        """Spatial dimension (matches paired :class:`FSF` ``input_dim``)."""
        return self.spatial_dim

    def gradient_and_hessian(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del w
        n = x.shape[0]
        h = x.new_zeros(n, self.embed_dim, self.embed_dim)
        return self.gradient_raw(x), h

    def gradient_raw(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Constant gradient, without allocating the (always zero) Hessian."""
        del w
        n = x.shape[0]
        g = self.direction.unsqueeze(0).expand(n, -1)
        if self.embed_dim == 3 and g.shape[-1] == 2:
            g = torch.cat([g, g.new_zeros(n, 1)], dim=-1)
        return g


class SlowHingePotential(nn.Module):
    """
    Hinge potential with a slow wander: φ = h·x + φ_slow.

    ``h`` is a fixed unit direction. ``φ_slow`` is a long-wavelength Fourier
    potential, so ∇φ = h + ∇φ_slow stays near ``h`` and varies only on scales
    the caller put in that bank. The companion Fourier potential is built
    separately, with its frequencies in the plane perpendicular to ``h``.
    """

    def __init__(self, direction, slow: FSF) -> None:
        super().__init__()
        if not isinstance(slow, FSF) or int(slow.dim) != 3:
            raise ValueError("slow hinge perturbation must be a 3-D FSF")
        d = _tensor(direction, dev="cpu", dt=torch.float64).reshape(-1)
        if int(d.numel()) != 3:
            raise ValueError(f"hinge direction must have length 3, got {int(d.numel())}")
        norm = float(d.norm())
        if norm < 1e-12:
            raise ValueError("hinge direction must be nonzero")
        d = (d / norm).to(dtype=curlew.dtype, device=curlew.device)
        self.register_buffer("direction", d)
        self.slow = slow
        self.spatial_dim = 3
        self.embed_dim = 3

    @property
    def dim(self) -> int:
        return self.spatial_dim

    def gradient_raw(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.slow.gradient_raw(x, w) + self.direction

    def gradient_and_hessian(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g, h = self.slow.gradient_and_hessian(x, w)
        return g + self.direction, h


def _is_hinge_slot(slot) -> bool:
    return isinstance(slot, (FixedDirectionPotential, SlowHingePotential))


EulerSlot = Union[FSF, FixedDirectionPotential, SlowHingePotential]


def _embed_grad(g: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """Append a zero component when embedding a 2-D gradient into R³."""
    if embed_dim == 3 and g.shape[-1] == 2:
        return torch.cat([g, g.new_zeros(g.shape[0], 1)], dim=-1)
    return g


def _fsf_cosine_bank(
    fsf: FSF,
    x: Optional[torch.Tensor],
    w: Optional[torch.Tensor],
    embed_dim: int,
    prune: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ``∇φ = Σ p_k Ωdir cos(Ωph·x + off)`` as ``(p, Om_ph, Om_dir, off)``.

    ``prune`` drops zero amplitudes only outside a grad graph. A zero is still
    ``0·cos``; indexing it out removes that parameter, so an identity start has
    no gradient. Pass ``False`` when the caller caches by mode
    (:meth:`EulerVelocity.ridge_decomposition`).

    Linear Cartesian modes are two parallel shears per frequency
    (``B cos θ`` and ``−A cos(θ − π/2)``). Cubic harmonics stay along the
    fundamental ``Ω`` and stretch the phase to ``m Ω``.
    """
    act = getattr(fsf, "activation", None)
    if act not in (None, "cubic"):
        raise NotImplementedError(
            f"shear_decomposition supports activation=None or 'cubic'; "
            f"{act!r} is not polynomial in sin and would need infinitely "
            f"many harmonics"
        )
    om_dir = _embed_grad(fsf.omega_spatial, embed_dim)
    if w is not None and x is None:
        raise ValueError("shear_decomposition needs `x` when lift `w` is set")
    off = fsf._lift_phase(x, w)
    om_ph = om_dir
    if act is None:
        # ∇φ = (B cos θ − A sin θ) Ω = B Ω cos(θ) + (−A) Ω cos(θ − π/2).
        # Two shears, same direction — they merge into one group.
        a = fsf._effective_amplitudes()
        b = fsf._effective_amplitudes_sin()
        p = torch.cat([b, -a])
        om_ph = torch.cat([om_dir, om_dir])
        om_dir = om_ph
        off = torch.cat([off, off - 0.5 * math.pi], dim=-1)
        # See the docstring: dropping a zero amplitude removes it from the graph.
        if prune and not p.requires_grad:
            mag = p.abs()
            keep = mag > (1e-18 * mag.max().clamp_min(1e-30))
            p, om_ph, om_dir = p[keep], om_ph[keep], om_dir[keep]
            off = off[:, keep] if off.dim() == 2 else off[keep]
    else:
        p = fsf._effective_amplitudes()
        if act == "cubic":
            bias = fsf.bias
            c1 = 1.5 * ((1.0 - bias * bias) - 0.25)
            c2 = -1.5 * bias
            c3 = torch.full_like(c1, 1.5 * 0.25)
            p = torch.cat([c1 * p, c2 * p, c3 * p])
            om_ph = torch.cat([om_dir, 2.0 * om_dir, 3.0 * om_dir])
            om_dir = torch.cat([om_dir, om_dir, om_dir])
            off = torch.cat([off, 2.0 * off - math.pi / 2, 3.0 * off], dim=-1)
    return p, om_ph, om_dir, off


def _off_sum(a: torch.Tensor, b: torch.Tensor, sign: float) -> torch.Tensor:
    """``a_i ± b_j`` with ``a, b`` either ``(F,)`` or ``(N, F)``."""
    if a.dim() == 1 and b.dim() == 1:
        return (a[:, None] + sign * b[None, :]).reshape(-1)
    if a.dim() == 1:
        a = a.expand(b.shape[0], -1)
    elif b.dim() == 1:
        b = b.expand(a.shape[0], -1)
    return (a[:, :, None] + sign * b[:, None, :]).reshape(a.shape[0], -1)


def _product_shears(
    p: torch.Tensor,
    om_a: torch.Tensor,
    dir_a: torch.Tensor,
    off_a: torch.Tensor,
    q: torch.Tensor,
    om_b: torch.Tensor,
    dir_b: torch.Tensor,
    off_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Product-to-sum of two cosine banks: ``2 F_α F_β`` shears.

    ``∇α × ∇β`` with ``∇φ = Σ p Ωdir cos(Ωph·x + off)`` expands as

        v = Σ_{ij} p_i q_j (Ωdir_i × Ωdir_j) cos θ_i cos ψ_j

    and ``cos θ cos ψ = ½ [cos(θ+ψ) + cos(θ−ψ)]``. Each pair is two shears
    with ``C ∝ Ωdir_i × Ωdir_j``, orthogonal to both ``Ωph_i ± Ωph_j``.
    """
    cross = torch.linalg.cross(dir_a[:, None, :], dir_b[None, :, :], dim=-1)
    C_pair = (0.5 * p[:, None] * q[None, :]).unsqueeze(-1) * cross
    C = torch.cat([C_pair.reshape(-1, C_pair.shape[-1]),
                   C_pair.reshape(-1, C_pair.shape[-1])])
    om = torch.cat([
        (om_a[:, None, :] + om_b[None, :, :]).reshape(-1, om_a.shape[-1]),
        (om_a[:, None, :] - om_b[None, :, :]).reshape(-1, om_a.shape[-1]),
    ])
    off = torch.cat([_off_sum(off_a, off_b, 1.0), _off_sum(off_a, off_b, -1.0)],
                    dim=-1)
    mag = C.norm(dim=-1)
    keep = mag > (1e-18 * mag.max().clamp_min(1e-30))
    # An all-zero product is the identity. Dropping it leaves nothing to differentiate.
    if (not bool(keep.any())) and C.requires_grad:
        return C, om, off
    if off.dim() == 1:
        return C[keep], om[keep], off[keep]
    return C[keep], om[keep], off[:, keep]


def _cat_offset(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concatenate phase offsets along the mode axis."""
    if a.dim() == 1 and b.dim() == 1:
        return torch.cat([a, b])
    if a.dim() == 1:
        a = a.expand(b.shape[0], -1)
    elif b.dim() == 1:
        b = b.expand(a.shape[0], -1)
    return torch.cat([a, b], dim=-1)


def _hinge_plus_slow_shears(
    stream: FSF,
    hinge: SlowHingePotential,
    x: Optional[torch.Tensor],
    w: Optional[torch.Tensor],
    embed_dim: int,
    *,
    sign: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shears for ∇(stream) × (h + ∇φ_slow), times ``sign``."""
    p, om, om_dir, off = _fsf_cosine_bank(stream, x, w, embed_dim)
    direction = _embed_grad(hinge.direction.unsqueeze(0), embed_dim)[0]
    cross = torch.linalg.cross(om_dir, direction.expand_as(om_dir), dim=-1)
    C_lin = sign * p.unsqueeze(-1) * cross
    q, om_b, dir_b, off_b = _fsf_cosine_bank(hinge.slow, x, w, embed_dim)
    C_slow, om_slow, off_slow = _product_shears(
        p, om, om_dir, off, q, om_b, dir_b, off_b,
    )
    C = torch.cat([C_lin, sign * C_slow])
    om_out = torch.cat([om, om_slow])
    return C, om_out, _cat_offset(off, off_slow)


def _ridge_bank(
    slot: FSF,
    x: Optional[torch.Tensor],
    w: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One FSF as ``(dir, amp, kap, off)`` for the normal shear split."""
    p, om_ph, om_dir, off = _fsf_cosine_bank(slot, x, w, 3, prune=False)
    mag = om_dir.norm(dim=-1)
    u = om_dir / mag.clamp_min(1e-30).unsqueeze(-1)
    # Om_ph is parallel to Om_dir (equal, or an integer multiple for cubic
    # harmonics), so its projection on `u` is the whole phase wavenumber.
    return u, p * mag, (om_ph * u).sum(-1), off


def _slow_hinge_ridge_bank(
    hinge: SlowHingePotential,
    x: Optional[torch.Tensor],
    w: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear hinge as a zero-wavenumber ridge, plus the slow Fourier bank."""
    u, amp, kap, off = _ridge_bank(hinge.slow, x, w)
    h = hinge.direction.reshape(1, 3)
    amp_h = h.new_ones(1)
    kap_h = h.new_zeros(1)
    if off.dim() == 1:
        off_h = h.new_zeros(1)
    else:
        off_h = off.new_zeros(off.shape[0], 1)
    return (
        torch.cat([h, u]),
        torch.cat([amp_h, amp]),
        torch.cat([kap_h, kap]),
        _cat_offset(off_h, off),
    )


def _project_velocity(v: torch.Tensor, spatial_dim: int) -> torch.Tensor:
    if spatial_dim == 2 and v.shape[-1] == 3:
        return v[:, :2]
    return v


def _project_jacobian(jv: torch.Tensor, spatial_dim: int) -> torch.Tensor:
    if spatial_dim == 2 and jv.shape[-1] == 3:
        return jv[:, :2, :2]
    return jv


def _slot_grad_hess(
    slot: EulerSlot, x: torch.Tensor, w: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    return slot.gradient_and_hessian(x, w)


def _slot_grad(
    slot: EulerSlot, x: torch.Tensor, w: Optional[torch.Tensor]
) -> torch.Tensor:
    """Gradient only — the velocity needs no Hessian (see :meth:`FSF.gradient_raw`)."""
    return slot.gradient_raw(x, w)


def _euler_cross(
    g_a: torch.Tensor,
    h_a: torch.Tensor,
    g_b: torch.Tensor,
    h_b: torch.Tensor,
    *,
    embed_dim: int,
    spatial_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """v = ∇α × ∇β with optional 2-D section embed."""
    g_a = _embed_grad(g_a, embed_dim)
    g_b = _embed_grad(g_b, embed_dim)
    v = torch.linalg.cross(g_a, g_b, dim=-1)
    jv = (
        torch.linalg.cross(h_a, g_b.unsqueeze(1).expand_as(h_a), dim=-1).mT
        + torch.linalg.cross(g_a.unsqueeze(1).expand_as(h_b), h_b, dim=-1).mT
    )
    return _project_velocity(v, spatial_dim), _project_jacobian(jv, spatial_dim)


def _default_hinge_for_dim(dim: int) -> FixedDirectionPotential:
    """Out-of-section direction for 2-D (embedded ``e_z``) or ``e_z`` in 3-D."""
    if dim == 2:
        return FixedDirectionPotential([0.0, 0.0, 1.0], spatial_dim=2)
    return FixedDirectionPotential(axis=2, dim=3)


def _parse_hinge(
    dim: int,
    *,
    hinge=None,
    section_axis: Optional[int] = None,
) -> FixedDirectionPotential:
    if hinge is not None:
        return FixedDirectionPotential(hinge, spatial_dim=dim)
    if section_axis is not None:
        return FixedDirectionPotential(axis=section_axis, dim=dim)
    return _default_hinge_for_dim(dim)


class EulerVelocity(nn.Module):
    """
    Divergence-free velocity ``v = ∇α × ∇β``.

    ``beta`` is the second slot. With ``alpha=None`` in 2-D it is the learnable
    stream function and ``alpha`` is a fixed hinge (default ``e_z``). A second
    :class:`FSF` is required in 3-D. Potentials are raw :class:`FSF` values, with
    no geomodel transform. ``w`` carries sheet labels; derivatives are spatial.

    There is no in-loop gauge. A constant field ``u`` does not commute with
    ``v`` (``[v, u] = (u·∇)v``), so subtracting one changes the map and the
    strain. Pin the restored frame after fitting
    (:func:`~curlew.kinematics.restoration.pin_anchor`,
    :func:`~curlew.kinematics.restoration.pin_mean`): a constant shift of the
    restored coordinates, with identity Jacobian.
    """

    def __init__(
        self,
        beta: EulerSlot,
        alpha: Optional[EulerSlot] = None,
        *,
        stream: Optional[Literal["alpha", "beta"]] = None,
        hinge: Optional[FixedDirectionPotential] = None,
    ) -> None:
        super().__init__()
        if alpha is None:
            if not isinstance(beta, FSF):
                raise TypeError("when alpha is omitted, beta must be an FSF")
            if beta.dim not in (2, 3):
                raise ValueError(f"beta.dim must be 2 or 3, got {beta.dim}")
            if beta.dim == 3:
                raise ValueError("alpha potential is required for dim=3")
            alpha = hinge or _default_hinge_for_dim(2)
            beta_slot, alpha_slot = beta, alpha
        elif isinstance(beta, FSF) and isinstance(alpha, FSF):
            if beta.dim != alpha.dim:
                raise ValueError("alpha and beta must have the same spatial dimension")
            if beta.dim not in (2, 3):
                raise ValueError(f"dim must be 2 or 3, got {beta.dim}")
            beta_slot, alpha_slot = beta, alpha
        elif (isinstance(beta, FSF) and _is_hinge_slot(alpha)) or (
            _is_hinge_slot(beta) and isinstance(alpha, FSF)
        ):
            beta_slot, alpha_slot = beta, alpha
        else:
            raise TypeError(
                "slots must be (FSF, FSF), (FSF,) in 2-D, or one FSF plus a hinge potential"
            )

        if stream is not None:
            if stream not in ("alpha", "beta"):
                raise ValueError(f"stream must be 'alpha' or 'beta', got {stream!r}")
            if not isinstance(beta_slot, FSF) or not isinstance(alpha_slot, FSF):
                raise ValueError("stream reordering requires two FSF potentials")

        self._stream = stream

        if isinstance(beta_slot, nn.Module):
            self.add_module("beta", beta_slot)
        else:
            self.beta = beta_slot
        if isinstance(alpha_slot, nn.Module):
            self.add_module("alpha", alpha_slot)
        else:
            self.alpha = alpha_slot

        self.dim = self._spatial_dim()
        self.embed_dim = 3 if self.dim == 2 else self.dim

    def _spatial_dim(self) -> int:
        for slot in (self.beta, self.alpha):
            if isinstance(slot, FSF):
                return int(slot.dim)
        for slot in (self.alpha, self.beta):
            if _is_hinge_slot(slot):
                return int(slot.spatial_dim)
        raise RuntimeError("no spatial slot found")

    def _fsf_slot(self, name: Literal["alpha", "beta"]) -> Optional[FSF]:
        slot = getattr(self, name)
        return slot if isinstance(slot, FSF) else None

    @property
    def stream(self) -> Optional[FSF]:
        """Learnable FSF when exactly one slot is fixed (else ``None``)."""
        fa, fb = self._fsf_slot("alpha"), self._fsf_slot("beta")
        if fa is not None and fb is None:
            return fa
        if fb is not None and fa is None:
            return fb
        if fa is not None and fb is not None:
            return fa if self._stream == "alpha" else fb if self._stream == "beta" else None
        return None

    @property
    def hinge(self) -> Optional[Union[FixedDirectionPotential, SlowHingePotential]]:
        for slot in (self.alpha, self.beta):
            if _is_hinge_slot(slot):
                return slot
        return None

    def _ordered_grad_hess(
        self, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        g_a, h_a = _slot_grad_hess(self.alpha, x, w)
        g_b, h_b = _slot_grad_hess(self.beta, x, w)
        if self.embed_dim == 3:
            h_a = _embed_grad_hess(h_a, g_a, x)
            h_b = _embed_grad_hess(h_b, g_b, x)
        return g_a, h_a, g_b, h_b

    def _raw(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor],
        jac: bool,
        t: Optional[float] = None,
    ):
        del t
        if not jac:
            # No Hessians on the position-only path. Building and discarding them dominated each sub-step.
            g_a = _slot_grad(self.alpha, x, w)
            g_b = _slot_grad(self.beta, x, w)
            if self._swap_slots():
                g_a, g_b = g_b, g_a
            g_a = _embed_grad(g_a, self.embed_dim)
            g_b = _embed_grad(g_b, self.embed_dim)
            v = torch.linalg.cross(g_a, g_b, dim=-1)
            return _project_velocity(v, self.dim), None

        g_a, h_a, g_b, h_b = self._ordered_grad_hess(x, w)
        if self._swap_slots():
            g_a, g_b = g_b, g_a
            h_a, h_b = h_b, h_a

        return _euler_cross(
            g_a, h_a, g_b, h_b, embed_dim=self.embed_dim, spatial_dim=self.dim
        )

    def _swap_slots(self) -> bool:
        """
        Whether the two slots exchange roles before the cross product.

        2-D with FSF on beta, hinge on alpha: swap → v = ∇β × ∇α = ∇⊥β (legacy).
        2-D with FSF on alpha, hinge on beta: no swap → v = ∇α × ∇β = ∇⊥α (A=α∇β).
        """
        if _is_hinge_slot(self.alpha) and isinstance(self.beta, FSF):
            return True
        return (
            self._stream == "beta"
            and isinstance(self.alpha, FSF)
            and isinstance(self.beta, FSF)
        )

    def forward(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        t: Optional[float] = None,
    ) -> torch.Tensor:
        """Velocity v(x[, w]) with shape (N, dim). ``t`` is ignored."""
        v, _ = self._raw(x, w, jac=False, t=t)
        return v

    def shear_decomposition(
        self,
        x: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ``(C, Om, off)`` with ``C[k] · Om[k] = 0`` and

            v(x) = Σ_k C[k] cos(Om[k] · x + off[k]).

        Each term is an exact shear (Quispel and McLaren 2003): its phase is
        invariant along its own flow, the step is a translation, and
        ``det(I − h sin(·) C Omᵀ) = 1`` at any step size.

        One learnable slot gives one cosine per Cartesian channel per mode
        (two, parallel, when both amplitudes are live). Two slots expand by
        the product-to-sum identity, up to ``8 F_α F_β`` terms. Sequential
        depth is the number of distinct directions. The normal split groups by
        ``∇P × ∇Q`` instead, so depth follows the number of rays.

        With ``w``, ``Ω_w · w`` is a per-point constant and ``off`` is
        ``(N, F)``. Without it, ``off`` is the learned phase, shape ``(F,)``
        or ``(2F,)``. ``activation="cubic"`` replaces each mode by harmonics at
        ``Ω, 2Ω, 3Ω``, which stay in one direction group. ``"tanh"`` and
        ``"softplus"`` are not trigonometric polynomials and are refused.

        Raises
        ------
        NotImplementedError
            A slot is not an :class:`FSF` or :class:`FixedDirectionPotential`,
            or the activation is not ``None`` or ``"cubic"``.
        """
        first, second = (
            (self.beta, self.alpha) if self._swap_slots() else (self.alpha, self.beta)
        )
        d = self.embed_dim
        # v = grad(first) x grad(second)
        if isinstance(second, FixedDirectionPotential) and isinstance(first, FSF):
            p, om, om_dir, off = _fsf_cosine_bank(first, x, w, d)
            direction = _embed_grad(second.direction.unsqueeze(0), d)[0]
            cross = torch.linalg.cross(om_dir, direction.expand_as(om_dir), dim=-1)
            C = p.unsqueeze(-1) * cross
        elif isinstance(first, FixedDirectionPotential) and isinstance(second, FSF):
            p, om, om_dir, off = _fsf_cosine_bank(second, x, w, d)
            direction = _embed_grad(first.direction.unsqueeze(0), d)[0]
            # d x Om = -(Om x d)
            cross = torch.linalg.cross(om_dir, direction.expand_as(om_dir), dim=-1)
            C = -(p.unsqueeze(-1) * cross)
        elif isinstance(first, FSF) and isinstance(second, SlowHingePotential):
            C, om, off = _hinge_plus_slow_shears(first, second, x, w, d, sign=1.0)
        elif isinstance(first, SlowHingePotential) and isinstance(second, FSF):
            C, om, off = _hinge_plus_slow_shears(second, first, x, w, d, sign=-1.0)
        elif isinstance(first, FSF) and isinstance(second, FSF):
            p, om_a, dir_a, off_a = _fsf_cosine_bank(first, x, w, d)
            q, om_b, dir_b, off_b = _fsf_cosine_bank(second, x, w, d)
            C, om, off = _product_shears(
                p, om_a, dir_a, off_a, q, om_b, dir_b, off_b,
            )
        else:
            raise NotImplementedError(
                "shear_decomposition needs FSF slots (optionally one "
                "FixedDirectionPotential hinge)"
            )
        return C[:, : self.dim], om[:, : self.dim], off

    def ridge_decomposition(
        self,
        x: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]]:
        """
        Both gradients as unexpanded ridges, or ``None`` for a hinge or a 2-D field.

        Returns ``((dir, amp, kap, off), ...)`` for the ordered slots of
        ``v = ∇P × ∇Q``:

            ∇P = Σ_i amp[i] dir[i] cos(kap[i] (dir[i] · x) + off[i]).

        ``dir`` is a unit vector, ``kap`` the signed wavenumber along it
        (``m |Ω|`` for cubic harmonic ``m``), and ``off`` is ``(K,)`` or
        ``(N, K)`` as in :meth:`shear_decomposition`.

        Grouping each bank into rays ``u_p``, ``v_q`` gives shears along
        ``n = u_p × v_q``. Depth is the number of distinct normals, at most
        ``D_a D_b``, independent of how many wavelengths a ray carries. The
        phase split's directions ``a u ± b v`` depend on the wavelength ratio
        and almost never merge. Zero-amplitude modes are kept, so the grouping
        depends only on the frequency buffers and can be cached while
        amplitudes train.

        Returns ``None`` for a fixed hinge or a 2-D field — there
        :meth:`shear_decomposition` already groups by ray. A slow hinge is a
        ridge bank: the linear direction is a zero-wavenumber mode in front
        of the perturbation.
        """
        first, second = (
            (self.beta, self.alpha) if self._swap_slots() else (self.alpha, self.beta)
        )
        if self.dim != 3:
            return None
        if isinstance(first, FSF) and isinstance(second, FSF):
            return _ridge_bank(first, x, w), _ridge_bank(second, x, w)
        if isinstance(first, FSF) and isinstance(second, SlowHingePotential):
            return _ridge_bank(first, x, w), _slow_hinge_ridge_bank(second, x, w)
        if isinstance(first, SlowHingePotential) and isinstance(second, FSF):
            return _slow_hinge_ridge_bank(first, x, w), _ridge_bank(second, x, w)
        return None

    def forward_and_jacobian(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        t: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return velocity (N, dim) and Jacobian ∂v/∂x (N, dim, dim)."""
        return self._raw(x, w, jac=True, t=t)


def _embed_grad_hess(
    h: torch.Tensor, g: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Pad a 2×2 spatial Hessian into 3×3 for section embed."""
    if h.shape[-1] == 2 and g.shape[-1] == 2:
        n = x.shape[0]
        out = x.new_zeros(n, 3, 3)
        out[:, :2, :2] = h
        return out
    return h


def _log_split_bands(length_scale_range, n_bands: int):
    """
    Wavelength bands, coarsest first (deformation order).

    One ``(lo, hi)`` is cut into ``n_bands`` log-equal disjoint bands. A
    sequence of pairs is used as given, so the bands may overlap.
    """
    if isinstance(length_scale_range[0], (list, tuple)):
        bands = [tuple(float(v) for v in b) for b in length_scale_range]
        if len(bands) != n_bands:
            raise ValueError(
                f"length_scale_range has {len(bands)} bands but n_features has "
                f"{n_bands} entries"
            )
        return bands
    lo, hi = (float(v) for v in length_scale_range)
    if not 0.0 < lo <= hi:
        raise ValueError(
            f"length_scale_range must satisfy 0 < min <= max, got {length_scale_range}"
        )
    edges = [
        math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / n_bands)
        for i in range(n_bands + 1)
    ]
    # exp/log perturbs the ends by an ulp. Pin them so n_features=[F] matches n_features=F.
    edges[0], edges[-1] = lo, hi
    return [(edges[i - 1], edges[i]) for i in range(n_bands, 0, -1)]


def euler_velocity(
    dim: int,
    *,
    n_features: Union[int, Sequence[int]] = 128,
    length_scale_range=(0.5, 3.0),
    modes_per_octave: Optional[float] = None,
    lift_dim: int = 0,
    stream: Optional[Literal["alpha", "beta"]] = None,
    hinge=None,
    section_axis: Optional[int] = None,
    cylindrical: bool = False,
    slow_hinge=None,
    slow_hinge_features: int = 16,
    slow_hinge_directions: int = 4,
    paired_frequencies: bool = False,
    pair_axis=None,
    seed: int = 42,
    alpha: Optional[Mapping] = None,
    beta: Optional[Mapping] = None,
    **fsf_kwargs,
) -> Union[EulerVelocity, list]:
    """
    :class:`FSF` potentials wrapped in :class:`EulerVelocity`.

    A list of ``n_features`` returns one velocity per band, coarsest first,
    for :class:`~curlew.kinematics.flow.FlowOffset` to compose.

    Parameters
    ----------
    dim : int
        2 or 3.
    n_features : int | sequence of int
        Modes. A sequence is a composite flow, coarse to fine. Default 128.
    length_scale_range : (float, float) | sequence of pairs | CSet
        Wavelength limits. With a list of ``n_features``, one pair is split
        log-equally; a list of pairs is used as given. A
        :class:`~curlew.core.CSet` is passed to
        :func:`~curlew.priors.spectral.spectral_band` (about ``hinge`` when
        ``cylindrical``).
    modes_per_octave : float, optional
        Replaces ``n_features`` by
        ``ceil(modes_per_octave · log2(L_max / L_min))``, rounded up to a
        multiple of ``n_directions``. Ignored for a list of ``n_features``.
    stream : {"alpha", "beta"}, optional
        Learnable slot. Default ``"alpha"`` puts the hinge on ``β``
        (``v = ∇⊥α``). In full 3-D it only orders the cross product, and the
        default stays ``"beta"``.
    hinge : array-like, optional
        Unit vector of the fixed slot in 3-D. Ignored in 2-D (always ``e_z``).
        Overrides ``section_axis``.
    section_axis : int, optional
        Coordinate axis of the fixed slot in 3-D when ``hinge`` is omitted.
    cylindrical : bool
        Draw the learnable bank in the plane normal to the hinge and embed
        it in R³. Requires a 3-D hinge.
    slow_hinge : (float, float), optional
        Wavelength band of a Fourier term added to the linear hinge, so
        ``φ = h·x + φ_slow``. Requires ``cylindrical=True``. On a composite
        flow it sits on the coarsest band. ``freq_damp=0``.
    slow_hinge_features, slow_hinge_directions : int
        Mode count and ray count of that term.
    paired_frequencies : bool
        Full 3-D only: pair directions so ``Ω^α_k × Ω^β_k`` is not zero.
    pair_axis : array-like, optional
        Axis used to build the orthogonal companion (default ``e_z``).
    alpha, beta : mapping, optional
        :class:`FSF` keywords for that slot. ``n_features`` is remapped to
        ``rff_features``.
    lift_dim, seed
        Forwarded to each :class:`FSF`. One learnable slot uses ``seed``, so
        it matches ``FSF(..., seed=s)``. Two slots use ``seed`` (β) and
        ``seed + 1`` (α). A ``"seed"`` in ``alpha`` or ``beta`` overrides.
    **fsf_kwargs
        Shared :class:`FSF` keywords (``scale``, ``freq_damp``, ``n_directions``).

    Returns
    -------
    EulerVelocity | list[EulerVelocity]
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")
    if cylindrical and dim != 3:
        raise ValueError("cylindrical applies in 3-D only")
    if cylindrical and hinge is None and section_axis is None:
        raise ValueError("cylindrical requires hinge or section_axis")
    from curlew.core import CSet
    import warnings
    if isinstance(length_scale_range, CSet):
        from curlew.priors.spectral import spectral_band
        band_axis = None
        if cylindrical and dim == 3:
            band_axis = _parse_hinge(
                3, hinge=hinge, section_axis=section_axis
            ).direction.detach().cpu().numpy()
        length_scale_range, band_info = spectral_band(
            length_scale_range, axis=band_axis, return_info=True
        )
        if not band_info.get("resolved", True):
            warnings.warn(
                "euler_velocity: data-driven wavelength band used unresolved λx "
                "(see spectral_band / spectral_moments).",
                UserWarning,
                stacklevel=2,
            )
    if modes_per_octave is not None and not isinstance(n_features, (list, tuple)):
        from curlew.priors.spectral import n_features_for_band
        n_features = n_features_for_band(
            length_scale_range, modes_per_octave, fsf_kwargs.get("n_directions")
        )

    # One velocity per band. FlowOffset composes them; a sum would rebuild one field.
    if isinstance(n_features, (list, tuple)):
        counts = [int(f) for f in n_features]
        if not counts or any(f < 1 for f in counts):
            raise ValueError("every n_features entry must be a positive int")
        bands = _log_split_bands(length_scale_range, len(counts))
        total = float(sum(counts))
        return [
            euler_velocity(
                dim,
                n_features=f_i,
                length_scale_range=band,
                lift_dim=lift_dim,
                stream=stream,
                hinge=hinge,
                section_axis=section_axis,
                cylindrical=cylindrical,
                slow_hinge=slow_hinge if i == 0 else None,
                slow_hinge_features=slow_hinge_features,
                slow_hinge_directions=slow_hinge_directions,
                paired_frequencies=paired_frequencies,
                pair_axis=pair_axis,
                # Two seeds are consumed per velocity (one per Euler slot).
                seed=seed + 2 * i,
                alpha=alpha,
                beta=beta,
                # out_scale is scale/sqrt(F) per field. Weight by sqrt(F_i/F) so the prior variance does not grow with the number of bands.
                scale=fsf_kwargs.get("scale", 1.0) * math.sqrt(f_i / total),
                **{k: v for k, v in fsf_kwargs.items() if k != "scale"},
            )
            for i, (f_i, band) in enumerate(zip(counts, bands))
        ]

    if stream is None:
        full_3d = dim == 3 and hinge is None and section_axis is None
        stream = "beta" if full_3d else "alpha"
    if stream not in ("alpha", "beta"):
        raise ValueError(f"stream must be 'alpha' or 'beta', got {stream!r}")
    if paired_frequencies and dim != 3:
        raise ValueError("paired_frequencies applies to full 3-D Euler only")
    if paired_frequencies and (hinge is not None or section_axis is not None):
        raise ValueError("paired_frequencies cannot be combined with hinge or section_axis")
    if paired_frequencies and cylindrical:
        raise ValueError("paired_frequencies cannot be combined with cylindrical")
    if paired_frequencies and (
        (alpha and "n_features" in alpha) or (beta and "n_features" in beta)
    ):
        raise ValueError(
            "paired_frequencies cannot be combined with per-slot n_features"
        )
    xavier = fsf_kwargs.pop("xavier_amplitudes", False)
    stochastic_scales = fsf_kwargs.pop("stochastic_scales", False)
    scale = fsf_kwargs.pop("scale", 1.0)
    wavelength_sampling = fsf_kwargs.get("wavelength_sampling", "log")
    h = HSet()

    paired_buffers = None
    if paired_frequencies:
        proj_a, proj_b, freq_scaling, out_scale = _build_paired_fourier_modes(
            rff_features=n_features,
            length_scale_range=length_scale_range,
            input_dim=dim,
            lift_dim=lift_dim,
            stochastic_scales=stochastic_scales,
            scale=scale,
            seed=seed,
            pair_axis=pair_axis,
            wavelength_sampling=wavelength_sampling,
        )
        paired_buffers = (proj_a, proj_b, freq_scaling, out_scale)

    fixed_3d = None
    if dim == 3 and (hinge is not None or section_axis is not None):
        fixed_3d = _parse_hinge(3, hinge=hinge, section_axis=section_axis)

    def _fsf_kwargs(slot: str, sole: bool = False):
        # The only learnable slot takes `seed`; with two, alpha takes seed + 1.
        extra = dict(alpha or {}) if slot == "alpha" else dict(beta or {})
        n_feat = extra.pop("n_features", n_features)
        lsr = extra.pop("length_scale_range", length_scale_range)
        sl_scale = extra.pop("scale", scale)
        sl_seed = extra.pop("seed", seed if (sole or slot == "psi") else seed + 1)
        sl_xavier = extra.pop("xavier_amplitudes", xavier)
        sl_stoch = extra.pop("stochastic_scales", stochastic_scales)
        kw = dict(
            H=h.copy() if slot == "alpha" else h,
            input_dim=dim,
            lift_dim=lift_dim,
            rff_features=n_feat,
            length_scale_range=lsr,
            seed=sl_seed,
            xavier_amplitudes=sl_xavier,
            stochastic_scales=sl_stoch,
            scale=sl_scale,
            **fsf_kwargs,
            **extra,
        )
        if paired_buffers is not None:
            proj = paired_buffers[0] if slot == "psi" else paired_buffers[1]
            kw.update(
                fourier_projection=proj,
                freq_scaling=paired_buffers[2],
                out_scale=paired_buffers[3],
            )
        elif cylindrical:
            proj, freq, osc = _build_cylindrical_fourier_modes(
                hinge=fixed_3d.direction,
                rff_features=n_feat,
                length_scale_range=lsr,
                lift_dim=lift_dim,
                stochastic_scales=sl_stoch,
                scale=sl_scale,
                seed=sl_seed,
                stratify_directions=bool(kw.get("stratify_directions", False)),
                n_directions=kw.get("n_directions"),
                wavelength_sampling=kw.get("wavelength_sampling", "log"),
            )
            kw.update(
                fourier_projection=proj,
                freq_scaling=freq,
                out_scale=osc,
            )
        return kw

    if dim == 2:
        if hinge is not None or section_axis is not None:
            raise ValueError(
                "hinge and section_axis apply in 3-D only; in 2-D the fixed "
                "Euler slot is always the section normal (internal e_z embed)"
            )
        fixed = _default_hinge_for_dim(2)
        if stream == "beta":
            return EulerVelocity(FSF("psi", **_fsf_kwargs("psi")), fixed)
        return EulerVelocity(fixed, FSF("alpha", **_fsf_kwargs("alpha", sole=True)))

    if slow_hinge is not None and fixed_3d is None:
        raise ValueError("slow_hinge requires hinge or section_axis in 3-D")
    if slow_hinge is not None and not cylindrical:
        raise ValueError(
            "slow_hinge requires cylindrical=True so the Fourier potential "
            "stays in the plane perpendicular to the hinge"
        )
    hinge_slot = fixed_3d
    if slow_hinge is not None:
        band = tuple(float(v) for v in slow_hinge)
        if len(band) != 2 or not 0.0 < band[0] <= band[1]:
            raise ValueError(
                f"slow_hinge must be a wavelength pair (min, max), got {slow_hinge!r}"
            )
        if int(slow_hinge_features) < 1 or int(slow_hinge_directions) < 1:
            raise ValueError("slow_hinge_features and slow_hinge_directions must be positive")
        hinge_slot = SlowHingePotential(
            fixed_3d.direction,
            FSF(
                "hinge",
                H=HSet(),
                input_dim=3,
                rff_features=int(slow_hinge_features),
                length_scale_range=band,
                n_directions=int(slow_hinge_directions),
                activation=None,
                freq_damp=0.0,
                seed=int(seed) + 17,
                xavier_amplitudes=False,
                scale=1.0,
            ),
        )

    if fixed_3d is not None:
        if stream == "beta":
            return EulerVelocity(FSF("psi", **_fsf_kwargs("psi")), hinge_slot)
        return EulerVelocity(hinge_slot, FSF("alpha", **_fsf_kwargs("alpha", sole=True)))

    # Fully 3-D: two FSF potentials (hinge kwargs omitted).
    return EulerVelocity(
        FSF("psi", **_fsf_kwargs("psi")),
        FSF("alpha", **_fsf_kwargs("alpha")),
        stream=stream,
    )
