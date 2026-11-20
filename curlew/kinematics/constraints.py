"""Priors for :class:`~curlew.kinematics.restoration.RestorationField`.

Each constraint implements ``raw(field, cache) -> scalar``. A float weight is
used as given. A string ``"w"`` becomes ``w / v`` the first time the raw loss
``v`` reaches ``floor``, and is 0 until then. Pass a list to
:func:`~curlew.geology.restore` (``constraints=``) or assign
``field.constraints`` before training.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import curlew
import numpy as np
import torch
import torch.nn as nn

from curlew.core import _tensor

Weight = Union[float, str]
Points = Union[torch.Tensor, Sequence, "np.ndarray"]


class RestorationCache:
    """Lazy per-:meth:`loss` cache of inverse-map evaluations shared between constraints."""

    def __init__(self, field: nn.Module) -> None:
        self.field = field
        self._inv: dict[int, torch.Tensor] = {}
        self._inv_jac: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def inverse_map(self, pts: torch.Tensor) -> torch.Tensor:
        key = id(pts)
        if key not in self._inv:
            self._inv[key] = self.field.integrator.inverse_map(pts)
        return self._inv[key]

    def inverse_map_with_jacobian(
        self, pts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = id(pts)
        if key not in self._inv_jac:
            self._inv_jac[key] = self.field.integrator.inverse_map_with_jacobian(pts)
        return self._inv_jac[key]


class RestorationConstraint:
    """
    Base class. Subclasses implement :meth:`raw`.

    A string weight waits for ``floor`` because an identity map makes several
    of these losses exactly zero; resolving against that first value would
    give an infinite weight. ``floor`` of ``1e-4`` is about 1 % in log-strain.
    """

    floor: float = 1e-4

    def __init__(
        self,
        weight: Weight = 1.0,
        *,
        floor: Optional[float] = None,
        label: Optional[str] = None,
    ) -> None:
        self.weight = weight
        if floor is not None:
            self.floor = float(floor)
        self.label = label or type(self).__name__

    def raw(self, field, cache: Optional[RestorationCache]) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self,
        field,
        cache: Optional[RestorationCache] = None,
    ) -> tuple[str, torch.Tensor, float]:
        """``(label, raw_loss, weight)``. A string weight stays 0 until ``raw`` is finite and at least ``floor``, then resolves once."""
        loss = self.raw(field, cache)
        weight = self.weight
        if isinstance(weight, str):
            v = float(loss.detach())
            if not (np.isfinite(v) and v >= self.floor):
                return self.label, loss, 0.0
            weight = float(weight) / v
            self.weight = weight
        return self.label, loss, float(weight)


def _as_points(
    points: Optional[Points],
    *,
    field,
) -> Optional[torch.Tensor]:
    if points is None:
        return None
    pts = _tensor(points, dev=curlew.device, dt=curlew.dtype)
    if pts.dim() == 1:
        pts = pts.unsqueeze(0)
    return pts


def _as_horizon_indices(value) -> Optional[Tuple[int, int]]:
    """Return ``(i, j)`` if ``value`` is two horizon indices, else ``None``."""

    try:
        if value is None or len(value) != 2:
            return None
    except TypeError:
        return None
    a, b = value[0], value[1]
    if isinstance(a, (np.ndarray, list, tuple)) or isinstance(b, (np.ndarray, list, tuple)):
        return None
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
        return None
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a), int(b)
    return None


def _to_numpy_points(points) -> np.ndarray:
    if torch.is_tensor(points):
        return points.detach().cpu().numpy()
    return np.asarray(points)


def _resample_by_arclength(h: np.ndarray, n: int) -> np.ndarray:
    diffs = np.diff(h, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    if s[-1] == 0.0:
        return np.repeat(h[:1], n, axis=0)
    s = s / s[-1]
    s_new = np.linspace(0.0, 1.0, n)
    out = np.empty((n, h.shape[1]), dtype=h.dtype)
    for d in range(h.shape[1]):
        out[:, d] = np.interp(s_new, s, h[:, d])
    return out


def _curve_over_axis(h: np.ndarray, along: int, s_new: np.ndarray) -> np.ndarray:
    """Interpolate a point cloud as a graph over one coordinate."""

    order = np.argsort(h[:, along], kind="mergesort")
    sorted_h = h[order]
    s = sorted_h[:, along]
    uniq, inv = np.unique(s, return_inverse=True)
    acc = np.zeros((len(uniq), h.shape[1]), dtype=float)
    weight = np.zeros(len(uniq), dtype=float)
    np.add.at(acc, inv, sorted_h)
    np.add.at(weight, inv, 1.0)
    mean = acc / np.clip(weight[:, None], 1.0, None)
    out = np.empty((len(s_new), h.shape[1]), dtype=float)
    for d in range(h.shape[1]):
        if d == along:
            out[:, d] = s_new
        else:
            out[:, d] = np.interp(s_new, mean[:, along], mean[:, d])
    return out


def points_between_horizons(
    h0: Union[np.ndarray, torch.Tensor],
    h1: Union[np.ndarray, torch.Tensor],
    n_along: int = 32,
    n_between: int = 4,
    *,
    t_min: float = 0.1,
    t_max: float = 0.9,
    depth_axis: Optional[int] = None,
    parameterization: str = "arclength",
) -> torch.Tensor:
    """
    Samples of the unit between two traces, in one shared frame.

    ``arclength`` resamples ordered polylines. ``graph`` treats each cloud as
    a function of the non-depth coordinate (unordered ``CSet.eq``, or a layer
    after unfolding). Along-strike samples span the union of the two traces.
    ``n_between`` rows fill ``[t_min, t_max]``; one row sits at the midpoint.
    """
    assert n_along >= 2 and n_between >= 1
    assert 0.0 <= t_min <= t_max <= 1.0
    if parameterization not in ("arclength", "graph"):
        raise ValueError(
            f"parameterization must be 'arclength' or 'graph', got {parameterization!r}"
        )

    h0_np = np.asarray(_to_numpy_points(h0), dtype=float)
    h1_np = np.asarray(_to_numpy_points(h1), dtype=float)
    assert h0_np.ndim == 2 and h1_np.ndim == 2 and h0_np.shape[1] == h1_np.shape[1]
    dim = h0_np.shape[1]

    if parameterization == "graph" and dim == 2:
        dax = 1 if depth_axis is None else int(depth_axis) % dim
        along = 0 if dax != 0 else 1
        lo = min(float(h0_np[:, along].min()), float(h1_np[:, along].min()))
        hi = max(float(h0_np[:, along].max()), float(h1_np[:, along].max()))
        s_new = np.linspace(lo, hi, n_along)
        h0_r = _curve_over_axis(h0_np, along, s_new)
        h1_r = _curve_over_axis(h1_np, along, s_new)
    else:
        h0_r = _resample_by_arclength(h0_np, n_along)
        h1_r = _resample_by_arclength(h1_np, n_along)

    # A single row sits at the middle of [t_min, t_max], not at t_min.
    if n_between == 1:
        ts = np.array([0.5 * (t_min + t_max)])
    else:
        ts = np.linspace(t_min, t_max, n_between, dtype=float)
    grid = (
        (1.0 - ts[:, None, None]) * h0_r[None, :, :]
        + ts[:, None, None] * h1_r[None, :, :]
    )
    return _tensor(
        grid.reshape(n_between * n_along, h0_r.shape[1]),
        dev=curlew.device,
        dt=curlew.dtype,
    )


def _unimodular_inverse(J: torch.Tensor) -> torch.Tensor:
    """
    Adjugate of a batch of Jacobians with ``det = 1`` (no division, no LU).

    Exact for the shear integrator. For RK4 it is accurate to the scheme's
    own volume error. Above 3×3 this falls back to a general solve.
    """
    d = J.shape[-1]
    if d == 2:
        a, b = J[..., 0, 0], J[..., 0, 1]
        c, e = J[..., 1, 0], J[..., 1, 1]
        return torch.stack(
            [torch.stack([e, -b], -1), torch.stack([-c, a], -1)], dim=-2
        )
    if d == 3:
        # With det = 1 the rows of the inverse are the cross products of the
        # columns: inv[0] = a2 x a3, inv[1] = a3 x a1, inv[2] = a1 x a2.
        a1, a2, a3 = J[..., :, 0], J[..., :, 1], J[..., :, 2]
        return torch.stack(
            [
                torch.linalg.cross(a2, a3, dim=-1),
                torch.linalg.cross(a3, a1, dim=-1),
                torch.linalg.cross(a1, a2, dim=-1),
            ],
            dim=-2,
        )
    return torch.linalg.inv(J)


def layer_parallel_stretch_loss(F: torch.Tensor, depth_axis: int) -> torch.Tensor:
    """
    ``⟨Σ (ln σ)²⟩`` for material lines in the reference layering.

    ``F = Dφ`` at reference points. Layers are normal to ``depth_axis``, so
    the other columns of ``F`` are the in-plane line elements and ``σ`` their
    singular values (one in 2-D, two in 3-D). Zero when those lines keep their
    length, at any rotation. Thickness is not charged directly; with
    ``det F = 1`` it follows from the in-plane stretch. Layer-parallel shear
    is free. Unlike :func:`hencky_strain_loss`, the singular values are not
    projected to unit determinant: a change of bed length is the penalty.
    """
    d = F.shape[-1]
    dax = int(depth_axis) % d
    along = [k for k in range(d) if k != dax]
    s = torch.linalg.svdvals(F[..., :, along]).clamp_min(1e-12)
    return s.log().square().sum(dim=-1).mean()


def hencky_strain_loss(J: torch.Tensor) -> torch.Tensor:
    """
    ``⟨‖log U‖²_F⟩`` from the polar factor ``J = R U``.

    Invariant to rotation and to replacing ``J`` by ``J⁻¹``. Singular values
    are projected to ``det = 1`` first, so RK4's ``O(h⁴)`` volume error is not
    charged as strain.
    """
    s = torch.linalg.svdvals(J).clamp_min(1e-12)
    d = s.shape[-1]
    s = s / s.prod(dim=-1, keepdim=True).pow(1.0 / d)
    return s.log().square().sum(dim=-1).mean()


def orthogonal_thickness_loss(
    x0: torch.Tensor,
    J_inv: torch.Tensor,
    depth_axis: int,
    *,
    log: bool = True,
) -> torch.Tensor:
    """
    Penalise ``|J^T e_d| != 1`` (orthogonal layer-thickness preservation).
    """
    dim = x0.shape[-1]
    g0 = torch.zeros_like(x0)
    g0[:, int(depth_axis) % dim] = 1.0
    g = torch.bmm(J_inv.mT, g0.unsqueeze(-1)).squeeze(-1)
    m = g.norm(dim=-1).clamp_min(1e-8)
    if log:
        return torch.log(m).pow(2).mean()
    return (m - 1.0).pow(2).mean()


def layer_thickness_loss(
    depths: List[torch.Tensor],
    thicknesses: torch.Tensor,
    mode: str = "relative",
) -> torch.Tensor:
    """
    Match mean restored-depth gaps between consecutive horizons.
    """
    if len(depths) < 2:
        raise ValueError("layer thickness requires at least two horizons")
    mu = torch.stack([d.mean() for d in depths])
    T = thicknesses.reshape(-1).to(dtype=mu.dtype, device=mu.device)
    if T.numel() != len(depths) - 1:
        raise ValueError(
            f"need {len(depths) - 1} layer thicknesses, got {T.numel()}"
        )
    gaps = mu[1:] - mu[:-1]
    if mode == "relative":
        scale = gaps.pow(2).mean().detach().clamp_min(1e-8)
        a = (gaps * T).sum() / T.pow(2).sum().clamp_min(1e-8)
        res = gaps - a * T
    elif mode == "absolute":
        scale = T.pow(2).mean().detach().clamp_min(1e-8)
        res = gaps - T
    else:
        raise ValueError(
            f"mode must be 'relative' or 'absolute', got {mode!r}"
        )
    return res.pow(2).mean() / scale


class _SampledJacobianConstraint(RestorationConstraint):
    """
    One sample mode for a Jacobian prior:

    * ``points`` — present-day coordinates, inverse-mapped here
    * ``grid`` — one draw from a :class:`~curlew.geometry.Grid`
    * ``horizons=(i, j)`` — indices into the bound ``CSet.eq``; the two traces
      are inverse-mapped and the unit between them is filled
    * ``restored_layer=(i, j)`` with explicit ``horizons`` polylines — the same fill
    """

    def __init__(
        self,
        weight: Weight = 1.0,
        *,
        points: Optional[Points] = None,
        grid=None,
        restored_layer: Optional[Tuple[int, int]] = None,
        horizons: Optional[Union[Tuple[int, int], Sequence[Points]]] = None,
        n_along: int = 32,
        n_between: int = 4,
        t_min: float = 0.1,
        t_max: float = 0.9,
        floor: Optional[float] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(weight, floor=floor, label=label)
        index_pair = _as_horizon_indices(horizons)
        if index_pair is not None:
            if restored_layer is not None and tuple(restored_layer) != index_pair:
                raise ValueError(
                    f"{type(self).__name__} `horizons=` indices {index_pair} "
                    f"disagree with `restored_layer=` {tuple(restored_layer)}"
                )
            restored_layer = index_pair
            horizons = None
        elif restored_layer is not None:
            restored_layer = (int(restored_layer[0]), int(restored_layer[1]))

        self.points = points
        self.grid = grid
        self.restored_layer = restored_layer
        self.horizons = horizons
        self.n_along = int(n_along)
        self.n_between = int(n_between)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        modes = sum(x is not None for x in (points, grid, restored_layer))
        if modes != 1:
            raise ValueError(
                f"{type(self).__name__} requires exactly one of "
                "`points=`, `grid=`, `horizons=(i, j)`, or `restored_layer=`"
            )
        if restored_layer is not None and horizons is not None:
            i, j = restored_layer
            if i < 0 or j < 0 or i >= len(horizons) or j >= len(horizons):
                raise ValueError(
                    f"restored_layer indices {(i, j)} out of range for "
                    f"{len(horizons)} horizons"
                )

    def _evaluation_points(self, field) -> torch.Tensor:
        if self.points is not None:
            return _as_points(self.points, field=field)
        return _tensor(self.grid.draw(), dev=curlew.device, dt=curlew.dtype)

    def _horizon_traces(self, field) -> Sequence:
        if self.horizons is not None:
            return self.horizons
        C = getattr(field, "C", None)
        eq = None if C is None else getattr(C, "eq", None)
        if eq is None:
            raise ValueError(
                f"{type(self).__name__} horizon indices require bound "
                "CSet.eq traces, or explicit `horizons=` polylines"
            )
        return eq

    def _restored_layer_points(
        self, field, cache: Optional[RestorationCache]
    ) -> torch.Tensor:
        traces = self._horizon_traces(field)
        i, j = self.restored_layer
        if i < 0 or j < 0 or i >= len(traces) or j >= len(traces):
            raise ValueError(
                f"horizon indices {(i, j)} out of range for {len(traces)} traces"
            )
        hi = _as_points(traces[i], field=field)
        hj = _as_points(traces[j], field=field)
        if cache is not None:
            h0 = cache.inverse_map(hi)
            h1 = cache.inverse_map(hj)
        else:
            h0 = field.integrator.inverse_map(hi)
            h1 = field.integrator.inverse_map(hj)
        # Unordered CSet.eq samples become graphs over restored x; supplied
        # polylines keep their along-trace pairing.
        how = "graph" if self.horizons is None else "arclength"
        return points_between_horizons(
            h0,
            h1,
            n_along=self.n_along,
            n_between=self.n_between,
            t_min=self.t_min,
            t_max=self.t_max,
            depth_axis=getattr(field, "depth_axis", 1),
            parameterization=how,
        )

    def _restoration_jacobian(
        self, field, cache: Optional[RestorationCache]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample points and the restoration Jacobian ``DΦ⁻¹``."""
        if self.restored_layer is not None:
            x0 = self._restored_layer_points(field, cache)
            _, J_fwd = field.integrator.forward_map_with_jacobian(x0)
            return x0, _unimodular_inverse(J_fwd)
        pts = self._evaluation_points(field)
        if cache is not None:
            return cache.inverse_map_with_jacobian(pts)
        return field.integrator.inverse_map_with_jacobian(pts)


class OrthogonalThickness(_SampledJacobianConstraint):
    """
    ``mean(log² |Jᵀ e_d|)`` with ``log=True`` (default), so thickening and
    thinning cost the same. Sample modes are those of
    :class:`_SampledJacobianConstraint`.
    """

    def __init__(
        self,
        weight: Weight = 1.0,
        *,
        points: Optional[Points] = None,
        grid=None,
        restored_layer: Optional[Tuple[int, int]] = None,
        horizons: Optional[Union[Tuple[int, int], Sequence[Points]]] = None,
        n_along: int = 32,
        n_between: int = 4,
        t_min: float = 0.1,
        t_max: float = 0.9,
        log: bool = True,
        floor: Optional[float] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(
            weight,
            points=points,
            grid=grid,
            restored_layer=restored_layer,
            horizons=horizons,
            n_along=n_along,
            n_between=n_between,
            t_min=t_min,
            t_max=t_max,
            floor=floor,
            label=label or "thick_loss",
        )
        self.log = bool(log)

    def raw(self, field, cache: Optional[RestorationCache]) -> torch.Tensor:
        x0, J = self._restoration_jacobian(field, cache)
        return orthogonal_thickness_loss(
            x0, J, field.depth_axis, log=self.log
        )


class HenckyStrain(_SampledJacobianConstraint):
    """
    :func:`hencky_strain_loss` on the restoration map.

    Zero for a rigid motion, unchanged by the restored-frame pin, and the same
    for ``Φ`` and ``Φ⁻¹``. It picks a low-strain map among those that already
    fit the data; keep its weight below the contacts and bedding. Sample modes
    as :class:`OrthogonalThickness`.
    """

    def __init__(
        self,
        weight: Weight = 1.0,
        *,
        points: Optional[Points] = None,
        grid=None,
        restored_layer: Optional[Tuple[int, int]] = None,
        horizons: Optional[Union[Tuple[int, int], Sequence[Points]]] = None,
        n_along: int = 32,
        n_between: int = 4,
        t_min: float = 0.1,
        t_max: float = 0.9,
        floor: Optional[float] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(
            weight,
            points=points,
            grid=grid,
            restored_layer=restored_layer,
            horizons=horizons,
            n_along=n_along,
            n_between=n_between,
            t_min=t_min,
            t_max=t_max,
            floor=floor,
            label=label or "hencky_loss",
        )

    def raw(self, field, cache: Optional[RestorationCache]) -> torch.Tensor:
        _, J = self._restoration_jacobian(field, cache)
        return hencky_strain_loss(J)


class LayerParallelStretch(_SampledJacobianConstraint):
    """
    :func:`layer_parallel_stretch_loss`: no change of length in the reference
    layers (flat, normal to ``field.depth_axis``).

    On the middle of a competent bed this is a neutral surface; over a whole
    unit it is bed-length preservation. Rotation and layer-parallel shear are
    free, unlike :class:`HenckyStrain`. Sample modes as
    :class:`OrthogonalThickness`.
    """

    def __init__(
        self,
        weight: Weight = 1.0,
        *,
        points: Optional[Points] = None,
        grid=None,
        restored_layer: Optional[Tuple[int, int]] = None,
        horizons: Optional[Union[Tuple[int, int], Sequence[Points]]] = None,
        n_along: int = 32,
        n_between: int = 4,
        t_min: float = 0.1,
        t_max: float = 0.9,
        floor: Optional[float] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(
            weight,
            points=points,
            grid=grid,
            restored_layer=restored_layer,
            horizons=horizons,
            n_along=n_along,
            n_between=n_between,
            t_min=t_min,
            t_max=t_max,
            floor=floor,
            label=label or "layer_stretch_loss",
        )

    def raw(self, field, cache: Optional[RestorationCache]) -> torch.Tensor:
        # _restoration_jacobian returns D(φ⁻¹); the forward F is its inverse (det 1).
        _, J_inv = self._restoration_jacobian(field, cache)
        F = _unimodular_inverse(J_inv)
        return layer_parallel_stretch_loss(F, getattr(field, "depth_axis", 1))


class LayerThickness(RestorationConstraint):
    """
    Restored-depth gaps between ordered horizons (top to bottom).

    ``thicknesses`` has one entry per gap. ``mode="relative"`` fits a single
    scale, so only the ratios are constrained. ``mode="absolute"`` matches
    the given thicknesses.
    """

    def __init__(
        self,
        horizons: Sequence[Points],
        thicknesses: Points,
        weight: Weight = 1.0,
        *,
        mode: str = "relative",
        floor: Optional[float] = None,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(weight, floor=floor, label=label or "layer_thick_loss")
        if len(horizons) < 2:
            raise ValueError("LayerThickness requires at least two horizons")
        self.horizons = horizons
        self.thicknesses = _tensor(thicknesses, dev=curlew.device, dt=curlew.dtype)
        if mode not in ("relative", "absolute"):
            raise ValueError(f"mode must be 'relative' or 'absolute', got {mode!r}")
        self.mode = mode

    def raw(self, field, cache: Optional[RestorationCache]) -> torch.Tensor:
        chunks = [_as_points(h, field=field) for h in self.horizons]
        sizes = [c.shape[0] for c in chunks]
        pts = torch.cat(chunks, dim=0)
        if cache is not None:
            x0 = cache.inverse_map(pts)
        else:
            x0 = field.inverse_map(pts)
        # Gaps are in the reference scalar. They match raw depth only for a flat layer cake.
        v_ref, g_ref = field._reference_at(x0)
        depths = list(field._reattach(v_ref, g_ref, x0).split(sizes))
        return layer_thickness_loss(depths, self.thicknesses, mode=self.mode)
