"""
Pluggable priors for :class:`~curlew.fields.restoration.RestorationField`.

Each constraint implements ``raw(field, cache) -> scalar`` and is weighted
like curlew ``HSet`` terms (float literal or string ``"1.0"`` for auto-balance
on first evaluation).

Pass a list to :func:`~curlew.geology.restore` or assign to
``field.constraints`` before training::

    from curlew.fields.clebsch import clebsch_velocity
    from curlew.fields.restoration_constraints import (
        OrthogonalThickness, LayerThickness,
    )

    velocity = clebsch_velocity(2, n_features=128)
    event = restore(
        "fold", C=C, H=H, velocity=velocity,
        constraints=[
            OrthogonalThickness(
                weight="1.0",
                restored_layer=(3, 4),
                horizons=horizon_traces,
            ),
        ],
    )
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
    Base class for restoration priors.

    Subclasses implement :meth:`raw`.
    """

    def __init__(
        self,
        weight: Weight = 1.0,
        *,
        label: Optional[str] = None,
    ) -> None:
        self.weight = weight
        self.label = label or type(self).__name__

    def raw(self, field, cache: Optional[RestorationCache]) -> torch.Tensor:
        raise NotImplementedError

    def __call__(
        self,
        field,
        cache: Optional[RestorationCache] = None,
    ) -> tuple[str, torch.Tensor, float]:
        """
        Evaluate and return ``(label, raw_loss, resolved_weight)``.

        String weights are auto-balanced once against the first raw loss value.
        """
        loss = self.raw(field, cache)
        weight = self.weight
        if isinstance(weight, str):
            weight = float(weight) / max(float(loss.detach()), 1e-8)
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


def points_between_horizons(
    h0: Union[np.ndarray, torch.Tensor],
    h1: Union[np.ndarray, torch.Tensor],
    n_along: int = 32,
    n_between: int = 4,
    *,
    t_min: float = 0.1,
    t_max: float = 0.9,
) -> torch.Tensor:
    """
    Interior samples between two horizon polylines (arc-length resampled).

    ``h0`` and ``h1`` must live in the same coordinate frame — typically the
    reference/restored space after :meth:`inverse_map`.
    """
    assert n_along >= 2 and n_between >= 1
    assert 0.0 <= t_min <= t_max <= 1.0

    h0_np = h0.detach().cpu().numpy() if torch.is_tensor(h0) else np.asarray(h0)
    h1_np = h1.detach().cpu().numpy() if torch.is_tensor(h1) else np.asarray(h1)
    assert h0_np.ndim == 2 and h1_np.ndim == 2 and h0_np.shape[1] == h1_np.shape[1]

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

    h0_r = _resample_by_arclength(h0_np, n_along)
    h1_r = _resample_by_arclength(h1_np, n_along)
    ts = np.linspace(t_min, t_max, n_between, dtype=h0_r.dtype)
    grid = (
        (1.0 - ts[:, None, None]) * h0_r[None, :, :]
        + ts[:, None, None] * h1_r[None, :, :]
    )
    return _tensor(
        grid.reshape(n_between * n_along, h0_r.shape[1]),
        dev=curlew.device,
        dt=curlew.dtype,
    )


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


class OrthogonalThickness(RestorationConstraint):
    """
    Orthogonal-thickness preservation on a point sample.

    With ``log=True`` (default) the misfit is ``mean(log² |J^T e_d|)`` so stretch
    and shortening are penalised symmetrically.

    Evaluation modes (pick one):

    * ``points=`` — present-day coordinates (inverse-mapped internally).
    * ``grid=`` — random draw from a :class:`~curlew.geometry.Grid`.
    * ``restored_layer=(i, j), horizons=...`` — each step inverse-maps horizon
      polylines ``i`` and ``j`` to the current reference frame, samples the
      interior, forward-transports markers, and penalises local thickness stretch.
      No restored-depth targets or thickness literals are required.
    """

    def __init__(
        self,
        weight: Weight = 1.0,
        *,
        points: Optional[Points] = None,
        grid=None,
        restored_layer: Optional[Tuple[int, int]] = None,
        horizons: Optional[Sequence[Points]] = None,
        n_along: int = 32,
        n_between: int = 4,
        t_min: float = 0.1,
        t_max: float = 0.9,
        log: bool = True,
        label: Optional[str] = None,
    ) -> None:
        super().__init__(weight, label=label or "thick_loss")
        self.points = points
        self.grid = grid
        self.restored_layer = restored_layer
        self.horizons = horizons
        self.n_along = int(n_along)
        self.n_between = int(n_between)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.log = bool(log)

        modes = sum(
            x is not None for x in (points, grid, restored_layer)
        )
        if modes != 1:
            raise ValueError(
                "OrthogonalThickness requires exactly one of "
                "`points=`, `grid=`, or `restored_layer=`"
            )
        if restored_layer is not None:
            if horizons is None:
                raise ValueError(
                    "restored_layer= requires present-day `horizons=` polylines"
                )
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

    def _restored_layer_points(
        self, field, cache: Optional[RestorationCache]
    ) -> torch.Tensor:
        i, j = self.restored_layer
        hi = _as_points(self.horizons[i], field=field)
        hj = _as_points(self.horizons[j], field=field)
        if cache is not None:
            h0 = cache.inverse_map(hi)
            h1 = cache.inverse_map(hj)
        else:
            h0 = field.integrator.inverse_map(hi)
            h1 = field.integrator.inverse_map(hj)
        return points_between_horizons(
            h0,
            h1,
            n_along=self.n_along,
            n_between=self.n_between,
            t_min=self.t_min,
            t_max=self.t_max,
        )

    def raw(self, field, cache: Optional[RestorationCache]) -> torch.Tensor:
        if self.restored_layer is not None:
            x0 = self._restored_layer_points(field, cache)
            _, J_fwd = field.integrator.forward_map_with_jacobian(x0)
            J_inv = torch.linalg.inv(J_fwd)
            return orthogonal_thickness_loss(
                x0, J_inv, field.depth_axis, log=self.log
            )

        pts = self._evaluation_points(field)
        if cache is not None:
            x0, J = cache.inverse_map_with_jacobian(pts)
        else:
            x0, J = field.integrator.inverse_map_with_jacobian(pts)
        return orthogonal_thickness_loss(
            x0, J, field.depth_axis, log=self.log
        )


class LayerThickness(RestorationConstraint):
    """
    Constant (or ratio-known) layer thickness in restored depth.

    ``horizons`` is an ordered sequence of horizon polylines (top → bottom).
    ``thicknesses`` has length ``len(horizons) - 1``; in ``relative`` mode only
    the ratios matter (one scale factor is fitted).
    """

    def __init__(
        self,
        horizons: Sequence[Points],
        thicknesses: Points,
        weight: Weight = 1.0,
        *,
        mode: str = "relative",
        label: Optional[str] = None,
    ) -> None:
        super().__init__(weight, label=label or "layer_thick_loss")
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
        depths = list(x0[:, field.depth_axis].split(sizes))
        return layer_thickness_loss(depths, self.thicknesses, mode=self.mode)
