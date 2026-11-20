"""
GWN sheet labels and tangency confinement from open fault traces (2-D) or
triangle meshes (3-D).

Used with lifted velocity fields and
:class:`~curlew.kinematics.flow.FlowOffset`.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from curlew.core import _tensor
from curlew.geometry.gwn import (
    blended_tangency_polyline,
    closest_point_mesh,
    closest_point_polyline,
    gwn_mesh,
    gwn_polyline,
    mesh_boundary_vertices,
    smoothstep,
    smoothstep_scalar,
)


@dataclass
class SheetState:
    """Per-point fault quantities for one integration substep."""
    w_raw: torch.Tensor       # (N, n_faults) raw GWN labels
    w_active: torch.Tensor
    influence: torch.Tensor   # (N, n_faults) GWN corridor weight in [0, 1]
    confinement: torch.Tensor  # (N, n_faults) tangency projection strength λ
    normal: torch.Tensor
    mobility: Optional[torch.Tensor] = None  # (N,) slip weight in [0, 1]; None = symmetric


def _fault_geometry_dim(fault) -> int:
    """Return ``2`` for open polylines, ``3`` for ``(verts, faces)`` mesh tuples."""
    if isinstance(fault, (tuple, list)) and len(fault) == 2:
        verts, faces = fault
        v, f = _tensor(verts), _tensor(faces)
        if v.ndim == 2 and v.shape[1] == 3 and f.ndim == 2 and f.shape[1] == 3:
            return 3
        raise ValueError(
            "3-D faults must be (verts, faces) with verts (V, 3) and faces (F, 3)"
        )
    t = _tensor(fault)
    if t.ndim == 2 and t.shape[1] == 2 and t.shape[0] >= 2:
        return 2
    raise ValueError(
        "2-D faults must be open polylines (P >= 2, 2); for 3-D supply a "
        "triangle mesh (verts, faces)"
    )


_SHEET_MOBILITY_MODES = ("both", "positive", "negative")


def _parse_sheet_mobility(
    sheet_mobility: Union[str, Sequence[str]],
    n_faults: int,
) -> Tuple[Tuple[str, ...], bool]:
    """
    Normalise ``sheet_mobility`` to a length-``n_faults`` tuple.

    Returns ``(modes, from_scalar)`` where ``from_scalar`` is True when the
    caller passed a single string (velocity μ still uses ``mobility_fault``).
    """
    if isinstance(sheet_mobility, str):
        if sheet_mobility not in _SHEET_MOBILITY_MODES:
            raise ValueError(
                f"sheet_mobility must be one of {_SHEET_MOBILITY_MODES}, "
                f"got {sheet_mobility!r}"
            )
        return (sheet_mobility,) * n_faults, True
    mobs = tuple(sheet_mobility)
    if len(mobs) != n_faults:
        raise ValueError(
            f"sheet_mobility length {len(mobs)} != n_faults {n_faults}"
        )
    for j, m in enumerate(mobs):
        if m not in _SHEET_MOBILITY_MODES:
            raise ValueError(
                f"sheet_mobility[{j}] must be one of {_SHEET_MOBILITY_MODES}, "
                f"got {m!r}"
            )
    return mobs, False


class FaultLift(nn.Module):
    """
    Sheet labels and tangency confinement from open 2-D traces or 3-D meshes.

    Each entry of ``faults`` is a polyline ``(P, 2)`` or ``(verts, faces)``
    with ``verts (V, 3)`` and ``faces (F, 3)``. All faults share one dimension.
    The lift says which side of a cut a point is on; the slip is fitted.

    Parameters
    ----------
    faults : sequence
        One geometry per cut.
    winding_power : float
        Exponent ``p`` in ``(|w| / label_scale)^p`` for the GWN corridor.
        Sets tangency only when ``confine_mode="winding"``.
    corridor_ref : float, optional
        GWN scale for corridor weights. Default ``π`` (2-D) or ``2π`` (3-D).
    confine_mode : {"distance", "winding"}
        ``"distance"`` (default): ``λ = exp(-(d/σ)^p)`` from perpendicular
        distance. ``"winding"``: ``λ`` is the corridor weight.
    confine_width : float, optional
        ``σ`` for distance confinement, in coordinate units. Default is about
        8 % of the fault bounding-box span.
    confine_power : float
        ``p`` in ``exp(-(d/σ)^p)``. ``2`` is Gaussian.
    tip_taper : float
        Fraction of each fault's length over which labels fade at the tips.
        ``0`` disables it.
    lift_mode : {"independent", "nested"}
        ``"independent"`` (default): one axis per fault.
        ``"nested"``: one scalar, built in fault order.
    strike_tangent : {"blended", "closest"}
        2-D strike and normal at a query point.
    strike_blend_power : float
        Exponent on subtended-angle weights when ``strike_tangent="blended"``.
    sheet_mobility : {"both", "positive", "negative"} or a sequence
        Which GWN side may slip. One string applies to every fault.
    mobility_fault : int, optional
        Which fault's GWN scales the velocity when ``sheet_mobility`` is one
        string (default ``0``). Ignored for a per-fault sequence, where the
        weights of the non-``"both"`` faults are multiplied.
    advect_traces : sequence of bool, optional
        Lagrangian advection of each fault. Default ``[False, True, True, …]``:
        the first fault stays fixed.
    advect_offset : float
        If positive, carried geometry is advected from points offset into the
        hanging wall, so the cut does not slip along itself.
    activation_windows : sequence of (t_on, t_off), optional
        When each fault exists for ``t ∈ [0, 1]``. Used only while integrating.
    """

    def __init__(
        self,
        faults: Sequence,
        *,
        winding_power: float = 1.0,
        corridor_ref: Optional[float] = None,
        confine_mode: str = "distance",
        confine_width: Optional[float] = None,
        confine_power: float = 2.0,
        tip_taper: float = 0.15,
        lift_mode: str = "independent",
        strike_tangent: str = "blended",
        strike_blend_power: float = 1.0,
        sheet_mobility: Union[str, Sequence[str]] = "both",
        mobility_fault: int = 0,
        advect_traces: Optional[Sequence[bool]] = None,
        advect_offset: float = 0.0,
        activation_windows: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        super().__init__()
        if not faults:
            raise ValueError("faults must be non-empty")
        if lift_mode not in ("independent", "nested"):
            raise ValueError(f"unknown lift_mode {lift_mode!r}")
        if confine_mode not in ("distance", "winding"):
            raise ValueError(
                f"confine_mode must be 'distance' or 'winding', got {confine_mode!r}"
            )
        if confine_width is not None and confine_width <= 0:
            raise ValueError(f"confine_width must be > 0, got {confine_width}")
        if confine_power <= 0:
            raise ValueError(f"confine_power must be > 0, got {confine_power}")
        if winding_power <= 0:
            raise ValueError(f"winding_power must be > 0, got {winding_power}")
        if tip_taper < 0:
            raise ValueError(f"tip_taper must be >= 0, got {tip_taper}")
        dims = {_fault_geometry_dim(f) for f in faults}
        if len(dims) != 1:
            raise ValueError("all faults must share the same dimension (2 or 3)")
        self.dim = dims.pop()
        if self.dim == 2:
            if strike_tangent not in ("blended", "closest"):
                raise ValueError(f"unknown strike_tangent {strike_tangent!r}")
            if strike_blend_power <= 0:
                raise ValueError(
                    f"strike_blend_power must be > 0, got {strike_blend_power}"
                )
            self.strike_tangent = strike_tangent
            self.strike_blend_power = float(strike_blend_power)
            default_ref = math.pi
        else:
            if strike_tangent != "blended":
                raise ValueError("strike_tangent applies to 2-D traces only")
            if strike_blend_power != 1.0:
                raise ValueError("strike_blend_power applies to 2-D traces only")
            self.strike_tangent = "blended"
            self.strike_blend_power = 1.0
            default_ref = 2.0 * math.pi
        if corridor_ref is not None and corridor_ref <= 0:
            raise ValueError(f"corridor_ref must be > 0, got {corridor_ref}")
        self.corridor_ref = float(
            corridor_ref if corridor_ref is not None else default_ref
        )
        self.n_faults = len(faults)
        if not 0 <= mobility_fault < self.n_faults:
            raise ValueError(
                f"mobility_fault must be in [0, {self.n_faults}), got {mobility_fault}"
            )
        self.sheet_mobility, self._sheet_mobility_scalar = _parse_sheet_mobility(
            sheet_mobility, self.n_faults,
        )
        self.mobility_fault = int(mobility_fault)
        self.lift_mode = lift_mode
        self.winding_power = float(winding_power)
        self.confine_mode = confine_mode
        self.confine_width = None if confine_width is None else float(confine_width)
        self.confine_power = float(confine_power)
        self.tip_taper = float(tip_taper)
        if advect_traces is None:
            advect = (False,) + (True,) * (self.n_faults - 1)
        else:
            if len(advect_traces) != self.n_faults:
                raise ValueError(
                    f"advect_traces length {len(advect_traces)} != n_faults "
                    f"{self.n_faults}"
                )
            advect = tuple(bool(x) for x in advect_traces)
        self.advect_traces = advect
        if advect_offset < 0:
            raise ValueError(f"advect_offset must be >= 0, got {advect_offset}")
        self.advect_offset = float(advect_offset)
        if activation_windows is None:
            self.activation_windows = None
        else:
            if len(activation_windows) != self.n_faults:
                raise ValueError(
                    f"activation_windows length {len(activation_windows)} != "
                    f"n_faults {self.n_faults}"
                )
            wins = []
            for j, win in enumerate(activation_windows):
                t_on, t_off = float(win[0]), float(win[1])
                if not (0.0 <= t_on <= 1.0 and 0.0 <= t_off <= 1.0):
                    raise ValueError(
                        f"activation_windows[{j}] must lie in [0, 1], got {win}"
                    )
                if t_off < t_on:
                    raise ValueError(
                        f"activation_windows[{j}] must have t_off >= t_on, got {win}"
                    )
                wins.append((t_on, t_off))
            self.activation_windows = tuple(wins)
        for i, fault in enumerate(faults):
            if self.dim == 2:
                t = _tensor(fault)
                if t.ndim != 2 or t.shape[1] != 2 or t.shape[0] < 2:
                    raise ValueError(f"fault {i} must be (P >= 2, 2)")
                self.register_buffer(f"trace_{i}", t)
            else:
                verts, faces = fault
                v = _tensor(verts)
                f = _tensor(faces).long()
                if v.ndim != 2 or v.shape[1] != 3:
                    raise ValueError(f"fault {i} verts must be (V, 3)")
                if f.ndim != 2 or f.shape[1] != 3:
                    raise ValueError(f"fault {i} faces must be (F, 3)")
                self.register_buffer(f"trace_{i}", v)
                self.register_buffer(f"faces_{i}", f)
                self.register_buffer(
                    f"boundary_{i}", mesh_boundary_vertices(f),
                )

    def trace(self, i: int = 0) -> torch.Tensor:
        """Fault vertices — polyline stations (2-D) or mesh vertices (3-D)."""
        return getattr(self, f"trace_{i}")

    def faces(self, i: int = 0) -> torch.Tensor:
        """Triangle faces for fault ``i`` (3-D only)."""
        if self.dim != 3:
            raise AttributeError("faces are only defined for 3-D fault meshes")
        return getattr(self, f"faces_{i}")

    def _set_trace(self, i: int, verts: torch.Tensor) -> None:
        """In-place update of fault geometry (Lagrangian advection during integration)."""
        buf = getattr(self, f"trace_{i}")
        if verts.shape != buf.shape:
            raise ValueError(
                f"trace {i} shape {tuple(verts.shape)} != buffer {tuple(buf.shape)}"
            )
        buf.copy_(verts)

    def snapshot_traces(self) -> None:
        """Store present-day traces for :meth:`restore_traces` after integration."""
        self._trace_snapshot = [self.trace(i).clone() for i in range(self.n_faults)]

    def restore_traces(self) -> None:
        """Reset traces to the last :meth:`snapshot_traces` copy."""
        if not hasattr(self, "_trace_snapshot"):
            raise RuntimeError("restore_traces called without snapshot_traces")
        for i, tr in enumerate(self._trace_snapshot):
            self._set_trace(i, tr)

    @property
    def lift_dim(self) -> int:
        """Extra Euler-potential input dimension (1 for nested lift)."""
        return 1 if self.lift_mode == "nested" else self.n_faults

    def activation(self, t: Optional[float]) -> Optional[torch.Tensor]:
        """
        Per-fault temporal existence gate ``g_i(t) in [0, 1]`` (``None`` if disabled).

        ``g_i`` is a smoothstep rising from ``0`` at the nucleation time ``t_on``
        to ``1`` at ``t_off`` (fully formed) and held at ``1`` afterwards.  At the
        present (``t = 1``) every active fault is fully formed; toward the
        reference (``t → 0``) faults fade out, the later-nucleating ones first.
        """
        if self.activation_windows is None or t is None:
            return None
        gs = []
        for (t_on, t_off) in self.activation_windows:
            if t_off <= t_on:
                g = 1.0 if float(t) >= t_off else 0.0
            else:
                g = smoothstep_scalar((float(t) - t_on) / (t_off - t_on))
            gs.append(g)
        ref = self.trace(0)
        return torch.tensor(gs, device=ref.device, dtype=ref.dtype)

    def _hanging_weight(self, w: torch.Tensor) -> torch.Tensor:
        """Soft hanging-wall weight in [0, 1] from raw GWN label."""
        return self._mobility_weight(w, "positive")

    def _mobility_weight(
        self,
        w: torch.Tensor,
        mobility: str,
    ) -> torch.Tensor:
        """Slip weight μ ∈ [0, 1] from normalised GWN label."""
        if mobility == "both":
            return w.new_ones(w.shape)
        if mobility not in ("positive", "negative"):
            raise ValueError(
                f"mobility must be 'both', 'positive', or 'negative', got {mobility!r}"
            )
        s = (w / self.corridor_ref).clamp(-1.0, 1.0)
        t = ((s + 1.0) * 0.5) if mobility == "positive" else ((1.0 - s) * 0.5)
        return smoothstep(t) ** self.winding_power

    def _combine_mobility_weights(
        self, w_raw: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Scalar (N,) mobility from per-fault modes, or None when fully symmetric.

        Scalar ``sheet_mobility``: μ from ``mobility_fault`` only (legacy).
        Per-fault sequence: product of each non-``"both"`` fault's μ(w_i).
        """
        if all(m == "both" for m in self.sheet_mobility):
            return None
        if self._sheet_mobility_scalar:
            i = self.mobility_fault
            return self._mobility_weight(w_raw[:, i], self.sheet_mobility[i])
        mu = None
        for i, mob in enumerate(self.sheet_mobility):
            if mob == "both":
                continue
            m = self._mobility_weight(w_raw[:, i], mob)
            mu = m if mu is None else mu * m
        return mu

    def _velocity_mobility(self, w_raw: torch.Tensor) -> Optional[torch.Tensor]:
        """Combined μ for scaling confined velocity (None when symmetric)."""
        return self._combine_mobility_weights(w_raw)

    def _tip_taper_weight(
        self,
        x: torch.Tensor,
        geom: torch.Tensor,
        *,
        boundary: Optional[torch.Tensor] = None,
        closest: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Smooth weight in [0, 1]: 0 at the tips, 1 in the fault interior."""
        if self.tip_taper <= 0:
            return x.new_ones(x.shape[0])
        if self.dim == 2:
            L = (geom[1:] - geom[:-1]).norm(dim=-1).sum().clamp_min(1e-12)
            r = self.tip_taper * L
            p = x.detach()
            d = torch.minimum(
                (p - geom[0]).norm(dim=-1),
                (p - geom[-1]).norm(dim=-1),
            )
            return smoothstep(d / r)
        if boundary is None or len(boundary) == 0:
            return x.new_ones(x.shape[0])
        bnd = geom[boundary]
        span = (bnd.max(dim=0).values - bnd.min(dim=0).values).norm().clamp_min(1e-12)
        r = self.tip_taper * span
        d = (closest.unsqueeze(1) - bnd.unsqueeze(0)).norm(dim=-1).min(dim=1).values
        return smoothstep(d / r)

    def _confine_sigma(self, geom: torch.Tensor) -> float:
        """Characteristic distance σ for ``confine_mode='distance'``."""
        if self.confine_width is not None:
            return self.confine_width
        span = (geom.max(dim=0).values - geom.min(dim=0).values).max().item()
        return max(0.08 * span, 1e-3)

    def _confinement_weight(
        self, dist: torch.Tensor, geom: torch.Tensor, tip: torch.Tensor,
    ) -> torch.Tensor:
        """Distance-based tangency projection strength λ ∈ [0, 1]."""
        sigma = self._confine_sigma(geom)
        return torch.exp(-(dist / sigma) ** self.confine_power) * tip

    def _strike_frame(self, x: torch.Tensor, trace: torch.Tensor):
        """Closest point and unit strike tangent (2-D confinement normals)."""
        if self.strike_tangent == "closest":
            return closest_point_polyline(x, trace)
        return blended_tangency_polyline(
            x, trace, blend_power=self.strike_blend_power,
        )

    def _geometry_state(
        self, x: torch.Tensor, gate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Corridor geometry from **current** fault positions (influence, λ, n, taper, live w).

        ``gate`` (per-fault, in ``[0, 1]``) scales each fault's tip-taper, which
        propagates to corridor influence, confinement λ, and the lift level.
        """
        w_live, infl, conf, nrm, taper = [], [], [], [], []
        for i in range(self.n_faults):
            tr = self.trace(i)
            if self.dim == 2:
                w = gwn_polyline(x, tr)
                _, tang, dist = self._strike_frame(x, tr)
                tp = self._tip_taper_weight(x, tr)
                n_i = torch.stack([-tang[:, 1], tang[:, 0]], dim=-1)
            else:
                f = self.faces(i)
                w = gwn_mesh(x, tr, f)
                cp, dist, n_i = closest_point_mesh(x, tr, f)
                tp = self._tip_taper_weight(
                    x, tr, boundary=getattr(self, f"boundary_{i}"), closest=cp,
                )
            if gate is not None:
                tp = tp * gate[i]
            corridor = (
                (w.abs() / self.corridor_ref).clamp(0, 1) ** self.winding_power * tp
            )
            lam = corridor if self.confine_mode == "winding" else \
                self._confinement_weight(dist, tr, tp)
            w_live.append(w)
            taper.append(tp)
            infl.append(corridor)
            conf.append(lam)
            nrm.append(n_i)
        return (
            torch.stack(w_live, dim=-1),
            torch.stack(infl, dim=-1),
            torch.stack(conf, dim=-1),
            torch.stack(nrm, dim=1),
            torch.stack(taper, dim=-1),
        )

    @torch.no_grad()
    def state(
        self, x: torch.Tensor, gate: Optional[torch.Tensor] = None,
    ) -> SheetState:
        """All per-point fault quantities, one polyline sweep per fault."""
        w_raw, infl, conf, nrm, taper = self._geometry_state(x, gate=gate)
        w_active = self._active_lift(w_raw, infl, taper)
        mobility = self._velocity_mobility(w_raw)
        return SheetState(
            w_raw, w_active, infl, conf, nrm, mobility,
        )

    @torch.no_grad()
    def state_with_material(
        self,
        x: torch.Tensor,
        w_active: torch.Tensor,
        w_raw: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> SheetState:
        """
        Lagrangian labels with live corridor geometry.

        ``w_active`` and ``w_raw`` are the present-day material labels.
        Normals, confinement and influence follow the advected trace. With a
        ``gate``, the frozen side (``w_raw``) is kept and the lift level is
        recomputed from the live influence, so the jump closes with the
        barrier. Mobility is unchanged.
        """
        _, infl, conf, nrm, taper = self._geometry_state(x, gate=gate)
        if gate is not None:
            w_active = self._active_lift(w_raw, infl, taper)
        mobility = self._velocity_mobility(w_raw)
        return SheetState(w_raw, w_active, infl, conf, nrm, mobility)

    @torch.no_grad()
    def material_labels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Present-day ``(w_active, w_raw)`` to freeze for Lagrangian integration."""
        st = self.state(x)
        return st.w_active, st.w_raw

    @torch.no_grad()
    def trace_material_labels(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Per-trace ``(w_active, w_raw)`` at each vertex for trace advection."""
        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(self.n_faults):
            st = self.state(self.trace(i))
            out.append((st.w_active, st.w_raw))
        return out

    @torch.no_grad()
    def advect_probe_points(
        self,
        fault_index: int,
        w_raw: torch.Tensor,
        *,
        points: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Points at which to sample velocity when advecting fault ``fault_index``.

        On the cut itself, confinement leaves mostly tangential motion (self-slip).
        With ``advect_offset > 0``, probes sit slightly on the hanging-wall side.
        """
        tr = points if points is not None else self.trace(fault_index)
        if self.advect_offset <= 0:
            return tr
        wf = 0 if w_raw.ndim == 1 else int(fault_index)
        w_side = w_raw[:, wf] if w_raw.ndim > 1 else w_raw
        sign = torch.where(w_side >= 0, 1.0, -1.0).unsqueeze(-1)
        if self.dim == 2:
            _, tang, _ = self._strike_frame(tr, self.trace(fault_index))
            n = torch.stack([-tang[:, 1], tang[:, 0]], dim=-1)
        else:
            _, _, fn = closest_point_mesh(
                tr, self.trace(fault_index), self.faces(fault_index),
            )
            n = fn
        return tr + self.advect_offset * sign * n

    def _active_lift(
        self,
        w_raw: torch.Tensor,
        infl: torch.Tensor,
        taper: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lift coordinates fed to ``v(x, w)``.

        ``independent``: tip-tapered GWN labels — the 2*pi jump across each
        cut shrinks continuously to zero at the tips.
        ``nested``: a single level accumulated in trace order; each
        hanging-wall crossing adds that fault's (tip-tapered) influence.
        """
        if self.lift_mode == "nested":
            level = w_raw.new_zeros(w_raw.shape[0])
            for i in range(self.n_faults):
                level = level + infl[:, i] * self._hanging_weight(w_raw[:, i])
            mu = self._combine_mobility_weights(w_raw)
            if mu is not None:
                level = level * mu
            return level.unsqueeze(-1)
        w_active = w_raw * taper
        if any(m != "both" for m in self.sheet_mobility):
            chans = []
            for i, mob in enumerate(self.sheet_mobility):
                wi = w_active[:, i]
                if mob != "both":
                    wi = wi * self._mobility_weight(w_raw[:, i], mob)
                chans.append(wi)
            w_active = torch.stack(chans, dim=-1)
        return w_active
