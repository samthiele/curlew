"""
Restoration scalar field: φ(x) = φ_ref(Φ⁻¹ x), ∇φ = Jᵀ ∇φ_ref(Φ⁻¹ x).

``φ_ref`` defaults to a flat layer cake ``x[depth_axis]``. It may be any
scalar field or another restore event; the query recurses. Value-type losses
use a frozen reference reattached by :meth:`RestorationField._reattach`.
Orientation losses differentiate through ``∇φ_ref``.
"""
from __future__ import annotations

import math
import warnings
import weakref

from typing import List, Optional, Sequence, Tuple, Union

import curlew
import torch
import torch.nn as nn
import torch.nn.functional as F

from curlew.core import (
    Pebble,
    _tensor,
    cache_iq_worst_pairs,
    iq_pair_loss,
    resolve_string_weights,
    sample_iq_pairs,
)
from curlew.fields import BaseNF
from curlew.kinematics.flow import FlowOffset
from curlew.fields.series import FSF


class _FlatReferenceStratigraphy:
    __slots__ = ("depth_axis",)

    def __init__(self, depth_axis: int):
        self.depth_axis = int(depth_axis)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(x)
        g[..., self.depth_axis] = 1.0
        return g

    def value_and_gradient(self, x: torch.Tensor):
        return x[..., self.depth_axis], self.gradient(x)

    def __repr__(self):
        return f"FlatReference(depth_axis={self.depth_axis})"


class _ScalarFieldReference:
    __slots__ = ("field",)

    def __init__(self, field):
        self.field = field

    def value_and_gradient(self, x: torch.Tensor):
        with torch.enable_grad():
            xr = x.detach().clone().requires_grad_(True)
            g, v = self.field.gradient(
                xr, normalize=False, transform=False, return_value=True,
            )
        return v.detach(), g.detach()

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        g, _ = self.field.gradient(
            x, normalize=False, transform=False, return_value=True,
            create_graph=torch.is_grad_enabled(),
        )
        return g

    def __repr__(self):
        return f"ScalarFieldReference({getattr(self.field, 'name', self.field)!r})"


class _EventReference:
    """Keeps the referenced RestorationField out of this module's parameter tree."""

    __slots__ = ("field",)

    def __init__(self, field):
        self.field = field

    def value_and_gradient(self, x: torch.Tensor):
        return self.field.value_and_gradient(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return self.field.stratigraphy_gradient(x, frozen_reference=False)

    def __repr__(self):
        return f"EventReference({self.field.name!r})"


def _reference_field(ref):
    return ref.field if isinstance(ref, _EventReference) else None


def _as_reference(ref, depth_axis: int):
    if ref is None:
        return _FlatReferenceStratigraphy(depth_axis)
    if isinstance(ref, RestorationField):
        return _EventReference(ref)
    if hasattr(ref, "value_and_gradient"):
        return ref
    return _ScalarFieldReference(ref)


class _SplitOptim:
    """Several optimisers over disjoint parameter sets, driven as one."""

    def __init__(self, *opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


def pin_anchor(integrator, point, *, axes: Optional[Sequence[int]] = None):
    """Hold one present-day point fixed in the restored frame (post-hoc)."""
    return integrator.pin(_tensor(point), reduce="anchor", axes=axes)


def pin_mean(integrator, points, *, axes: Optional[Sequence[int]] = None):
    """Remove mean displacement over ``points`` (post-hoc)."""
    return integrator.pin(_tensor(points), reduce="mean", axes=axes)


class RestorationField(BaseNF):
    """
    ``φ = φ_ref ∘ Φ⁻¹`` from a :class:`~curlew.kinematics.flow.FlowOffset`.

    The integrator is the single owner of velocity factors and optional
    :class:`~curlew.kinematics.lift.FaultLift`. Use :func:`~curlew.geology.restore`
    to wrap this in a :class:`~curlew.geology.geoevent.GeoEvent`.
    """

    def initField(
        self,
        *,
        velocity: Union[nn.Module, Sequence[nn.Module]],
        n_steps: Union[int, Sequence[int]] = 20,
        integrator: str = "rk4",
        shear_table: Optional[int] = None,
        shear_split: str = "normal",
        shear_scheme: str = "strang",
        depth_axis: int = -1,
        init_std: Optional[float] = 0.01,
        signed_normals: bool = False,
        sigma_floor: float = 0.0,
        eq_form: str = "variance",
        collinear: bool = False,
        eq_scale: Optional[float] = None,
        eq_sample: Optional[Union[int, float]] = None,
        eq_sampling: str = "stratified",
        learning_rate: float = 1e-3,
        optimizer=None,
        optimizer_kwargs: Optional[dict] = None,
        constraints: Optional[Sequence] = None,
        fault_lift: Optional[nn.Module] = None,
        reference_event=None,
        pin=None,
        pin_axes: Optional[Sequence[int]] = None,
        **_ignored,
    ):
        if velocity is None:
            raise TypeError("RestorationField requires a `velocity` argument")
        factors = list(velocity) if isinstance(velocity, (list, tuple)) else [velocity]
        if not factors:
            raise ValueError("velocity list is empty")
        for i, m in enumerate(factors):
            label = f"velocity[{i}]" if len(factors) > 1 else "velocity"
            if not hasattr(m, "dim"):
                raise TypeError(f"{label} must expose a `dim` attribute")
            if not callable(getattr(m, "forward", None)):
                raise TypeError(f"{label} must implement forward(x, w=None, t=None)")
            if not callable(getattr(m, "forward_and_jacobian", None)):
                raise TypeError(
                    f"{label} must implement forward_and_jacobian(x, w=None, t=None)"
                )
        steps = list(n_steps) if isinstance(n_steps, (list, tuple)) else [n_steps] * len(factors)
        if len(steps) != len(factors):
            raise ValueError(
                f"n_steps has {len(steps)} entries but there are {len(factors)} "
                "velocity factors"
            )
        if any(int(s) < 0 for s in steps):
            raise ValueError("every n_steps entry must be >= 0 (0: one-way shear sweep)")
        if eq_sample is not None:
            if isinstance(eq_sample, float):
                if not (0.0 < eq_sample <= 1.0):
                    raise ValueError(
                        f"eq_sample fraction must be in (0, 1], got {eq_sample}"
                    )
            else:
                eq_sample = int(eq_sample)
                if eq_sample < 2 and eq_sample > 0:
                    raise ValueError(
                        f"eq_sample count must be >= 2 (or <= 0 to disable), got {eq_sample}"
                    )

        self.depth_axis = int(depth_axis) % self.input_dim
        self.signed_normals = bool(signed_normals)
        self.sigma_floor = float(sigma_floor)
        if eq_form not in ("variance", "rms", "log"):
            raise ValueError(f"eq_form must be 'variance', 'rms' or 'log', got {eq_form!r}")
        if eq_form == "log" and not (eq_scale and eq_scale > 0):
            raise ValueError("eq_form='log' needs a positive eq_scale (a length, the floor of the spread)")
        self.eq_form = eq_form
        self.collinear = bool(collinear)
        self.eq_scale = 0.0 if eq_scale is None else float(eq_scale)
        if eq_sampling not in ("stratified", "random"):
            raise ValueError(
                f"eq_sampling must be 'stratified' or 'random', got {eq_sampling!r}"
            )
        self.eq_sample = eq_sample
        self.eq_sampling = eq_sampling
        self.n_steps = int(steps[0]) if len(steps) == 1 else [int(s) for s in steps]
        self.constraints: List = list(constraints or [])
        self._learning_rate = float(learning_rate)
        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        self._pin = pin
        self._pin_axes = None if pin_axes is None else tuple(int(a) for a in pin_axes)
        self._fitting = False
        self._pin_applied = False

        self._integrator = FlowOffset(
            velocity if isinstance(velocity, (list, tuple)) else factors[0],
            n_steps=self.n_steps,
            fault_lift=fault_lift,
            integrator=integrator,
            shear_table=shear_table,
            shear_split=shear_split,
            shear_scheme=shear_scheme,
        )
        self.add_module("_integrator", self._integrator)

        if init_std is not None:
            init_std = float(init_std)
            if init_std < 0:
                raise ValueError(f"init_std must be non-negative or None, got {init_std}")
            with torch.no_grad():
                for m in self._integrator.velocities.modules():
                    if isinstance(m, FSF):
                        if m.activation is None:
                            if init_std == 0.0:
                                m.A_mu.zero_()
                                m.B_mu.zero_()
                            else:
                                m.A_mu.normal_(0.0, init_std)
                                m.B_mu.normal_(0.0, init_std)
                        else:
                            if init_std == 0.0:
                                # Identity flow with a common phase.
                                m.C_mu.zero_()
                                m.phi_mu.zero_()
                                m.bias.zero_()
                            else:
                                m.C_mu.normal_(0.0, init_std)

        self._reference = _FlatReferenceStratigraphy(self.depth_axis)
        self._reference_event = None
        self._reference_pending = reference_event

        self.to(curlew.device)
        if reference_event not in (None, "parent"):
            self.set_reference(reference_event)
            self._reference_pending = None
        elif any(p.requires_grad for p in self._integrator.velocities.parameters()):
            self._init_optim()

    def _init_optim(self):
        """
        ``optimizer=None`` keeps the default (Adam at ``learning_rate``). A class
        uses that class with ``optimizer_kwargs``. A dict
        ``{"amplitude": (cls, kw), "shape": (cls, kw)}`` splits the parameters:
        FSF mode amplitudes (``A_mu``, ``B_mu``, ``C_mu``) under the velocities
        go to ``"amplitude"``, everything else to ``"shape"``. ``kw`` may set
        ``lr``; it defaults to ``learning_rate``.
        """
        opt, kw = self._optimizer, self._optimizer_kwargs or {}
        if opt is None:
            self.init_optim(lr=self._learning_rate)
            return
        if not isinstance(opt, dict):
            self.init_optim(method=opt, lr=self._learning_rate, **kw)
            return
        if set(opt) != {"amplitude", "shape"}:
            raise ValueError(
                f"optimizer dict needs keys 'amplitude' and 'shape', got {sorted(opt)}"
            )
        amp_ids = {
            id(getattr(m, n))
            for m in FSF.iter_in(self._integrator.velocities)
            for n in ("A_mu", "B_mu", "C_mu")
            if hasattr(m, n)
        }
        params = [p for p in self.parameters() if p.requires_grad]
        groups = {
            "amplitude": [p for p in params if id(p) in amp_ids],
            "shape": [p for p in params if id(p) not in amp_ids],
        }
        opts = []
        for key in ("amplitude", "shape"):
            if not groups[key]:
                continue
            cls, okw = opt[key] if isinstance(opt[key], tuple) else (opt[key], {})
            okw = dict(okw)
            lr = okw.pop("lr", self._learning_rate)
            opts.append(cls(groups[key], lr=lr, **okw))
        self.optim = _SplitOptim(*opts)

    @property
    def integrator(self):
        return self._integrator

    @property
    def velocity(self):
        if self._integrator.n_factors > 1:
            return self._integrator.velocities
        return self._integrator.velocity

    @property
    def fault_lift(self):
        return self._integrator.fault_lift

    @property
    def pinned(self) -> bool:
        return bool(self._integrator.pinned)

    def unpin(self) -> None:
        self._integrator.unpin()
        self._pin_applied = False

    def _ensure_pinned(self) -> None:
        """Apply ``restore(pin=...)`` on first evaluation if fit never finished."""
        if self._pin is None or self._fitting or self._pin_applied:
            return
        self.pin_frame()

    def fit(self, *args, **kwargs):
        self._fitting = True
        try:
            return super().fit(*args, **kwargs)
        finally:
            self._fitting = False

    def pin_frame(
        self,
        pin=None,
        *,
        axes: Optional[Sequence[int]] = None,
        points: Optional[torch.Tensor] = None,
    ):
        spec = self._pin if pin is None else pin
        if spec is None:
            raise ValueError("no pin specified; pass pin= here or to restore(pin=...)")
        if axes is None:
            axes = self._pin_axes
        if not isinstance(self._reference, _FlatReferenceStratigraphy):
            warnings.warn(
                f"RestorationField {self.name!r} has a non-flat reference "
                "stratigraphy, so a translation of the restored frame is not a "
                "symmetry of the loss.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.unpin()
        if isinstance(spec, str):
            if spec != "mean":
                raise ValueError(
                    f"unknown pin {spec!r}; expected a point, a set of points, "
                    "'mean', or the name of a GeoEvent anchor"
                )
            pts = points if points is not None else self._pin_default_points()
            out = pin_mean(self._integrator, pts, axes=axes)
        else:
            pts = _tensor(spec, dev=curlew.device, dt=curlew.dtype)
            if pts.dim() == 1:
                pts = pts.unsqueeze(0)
            if pts.shape[0] == 1:
                out = pin_anchor(self._integrator, pts, axes=axes)
            else:
                out = pin_mean(self._integrator, pts, axes=axes)
        self._pin_applied = True
        return out

    def _pin_default_points(self) -> torch.Tensor:
        c = self.C
        chunks = []
        if c is not None:
            for attr in ("gp", "gop", "vp"):
                pts = getattr(c, attr, None)
                if pts is not None:
                    chunks.append(pts)
            if getattr(c, "eq", None) is not None:
                chunks += [t for t in c.eq if t is not None and t.shape[0] > 0]
        if not chunks:
            raise ValueError(
                "pin='mean' needs points: the bound CSet has no gp/gop/vp/eq "
                "constraints. Pass points= explicitly."
            )
        return torch.cat(chunks, dim=0)

    @property
    def reference(self):
        return self._reference

    @property
    def reference_event(self):
        return self._reference_event

    def set_reference(self, reference) -> None:
        from curlew.geology.geoevent import GeoEvent

        self._reference_event = None
        if reference is None:
            self._reference = _FlatReferenceStratigraphy(self.depth_axis)
            return

        field = reference
        if isinstance(reference, GeoEvent):
            self._reference_event = reference
            field = reference.field

        if isinstance(field, RestorationField):
            node = field
            while node is not None:
                if node is self:
                    name = getattr(reference, "name", field.name)
                    raise ValueError(
                        f"RestorationField {self.name!r} cannot reference "
                        f"{name!r}: {self.name!r} already appears upstream in "
                        "its reference chain (would create a cycle)."
                    )
                node = _reference_field(node._reference)

        self._reference = _as_reference(field, self.depth_axis)
        if self.optim is None and any(
            p.requires_grad for p in self._integrator.velocities.parameters()
        ):
            self._init_optim()

    def resolve_reference(self, parent_event) -> None:
        if getattr(self, "_reference_pending", None) == "parent":
            if parent_event is None:
                raise ValueError(
                    f"RestorationField {self.name!r} expects reference='parent' "
                    "but this GeoEvent has no parent"
                )
            self.set_reference(parent_event)
            self._reference_pending = None

    def _ensure_reference_wired(self) -> None:
        if getattr(self, "_reference_pending", None) == "parent":
            raise RuntimeError(
                f"RestorationField {self.name!r} was created with "
                f"reference='parent', but the parent restore event is not "
                f"wired yet. Build GeoModel([parent, {self.name!r}, ...]) or "
                f"call set_reference(parent) before evaluating the field."
            )

    def _reference_at(self, x0: torch.Tensor):
        self._ensure_reference_wired()
        with torch.no_grad():
            return self._reference.value_and_gradient(x0.detach())

    def _reference_gradient_at(
        self, x0: torch.Tensor, *, frozen: bool = False
    ) -> torch.Tensor:
        self._ensure_reference_wired()
        if frozen:
            with torch.no_grad():
                return self._reference.value_and_gradient(x0.detach())[1]
        return self._reference.gradient(x0)

    @staticmethod
    def _reattach(v_ref, g_ref, x0):
        return v_ref + (g_ref * (x0 - x0.detach())).sum(-1)

    def inverse_map(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_reference_wired()
        self._ensure_pinned()
        return self._integrator.inverse_map(x)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_pinned()
        return self._integrator.forward_map(x)

    def _evaluate_batch(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.inverse_map(x)
        v_ref, g_ref = self._reference_at(x0)
        return self._reattach(v_ref, g_ref, x0)

    def _gradient_batch(self, x: torch.Tensor) -> torch.Tensor:
        x0, j_full = self._integrator.inverse_map_with_jacobian(x)
        g_ref = self._reference_gradient_at(x0)
        return torch.bmm(j_full.mT, g_ref.unsqueeze(-1)).squeeze(-1)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return self._evaluate_batch(x)

    def restored_depth(self, x: torch.Tensor) -> torch.Tensor:
        return self.evaluate(x)

    def value_and_gradient(self, x: torch.Tensor):
        self._ensure_reference_wired()
        x0, j_full = self._integrator.inverse_map_with_jacobian(x)
        v_ref, g_ref = self._reference_at(x0)
        g = torch.bmm(j_full.mT, g_ref.unsqueeze(-1)).squeeze(-1)
        return self._reattach(v_ref, g_ref, x0), g

    def gradient(
        self,
        coords: torch.Tensor,
        normalize: bool = True,
        transform: bool = True,
        return_value: bool = False,
        retain_graph: bool = False,
        create_graph: bool = False,
        accumulate: bool = True,
    ):
        del retain_graph, create_graph, accumulate
        if transform and self.transform is not None:
            coords = self.transform(coords, end=transform)
        if return_value:
            v, g = self.value_and_gradient(coords)
        else:
            g = self._gradient_batch(coords)
            v = None
        if normalize:
            g = F.normalize(g, dim=-1)
        if return_value:
            return g, v
        return g

    def total_inverse_map(self, x: torch.Tensor) -> torch.Tensor:
        ref = _reference_field(self._reference)
        if ref is None:
            return self.inverse_map(x)
        return ref.total_inverse_map(self.inverse_map(x))

    def total_inverse_map_with_jacobian(self, x: torch.Tensor):
        self._ensure_pinned()
        ref = _reference_field(self._reference)
        if ref is None:
            return self._integrator.inverse_map_with_jacobian(x)
        x0, j = self._integrator.inverse_map_with_jacobian(x)
        x0, j_ref = ref.total_inverse_map_with_jacobian(x0)
        return x0, torch.bmm(j_ref, j)

    def total_forward_map(self, y: torch.Tensor) -> torch.Tensor:
        ref = _reference_field(self._reference)
        if ref is None:
            return self.forward_map(y)
        return self.forward_map(ref.total_forward_map(y))

    def retro_displacement(self, x: torch.Tensor) -> torch.Tensor:
        return self._integrator.disp(x, None)

    def stratigraphy_gradient(
        self, x: torch.Tensor, *, frozen_reference: bool = False
    ) -> torch.Tensor:
        self._ensure_reference_wired()
        x0, j_full = self._integrator.inverse_map_with_jacobian(x)
        g_ref = self._reference_gradient_at(x0, frozen=frozen_reference)
        return torch.bmm(j_full.mT, g_ref.unsqueeze(-1)).squeeze(-1)

    def predict_normals(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.stratigraphy_gradient(x), dim=-1)

    @staticmethod
    def _normal_loss(
        n_pred: torch.Tensor, n_obs: torch.Tensor, *, signed: bool
    ) -> torch.Tensor:
        """Squared angle between predicted and observed normals (θ²)."""
        o = F.normalize(n_obs, dim=-1)
        n = F.normalize(n_pred, dim=-1)
        dot = (n * o).sum(-1)
        if n.shape[-1] == 2:
            cross = n[..., 0] * o[..., 1] - n[..., 1] * o[..., 0]
            sin_mag = cross.abs()
        else:
            sin_mag = torch.linalg.cross(n, o, dim=-1).norm(dim=-1)
        theta = torch.atan2(sin_mag.clamp_min(1e-12), dot)
        if not signed:
            theta = torch.minimum(theta, math.pi - theta)
        return theta.pow(2).mean()

    @staticmethod
    def _eq_trace_loss(
        depths: torch.Tensor, sigma_floor: float, form: str = "variance", scale: float = 0.0
    ) -> torch.Tensor:
        """
        Spread of one horizon's restored depths about their detached mean.

        ``"variance"`` is the variance ``V`` above ``sigma_floor²``. Its pull
        shrinks with the residual. ``"rms"`` is ``√(V + ε²) − ε``: the pull
        is the residual over the current spread. ``"log"`` is
        ``log(1 + V / ε²)``: the pull grows as ``V`` shrinks, down to ``ε``.
        ``ε`` is ``scale`` (``eq_scale``).
        """
        mu = depths.mean().detach()
        mse = torch.relu((depths - mu).pow(2).mean() - sigma_floor**2)
        if form == "variance":
            return mse
        if form == "rms":
            return torch.sqrt(mse + scale**2 + 1e-24) - scale
        return torch.log1p(mse / scale**2)

    @staticmethod
    def _line_variance(points: torch.Tensor) -> torch.Tensor:
        """Variance about the best-fit line: ``tr(cov) − λ_max``.

        That is the smaller eigenvalue in 2-D and the sum of the two smaller
        ones in 3-D. The sum stays defined when the points are colinear and
        both perpendicular eigenvalues are zero.
        """
        centred = points - points.mean(dim=0)
        cov = centred.transpose(-1, -2) @ centred / points.shape[0]
        return cov.diagonal(dim1=-2, dim2=-1).sum(-1) - torch.linalg.eigvalsh(cov)[..., -1]

    @staticmethod
    def _strain_shape(J: torch.Tensor) -> torch.Tensor:
        """``B = F Fᵀ`` scaled to unit determinant. ``J`` is ``DΦ⁻¹``."""
        F_map = torch.linalg.inv(J)
        B = torch.matmul(F_map, F_map.transpose(-1, -2))
        det = torch.linalg.det(B).clamp_min(1e-12)
        dim = B.shape[-1]
        scale = det.sqrt() if dim == 2 else det.pow(1.0 / dim)
        return B / scale.unsqueeze(-1).unsqueeze(-1)

    @staticmethod
    def _axis_ratio(B_hat: torch.Tensor) -> torch.Tensor:
        """Major/minor eigenvalue ratio of a 2-D unit-determinant ellipse."""
        tr = B_hat.diagonal(dim1=-2, dim2=-1).sum(-1)
        disc = (tr * tr - 4).clamp_min(0)
        lam_max = (tr + disc.sqrt()) * 0.5
        return lam_max.pow(2).clamp_min(1.0)

    @staticmethod
    def _observed_ellipse(evecs_obs: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """Unit-determinant ellipse from minor-then-major columns and their ratio."""
        obs = F.normalize(evecs_obs, dim=-2)
        r = ratio.reshape(-1).clamp_min(1e-12)
        n_min = obs[..., 0]
        n_maj = obs[..., 1]
        return (
            r.rsqrt()[:, None, None] * n_min.unsqueeze(-1) * n_min.unsqueeze(-2)
            + r.sqrt()[:, None, None] * n_maj.unsqueeze(-1) * n_maj.unsqueeze(-2)
        )

    @staticmethod
    def _strain_ellipse_loss(
        J: torch.Tensor,
        evecs_obs: torch.Tensor = None,
        ratio_obs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Misfit of ``B = F Fᵀ`` (``F = (DΦ⁻¹)⁻¹``) against an observed ellipse.

        ``B`` is scaled to unit determinant, so the isotropic scale drops out.
        Columns of ``evecs_obs`` are minor then major. A ratio alone is compared
        through the trace (2-D; the axes are unseen). Eigenvectors alone are
        compared with the predicted ratio frozen, so only the axes are charged
        and a circle, which is the identity at this scaling, contributes no
        orientation. 2-D only.
        """
        if evecs_obs is None and ratio_obs is None:
            raise ValueError("strain loss needs eigenvectors or an eigenvalue ratio")
        if J.shape[-1] != 2:
            raise NotImplementedError(
                "strain-ellipse loss is 2-D only: a single eigenvalue ratio and "
                "two eigenvector columns do not determine a 3-D strain ellipsoid"
            )
        B_hat = RestorationField._strain_shape(J)
        if evecs_obs is None:
            tr = B_hat.diagonal(dim1=-2, dim2=-1).sum(-1)
            ratio = ratio_obs.reshape(-1).clamp_min(1e-12)
            target = ratio.sqrt() + ratio.rsqrt()
            return (tr - target).pow(2).mean()
        if ratio_obs is None:
            # Freeze the predicted ratio so its derivative (singular on a
            # circle) never enters; only the axes remain in the graph.
            ratio = RestorationField._axis_ratio(B_hat).detach()
            B_obs = RestorationField._observed_ellipse(evecs_obs, ratio)
        else:
            B_obs = RestorationField._observed_ellipse(evecs_obs, ratio_obs)
        return (B_hat - B_obs).pow(2).sum(dim=(-2, -1)).mean()

    def _eq_sample_count(self, n: int) -> int:
        s = self.eq_sample
        if s is None or s <= 0:
            return n
        if isinstance(s, float):
            k = int(round(s * n))
        else:
            k = int(s)
        return max(2, min(k, n))

    @staticmethod
    @torch.no_grad()
    def _eq_strata(points: torch.Tensor, k: int):
        """
        ``k`` compact strata of an unordered point set, by farthest-point sampling.

        Each point joins its nearest centre. The start is the point farthest
        from the centroid, so the partition is deterministic. Returns
        ``(members, offsets, counts)``: stratum ``j`` is
        ``members[offsets[j]:offsets[j]+counts[j]]``. Every stratum contains
        its centre.
        """
        n = points.shape[0]
        x = points.reshape(n, -1)
        dist = (x - x.mean(0)).pow(2).sum(-1)
        label = torch.zeros(n, dtype=torch.long, device=x.device)
        centre = int(dist.argmax())
        dist = (x - x[centre]).pow(2).sum(-1)
        # Mark the centre -1 so repeated points are not re-picked, which would empty its stratum.
        dist[centre] = -1.0
        for j in range(1, k):
            centre = int(dist.argmax())
            d = (x - x[centre]).pow(2).sum(-1)
            closer = d < dist
            closer[centre] = True
            label = torch.where(closer, j, label)
            dist = torch.minimum(dist, d)
            dist[centre] = -1.0
        members = torch.argsort(label, stable=True)
        counts = torch.bincount(label, minlength=k)
        offsets = torch.cumsum(counts, 0) - counts
        return members, offsets, counts

    def _subsample_eq_trace(self, trace: torch.Tensor) -> torch.Tensor:
        """
        ``k`` points of an ``eq`` horizon for this iteration.

        ``"stratified"`` (default) draws one member from each of ``k`` strata
        (:meth:`_eq_strata`), so each iteration covers the horizon and a region
        is weighted by its extent. The strata are cached. ``"random"`` draws
        a uniform subset without replacement.
        """
        n = int(trace.shape[0])
        k = self._eq_sample_count(n)
        if k >= n:
            return trace
        if self.eq_sampling == "random":
            idx = torch.randperm(n, device=trace.device)[:k]
            return trace.index_select(0, idx)

        cache = self.__dict__.setdefault("_eq_strata_cache", {})
        hit = cache.get(id(trace))
        if hit is None or hit[0]() is not trace or hit[1] != k:
            strata = self._eq_strata(trace.detach(), k)
            cache[id(trace)] = (weakref.ref(trace), k, strata)
        else:
            strata = hit[2]
        members, offsets, counts = strata
        u = torch.rand(k, device=trace.device, dtype=trace.dtype)
        pick = offsets + (u * counts).long().clamp_max(counts - 1)
        return trace.index_select(0, members[pick])

    def __getstate__(self):
        # Weak-ref caches cannot be pickled. They are rebuilt on first use.
        state = self.__dict__.copy()
        for key in ("_eq_strata_cache", "_staged_terms"):
            state.pop(key, None)
        return state

    def _active(self, h_weight) -> bool:
        return isinstance(h_weight, str) or (h_weight is not None and h_weight > 0)

    def _push_term(self, pebble: Pebble, key: str, value: torch.Tensor, h_weight):
        if not self._active(h_weight):
            return pebble
        staged = getattr(self, "_staged_terms", None)
        if staged is not None:  # weights resolved together in _push_staged
            if isinstance(value, torch.Tensor):
                staged.append((key, value))
            return pebble
        if not isinstance(value, torch.Tensor) or float(value.detach()) <= 0:
            return pebble
        h = getattr(self.H, key)
        if isinstance(h, str):
            h = float(h) * (1.0 / max(float(value.detach()), 1e-8))
            setattr(self.H, key, h)
        if h is not None and h > 0:
            pebble.push(self.name, key, value, weight=h, optim=self.optim)
        return pebble

    def _push_staged(self, pebble: Pebble) -> Pebble:
        """Resolve string weights of the staged terms together (``HSet.resolve_after > 0``) and push them."""
        staged, self._staged_terms = self._staged_terms, None
        values = {key: float(value.detach()) for key, value in staged}
        weights = resolve_string_weights(self, self.H, values)
        for key, value in staged:
            w = weights[key]
            if w is not None and w > 0 and values[key] > 0:
                pebble.push(self.name, key, value, weight=w, optim=self.optim)
        return pebble

    def _push_constraint(self, pebble: Pebble, constraint, cache) -> Pebble:
        label, raw, weight = constraint(self, cache)
        if not isinstance(raw, torch.Tensor) or float(raw.detach()) <= 0:
            return pebble
        if weight > 0:
            pebble.push(self.name, label, raw, weight=weight, optim=self.optim)
        return pebble

    def _maybe_transform(self, pts, transform):
        if transform and self.transform is not None:
            return self.transform(pts, end=transform)
        return pts

    def loss(self, transform: bool = True):
        """
        CSet terms after one pullback.

        ``gp``/``gv`` Nanson normals; ``gop``/``gov`` unsigned normals;
        ``vp``/``vv`` restored depth; ``eq`` per-trace depth variance;
        ``iq`` restored-depth ordering; ``sp`` with ``se`` and/or ``sr``
        strain ellipses (a missing one is left out). Plus :attr:`constraints`
        and ``H.energy_loss``.
        """
        if self.C is None:
            return Pebble()

        c = self.C
        h = self.H
        pebble = Pebble()
        self._ensure_reference_wired()
        # with HSet.resolve_after, terms are collected and weighted together at the end
        self._staged_terms = [] if getattr(h, "resolve_after", 0) else None

        chunks: List[torch.Tensor] = []
        sizes: List[int] = []
        kinds: List[str] = []

        has_gp = c.gp is not None and c.gv is not None and self._active(h.grad_loss)
        has_gop = c.gop is not None and c.gov is not None and self._active(h.ori_loss)
        has_vp = c.vp is not None and c.vv is not None and self._active(h.value_loss)

        if has_gp:
            chunks.append(self._maybe_transform(c.gp, transform))
            sizes.append(c.gp.shape[0])
            kinds.append("gp")
        if has_gop:
            chunks.append(self._maybe_transform(c.gop, transform))
            sizes.append(c.gop.shape[0])
            kinds.append("gop")
        if has_vp:
            chunks.append(self._maybe_transform(c.vp, transform))
            sizes.append(c.vp.shape[0])
            kinds.append("vp")

        n_eq = 0
        if c.eq is not None and self._active(h.eq_loss):
            for trace in c.eq:
                if trace is None or trace.shape[0] < 2:
                    continue
                tr = self._subsample_eq_trace(trace)
                chunks.append(self._maybe_transform(tr, transform))
                sizes.append(tr.shape[0])
                kinds.append("eq")
                n_eq += 1

        has_axes = c.se is not None
        has_ratio = c.sr is not None
        if (has_axes or has_ratio) and c.sp is None and self._active(h.strain_loss):
            raise ValueError(
                "strain eigenvectors or eigenvalue ratios need positions in CSet.sp"
            )
        has_strain = (
            c.sp is not None
            and (has_axes or has_ratio)
            and self._active(h.strain_loss)
        )
        if has_strain:
            chunks.append(self._maybe_transform(c.sp, transform))
            sizes.append(c.sp.shape[0])
            kinds.append("strain")

        has_iq = c.iq is not None and self._active(h.iq_loss)
        n_iq = 0
        ns = None
        six_list = eix_list = None
        reuse_frac = 0
        if has_iq:
            reuse_frac = getattr(h, "reuse_worst_half", 0) or 0
            ns = c.iq[0]
            six_list, eix_list, pts_list = sample_iq_pairs(
                c.iq, reuse_frac, getattr(self, "_last_iq_worst_indices", None)
            )
            all_pts = torch.cat(pts_list, dim=0)
            all_pts = self._maybe_transform(all_pts, transform)
            n_iq = len(pts_list) // 2
            chunks.append(all_pts)
            sizes.append(all_pts.shape[0])
            kinds.append("iq")

        if chunks:
            x_all = torch.cat(chunks, dim=0)
            need_j = has_gp or has_gop or has_strain or bool(self.constraints)
            # The Jacobian is built only when a constraint or a strain ellipse needs the matrix. Otherwise it is only ever contracted with g_ref.
            need_matrix = bool(self.constraints) or has_strain
            # gp/gop are a prefix of x_all, so the normal transport stops after n_ori rows.
            n_ori_chunks = sum(1 for k in kinds[:2] if k in ("gp", "gop"))
            n_ori = sum(sizes[:n_ori_chunks])
            if need_j:
                if need_matrix:
                    x0, j = self._integrator.inverse_map_with_jacobian(x_all)
                    pull = lambda g: torch.bmm(
                        j[:n_ori].mT, g.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    j = None
                    x0, pull = self._integrator.inverse_map_with_pullback(
                        x_all, jac_rows=n_ori
                    )
                if n_ori:
                    g_ref = self._reference_gradient_at(x0[:n_ori])
                    n_pred = F.normalize(pull(g_ref), dim=-1)
                else:
                    n_pred = None
                v_ref, g_ref_v = self._reference_at(x0)
                depths = self._reattach(v_ref, g_ref_v, x0)
            else:
                x0 = self.inverse_map(x_all)
                v_ref, g_ref_v = self._reference_at(x0)
                depths = self._reattach(v_ref, g_ref_v, x0)
                n_pred = None

            from curlew.kinematics.constraints import RestorationCache

            cache = RestorationCache(self)
            if need_j:
                # share this pullback with thickness priors keyed by the same tensors
                offset = 0
                for pts, sz in zip(chunks, sizes):
                    sl = slice(offset, offset + sz)
                    cache._inv[id(pts)] = x0[sl]
                    if j is not None:
                        cache._inv_jac[id(pts)] = (x0[sl], j[sl])
                    offset += sz

            # n_pred now covers only the orientation prefix, so it splits over
            # those chunks alone; the non-orientation kinds no longer pop it.
            parts_n = (
                list(n_pred.split(sizes[:n_ori_chunks]))
                if n_pred is not None
                else None
            )
            parts_x = list(x0.split(sizes))
            parts_d = list(depths.split(sizes))
            eq_terms = []
            offset = 0
            for i, (kind, sz) in enumerate(zip(kinds, sizes)):
                if kind == "gp":
                    pebble = self._push_term(
                        pebble, "grad_loss",
                        self._normal_loss(
                            parts_n.pop(0), c.gv, signed=self.signed_normals
                        ),
                        h.grad_loss,
                    )
                    parts_d.pop(0)
                elif kind == "gop":
                    pebble = self._push_term(
                        pebble, "ori_loss",
                        self._normal_loss(parts_n.pop(0), c.gov, signed=False),
                        h.ori_loss,
                    )
                    parts_d.pop(0)
                elif kind == "vp":
                    depths_vp = parts_d.pop(0)
                    target = c.vv.flatten()
                    if target.shape[0] == 1 and depths_vp.shape[0] > 1:
                        target = target.expand_as(depths_vp)
                    pebble = self._push_term(
                        pebble, "value_loss",
                        (depths_vp - target).pow(2).mean(),
                        h.value_loss,
                    )
                elif kind == "eq":
                    depth_term = self._eq_trace_loss(
                        parts_d.pop(0), self.sigma_floor, self.eq_form, self.eq_scale
                    )
                    if self.collinear and self.eq_form == "variance":
                        # Same units as the depth variance. Charge only scatter beyond the present-day line; a crooked trace keeps that allowance.
                        depth_term = depth_term + torch.relu(
                            self._line_variance(parts_x[i])
                            - self._line_variance(chunks[i]).detach()
                        )
                    eq_terms.append(depth_term)
                elif kind == "iq":
                    d = parts_d.pop(0).view(n_iq, 2, ns)
                    start_vals = d[:, 0, :].reshape(-1)
                    end_vals = d[:, 1, :].reshape(-1)
                    delta_all, l_iq = iq_pair_loss(
                        start_vals, end_vals, c._iq_low_clamp, c._iq_high_clamp,
                        getattr(c, "_iq_shift", None),
                    )
                    pebble = self._push_term(pebble, "iq_loss", l_iq, h.iq_loss)
                    self._last_iq_worst_indices = cache_iq_worst_pairs(
                        delta_all, ns, n_iq, six_list, eix_list, reuse_frac
                    )
                elif kind == "strain":
                    parts_d.pop(0)
                    pebble = self._push_term(
                        pebble, "strain_loss",
                        self._strain_ellipse_loss(
                            j[offset:offset + sz], c.se, c.sr
                        ),
                        h.strain_loss,
                    )
                offset += sz
            if eq_terms:
                pebble = self._push_term(
                    pebble, "eq_loss", torch.stack(eq_terms).mean(), h.eq_loss
                )
        else:
            cache = None

        if self.constraints:
            from curlew.kinematics.constraints import RestorationCache

            if cache is None:
                cache = RestorationCache(self)
            for constraint in self.constraints:
                pebble = self._push_constraint(pebble, constraint, cache)

        if self._active(h.energy_loss):
            l_en = FSF.kinetic_energy_on(self._integrator.velocities)
            if l_en is not None:
                pebble = self._push_term(pebble, "energy_loss", l_en, h.energy_loss)

        if self._staged_terms is not None:
            pebble = self._push_staged(pebble)
        return pebble

    def __repr__(self):
        ref = getattr(self.reference_event, "name", None)
        return (
            f"RestorationField(name={self.name!r}, dim={self.input_dim}, "
            f"depth_axis={self.depth_axis}, n_steps={self.n_steps}, "
            f"reference={ref!r})"
        )
