"""
Restoration scalar field: present-day stratigraphy as pullback of a flat reference
layer cake under a learned volume-preserving diffeomorphism (Clebsch flow).

This is the curlew restoration scalar field: the **field** defines
φ(x) = (Φ⁻¹ x)[depth_axis] (restored depth at modern coordinates), not a separate
kinematic offset.  Constraints on :class:`~curlew.core.CSet` are reinterpreted in
:meth:`loss` (Nanson normals, restored depths, horizon flatness on ``eq`` traces).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import curlew
import torch
import torch.nn as nn
import torch.nn.functional as F

from curlew.core import (
    HSet,
    Pebble,
    cache_iq_worst_pairs,
    iq_pair_loss,
    sample_iq_pairs,
)
from curlew.fields import BaseNF
from curlew.fields.series import FSF


class RestorationField(BaseNF):
    """
    Implicit scalar field from a learned diffeomorphic restoration map.

    Reference stratigraphy is a flat layer cake φ₀(x) = x[depth_axis].  At
    present-day positions x₁ the model evaluates

        φ(x₁) = (Φ⁻¹ x₁)[depth_axis]

    where Φ is obtained by integrating a :class:`~curlew.fields.clebsch.ClebschVelocity`
    field.  Bedding normals follow Nanson's relation on the inverse Jacobian.

    Parameters passed through ``initField`` / constructor keywords
    ----------------------------------------------------------------
    velocity, n_steps, depth_axis, init_std, signed_normals, sigma_floor,
    learning_rate, fault_lift, constraints.

    Notes
    -----
    ``velocity`` must be supplied explicitly (e.g. from
    :func:`~curlew.fields.clebsch.clebsch_velocity` or
    :func:`~curlew.fields.clebsch.temporal_clebsch_velocity`); it is not
    constructed automatically.  Set ``H.kl_loss`` (default 0, disabled) to weight
    the Bayes-by-Backprop complexity cost from nested :class:`FSF` potentials
    in :meth:`loss`.

    Use :func:`~curlew.geology.restore` to wrap this field in a :class:`GeoEvent`.
    Like :func:`~curlew.geology.strati`, ``restore`` builds a generative event
    (an :class:`~curlew.geology.interactions.Overprint` over the restored-depth
    scalar), so isosurfaces/lithology work exactly as for any other
    stratigraphic package; it additionally sets ``event.deformation =
    event.field.integrator`` so younger events are undeformed through the same
    :class:`~curlew.geology.interactions.FlowOffset` (and parameters) used for
    scalar evaluation. :meth:`retro_displacement` remains a convenience wrapper
    around ``self.integrator.disp(x, None)`` for direct use.
    """

    def initField(
        self,
        *,
        velocity: nn.Module,
        n_steps: int = 20,
        depth_axis: int = -1,
        init_std: float = 0.01,
        signed_normals: bool = False,
        sigma_floor: float = 1.0,
        learning_rate: float = 1e-3,
        constraints: Optional[Sequence] = None,
        fault_lift: Optional[nn.Module] = None,
    ):
        """
        Wire a pre-built velocity field and internal flow integrator.

        Parameters
        ----------
        velocity
            Velocity module (must expose ``dim``, ``forward``, and
            ``forward_and_jacobian``).
        n_steps
            RK2 / Magnus integration steps over ``t ∈ [0, 1]``.
        depth_axis
            Reference coordinate index treated as stratigraphic depth.
        init_std
            Std of ``A`` on 3-D potentials after construction (breaks α×β degeneracy).
        signed_normals
            If True, bedding normal loss uses ``1 − cos θ``; else ``1 − cos² θ``.
        sigma_floor
            Hinge for horizon / ``eq`` trace flatness in restored depth (restored-depth
            units; see :meth:`_eq_trace_loss`).
        constraints
            Optional sequence of :class:`~curlew.fields.restoration_constraints.RestorationConstraint`
            instances (thickness priors, etc.) evaluated each :meth:`loss` call.
        learning_rate
            Adam learning rate for :meth:`init_optim`.
        fault_lift
            Optional :class:`~curlew.fields.lift.FaultLift` for lifted restoration.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if velocity is None:
            raise TypeError("RestorationField requires a `velocity` argument")
        if not hasattr(velocity, "dim"):
            raise TypeError("velocity must expose a `dim` attribute")
        if not callable(getattr(velocity, "forward", None)):
            raise TypeError("velocity must implement forward(x, w=None, t=None)")
        if not callable(getattr(velocity, "forward_and_jacobian", None)):
            raise TypeError("velocity must implement forward_and_jacobian(x, w=None, t=None)")

        self.depth_axis = int(depth_axis) % self.input_dim
        self.signed_normals = bool(signed_normals)
        self.sigma_floor = float(sigma_floor)
        self.n_steps = int(n_steps)
        self.constraints: List = list(constraints or [])
        self._learning_rate = float(learning_rate)

        self.velocity = velocity
        self.add_module("velocity", velocity)
        if fault_lift is not None:
            self.fault_lift = fault_lift
            self.add_module("fault_lift", fault_lift)

        if self.input_dim == 3 and init_std > 0:
            with torch.no_grad():
                for m in self.velocity.modules():
                    if isinstance(m, FSF):
                        m.A_mu.data.normal_(0.0, init_std)

        from curlew.geology.interactions import FlowOffset

        self._integrator = FlowOffset(
            self.velocity, n_steps=self.n_steps, fault_lift=fault_lift,
        )
        self.add_module("_integrator", self._integrator)

        self.to(curlew.device)
        if any(p.requires_grad for p in self.velocity.parameters()):
            self.init_optim(lr=learning_rate)

    # ── Flow / scalar evaluation ────────────────────────────────────────────

    @property
    def integrator(self):
        """Internal :class:`~curlew.geology.interactions.FlowOffset` (Φ, Φ⁻¹)."""
        return self._integrator

    def inverse_map(self, x: torch.Tensor) -> torch.Tensor:
        """Present → reference coordinates ``Φ⁻¹(x)``."""
        return self._integrator.inverse_map(x)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """Reference → present coordinates ``Φ(x)``."""
        return self._integrator.forward_map(x)

    def restored_depth(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar φ(x) = restored stratigraphic depth at present-day ``x``."""
        return self.inverse_map(x)[:, self.depth_axis]

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate restored depth φ(x) — :class:`BaseNF` entry point."""
        return self.restored_depth(x)

    def retro_displacement(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper: ``x_paleo = x + retro_displacement(x)``.

        Equivalent to ``self.integrator.disp(x, None)``, which is what
        :meth:`~curlew.geology.geoevent.GeoEvent.undeform` calls (via the
        ordinary ``deformation`` path) once ``event.deformation`` is set to
        ``self.integrator`` by :func:`~curlew.geology.restore`.
        """
        return self._integrator.disp(x, None)

    @staticmethod
    def _depth_gradient(x0: torch.Tensor, depth_axis: int) -> torch.Tensor:
        """∇φ₀ for reference stratigraphy φ₀(x) = x[depth_axis]."""
        g = torch.zeros_like(x0)
        g[:, depth_axis] = 1.0
        return g

    @staticmethod
    def _nanson(J_inv: torch.Tensor, g0: torch.Tensor) -> torch.Tensor:
        """Transport reference covector to present: ``normalise(J⁻ᵀ g₀)``."""
        n = torch.bmm(J_inv.mT, g0.unsqueeze(-1)).squeeze(-1)
        return F.normalize(n, dim=-1)

    def predict_normals(self, x: torch.Tensor) -> torch.Tensor:
        """Unit bedding normals at present-day positions (Nanson)."""
        x0, j_inv = self._integrator.inverse_map_with_jacobian(x)
        return self._nanson(j_inv, self._depth_gradient(x0, self.depth_axis))

    @staticmethod
    def _normal_loss(
        n_pred: torch.Tensor, n_obs: torch.Tensor, *, signed: bool
    ) -> torch.Tensor:
        d = (n_pred * F.normalize(n_obs, dim=-1)).sum(-1)
        return (1.0 - d).mean() if signed else (1.0 - d.pow(2)).mean()

    @staticmethod
    def _eq_trace_loss(depths: torch.Tensor, sigma_floor: float) -> torch.Tensor:
        """
        Flatten each ``eq`` trace in restored depth.

        Per-trace mean is detached so the loss reduces spread along the trace
        through the flow parameters only (detached per-trace mean).
        """
        mu = depths.mean().detach()
        mse = (depths - mu).pow(2).mean()
        return torch.relu(mse - sigma_floor**2)

    def _active(self, h_weight) -> bool:
        return isinstance(h_weight, str) or (h_weight is not None and h_weight > 0)

    def _push_term(self, pebble: Pebble, key: str, value: torch.Tensor, h_weight):
        """Resolve string hyperparameters and append one loss term."""
        if not self._active(h_weight):
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

    def _push_constraint(self, pebble: Pebble, constraint, cache) -> Pebble:
        label, raw, weight = constraint(self, cache)
        if not isinstance(raw, torch.Tensor) or float(raw.detach()) <= 0:
            return pebble
        if weight > 0:
            pebble.push(self.name, label, raw, weight=weight, optim=self.optim)
        return pebble

    def loss(self, transform: bool = True):
        """
        Restoration losses from bound :class:`~curlew.core.CSet` constraints.

        ``gp``/``gv`` → bedding normals (Nanson, weight ``H.grad_loss``).
        ``gop``/``gov`` → sign-invariant normals (weight ``H.ori_loss``).
        ``vp``/``vv`` → restored depth MSE (weight ``H.value_loss``).
        ``eq`` traces → restored-depth variance along each trace, hinged by
        ``sigma_floor`` (weight ``H.eq_loss``; see :meth:`_eq_trace_loss`).
        Pluggable :attr:`constraints` (thickness priors, etc.) are evaluated
        after the core ``CSet`` terms.
        Bayes-by-Backprop complexity cost from every nested ``FSF`` potential
        in ``self.velocity`` (weight ``H.kl_loss``; 0 by default).
        Analytic path (kinetic) energy of the velocity potentials
        (weight ``H.energy_loss``; 0 by default — see
        :meth:`~curlew.fields.series.FSF.kinetic_energy` and, for temporal
        potentials, the path-averaged
        :meth:`~curlew.fields.series.TemporalFSF.kinetic_energy`).
        ``iq`` pairs → restored-depth ordering via ``Φ⁻¹`` (weight
        ``H.iq_loss``; same :class:`~curlew.core.CSet` format as scalar fields).

        Standard scalar-interpolator terms (``mono_loss``, ``thick_loss``,
        ``flat_loss``) are not used.
        """
        if self.C is None:
            return Pebble()

        c = self.C
        h = self.H
        pebble = Pebble()

        # ── Bedding normals (gp / gv) ───────────────────────────────────────
        if (
            c.gp is not None
            and c.gv is not None
            and self._active(h.grad_loss)
        ):
            gp = c.gp
            if transform and self.transform is not None:
                gp = self.transform(gp, end=transform)
            n_pred = self.predict_normals(gp)
            l_grad = self._normal_loss(
                n_pred, c.gv, signed=self.signed_normals
            )
            pebble = self._push_term(pebble, "grad_loss", l_grad, h.grad_loss)

        # ── Orientation-only normals (gop / gov) ────────────────────────────
        if (
            c.gop is not None
            and c.gov is not None
            and self._active(h.ori_loss)
        ):
            gop = c.gop
            if transform and self.transform is not None:
                gop = self.transform(gop, end=transform)
            n_pred = self.predict_normals(gop)
            l_ori = self._normal_loss(n_pred, c.gov, signed=False)
            pebble = self._push_term(pebble, "ori_loss", l_ori, h.ori_loss)

        # ── Fused inverse map for value + eq traces ─────────────────────────
        chunks: List[torch.Tensor] = []
        sizes: List[int] = []
        has_vp = (
            c.vp is not None
            and c.vv is not None
            and self._active(h.value_loss)
        )
        eq_traces: List[torch.Tensor] = []
        if c.eq is not None and self._active(h.eq_loss):
            for trace in c.eq:
                if trace is not None and trace.shape[0] >= 2:
                    eq_traces.append(trace)

        if has_vp:
            vp = c.vp
            if transform and self.transform is not None:
                vp = self.transform(vp, end=transform)
            chunks.append(vp)
            sizes.append(vp.shape[0])

        eq_sizes: List[int] = []
        if eq_traces:
            for trace in eq_traces:
                tr = trace
                if transform and self.transform is not None:
                    tr = self.transform(tr, end=transform)
                eq_sizes.append(tr.shape[0])
                chunks.append(tr)
                sizes.append(tr.shape[0])

        if chunks:
            x0 = self.inverse_map(torch.cat(chunks, dim=0))
            depth_parts = list(x0[:, self.depth_axis].split(sizes))

            if has_vp:
                depths_vp = depth_parts.pop(0)
                target = c.vv.flatten()
                if target.shape[0] == 1 and depths_vp.shape[0] > 1:
                    target = target.expand_as(depths_vp)
                l_val = (depths_vp - target).pow(2).mean()
                pebble = self._push_term(pebble, "value_loss", l_val, h.value_loss)

            if eq_traces and self._active(h.eq_loss):
                eq_terms = [
                    self._eq_trace_loss(d, self.sigma_floor)
                    for d in depth_parts
                ]
                if eq_terms:
                    l_eq = torch.stack(eq_terms).mean()
                    pebble = self._push_term(pebble, "eq_loss", l_eq, h.eq_loss)

        # ── Inequality constraints (restored depth via inverse_map) ───────────
        if c.iq is not None and self._active(h.iq_loss):
            reuse_frac = getattr(h, "reuse_worst_half", 0) or 0
            ns = c.iq[0]
            six_list, eix_list, pts_list = sample_iq_pairs(
                c.iq, reuse_frac, getattr(self, "_last_iq_worst_indices", None)
            )
            all_pts = torch.cat(pts_list, dim=0)
            if transform and self.transform is not None:
                all_pts = self.transform(all_pts, end=transform)
            n_iq = len(pts_list) // 2
            depths = self.restored_depth(all_pts).view(n_iq, 2, ns)
            start_vals = depths[:, 0, :].reshape(-1)
            end_vals = depths[:, 1, :].reshape(-1)
            delta_all, l_iq = iq_pair_loss(
                start_vals, end_vals, c._iq_low_clamp, c._iq_high_clamp
            )
            pebble = self._push_term(pebble, "iq_loss", l_iq, h.iq_loss)
            self._last_iq_worst_indices = cache_iq_worst_pairs(
                delta_all, ns, n_iq, six_list, eix_list, reuse_frac
            )

        # ── Pluggable restoration constraints ────────────────────────────────
        if self.constraints:
            from curlew.fields.restoration_constraints import RestorationCache

            cache = RestorationCache(self)
            for constraint in self.constraints:
                pebble = self._push_constraint(pebble, constraint, cache)

        # ── Bayes-by-Backprop complexity cost (nested FSF potentials) ────────
        if self._active(h.kl_loss):
            l_kl = FSF.kl_loss_on(self.velocity)
            if l_kl is not None:
                pebble = self._push_term(pebble, "kl_loss", l_kl, h.kl_loss)

        # ── Path (kinetic) energy of the velocity potentials ─────────────────
        if self._active(h.energy_loss):
            l_en = FSF.kinetic_energy_on(self.velocity)
            if l_en is not None:
                pebble = self._push_term(pebble, "energy_loss", l_en, h.energy_loss)

        return pebble

    def __repr__(self):
        return (
            f"RestorationField(name={self.name!r}, dim={self.input_dim}, "
            f"depth_axis={self.depth_axis}, n_steps={self.n_steps})"
        )
