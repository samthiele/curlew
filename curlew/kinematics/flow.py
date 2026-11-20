"""
Integration of Euler (or other divergence-free) velocity fields.

:class:`FlowOffset` is the single owner of velocity factors, optional
:class:`~curlew.kinematics.lift.FaultLift`, and the restored-frame pin. One
Lagrangian fault session wraps a whole composite (K-band) traversal.
"""
from __future__ import annotations

import math
import weakref
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

import curlew
from curlew import _tensor
from curlew.core import LearnableBase
from curlew.kinematics.lift import SheetState
from curlew.fields.series import _sincos


def _checkVectorShape(out: torch.Tensor, refPoints: torch.Tensor) -> torch.Tensor:
    if not isinstance(out, torch.Tensor):
        out = _tensor(out)
    if out.dim() == 1:
        raise ValueError("Vector field expected, got scalar field?")
    if out.shape[-1] != refPoints.shape[-1]:
        raise ValueError(
            f"Latent field output dimension {out.shape} does not match "
            f"coordinate dimension {refPoints.shape}"
        )
    return out


def rk2_step(x, v0, v_mid, sign, dt):
    """One midpoint update: ``x + sign * dt * v_mid`` after ``v0`` built ``x_mid``."""
    del v0
    return x + sign * dt * v_mid


def _same_tensor(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact equality that is ``False`` (not an error) across shape/device/dtype."""
    return (
        a.shape == b.shape and a.device == b.device and a.dtype == b.dtype
        and bool(torch.equal(a, b.detach()))
    )


class FlowOffset(LearnableBase):
    """
    Integrate one or more velocities over ``τ ∈ [0, 1]``.

    ``velocity`` exposes ``dim``, ``forward`` and ``forward_and_jacobian``.
    A sequence is a composition of unit flows, coarse to fine, not a sum.
    ``n_steps`` broadcasts. Global ``τ`` labels :meth:`map_to_time` and
    :class:`~curlew.kinematics.lift.FaultLift` windows.

    ``n_steps=0`` (shear only) is a one-way sweep: each direction group once,
    at full amplitude, in :meth:`_shear_groups` order; the inverse reverses
    it. The leading interaction between groups is kept (a palindrome cancels
    it). Still exactly volume preserving and invertible.

    ``shear_scheme`` (shear, ``n_steps ≥ 1``): ``"strang"`` (default) is ``N``
    palindromes, second order; ``"lie"`` is ``N`` one-way sweeps at ``1/N``,
    first order, and at ``N = 1`` the same map as ``n_steps=0``;
    ``"yoshida"`` is three palindromes of durations ``w1, w0, w1``
    (``w0 < 0``), fourth order. All three stay exactly volume preserving and
    converge to the stationary flow as ``N`` grows.

    ``fault_lift`` is optional. One snapshot wraps the whole traversal, so
    composite bands share frozen labels and continuously advected traces.
    """

    _SHEAR_SCHEMES = ("strang", "lie", "yoshida")

    def __init__(
        self,
        velocity: Union[nn.Module, Sequence[nn.Module]],
        *,
        n_steps: Union[int, Sequence[int]] = 20,
        direction: float = -1.0,
        lift_w: Optional[torch.Tensor] = None,
        fault_lift: Optional[nn.Module] = None,
        confine: Optional[bool] = None,
        lagrangian_faults: Optional[bool] = None,
        integrator: str = "rk4",
        shear_table: Optional[int] = None,
        shear_split: str = "normal",
        shear_scheme: str = "strang",
    ):
        super().__init__()
        if shear_scheme not in self._SHEAR_SCHEMES:
            raise ValueError(
                f"shear_scheme must be one of {self._SHEAR_SCHEMES}, got {shear_scheme!r}"
            )
        # How n_steps >= 1 sweeps are composed; see :meth:`_shear_schedule`.
        self.shear_scheme = shear_scheme
        if shear_split not in ("normal", "phase"):
            raise ValueError(
                f"shear_split must be 'normal' or 'phase', got {shear_split!r}"
            )
        # Normal split for two FSF slots; a hinge or 2-D field stays on the phase split.
        self.shear_split = shear_split
        if integrator not in ("rk2", "rk4", "shear"):
            raise ValueError(
                f"integrator must be 'rk2', 'rk4' or 'shear', got {integrator!r}"
            )
        self.integrator = integrator
        # Size of the per-group 1-D profile table; None = evaluate the modes
        # directly. See :meth:`_shear_tables`.
        self.shear_table = None if shear_table is None else int(shear_table)
        if isinstance(velocity, (list, tuple, nn.ModuleList)):
            mods = list(velocity)
            if not mods:
                raise ValueError("velocity list is empty")
        else:
            mods = [velocity]

        self.dim = None
        for i, m in enumerate(mods):
            if not hasattr(m, "dim"):
                raise TypeError(f"velocity[{i}] must expose a `dim` attribute")
            if not callable(getattr(m, "forward", None)):
                raise TypeError(
                    f"velocity[{i}] must implement forward(x, w=None, t=None)"
                )
            if self.dim is None:
                self.dim = int(m.dim)
            elif int(m.dim) != self.dim:
                raise ValueError(
                    f"all velocity factors must share `dim`; velocity[{i}] has "
                    f"{int(m.dim)}, expected {self.dim}"
                )
        self.velocities = nn.ModuleList(mods)

        n_factors = len(mods)
        if isinstance(n_steps, (list, tuple)):
            steps = [int(s) for s in n_steps]
            if len(steps) != n_factors:
                raise ValueError(
                    f"n_steps has {len(steps)} entries but there are "
                    f"{n_factors} velocity factors"
                )
        else:
            steps = [int(n_steps)] * n_factors
        self._check_steps(steps, fault_lift)
        self.steps = steps

        self.direction = float(direction)
        if fault_lift is not None:
            self.fault_lift = fault_lift
            self.add_module("fault_lift", fault_lift)
        else:
            self.fault_lift = None
        self.confine = (fault_lift is not None) if confine is None else bool(confine)
        if lagrangian_faults is None:
            lagrangian_faults = fault_lift is not None
        if lagrangian_faults and fault_lift is None:
            raise ValueError("lagrangian_faults requires fault_lift")
        self.lagrangian_faults = bool(lagrangian_faults)
        self.lift_w = None
        if fault_lift is not None and lift_w is not None:
            raise ValueError("pass fault_lift or lift_w, not both")
        if lift_w is not None:
            w = _tensor(lift_w, dev=curlew.device, dt=curlew.dtype)
            if w.dim() == 1:
                w = w.unsqueeze(0)
            self.register_buffer("lift_w", w)

        self.register_buffer(
            "frame_offset",
            torch.zeros(self.dim, dtype=curlew.dtype, device=curlew.device),
        )

    def eval(self, x, G):
        return self.disp(x, G)

    @property
    def pinned(self) -> bool:
        return bool(self.frame_offset.any())

    def unpin(self) -> "FlowOffset":
        with torch.no_grad():
            self.frame_offset.zero_()
        return self

    @torch.no_grad()
    def pin(
        self,
        points: torch.Tensor,
        *,
        reduce: str = "anchor",
        axes: Optional[Sequence[int]] = None,
    ) -> "FlowOffset":
        if reduce not in ("anchor", "mean"):
            raise ValueError(f"reduce must be 'anchor' or 'mean', got {reduce!r}")
        pts = points if isinstance(points, torch.Tensor) else _tensor(points)
        if pts.dim() == 1:
            pts = pts.unsqueeze(0)
        if reduce == "anchor" and pts.shape[0] != 1:
            raise ValueError(
                f"reduce='anchor' expects a single point, got {pts.shape[0]}; "
                "use reduce='mean' to average over several"
            )
        self.unpin()
        disp = self.inverse_map(pts) - pts
        c = disp[0] if reduce == "anchor" else disp.mean(0)
        if axes is not None:
            mask = torch.zeros_like(c)
            mask[list(axes)] = 1.0
            c = c * mask
        self.frame_offset.copy_(c)
        return self

    @property
    def n_factors(self) -> int:
        return len(self.velocities)

    @property
    def velocity(self) -> nn.Module:
        if len(self.velocities) > 1:
            raise AttributeError(
                f"this is a composite flow with {len(self.velocities)} factors; "
                "use `.velocities` instead of `.velocity`"
            )
        return self.velocities[0]

    @property
    def n_steps(self) -> Union[int, List[int]]:
        return self.steps[0] if len(self.steps) == 1 else list(self.steps)

    @n_steps.setter
    def n_steps(self, value: Union[int, Sequence[int]]) -> None:
        if isinstance(value, (list, tuple)):
            steps = [int(s) for s in value]
            if len(steps) != self.n_factors:
                raise ValueError(
                    f"n_steps has {len(steps)} entries but there are "
                    f"{self.n_factors} velocity factors"
                )
        else:
            steps = [int(value)] * self.n_factors
        self._check_steps(steps, self.fault_lift)
        self.steps = steps

    def _check_steps(self, steps: Sequence[int], fault_lift) -> None:
        # 0 means a one-way shear sweep, which only the shear integrator has.
        if any(s < 0 for s in steps):
            raise ValueError("every n_steps entry must be >= 0")
        if 0 in steps:
            if self.integrator != "shear":
                raise ValueError(
                    "n_steps=0 (a one-way shear sweep) needs integrator='shear'"
                )
            if fault_lift is not None:
                raise ValueError("n_steps=0 is not supported with a fault_lift")

    def _plan(self, sign: float) -> List[Tuple[nn.Module, int, Tuple[float, float]]]:
        k = self.n_factors
        mods = list(self.velocities)
        order = range(k) if sign > 0 else range(k - 1, -1, -1)
        plan = []
        for i in order:
            lo, hi = i / k, (i + 1) / k
            span = (lo, hi) if sign > 0 else (hi, lo)
            plan.append((mods[i], self.steps[i], span))
        return plan

    def _sign(self) -> float:
        return 1.0 if self.direction > 0 else -1.0

    def _open_fault_session(self, x):
        if self.fault_lift is None or not self.lagrangian_faults:
            return None, None, False
        self.fault_lift.snapshot_traces()
        return (
            self.fault_lift.material_labels(x),
            self.fault_lift.trace_material_labels(),
            True,
        )

    def _close_fault_session(self, opened: bool) -> None:
        if opened:
            self.fault_lift.restore_traces()

    def __getstate__(self):
        # Weak-ref caches cannot be pickled. They are rebuilt from frozen buffers.
        state = self.__dict__.copy()
        for key in ("_shear_group_cache", "_ridge_cache"):
            state.pop(key, None)
        return state

    def _run(
        self,
        x: torch.Tensor,
        *,
        sign: float,
        jac: bool = False,
        duration: float = 1.0,
        G=None,
        stack=None,
        jac_rows=None,
    ):
        material, trace_mat, opened = self._open_fault_session(x)
        J = None
        try:
            for vel, n, span in self._plan(sign):
                if jac:
                    x, Jf = self._integrate(
                        x, sign=sign, jac=True, duration=duration,
                        n_steps=n, G=G, velocity=vel, t_span=span,
                        material=material, trace_mat=trace_mat,
                    )
                    J = Jf if J is None else torch.bmm(Jf, J)
                else:
                    x = self._integrate(
                        x, sign=sign, jac=False, duration=duration,
                        n_steps=n, G=G, velocity=vel, t_span=span,
                        material=material, trace_mat=trace_mat, stack=stack,
                        jac_rows=jac_rows,
                    )
        finally:
            self._close_fault_session(opened)
        return (x, J) if jac else x

    def _velocity_at(
        self,
        x: torch.Tensor,
        t: float,
        *,
        jac: bool = False,
        G=None,
        material: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        gate: Optional[torch.Tensor] = None,
        velocity: Optional[nn.Module] = None,
    ):
        del G
        vel = velocity if velocity is not None else self.velocities[0]
        if self.fault_lift is not None:
            return self._velocity_with_fault(
                x, t, jac=jac, material=material, gate=gate, velocity=vel,
            )
        w = self.lift_w
        if w is not None and w.shape[0] == 1 and x.shape[0] != 1:
            w = w.expand(x.shape[0], -1)
        if jac:
            if not hasattr(vel, "forward_and_jacobian"):
                raise AttributeError(
                    "velocity must implement forward_and_jacobian for Jacobian integration"
                )
            v, jv = vel.forward_and_jacobian(x, w, t)
            return _checkVectorShape(v, x), jv
        return _checkVectorShape(vel(x, w, t), x)

    def _velocity_with_fault(
        self,
        x: torch.Tensor,
        t: float,
        *,
        jac: bool = False,
        material: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        gate: Optional[torch.Tensor] = None,
        velocity: Optional[nn.Module] = None,
    ):
        fl = self.fault_lift
        vel = velocity if velocity is not None else self.velocities[0]
        if material is None:
            st = fl.state(x, gate=gate)
        else:
            w_active, w_raw = material
            st = fl.state_with_material(x, w_active, w_raw, gate=gate)
        if jac:
            v, jv = vel.forward_and_jacobian(x, st.w_active, t)
        else:
            v, jv = vel(x, st.w_active, t), None
        v = _checkVectorShape(v, x)
        if self.confine:
            if jac:
                v, jv = self._confine_blended(v, st, jv=jv)
            else:
                v = self._confine_blended(v, st)
        if st.mobility is not None:
            v = v * st.mobility.unsqueeze(-1)
            if jac:
                jv = jv * st.mobility.unsqueeze(-1).unsqueeze(-1)
        return (v, jv) if jac else v

    @staticmethod
    def _confine_blended(
        v: torch.Tensor, st: SheetState, jv: Optional[torch.Tensor] = None,
    ):
        lam = st.confinement.clamp(0.0, 1.0 - 1e-9)
        alpha = lam / (1.0 - lam)
        n = st.normal
        dim = v.shape[-1]
        eye = torch.eye(dim, dtype=v.dtype, device=v.device)
        outer = alpha.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
            "nfd,nfe->nfde", n, n,
        )
        m = eye.unsqueeze(0).expand(v.shape[0], -1, -1) + outer.sum(dim=1).to(dtype=v.dtype)
        v_out = torch.linalg.solve(m, v.unsqueeze(-1)).squeeze(-1)
        if jv is None:
            return v_out
        jv_out = torch.linalg.solve(m, jv.to(dtype=m.dtype)).to(dtype=jv.dtype)
        return v_out, jv_out

    def _integrate(
        self,
        x: torch.Tensor,
        *,
        sign: float,
        jac: bool = False,
        duration: float = 1.0,
        n_steps: Optional[int] = None,
        G=None,
        velocity: Optional[nn.Module] = None,
        t_span: Optional[Tuple[float, float]] = None,
        material=None,
        trace_mat=None,
        stack=None,
        jac_rows=None,
        record=None,
    ):
        if self.fault_lift is not None:
            return self._integrate_with_fault(
                x, sign=sign, jac=jac, duration=duration, n_steps=n_steps,
                velocity=velocity, t_span=t_span,
                material=material, trace_mat=trace_mat,
            )
        if self.integrator == "shear":
            return self._integrate_shear(
                x, sign=sign, jac=jac, duration=duration, n_steps=n_steps,
                velocity=velocity, stack=stack, jac_rows=jac_rows,
                record=record,
            )
        if self.integrator == "rk4":
            return self._integrate_rk4(
                x, sign=sign, jac=jac, duration=duration, n_steps=n_steps,
                G=G, velocity=velocity, t_span=t_span,
            )
        n = self.steps[0] if n_steps is None else max(1, int(n_steps))
        dt = duration / n
        h = sign * dt
        t_start, t_end = t_span if t_span is not None else (
            (0.0, 1.0) if sign > 0 else (1.0, 0.0)
        )
        dtau = (t_end - t_start) / n
        eye = J = None
        if jac:
            dim = self.dim if self.dim is not None else x.shape[-1]
            eye = torch.eye(dim, dtype=x.dtype, device=x.device)
            J = eye.expand(len(x), -1, -1).clone()

        for k in range(n):
            t_k = t_start + k * dtau
            t_mid = t_start + (k + 0.5) * dtau
            at = lambda p, t: self._velocity_at(
                p, t, jac=jac, G=G, velocity=velocity
            )
            if jac:
                v0, L0 = at(x, t_k)
                x_mid = x + (h / 2) * v0
                v_mid, L_mid = at(x_mid, t_mid)
                # d(x + h v_mid)/dx = I + h L_mid (I + (h/2) L0)
                d_mid = eye + (h / 2) * L0
                J = torch.bmm(eye + h * torch.bmm(L_mid, d_mid), J)
            else:
                v0 = at(x, t_k)
                x_mid = x + (h / 2) * v0
                v_mid = at(x_mid, t_mid)
            x = rk2_step(x, v0, v_mid, sign, dt)

        return (x, J) if jac else x

    def _integrate_rk4(
        self, x, *, sign, jac=False, duration=1.0, n_steps=None, G=None,
        velocity=None, t_span=None,
    ):
        """
        RK4, with the chain-rule Jacobian of the discrete step.

        ``det J = 1 + O(dt⁴)``. Use ``integrator="shear"`` for exact volume.
        Four velocity evaluations per step; the position-only path evaluates
        none of the Jacobians.
        """
        n = self.steps[0] if n_steps is None else max(1, int(n_steps))
        dt = duration / n
        t_start, t_end = t_span if t_span is not None else (
            (0.0, 1.0) if sign > 0 else (1.0, 0.0)
        )
        dtau = (t_end - t_start) / n
        eye = J = None
        if jac:
            eye = torch.eye(self.dim, dtype=x.dtype, device=x.device)
            J = eye.expand(len(x), -1, -1).clone()

        h = sign * dt
        for k in range(n):
            t0 = t_start + k * dtau
            tm = t_start + (k + 0.5) * dtau
            t1 = t_start + (k + 1) * dtau
            at = lambda p, t: self._velocity_at(
                p, t, jac=jac, G=G, velocity=velocity
            )
            if jac:
                k1, L1 = at(x, t0)
                k2, L2 = at(x + (h / 2) * k1, tm)
                k3, L3 = at(x + (h / 2) * k2, tm)
                k4, L4 = at(x + h * k3, t1)
                # d(k_i)/dx by the chain rule through each stage's own argument
                d1 = L1
                d2 = torch.bmm(L2, eye + (h / 2) * d1)
                d3 = torch.bmm(L3, eye + (h / 2) * d2)
                d4 = torch.bmm(L4, eye + h * d3)
                J = torch.bmm(eye + (h / 6) * (d1 + 2 * d2 + 2 * d3 + d4), J)
            else:
                k1 = at(x, t0)
                k2 = at(x + (h / 2) * k1, tm)
                k3 = at(x + (h / 2) * k2, tm)
                k4 = at(x + h * k3, t1)
            x = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return (x, J) if jac else x

    @staticmethod
    @torch.no_grad()
    def _shear_groups(
        om: torch.Tensor,
        tol: Optional[int] = None,
        wavenumber: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Indices of modes that share a wave direction, longest wavelength first.

        Parallel ``Ω`` share the invariant ``d·x``, so their shears commute and
        stay rank-one after they are summed. ``tol`` is the angle threshold
        (default ``100 ε``). Cubic harmonics are parallel only to a few ulps;
        a fixed ``1e-9`` grid splits them in float32. ``wavenumber`` is ``|Ω|``
        per row when ``om`` holds unit directions.
        """
        if om.shape[0] == 0:
            return []
        k = om.norm(dim=-1) if wavenumber is None else wavenumber.abs()
        if tol is None:
            tol = 100.0 * torch.finfo(om.dtype).eps
        mag = om.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        u = om / mag
        # The Gram matrix is too large above a few thousand terms. Hashing can miss a merge across a cell boundary.
        if om.shape[0] > 4096:
            return FlowOffset._by_wavelength(
                FlowOffset._shear_groups_hashed(u, float(tol)), k
            )
        # Cluster on |u·v|. A rounded key splits directions that straddle a cell.
        # abs identifies +d with -d; the phase absorbs the sign.
        close = (u @ u.T).abs() >= 1.0 - tol
        groups: List[torch.Tensor] = []
        unassigned = torch.ones(len(u), dtype=torch.bool, device=u.device)
        for i in range(len(u)):
            if not unassigned[i]:
                continue
            members = close[i] & unassigned
            unassigned &= ~members
            groups.append(torch.nonzero(members, as_tuple=True)[0])
        return FlowOffset._by_wavelength(groups, k)

    @staticmethod
    def _by_wavelength(groups: List[torch.Tensor], k: torch.Tensor) -> List[torch.Tensor]:
        """Stable sort of groups by their smallest wavenumber, ascending."""
        lo = [float(k[g].min()) for g in groups]
        return [groups[i] for i in sorted(range(len(groups)), key=lo.__getitem__)]

    @staticmethod
    @torch.no_grad()
    def _shear_groups_hashed(u: torch.Tensor, tol: float) -> List[torch.Tensor]:
        """Bucket unit directions by a rounded key; identifies antipodes."""
        idx = u.abs().argmax(dim=-1, keepdim=True)
        flip = u.gather(-1, idx) < 0
        uc = torch.where(flip, -u, u)
        scale = 1.0 / max(tol * 10.0, 1e-12)
        key = torch.round(uc * scale).to(torch.int64)
        _, inv = torch.unique(key, dim=0, return_inverse=True)
        return [
            torch.nonzero(inv == g, as_tuple=True)[0]
            for g in range(int(inv.max()) + 1)
        ]

    def _shear_plan(self, velocity, x, w):
        """
        Per-group ``(Cg, a, off, dhat, Cg·a)``, cached on the frequency buffers.

        Every ``Ω`` in a group is parallel to ``dhat``, so the phases are
        ``(x·dhat) a + off``. The cache is a weak-key dictionary: ``id()`` is
        reused after a module is freed.
        """
        vel = velocity if velocity is not None else self.velocities[0]
        if not hasattr(vel, "shear_decomposition"):
            raise TypeError(
                f"integrator='shear' needs a velocity exposing "
                f"shear_decomposition(); {type(vel).__name__} does not"
            )
        C, om, off = vel.shear_decomposition(x, w)
        if om.shape[0] == 0:
            return []

        # The row set changes when a pruned zero amplitude starts training.
        cache = self.__dict__.setdefault("_shear_group_cache", weakref.WeakKeyDictionary())
        hit = cache.get(vel)
        if hit is not None and _same_tensor(hit[0], om):
            groups = hit[1]
        else:
            groups = self._shear_groups(om.detach())
            cache[vel] = (om.detach().clone(), groups)

        plan = []
        for idx in groups:
            omg = om[idx]
            dhat = (omg[0] / omg[0].norm().clamp_min(1e-30)).detach()
            a = omg @ dhat                       # signed magnitudes along dhat
            Cg = C[idx]
            plan.append((
                Cg, a,
                off[idx] if off.dim() == 1 else off[:, idx],
                dhat, Cg * a.unsqueeze(-1),
            ))
        return plan

    @staticmethod
    def _shear_tables(plan, x, reach: float, T: int):
        """
        Each group's displacement on a 1-D grid in ``s = dhat·x``.

        The profile is sampled once per traversal (``O(T · modes)``) and then
        read in ``O(N)``. Linear interpolation of vectors orthogonal to
        ``dhat`` stays orthogonal, so ``det = 1`` is unchanged. The table is
        padded by the furthest drift, ``reach · Σ |C_k|``. Returns
        ``[(table, s0, ds), ...]`` with ``table`` of shape ``(T, d)``.
        """
        # Pad by the furthest a point can drift, so lookups stay on the table.
        drift = sum(
            reach * float(Cg.norm(dim=-1).sum()) for Cg, _, _, _, _ in plan
        )
        tables = []
        for Cg, a, off_g, dhat, _ in plan:
            s = x @ dhat
            s0 = float(s.min()) - drift
            s1 = float(s.max()) + drift
            ds = (s1 - s0) / (T - 1)
            grid = torch.arange(T, dtype=x.dtype, device=x.device) * ds + s0
            # (T, modes) phases -> (T, d) displacement profile
            tab = torch.cos(grid.unsqueeze(-1) * a + off_g) @ Cg
            # Cancellation leaves a residual along dhat, and the slope amplifies it into det J − 1.
            # Re-projection is exact and commutes with d/ds.
            tab = tab - dhat * (tab @ dhat).unsqueeze(-1)
            tables.append((tab, s0, ds))
        return tables

    @staticmethod
    def _table_lookup(tab, s0: float, ds: float, s: torch.Tensor):
        """Linear interpolation plus the interpolant's *own* (exact) slope."""
        T = tab.shape[0]
        u = (s - s0) / ds
        i0 = u.floor().long().clamp(0, T - 2)
        w = (u - i0).unsqueeze(-1)
        lo, hi = tab[i0], tab[i0 + 1]
        return lo + (hi - lo) * w, (hi - lo) / ds

    @staticmethod
    def _scaled_plan(plan, h: float, mults) -> dict:
        """
        Fold ``h · mult`` into ``C`` so the inner step is one ``addmm``.

        The schedule uses two lengths (a half step and a fused full step).
        The launch count, not the arithmetic, limits this integrator on a GPU.
        """
        return {
            (gi, mult): (Cg * (h * mult), Cga * (h * mult))
            for gi, (Cg, _, _, _, Cga) in enumerate(plan)
            for mult in mults
        }

    @staticmethod
    def _shear_schedule(
        n: int, m: int, sign: float = 1.0, scheme: str = "strang",
    ) -> List[Tuple[int, float]]:
        """
        ``(group, half-step multiple)`` pairs, with adjacent equal groups fused.

        ``n == 0`` is one pass at the full step, reversed when ``sign < 0``.
        A palindrome meets itself at the centre and at each step seam; a group
        commutes with itself, so those pairs become one full step. One group
        is a single sub-step for any ``n``.

        ``"lie"`` repeats the one-way sweep ``n`` times at ``1/n`` (nothing
        fuses). ``"yoshida"`` replaces each Strang step by three of durations
        ``w1, w0, w1``, ``w1 = 1/(2 − 2^(1/3))``, ``w0 = 1 − 2 w1 < 0``.
        """
        if m == 1:
            return [(0, 1.0)]
        if n == 0:
            order = range(m) if sign > 0 else range(m - 1, -1, -1)
            return [(i, 1.0) for i in order]
        if scheme == "lie":
            order = list(range(m) if sign > 0 else range(m - 1, -1, -1))
            return [(i, 1.0) for _ in range(n) for i in order]
        if scheme == "yoshida":
            w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
            w0 = 1.0 - 2.0 * w1
            palindrome = list(range(m)) + list(range(m - 1, -1, -1))
            fused: List[Tuple[int, float]] = []
            for _ in range(n):
                for w in (w1, w0, w1):
                    for i in palindrome:
                        if fused and fused[-1][0] == i:
                            fused[-1] = (i, fused[-1][1] + w)
                        else:
                            fused.append((i, w))
            return fused
        sched: List[Tuple[int, float]] = []
        for s in range(n):
            sched.append((0, 1.0 if s == 0 else 2.0))   # fused with previous seam
            sched.extend((i, 1.0) for i in range(1, m - 1))
            sched.append((m - 1, 2.0))                  # fused sweep centre
            sched.extend((i, 1.0) for i in range(m - 2, 0, -1))
        sched.append((0, 1.0))
        return sched

    @staticmethod
    def _shear_step(
        n: int, m: int, sign: float, duration: float, scheme: str = "strang",
    ) -> float:
        """The unit of :meth:`_shear_schedule`: full duration for one group or ``n == 0``, ``1/n`` for ``"lie"``, else half a Strang step."""
        if m == 1 or n == 0:
            return sign * duration
        return sign * duration / n if scheme == "lie" else sign * duration / n / 2.0

    def _integrate_shear(
        self, x, *, sign, jac=False, duration=1.0, n_steps=None, velocity=None,
        stack=None, jac_rows=None, record=None,
    ):
        """
        Explicit shear splitting (Quispel and McLaren 2003).

        Each term with ``C · Ω = 0`` integrates as a translation at any step
        size, with ``det = 1``. Strang over non-commuting groups is second
        order; :meth:`_shear_groups` is what keeps the depth at the number of
        directions rather than the number of modes.
        """
        n = self.steps[0] if n_steps is None else max(0, int(n_steps))   # 0: one-way sweep
        w = self.lift_w
        if w is not None and w.shape[0] == 1 and x.shape[0] != 1:
            w = w.expand(x.shape[0], -1)
        if self.shear_split == "normal":
            rplan = self._ridge_plan(velocity, x, w)
            if rplan is not None:
                return self._integrate_ridge_shear(
                    x, rplan, sign=sign, jac=jac, duration=duration, n=n,
                    stack=stack, jac_rows=jac_rows, record=record,
                )
        plan = self._shear_plan(velocity, x, w)
        if not plan:
            if jac:
                J = (
                    torch.eye(self.dim, dtype=x.dtype, device=x.device)
                    .expand(len(x), -1, -1)
                    .clone()
                )
                return x, J
            return x
        m = len(plan)
        scheme = getattr(self, "shear_scheme", "strang")   # pickles from before it existed
        sched = self._shear_schedule(n, m, sign, scheme)
        h = self._shear_step(n, m, sign, duration, scheme)
        scaled = self._scaled_plan(plan, h, {mult for _, mult in sched})

        J = None
        if jac:
            J = (
                torch.eye(self.dim, dtype=x.dtype, device=x.device)
                .expand(len(x), -1, -1)
                .clone()
            )

        # Tabulated profiles: only when the offset is per-mode (no lift `w`,
        # which would make the profile per-point and not a 1-D function).
        tables = None
        if self.shear_table and all(o.dim() == 1 for _, _, o, _, _ in plan):
            tables = self._shear_tables(plan, x, abs(duration), self.shear_table)

        for gi, mult in sched:
            _, a, off_g, dhat, _ = plan[gi]
            hCg, hCga = scaled[(gi, mult)]
            if tables is not None:
                tab, s0, ds = tables[gi]
                U, dU = self._table_lookup(tab, s0, ds, x @ dhat)
                step = h * mult
                if jac:
                    J = J + (step * dU).unsqueeze(-1) * torch.einsum(
                        "d,ndk->nk", dhat, J
                    ).unsqueeze(-2)
                elif stack is not None:
                    q = dU if jac_rows is None else dU[:jac_rows]
                    stack.append((step * q, dhat))
                x = x + step * U
                if record is not None:
                    record.append(x.detach().clone())
                continue
            # All Om in this group are parallel to dhat, so one projection feeds
            # every mode's phase; `addr` fuses the outer product and the offset.
            p = torch.addr(off_g, x @ dhat, a)
            if jac:
                sin_p, cos_p = _sincos(p)
                # Rank-one update J <- (I + u dhatᵀ) J. u · dhat = 0, so det stays 1.
                u = (-sin_p) @ hCga
                J = J + u.unsqueeze(-1) * torch.einsum(
                    "d,ndk->nk", dhat, J
                ).unsqueeze(-2)
                x = torch.addmm(x, cos_p, hCg)
            else:
                if stack is not None:
                    # `sin` is needed only for the rows that will be pulled
                    # back; `cos` moves every point, so it stays full width.
                    q = p if jac_rows is None else p[:jac_rows]
                    stack.append(((-torch.sin(q)) @ hCga, dhat))
                x = torch.addmm(x, torch.cos(p), hCg)
            if record is not None:
                record.append(x.detach().clone())

        return (x, J) if jac else x

    # ── normal split: full 3-D Euler fields ─────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def _ridge_geometry(
        dir_a: torch.Tensor,
        dir_b: torch.Tensor,
        kap_a: Optional[torch.Tensor] = None,
        kap_b: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Rays of each bank, then groups of ray pairs that share a normal.

        Built once from the frequency buffers. Amplitudes are gathered in
        :meth:`_ridge_plan`.

        Parallel modes share a ray ``u_p``. With ``t_i = dir_i · u_p`` and
        ``s = u_p · x``,

            ∇P = Σ_p u_p f_p(s),   f_p = Σ_{i∈p} amp_i t_i cos(kap_i t_i s + off_i),

        and ``v = Σ_{p,q} f_p g_q (u_p × v_q)``. Rays are longest wavelength
        first; normals are taken ``p``-major.

        Pair ``(p, q)`` has ``u_p × v_q = c n``. Pairs with parallel ``n`` are
        one group: every phase direction is re-projected onto that group's
        ``n``-perp,         so ``n · ∇ψ = 0`` exactly rather than to ``tol``. Pairs
        with ``|u × v|`` below ``100 ε`` are dropped.
        """
        def rays(d, kap):
            groups = FlowOffset._shear_groups(d, wavenumber=kap)
            U = torch.stack([d[g[0]] for g in groups])
            ray_of = torch.empty(d.shape[0], dtype=torch.long, device=d.device)
            for r, g in enumerate(groups):
                ray_of[g] = r
            proj = (d * U[ray_of]).sum(-1)
            return groups, U, ray_of, proj

        ga, U, ray_a, proj_a = rays(dir_a, kap_a)
        gb, V, ray_b, proj_b = rays(dir_b, kap_b)
        Ra, Rb = len(ga), len(gb)

        cross = torch.linalg.cross(
            U[:, None, :].expand(Ra, Rb, 3), V[None, :, :].expand(Ra, Rb, 3), dim=-1
        ).reshape(-1, 3)
        mag = cross.norm(dim=-1)
        # Fused multiply-add leaves u × u a few ulps (torch >= 2.9, CPU). A smaller cut invents a group.
        live = torch.nonzero(mag > 100.0 * torch.finfo(mag.dtype).eps, as_tuple=True)[0]
        groups = []
        if live.numel():
            nhat_all = cross[live] / mag[live].unsqueeze(-1)
            # |nhat| would order groups by rounding. The pair index keeps p-major order.
            for g in FlowOffset._shear_groups(nhat_all, wavenumber=live.to(mag.dtype)):
                pairs = live[g]
                p_idx, q_idx = pairs // Rb, pairs % Rb
                nhat = nhat_all[g[0]]
                c = cross[pairs] @ nhat                       # signed |u x v|
                S = U[p_idx] - (U[p_idx] @ nhat).unsqueeze(-1) * nhat
                T = V[q_idx] - (V[q_idx] @ nhat).unsqueeze(-1) * nhat
                a_modes = torch.cat([ga[int(p)] for p in p_idx])
                b_modes = torch.cat([gb[int(q)] for q in q_idx])
                a_seg = torch.cat([
                    torch.full((len(ga[int(p)]),), k, dtype=torch.long, device=U.device)
                    for k, p in enumerate(p_idx)
                ])
                b_seg = torch.cat([
                    torch.full((len(gb[int(q)]),), k, dtype=torch.long, device=U.device)
                    for k, q in enumerate(q_idx)
                ])
                groups.append(dict(
                    nhat=nhat, c=c, S=S, T=T, p=p_idx, q=q_idx,
                    a_modes=a_modes, a_seg=a_seg, b_modes=b_modes, b_seg=b_seg,
                ))
        return dict(
            ray_a=ray_a, proj_a=proj_a, Ra=Ra,
            ray_b=ray_b, proj_b=proj_b, Rb=Rb, groups=groups,
        )

    def _ridge_plan(self, velocity, x, w):
        """
        Per-call coefficients for the normal split, or ``None`` without a ridge form.

        ``F = cos(phase) @ Ea`` with ``Ea[i, k] = amp_i t_i`` when mode ``i``
        belongs to pair ``k``'s ray, and ``Eka = Ea · kap`` so
        ``F' = −sin(phase) @ Eka``.
        """
        vel = velocity if velocity is not None else self.velocities[0]
        if not hasattr(vel, "ridge_decomposition"):
            return None
        banks = vel.ridge_decomposition(x, w)
        if banks is None:
            return None
        (dir_a, amp_a, kap_a, off_a), (dir_b, amp_b, kap_b, off_b) = banks

        cache = self.__dict__.setdefault("_ridge_cache", weakref.WeakKeyDictionary())
        hit = cache.get(vel)
        dirs = (dir_a, dir_b, kap_a, kap_b)
        if hit is not None and all(_same_tensor(h, d) for h, d in zip(hit[0], dirs)):
            geo = hit[1]
        else:
            dirs = tuple(d.detach().clone() for d in dirs)
            geo = self._ridge_geometry(*dirs)
            cache[vel] = (dirs, geo)

        # Ray-projected coefficients (see `_ridge_geometry`).
        ramp_a, rkap_a = amp_a * geo["proj_a"], kap_a * geo["proj_a"]
        ramp_b, rkap_b = amp_b * geo["proj_b"], kap_b * geo["proj_b"]

        def seg_matrix(vals, seg, P):
            M = vals.new_zeros(vals.shape[0], P)
            return M.index_put((torch.arange(vals.shape[0], device=vals.device), seg), vals)

        def offs(off, modes):
            return off[modes] if off.dim() == 1 else off[:, modes]

        plan = []
        for g in geo["groups"]:
            P = g["c"].shape[0]
            am, bm = g["a_modes"], g["b_modes"]
            ka, kb = rkap_a[am], rkap_b[bm]
            Ea = seg_matrix(ramp_a[am], g["a_seg"], P)
            Eb = seg_matrix(ramp_b[bm], g["b_seg"], P)
            plan.append(dict(
                g, oa=offs(off_a, am), ob=offs(off_b, bm),
                # One addmm for the group: Wa[:, i] = kap_i S_{seg(i)}.
                Wa=(g["S"][g["a_seg"]] * ka.unsqueeze(-1)).T.contiguous(),
                Wb=(g["T"][g["b_seg"]] * kb.unsqueeze(-1)).T.contiguous(),
                Ea=Ea, Eb=Eb, Eka=Ea * ka.unsqueeze(-1), Ekb=Eb * kb.unsqueeze(-1),
            ))
        return dict(
            groups=plan, geo=geo,
            rays=((ramp_a, rkap_a, off_a), (ramp_b, rkap_b, off_b)),
        )

    @staticmethod
    def _ridge_tables(rplan, x, reach: float, T: int):
        """
        Ray profiles ``f_p``, ``g_q`` on one grid each, shape ``(R, T)``.

        A ray is shared by every group that touches it, so ``R_a + R_b``
        tables cover the split. Each grid is padded by the furthest drift,
        ``reach · Σ |c| max|f| max|g|``. Interpolation keeps ``det = 1`` for
        the same reason as :meth:`_shear_tables`: the step direction stays
        ``n``. Returns ``((tab_a, s0_a, ds_a), (tab_b, s0_b, ds_b))``.
        """
        geo = rplan["geo"]
        (ramp_a, rkap_a, off_a), (ramp_b, rkap_b, off_b) = rplan["rays"]
        bound_a = ramp_a.detach().abs().new_zeros(geo["Ra"]).index_add(
            0, geo["ray_a"], ramp_a.detach().abs())
        bound_b = ramp_b.detach().abs().new_zeros(geo["Rb"]).index_add(
            0, geo["ray_b"], ramp_b.detach().abs())
        drift = reach * sum(
            float((g["c"].abs() * bound_a[g["p"]] * bound_b[g["q"]]).sum())
            for g in rplan["groups"]
        )

        def build(R, ray, ramp, rkap, off, key_ray, key_dir):
            lo = torch.full((R,), math.inf, dtype=x.dtype, device=x.device)
            hi = torch.full((R,), -math.inf, dtype=x.dtype, device=x.device)
            xd = x.detach()
            for g in rplan["groups"]:
                s = xd @ g[key_dir].T                                # (N, P)
                lo = lo.scatter_reduce(0, g[key_ray], s.amin(0), "amin")
                hi = hi.scatter_reduce(0, g[key_ray], s.amax(0), "amax")
            unused = ~torch.isfinite(lo)
            lo, hi = lo.masked_fill(unused, 0.0), hi.masked_fill(unused, 1.0)
            s0 = lo - drift
            ds = ((hi + drift) - s0).clamp_min(1e-12) / (T - 1)
            grid = s0.unsqueeze(-1) + ds.unsqueeze(-1) * torch.arange(
                T, dtype=x.dtype, device=x.device)
            vals = torch.cos(grid[ray] * rkap.unsqueeze(-1) + off.unsqueeze(-1))
            tab = (vals * ramp.unsqueeze(-1)).new_zeros(R, T).index_add(
                0, ray, vals * ramp.unsqueeze(-1))
            return tab, s0, ds

        return (
            build(geo["Ra"], geo["ray_a"], ramp_a, rkap_a, off_a, "p", "S"),
            build(geo["Rb"], geo["ray_b"], ramp_b, rkap_b, off_b, "q", "T"),
        )

    @staticmethod
    def _ridge_lookup(table, rays, s):
        """Interpolate rays ``rays`` (P,) at ``s`` (N, P): value and exact slope."""
        tab, s0, ds = table
        T = tab.shape[1]
        u = (s - s0[rays]) / ds[rays]
        i0 = u.floor().long().clamp(0, T - 2)
        wt = u - i0
        flat = tab[rays].reshape(-1)
        idx = i0 + T * torch.arange(len(rays), device=s.device)
        lo, hi = flat[idx], flat[idx + 1]
        return lo + (hi - lo) * wt, (hi - lo) / ds[rays]

    def _integrate_ridge_shear(
        self, x, rplan, *, sign, jac, duration, n, stack, jac_rows, record,
    ):
        """
        Normal-split shears of ``v = ∇P × ∇Q``.

        ``v = Σ_g ψ_g n_g`` with ``n · ∇ψ = 0``, so each sub-step is the
        translation ``x + h ψ n`` and ``det(I + h n ∇ψᵀ) = 1``. Depth is the
        number of distinct normals, at most ``R_a R_b``, and a sub-step costs
        ``O(N (K_p + K_q))``. The phase split's depth is about ``2 F_a F_b``.
        """
        groups = rplan["groups"]
        eye = lambda: (
            torch.eye(self.dim, dtype=x.dtype, device=x.device)
            .expand(len(x), -1, -1).clone()
        )
        if not groups:
            return (x, eye()) if jac else x
        m = len(groups)
        scheme = getattr(self, "shear_scheme", "strang")   # pickles from before it existed
        sched = self._shear_schedule(n, m, sign, scheme)
        h = self._shear_step(n, m, sign, duration, scheme)
        J = eye() if jac else None
        tables = None
        if self.shear_table and all(
            g["oa"].dim() == 1 and g["ob"].dim() == 1 for g in groups
        ):
            tables = self._ridge_tables(rplan, x, abs(duration), self.shear_table)

        rows = None
        if jac:
            rows = x.shape[0]
        elif stack is not None:
            rows = x.shape[0] if jac_rows is None else jac_rows

        for gi, mult in sched:
            g = groups[gi]
            c = g["c"] * (h * mult)
            if tables is not None:
                F, dF = self._ridge_lookup(tables[0], g["p"], x @ g["S"].T)
                G, dG = self._ridge_lookup(tables[1], g["q"], x @ g["T"].T)
                if rows is not None:
                    dF, dG = dF[:rows], dG[:rows]
            else:
                pa = torch.addmm(g["oa"], x, g["Wa"])            # (N, Ma)
                pb = torch.addmm(g["ob"], x, g["Wb"])
                if rows is None:
                    F, G = torch.cos(pa) @ g["Ea"], torch.cos(pb) @ g["Eb"]
                elif rows == x.shape[0]:
                    sa, ca = _sincos(pa)
                    sb, cb = _sincos(pb)
                    F, G = ca @ g["Ea"], cb @ g["Eb"]
                    dF, dG = -(sa @ g["Eka"]), -(sb @ g["Ekb"])
                else:
                    F, G = torch.cos(pa) @ g["Ea"], torch.cos(pb) @ g["Eb"]
                    dF = -(torch.sin(pa[:rows]) @ g["Eka"])
                    dG = -(torch.sin(pb[:rows]) @ g["Ekb"])
            psi = (F * G) @ c                                    # (N,)
            if rows is not None:
                grad = (c * dF * G[:rows]) @ g["S"] + (c * F[:rows] * dG) @ g["T"]
                if jac:
                    # J <- (I + n grad^T) J, as an outer-product update.
                    J = J + g["nhat"].view(1, -1, 1) * torch.einsum(
                        "nd,ndk->nk", grad, J
                    ).unsqueeze(-2)
                else:
                    # Same layout as the phase split: S = I + u dhat^T with
                    # (u, dhat) = (n, grad), so S^T g = g + grad (n . g).
                    stack.append((g["nhat"], grad))
            x = torch.addr(x, psi, g["nhat"])
            if record is not None:
                record.append(x.detach().clone())
        return (x, J) if jac else x

    @torch.no_grad()
    def stability(self, x: torch.Tensor) -> float:
        """
        Worst ``‖DΦ_h − I‖₂`` of a discrete step along the trajectory.

        ``ρ < 1`` is an Ostrowski injectivity certificate for that step.
        ``DΦ_h`` is the chain-rule Jacobian of the scheme in use, at every
        sub-step, not the Euler bound ``dt ‖∇v‖`` at the start. A
        volume-preserving step stays a diffeomorphism when ``ρ ≥ 1``; the
        number is then a margin.
        """
        if not isinstance(x, torch.Tensor):
            x = _tensor(x)
        else:
            x = x.detach()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.shape[0] == 0:
            return 0.0
        x = x.clone()
        sign = self._sign()
        dim = self.dim if self.dim is not None else x.shape[-1]
        eye = torch.eye(dim, dtype=x.dtype, device=x.device)
        worst = 0.0
        material, trace_mat, opened = self._open_fault_session(x)
        try:
            for vel, n, span in self._plan(sign):
                t_start, t_end = span
                # a one-way shear sweep (n = 0) is a single discrete step
                n_sub = max(n, 1)
                dtau = (t_end - t_start) / n_sub
                dt = 1.0 / n_sub
                for k in range(n_sub):
                    t_span = (t_start + k * dtau, t_start + (k + 1) * dtau)
                    x, j_h = self._integrate(
                        x,
                        sign=sign,
                        jac=True,
                        duration=dt,
                        n_steps=0 if n == 0 else 1,
                        velocity=vel,
                        t_span=t_span,
                        material=material,
                        trace_mat=trace_mat,
                    )
                    worst = max(
                        worst,
                        float(torch.linalg.matrix_norm(j_h - eye, ord=2).max()),
                    )
        finally:
            self._close_fault_session(opened)
        return worst

    def _integrate_with_fault(
        self,
        x: torch.Tensor,
        *,
        sign: float,
        jac: bool = False,
        duration: float = 1.0,
        n_steps: Optional[int] = None,
        velocity: Optional[nn.Module] = None,
        t_span: Optional[Tuple[float, float]] = None,
        material=None,
        trace_mat=None,
    ):
        n = self.steps[0] if n_steps is None else max(1, int(n_steps))
        dt = duration / n
        t_start, t_end = t_span if t_span is not None else (
            (0.0, 1.0) if sign > 0 else (1.0, 0.0)
        )
        dtau = (t_end - t_start) / n
        h = sign * dt
        eye = J = None
        if jac:
            dim = self.dim if self.dim is not None else x.shape[-1]
            eye = torch.eye(dim, dtype=x.dtype, device=x.device)
            J = eye.expand(len(x), -1, -1).clone()

        fl = self.fault_lift
        lag = self.lagrangian_faults
        owns = material is None and lag
        if owns:
            fl.snapshot_traces()
            material = fl.material_labels(x)
            trace_mat = fl.trace_material_labels()

        try:
            for k in range(n):
                t_k = t_start + k * dtau
                t_mid = t_start + (k + 0.5) * dtau
                gate_k = fl.activation(t_k) if lag else None
                gate_mid = fl.activation(t_mid) if lag else None
                if lag:
                    self._advect_fault_traces(
                        sign, dt, t_k, t_mid, trace_mat, gate_k, gate_mid,
                        velocity=velocity,
                    )
                if jac:
                    v0, L0 = self._velocity_at(
                        x, t_k, jac=True, material=material, gate=gate_k,
                        velocity=velocity,
                    )
                    x_mid = x + (h / 2) * v0
                    v_mid, L_mid = self._velocity_at(
                        x_mid, t_mid, jac=True, material=material, gate=gate_mid,
                        velocity=velocity,
                    )
                    d_mid = eye + (h / 2) * L0
                    J = torch.bmm(eye + h * torch.bmm(L_mid, d_mid), J)
                else:
                    v0 = self._velocity_at(
                        x, t_k, jac=False, material=material, gate=gate_k,
                        velocity=velocity,
                    )
                    x_mid = x + (h / 2) * v0
                    v_mid = self._velocity_at(
                        x_mid, t_mid, jac=False, material=material, gate=gate_mid,
                        velocity=velocity,
                    )
                x = rk2_step(x, v0, v_mid, sign, dt)
        finally:
            if owns:
                fl.restore_traces()
        return (x, J) if jac else x

    @torch.no_grad()
    def _advect_fault_traces(
        self,
        sign: float,
        dt: float,
        t_k: float,
        t_mid: float,
        trace_mat: list,
        gate_k: Optional[torch.Tensor] = None,
        gate_mid: Optional[torch.Tensor] = None,
        velocity: Optional[nn.Module] = None,
    ) -> None:
        fl = self.fault_lift
        for i in range(fl.n_faults):
            if not fl.advect_traces[i]:
                continue
            tr = fl.trace(i)
            w_a, w_r = trace_mat[i]
            p0 = fl.advect_probe_points(i, w_r)
            v0 = self._velocity_at(
                p0, t_k, material=(w_a, w_r), gate=gate_k, velocity=velocity,
            )
            tr_mid = tr + sign * (dt / 2) * v0
            p_mid = fl.advect_probe_points(i, w_r, points=tr_mid)
            v_mid = self._velocity_at(
                p_mid, t_mid, material=(w_a, w_r), gate=gate_mid, velocity=velocity,
            )
            fl._set_trace(i, tr + sign * dt * v_mid)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x + self.frame_offset, sign=+1.0)

    def inverse_map(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x, sign=-1.0) - self.frame_offset

    def forward_map_with_jacobian(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._run(x + self.frame_offset, sign=+1.0, jac=True)

    def inverse_map_with_jacobian(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, j = self._run(x, sign=-1.0, jac=True)
        return x0 - self.frame_offset, j

    def inverse_map_with_pullback(self, x: torch.Tensor, jac_rows: Optional[int] = None):
        """
        ``(x0, pull)`` with ``pull(g) = Jᵀ g``, without building ``J``.

        Nanson transport and orthogonal thickness each contract ``J`` with one
        covector. A shear step is ``I + u dhatᵀ`` with ``u · dhat = 0``, so
        the transpose is ``g → g + dhat (u · g)``. The forward sweep records
        ``(u, dhat)`` and ``pull`` replays them in reverse, which also chains
        a composite flow. Other integrators build ``J`` and contract it.

        ``jac_rows`` limits the pullback to a prefix. Orientation rows are
        concatenated first; value, eq and iq points are only advected. One
        sweep still moves every point. Two traversals cost more at every ratio.
        """
        if self.integrator != "shear" or self.fault_lift is not None:
            x0, j = self.inverse_map_with_jacobian(x)
            if jac_rows is not None:
                j = j[:jac_rows]
            return x0, lambda g: torch.bmm(j.mT, g.unsqueeze(-1)).squeeze(-1)

        stack: List[Tuple[torch.Tensor, torch.Tensor]] = []
        x0 = self._run(x, sign=-1.0, stack=stack, jac_rows=jac_rows)

        def pull(g: torch.Tensor) -> torch.Tensor:
            # Sᵀg = g + dhat (u·g). addcmul is one launch; vecdot expands to three.
            for u, dhat in reversed(stack):
                g = torch.addcmul(g, dhat, (u * g).sum(-1, keepdim=True))
            return g

        return x0 - self.frame_offset, pull

    @torch.no_grad()
    def map_to_time(
        self,
        x: torch.Tensor,
        t: float,
        *,
        forward: bool = True,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if t <= 0:
            return x.clone()
        sign = +1.0 if forward else -1.0
        k = self.n_factors
        remaining = min(float(t), 1.0) * k
        material, trace_mat, opened = self._open_fault_session(x)
        try:
            for vel, n, span in self._plan(sign):
                if remaining <= 0:
                    break
                frac = min(1.0, remaining)
                lo, hi = span
                x = self._integrate(
                    x, sign=sign, duration=frac,
                    n_steps=n if n_steps is None else n_steps,
                    velocity=vel, t_span=(lo, lo + frac * (hi - lo)),
                    material=material, trace_mat=trace_mat,
                )
                remaining -= 1.0
        finally:
            self._close_fault_session(opened)
        return x

    def disp(self, X: torch.Tensor, G=None) -> torch.Tensor:
        x = X if isinstance(X, torch.Tensor) else _tensor(X)
        sign = self._sign()
        x_end = self._run(x, sign=sign, duration=abs(self.direction), G=G)
        if sign < 0:
            x_end = x_end - self.frame_offset
        return x_end - x

    def learnable(self):
        return any(p.requires_grad for p in self.velocities.parameters())

    def freeze_factor(self, index: int, frozen: bool = True) -> "FlowOffset":
        for p in self.velocities[index].parameters():
            p.requires_grad_(not frozen)
        return self

    def __repr__(self):
        vel = (
            repr(self.velocities[0]) if len(self.velocities) == 1
            else f"[{len(self.velocities)} factors]"
        )
        return (
            f"FlowOffset(n_steps={self.n_steps}, direction={self.direction}, "
            f"velocity={vel})"
        )
