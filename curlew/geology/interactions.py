"""
Functions defining how different scalar fields interact with each other to create generative, kinematic and hybrid
events. 

This includes Deformation classes, which implement various types of displacment (e.g., fault offset, dyke offset, etc.), and Overprint
classes that determine which scalar field values, structure IDs and property values populate the final (modern-day) outputs. These are 
the "glue" that bind different scalar fields into a potentially complex multi-event geomodel.
"""
import numpy as np
import torch
from torch import nn
import curlew
from curlew import _tensor
from curlew.core import LearnableBase
from curlew.fields import BaseSF
from curlew.fields.lift import SheetState
from typing import List, Optional, Tuple, Union

def _checkVectorShape(out: torch.Tensor, refPoints: torch.Tensor) -> torch.Tensor:
    """Check vector field shapes"""
    if not isinstance(out, torch.Tensor):
        out = _tensor(out) # ensure out is a tensor and on the correct device
    if out.dim() == 1: raise ValueError(f"Vector field expected, got scalar field?")
    if out.shape[-1] != refPoints.shape[-1]:
        raise ValueError(f"Latent field output dimension {out.shape} does not match coordinate dimension {refPoints.shape}")
    return out

# OVERPRINTING RELATIONS -- function used to create new rocks (generative fields)
# -----------------------------------------------------------------------------------
class Overprint(LearnableBase):
    """
    Base class for combining predictions from two consecutive scalar fields and "overprinting" some older
    scalar values to form unconformities or intrusions.
    """
    def __init__(self, threshold : Union[str, list] = 0, mode : str = 'above', defaultDomain : str ='child',
                 lithoSharpness : float = 1.0, structureSharpness : float = 1.0):
        """
        Create a new "overprint" object for applying overprinting stratigraphic (e.g., unconformities) and
        igneous (e.g., dykes, intrusions) events.

        Parameters
        ----------
        threshold : float, tuple, str
            The threshold above which the child field will "overprint" the parent one. Can be a single value
            (integer or string representing relevant isosurface name) to overprint older rocks above or below
            a threshold, or a tuple (of values or isosurface names) representing a range of values (see "within" and
            "outside" modes). If mode is "in" then threshold have a length of any multiple of two (to create dyke swarms).
        mode : str
            The overprinting mode. Options are:
                - `"above"`: replace all regions greater than the provided threshold). Useful for e.g., unconformities.
                - `"below"`: replace all regions less than than the provided threshold). Useful for e.g., intrusions.
                - `"in"`: replace all regions between the provided thresholds (must be a tuple containing two values). Used for e.g., dykes.
                - `"out"`: replace all regions outside the provided thresholds (must be a tuple containing two values). Not sure why this would be used.
        
        defaultDomain : str
            The default domain to use if no domain is provided. Options are:
                - `"child"`: use the child field as the domain. This is the default as for most
                   unconformably surfaces the unconformity base is parallel to the overlying bedding. 
                - `"parent"`: use the parent field to define the domain boundary (erosional surface). This can
                   be useful if the erosional surface is parallel to the older bedding and younger units onlap onto this.
            Note that for domain boundaries (i.e. GeoEvent instances with a defined `parent2` field), this parameter will
            have no effect as the domain boundary is defined by a separate (third) field. 
        lithoSharpness : float
            Sigmoid sharpness for ``softLithoID`` at this overprint boundary (and isosurface lithology assignment on
            the associated generative event). Larger values give harder lithology transitions, smaller values give smoother transitions
            that can improve convergence for models learning using lithology IDs.
        structureSharpness : float
            Sigmoid sharpness for ``softStructureID`` at this overprint boundary. Behaviour is the same as for `lithoSharpness`.
        """
        super().__init__()
        self.threshold = threshold
        self.mode = mode
        self.defaultDomain = defaultDomain.lower()
        self.lithoSharpness = lithoSharpness
        self.structureSharpness = structureSharpness
        
    def apply(self, parent, child, domain=None ):
        """
        Combine two scalar fields, keeping the parent field where the child field is below a threshold.

        The output will have two dimensions: the first represents the scalar value, and the second 
        represents the ID of the event responsible for this value.

        Parameters
        ----------
        parent : curlew.core.Geode
            A `curlew.core.Geode` (output object) from the older `curlew.geology.geoevent.GeoEvent`.
        child : curlew.core.Geode
            A `curlew.core.Geode` (output object) from the younger `curlew.geology.geoevent.GeoEvent`.
        domain : torch.Tensor, optional
            An (N,) implicit field used as the overprint domain (see ``self.mode``). If None (default),
            ``child.scalar`` is used for a typical unconformity or intrusion.
        Returns
        -------
        curlew.core.Geode
            Combined parent/child outputs with overprint applied.
        """
        assert self.thresh is not None, "`self.thresh` must be defined (by e.g. evaluating an isosurface) before calling `overprint`."
        if domain is None: 
            if self.defaultDomain == 'child':domain = child.scalar # child field determines the domain
            elif self.defaultDomain == 'parent': 
                assert hasattr(self, 'domainEvent'), "updateThresh must be called before apply when defaultDomain is 'parent'."
                domain = parent.fields[self.domainEvent.name]
            else: raise ValueError(f"Invalid default domain: {self.defaultDomain}. Should be 'child' or 'parent'.")

        mask, soft_litho_weight, soft_structure_weight = self._overprint_weights(domain)

        # combine results and return an updated Geode object
        return parent.combine(
            child, mask,
            soft_litho_weight=soft_litho_weight,
            soft_structure_weight=soft_structure_weight,
        )

    def _soft_weight(self, domain, thresh, sharpness):
        """Sigmoid-smoothed overprint weight for a scalar threshold or interval."""
        if isinstance(thresh, (float, int)):
            return torch.sigmoid(sharpness * (domain - thresh))

        soft_weight = torch.zeros(len(domain), device=curlew.device, dtype=curlew.dtype)
        for i in np.arange(len(thresh), step=2):
            T = thresh[i:(i+2)]
            t_lo, t_hi = float(np.min(T)), float(np.max(T))
            soft_interval = (
                torch.sigmoid(sharpness * (domain - t_lo))
                * torch.sigmoid(sharpness * (t_hi - domain))
            )
            soft_weight = torch.maximum(soft_weight, soft_interval)
        return soft_weight

    def _overprint_weights(self, domain):
        """
        Hard and soft overprint weights from the domain scalar field.

        Returns a hard 0/1 mask (for ``lithoID``, ``structureID``, etc.) and sigmoid-smoothed
        weights for ``softLithoID`` and ``softStructureID`` using ``lithoSharpness`` and
        ``structureSharpness`` respectively.
        """
        if isinstance(self.thresh, list):
            thresh = float(self.thresh[0]) if len(self.thresh) == 1 else self.thresh
        else:
            thresh = float(self.thresh)

        if isinstance(thresh, (float, int)):
            mask = (domain > thresh)
        else:
            mask = torch.zeros(len(domain), device=curlew.device, dtype=curlew.dtype)
            for i in np.arange(len(thresh), step=2):
                T = thresh[i:(i+2)]
                t_lo, t_hi = float(np.min(T)), float(np.max(T))
                lower_mask = domain > t_lo
                upper_mask = domain < t_hi
                mask = torch.logical_or(mask, lower_mask * upper_mask)

        soft_litho_weight = self._soft_weight(domain, thresh, self.lithoSharpness)
        soft_structure_weight = self._soft_weight(domain, thresh, self.structureSharpness)

        mask = mask.type(curlew.dtype)
        if ('below' in self.mode.lower()) or ('out' in self.mode.lower()):
            mask = 1 - mask
            soft_litho_weight = 1 - soft_litho_weight
            soft_structure_weight = 1 - soft_structure_weight

        return mask, soft_litho_weight, soft_structure_weight

    def getBaseField(self, field):
        """ Recurse backwards through the model field to find the 
            generative event that determines the base of the domain. Typically
            this is the parent field, though we want to skip through deformation events."""
        if self.defaultDomain == 'child':
            return field # that's very easy
        else:
            assert field.parent is not None, "Parent field is not defined yet `defaultDomain` is 'parent'."
            if field.parent.overprint is not None:
                return field.parent # easy; parent is a generative event
            else:
                return self.getBaseField(field.parent) # recurse backwards
    
    def updateThresh(self, field):
        """
        Resolve ``self.threshold`` to numeric values and store them in ``self.thresh``.

        Parameters
        ----------
        field : curlew.geology.geoevent.GeoEvent
            The generative event whose isosurfaces supply named thresholds. Also sets
            ``self.domainEvent`` when ``defaultDomain`` is ``'parent'``.
        """
        self.domainEvent = self.getBaseField(field)
        field = self.domainEvent
        if isinstance(self.threshold, (tuple, list)):
            self.thresh = [
                field.getIsovalue(t) if isinstance(t, str) else t
                for t in self.threshold
            ]
        elif isinstance(self.threshold, str):
            assert self.threshold in field.isosurfaces, f"Isosurface {self.threshold} not found in field {field.name}"
            self.thresh = field.getIsovalue(self.threshold)
        else:
            self.thresh = self.threshold
        
    def __repr__(self):
        return f"Overprint(mode='{self.mode}', thresh={self.threshold})"

# OFFSETTING RELATIONS - functions used to move things around (kinematic fields; these are the real shakers and movers).
# ------------------------------------------------------------------------------------------------------------------------
class OffsetBase(LearnableBase):
    """
    Class from which all offset classess should inherit. 
    """
    def eval(self, x, G):
        """
        Get displacement vectors for points `X` based on GeoEvent `G`. This will be called by the GeoEvent and return the results of self.disp(...).
        """
        o = self.disp(x, G)
        return o
    
    def learnable(self):
        """ Return true if this offset has learnable parameters (and an optimiser is initialised)."""
        return self.optim is not None
    
    def dss( self, x, G, normalize=False ):
        """
        Evaluate the scalar field gradient (ds) and value (s)  for the points `X` given GeoEvent `G`. Note that 
        this assumes `x` is already transformed into the local (paleo) coordinate system relevant for `G`.
        """
        # get gradient of scalar field at X and associated value
        # note that Transform = False here as the displacements are naturally defined
        # by the gradients in field coordinates (i.e. younger events do not effect the displacement associated with 
        # this event)
        ds, s = G.gradient( x, normalize=normalize, return_vals=True, transform=False, to_numpy=False, retain_graph=True )
        s = s.scalar

        # return gradient
        return ds, s

    def disp(self, X, G):
        """
        Compute displacement vectors for points `X` based on GeoEvent `G`. Child classess implementing specific types of offset should implement this function.
        """
        raise NotImplementedError
    
# ── Diffeomorphic flow integration (Clebsch / velocity-field deformation) ─────

def expm_traceless(A: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix exponential. Uses the closed form for trace-free 2×2 matrices
  (``A² = −det(A) I``); falls back to ``torch.linalg.matrix_exp`` otherwise.
    """
    if A.shape[-1] != 2:
        return torch.linalg.matrix_exp(A)
    mu2 = A[..., 0, 0] ** 2 + A[..., 0, 1] * A[..., 1, 0]
    small = mu2.abs() < 1e-12
    s = torch.where(small, torch.ones_like(mu2), mu2.abs()).sqrt()
    pos = mu2 >= 0
    c = torch.where(pos, torch.cosh(s), torch.cos(s))
    f = torch.where(pos, torch.sinh(s), torch.sin(s)) / s
    c = torch.where(small, 1.0 + mu2 / 2.0, c)
    f = torch.where(small, 1.0 + mu2 / 6.0, f)
    eye = torch.eye(2, dtype=A.dtype, device=A.device)
    return c[..., None, None] * eye + f[..., None, None] * A

class FlowOffset(OffsetBase):
    """
    Displacement by RK2 integration of a velocity field over ``t ∈ [0, 1]``.

    Two flavours of "velocity" are supported:

    - **Spatial velocity module** (pass ``velocity``): an ``nn.Module`` exposing
      ``dim``, ``forward(x, w=None, t=None)`` and (for Jacobian integration)
      ``forward_and_jacobian(x, w=None, t=None)`` — e.g.
      :class:`~curlew.fields.clebsch.ClebschVelocity`, used by
      :func:`~curlew.geology.restore` for diffeomorphic restoration.
    - **``GeoEvent``-derived velocity** (leave ``velocity=None``): a subclass
      overrides :meth:`_velocity_at` to derive the instantaneous velocity from
      the owning :class:`~curlew.geology.geoevent.GeoEvent` ``G`` (e.g. the
      gradient of ``G``'s own scalar field), re-evaluated at the *current*
      integration position on every RK2 sub-step even though it doesn't
      actually depend on ``t``. Used by :class:`SheetOffset` and
      :class:`FaultOffset`.

    Positions are advanced with RK2 (midpoint method); the deformation Jacobian
    is accumulated with a Magnus-midpoint update when requested (spatial
    velocity modules only — see :meth:`_velocity_at`).

    If ``curlew.compile`` is ``True`` and a spatial ``velocity`` module was
    given, :meth:`_integrate` (the RK2 loop) is wrapped with ``torch.compile``
    at construction time. This is only done for the spatial-velocity path —
    ``SheetOffset``/``FaultOffset`` remain eager (see their nested
    ``autograd.grad`` call in ``_velocity_at``).

    Conventions:

    - ``inverse_map(x)`` maps present-day coordinates to the reference frame
      (integration from ``t = 1 → 0``).
    - ``forward_map(x)`` maps reference to present (``t = 0 → 1``).
    - :meth:`disp` returns the displacement added by
      :meth:`~curlew.geology.geoevent.GeoEvent.undeform`:
      ``x_paleo = x_present + disp(x_present)``.

    Parameters
    ----------
    velocity : torch.nn.Module, optional
        Spatial velocity field (e.g. :class:`~curlew.fields.clebsch.ClebschVelocity`).
        ``None`` (default for :class:`SheetOffset`/:class:`FaultOffset`) if a
        subclass overrides :meth:`_velocity_at` instead.
    n_steps : int
        Number of RK2 / Magnus substeps over the unit time interval.
    direction : float
        Sign controls integration direction: negative (default ``-1``)
        integrates for restoration (present → reference), as used by
        :meth:`~curlew.geology.geoevent.GeoEvent.undeform`; positive integrates
        forward in time (reference → present). Magnitude scales the total
        integrated duration relative to the unit interval — this generalises
        the historic ``dt`` (total elapsed pseudo-time) used by
        :class:`SheetOffset`/:class:`FaultOffset`; leave at ``±1`` unless a
        fractional/multiple displacement is genuinely wanted.
    lift_w : torch.Tensor, optional
        Fixed lift coordinates ``w`` passed to the velocity on every evaluation
        (e.g. fault sheet labels); ``None`` for a pure spatial field.
    fault_lift : :class:`~curlew.fields.lift.FaultLift`, optional
        Dynamic GWN sheet machinery (2-D traces or 3-D meshes).  When set,
        overrides ``lift_w`` and enables Lagrangian label freezing plus
        tangency confinement during integration.
    confine : bool, optional
        Project velocity onto fault tangents inside corridors (default: True
        when ``fault_lift`` is set).
    lagrangian_faults : bool, optional
        Freeze present-day sheet labels and co-advect carried fault geometry
        (default: True when ``fault_lift`` is set).
    """

    def __init__(
        self,
        velocity: Optional[nn.Module] = None,
        *,
        n_steps: int = 20,
        direction: float = -1.0,
        lift_w: Optional[torch.Tensor] = None,
        fault_lift: Optional[nn.Module] = None,
        confine: Optional[bool] = None,
        lagrangian_faults: Optional[bool] = None,
    ):
        super().__init__()
        self.dim = None
        if velocity is not None:
            if not hasattr(velocity, "dim"):
                raise TypeError("velocity must expose a `dim` attribute")
            if not callable(getattr(velocity, "forward", None)):
                raise TypeError("velocity must implement forward(x, w=None, t=None)")
            self.dim = int(velocity.dim)
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.velocity = velocity
        self.n_steps = int(n_steps)
        self.direction = float(direction)
        self.fault_lift = fault_lift
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

        # Only the spatial-velocity-module path is safe to torch.compile: its
        # default `_velocity_at` is pure closed-form tensor math (e.g.
        # FSF.gradient_and_hessian), with no `G`-dependence. SheetOffset/FaultOffset
        # (velocity=None) override `_velocity_at` to call `G.gradient(...,
        # create_graph=True)` on every sub-step instead — compiling through that
        # nested autograd.grad is unreliable, so it is intentionally left eager.
        #
        # `dynamic=True` is required: `RestorationField.loss()` calls this on
        # differently-sized batches (gp/gv normals vs. each `eq` trace) every
        # epoch. Without it, torch.compile specialises to each exact batch size
        # and throws the graph away whenever the size changes, so alternating
        # between shapes every epoch triggers a full recompile every epoch
        # forever (each taking seconds) instead of ever settling — in practice
        # indistinguishable from the training loop hanging.
        if curlew.compile and self.velocity is not None and self.fault_lift is None and hasattr(torch, "compile"):
            self._integrate = torch.compile(self._integrate, dynamic=True)

    def _sign(self) -> float:
        return 1.0 if self.direction > 0 else -1.0

    def _velocity_at(
        self,
        x: torch.Tensor,
        t: float,
        *,
        jac: bool = False,
        G=None,
        material: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        gate: Optional[torch.Tensor] = None,
    ):
        """
        Velocity at ``x`` (and Jacobian if ``jac``). Default implementation
        evaluates the spatial ``self.velocity`` module; ``G`` is accepted for
        the shared calling convention but unused here. Subclasses without a
        ``velocity`` module (:class:`SheetOffset`, :class:`FaultOffset`)
        override this to derive velocity from ``G`` instead.
        """
        if self.velocity is None:
            raise NotImplementedError(
                "_velocity_at must be overridden when no velocity module is provided."
            )
        if self.fault_lift is not None:
            return self._velocity_with_fault(
                x, t, jac=jac, material=material, gate=gate,
            )
        w = self.lift_w
        if w is not None and w.shape[0] == 1 and x.shape[0] != 1:
            w = w.expand(x.shape[0], -1)
        if jac:
            if not hasattr(self.velocity, "forward_and_jacobian"):
                raise AttributeError(
                    "velocity must implement forward_and_jacobian for Jacobian integration"
                )
            v, jv = self.velocity.forward_and_jacobian(x, w, t)
            return _checkVectorShape(v, x), jv
        v = self.velocity(x, w, t)
        return _checkVectorShape(v, x)

    def _velocity_with_fault(
        self,
        x: torch.Tensor,
        t: float,
        *,
        jac: bool = False,
        material: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        gate: Optional[torch.Tensor] = None,
    ):
        fl = self.fault_lift
        if material is None:
            st = fl.state(x, gate=gate)
        else:
            w_active, w_raw = material
            st = fl.state_with_material(x, w_active, w_raw, gate=gate)
        if jac:
            v, jv = self.velocity.forward_and_jacobian(x, st.w_active, t)
        else:
            v, jv = self.velocity(x, st.w_active, t), None
        v = _checkVectorShape(v, x)
        if self.confine:
            v = self._confine_blended(v, st)
        if st.mobility is not None:
            v = v * st.mobility.unsqueeze(-1)
        return (v, jv) if jac else v

    @staticmethod
    def _confine_blended(v: torch.Tensor, st: SheetState) -> torch.Tensor:
        """Joint tangency projection over all faults (2-D or 3-D)."""
        lam = st.confinement.clamp(0.0, 1.0 - 1e-9)
        alpha = lam / (1.0 - lam)
        n = st.normal
        dim = v.shape[-1]
        eye = torch.eye(dim, dtype=v.dtype, device=v.device)
        outer = alpha.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
            "nfd,nfe->nfde", n, n,
        )
        m = eye.unsqueeze(0).expand(v.shape[0], -1, -1) + outer.sum(dim=1).to(dtype=v.dtype)
        return torch.linalg.solve(m, v.unsqueeze(-1)).squeeze(-1)

    def _integrate(
        self,
        x: torch.Tensor,
        *,
        sign: float,
        jac: bool = False,
        duration: float = 1.0,
        n_steps: Optional[int] = None,
        G=None,
    ):
        if self.fault_lift is not None:
            return self._integrate_with_fault(
                x, sign=sign, jac=jac, duration=duration, n_steps=n_steps,
            )
        n = self.n_steps if n_steps is None else max(1, int(n_steps))
        dt = duration / n
        t0 = 0.0 if sign > 0 else 1.0
        J = None
        if jac:
            dim = self.dim if self.dim is not None else x.shape[-1]
            J = (
                torch.eye(dim, dtype=x.dtype, device=x.device)
                .expand(len(x), -1, -1)
                .clone()
            )

        for k in range(n):
            t_k = t0 + sign * k * dt
            t_mid = t0 + sign * (k + 0.5) * dt
            v0 = self._velocity_at(x, t_k, jac=False, G=G)
            x_mid = x + sign * (dt / 2) * v0
            if jac:
                v_mid, Jv = self._velocity_at(x_mid, t_mid, jac=True, G=G)
                J = torch.bmm(expm_traceless(sign * dt * Jv.to(dtype=J.dtype)), J)
            else:
                v_mid = self._velocity_at(x_mid, t_mid, jac=False, G=G)
            x = x + sign * dt * v_mid

        return (x, J) if jac else x

    def _integrate_with_fault(
        self,
        x: torch.Tensor,
        *,
        sign: float,
        jac: bool = False,
        duration: float = 1.0,
        n_steps: Optional[int] = None,
    ):
        n = self.n_steps if n_steps is None else max(1, int(n_steps))
        dt = duration / n
        t0 = 0.0 if sign > 0 else 1.0
        J = None
        if jac:
            dim = self.dim if self.dim is not None else x.shape[-1]
            J = (
                torch.eye(dim, dtype=x.dtype, device=x.device)
                .expand(len(x), -1, -1)
                .clone()
            )

        fl = self.fault_lift
        lag = self.lagrangian_faults
        material = None
        trace_mat = None
        if lag:
            fl.snapshot_traces()
            material = fl.material_labels(x)
            trace_mat = fl.trace_material_labels()

        try:
            for k in range(n):
                t_k = t0 + sign * k * dt
                t_mid = t0 + sign * (k + 0.5) * dt
                gate_k = fl.activation(t_k) if lag else None
                gate_mid = fl.activation(t_mid) if lag else None
                if lag:
                    self._advect_fault_traces(
                        sign, dt, t_k, t_mid, trace_mat, gate_k, gate_mid,
                    )
                v0 = self._velocity_at(
                    x, t_k, jac=False, material=material, gate=gate_k,
                )
                x_mid = x + sign * (dt / 2) * v0
                if jac:
                    v_mid, Jv = self._velocity_at(
                        x_mid, t_mid, jac=True, material=material, gate=gate_mid,
                    )
                    step = sign * dt * Jv.to(dtype=J.dtype)
                    if J.shape[-1] == 2:
                        J = torch.bmm(expm_traceless(step), J)
                    else:
                        J = torch.bmm(torch.linalg.matrix_exp(step), J)
                else:
                    v_mid = self._velocity_at(
                        x_mid, t_mid, jac=False, material=material, gate=gate_mid,
                    )
                x = x + sign * dt * v_mid
        finally:
            if lag:
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
    ) -> None:
        fl = self.fault_lift
        for i in range(fl.n_faults):
            if not fl.advect_traces[i]:
                continue
            tr = fl.trace(i)
            w_a, w_r = trace_mat[i]
            p0 = fl.advect_probe_points(i, w_r)
            v0 = self._velocity_at(p0, t_k, material=(w_a, w_r), gate=gate_k)
            tr_mid = tr + sign * (dt / 2) * v0
            p_mid = fl.advect_probe_points(i, w_r, points=tr_mid)
            v_mid = self._velocity_at(
                p_mid, t_mid, material=(w_a, w_r), gate=gate_mid,
            )
            fl._set_trace(i, tr + sign * dt * v_mid)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        """Reference → present (``t = 0 → 1``)."""
        return self._integrate(x, sign=+1.0)

    def inverse_map(self, x: torch.Tensor) -> torch.Tensor:
        """Present → reference (``t = 1 → 0``)."""
        return self._integrate(x, sign=-1.0)

    def forward_map_with_jacobian(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(Phi(x), grad Phi)`` with ``det J = 1`` for div-free velocities."""
        return self._integrate(x, sign=+1.0, jac=True)

    def inverse_map_with_jacobian(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(Phi^{-1}(x), grad Phi^{-1})``."""
        return self._integrate(x, sign=-1.0, jac=True)

    @torch.no_grad()
    def map_to_time(
        self,
        x: torch.Tensor,
        t: float,
        *,
        forward: bool = True,
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Partial map over duration ``t ∈ [0, 1]`` (animation / probing)."""
        if t <= 0:
            return x.clone()
        sign = +1.0 if forward else -1.0
        return self._integrate(
            x, sign=sign, duration=min(float(t), 1.0), n_steps=n_steps
        )

    def disp(self, X: torch.Tensor, G=None) -> torch.Tensor:
        """
        Displacement for :meth:`~curlew.geology.geoevent.GeoEvent.undeform`.

        With default ``direction = -1``, returns ``inverse_map(X) - X`` (scaled
        by ``abs(direction)`` when its magnitude isn't 1). ``G`` is forwarded
        to :meth:`_velocity_at` on every sub-step — used by
        :class:`SheetOffset`/:class:`FaultOffset`; ignored by the default
        spatial-velocity-module implementation.
        """
        x = X if isinstance(X, torch.Tensor) else _tensor(X)
        sign = self._sign()
        x_end = self._integrate(x, sign=sign, duration=abs(self.direction), G=G)
        return x_end - x

    def learnable(self):
        """True when the wrapped velocity (if any) has trainable parameters."""
        if self.velocity is None:
            return any(p.requires_grad for p in self.parameters())
        return any(p.requires_grad for p in self.velocity.parameters())

    def loss(self):
        """Placeholder — event-specific restoration losses are added later."""
        from curlew.core import Pebble

        return Pebble()

    def __repr__(self):
        return (
            f"FlowOffset(n_steps={self.n_steps}, direction={self.direction}, "
            f"velocity={self.velocity!r})"
        )


class SheetOffset(FlowOffset):
    """
    Dyke/sill-style opening in the gradient direction of the host scalar field,
    integrated with :class:`FlowOffset`'s RK2 scheme (default ``n_steps=1``,
    ``dt=-1.0``). The "velocity" (dyke-normal opening rate) is not
    time-dependent; it is simply re-evaluated at the current integration
    position on each RK2 sub-step.
    """

    def __init__(
        self,
        contact=(-1, 1),
        aperture=1,
        polarity=1,
        *,
        n_steps=1,
        dt=-1.0,
    ):
        super().__init__(velocity=None, n_steps=n_steps, direction=dt)
        assert len(contact) == 2, (
            "Contact must be a list or tuple of length two, representing the lower and upper "
            "surface of this intrusion."
        )
        self.contact = contact
        self.aperture = aperture
        self.polarity = polarity

    def _velocity_at(self, x, t, *, jac=False, G=None):
        """Infinite dyke displacement (evaluated at ``x``; ``G`` supplies the host scalar field)."""
        if jac:
            raise NotImplementedError("SheetOffset does not support Jacobian integration")
        ds, s = self.dss(x, G)
        s0, s1 = G.getIsovalues(self.contact)
        a = np.abs(s1 - s0)
        if self.polarity > 0:
            mask = s > min(s0, s1) #  move hangingwall up
        else:
            mask = s < max(s0, s1) #  move footwall down
            ds = -ds # need to reverse the gradient!

        v = ds.clone()
        v[~mask] = 0
        return (a * self.aperture) * v

    def __repr__(self):
        return (
            f"SheetOffset(contact={self.contact}, aperture={self.aperture}, "
            f"polarity={self.polarity}, n_steps={self.n_steps}, dt={self.direction})"
        )

class FaultOffset(FlowOffset):
    """
    Fault-related displacement from the gradient of the GeoEvent's implicit surface, integrated
    with :class:`FlowOffset`'s RK2 scheme (default ``n_steps=2``, ``dt=-1.0``). The instantaneous
    "velocity" at each RK2 sub-step is the mode-II slip vector constructed from ``dss`` (same
    construction as the historical single-step fault offset), re-evaluated at the current
    integration position — it is not actually time-dependent. For strongly curved faults,
    increase ``n_steps`` (and/or reduce ``dt``) instead of using a separate corrector pass.
    """

    def __init__(
        self,
        shortening,
        offset=0,
        offsetRange=None,
        contact=0,
        width=1e-5,
        modifier=None,
        polarity=1,
        *,
        n_steps=1,
        dt=-1.0,
    ):
        """
        Create a new fault offset object.

        Parameters
        ----------
        shortening : torch.tensor
            The principal shortening direction. This is used to determine slip direction on the fault, through projection onto
            the tangent of the fault plane.
        offset : float | tuple
            The mode II shear offset on the fault. Defaults to 0. If a float is passed
            then exactly this offset is used. Otherwise, a tuple should be passed in which
            the first element is a learnable parameter, and the second two give the allowed
            range of values, such that `offset = torch.clamp( offset[0], offset[1], offset[2] )`.
        offsetRange : tuple
            A tuple specifying the minimum and maximum allowed offset. Must be defined if offset is a learnable parameter,
            such that `applied_offset = torch.clamp( offset, min(offsetRange), max(offsetRange))
        contact : float | str
            The isosurface value (or name) defining the value used to define the fault surface. Default is zero.
        width : float | tuple
            The scaling factor for the sigmoid function used to determine the sign of
            the displacement across the fault. Use high values to get shear-zone like
            ductile deformation, and low values to get sharp "brittle" offsets. Default is 1e-5.

            A tuple can also be passed to use two sigmoid functions, one for an outer ductile
            deformation (e.g., drag folds) and another for an inner more-brittle deformation.
            This tuple should contain the following: `(outer_sharpness, inner_sharpness, proportion)`,
            where proportion (0 to 1) defines the strain partioning between the ductile and the brittle parts.
        modifier : str | None
            Name of an implicit field on the parent :class:`~curlew.geology.geoevent.GeoEvent` (e.g. from
            ``addField``) that is evaluated at all ``x`` and used to scale the applied offset. Used to
            e.g. implement finite faults where slip decays inside an ellipsoidal patch.
        polarity : int, optional
            The polarity of the fault offset. If 1 (default), the hangingwall is moved and the footwall is fixed.
            If -1, the footwall is moved and the hangingwall is fixed.
        n_steps : int, optional
            The number of RK2 substeps to take. Default is 1 (assumes quite a smooth displacement field!).
        dt : float, optional
            The time step size. Default is -1.0 (i.e. reconstruct from modern to paleo-coords), though +1.0 can be useful to move from paleo to modern coords.
        """
        super().__init__(velocity=None, n_steps=n_steps, direction=dt)
        self.shortening = shortening
        self.offset = offset
        self.offsetRange = offsetRange
        self.contact = contact
        self.width = width
        self.polarity = polarity
        self.modifier = modifier

    def _resolve_modifier(self, G):
        m = self.modifier
        if m is None:
            return None
        if isinstance(m, str):
            return G.getField(m)
        raise TypeError(
            f"FaultOffset modifier must be a field name (str) or None, got {type(m).__name__}."
        )

    def _velocity_at(self, x, t, *, jac=False, G=None):
        if jac:
            raise NotImplementedError("FaultOffset does not support Jacobian integration")

        # get field values and gradient at evaluation points
        ds, s = self.dss(x, G, normalize=True)

        # get contact surface values
        contact = self.contact
        if isinstance(contact, str):
            contact = G.getIsovalue(contact)
        s_adj = s - contact

        # calculate slip direction vector
        slip = self.shortening[None, :] - (
            torch.sum(self.shortening * ds, dim=-1, keepdim=True)
        ) * ds
        slip = slip / (torch.norm(slip, dim=1) + 1e-6)[:, None]

        # scale by offset magnitude
        off = self.offset
        if self.offsetRange is not None:
            off = torch.clamp(off, min(self.offsetRange), max(self.offsetRange))
        off = off * slip

        # apply sign flip and sigmoid scaling (for ductile faults)
        s_scale = s_adj.clone()
        if self.polarity < 0:
            s_scale = -s_scale
        if isinstance(self.width, tuple):
            s1, s2w, p = self.width
            scale = (1 - p) * torch.sigmoid(
                s_scale * 4 / np.clip(s1, 1e-6, np.inf)
            ) + p * torch.sigmoid(s_scale * 4 / np.clip(s2w, 1e-6, np.inf))
        else:
            scale = torch.sigmoid(
                s_scale * 4 / np.clip(self.width, 1e-6, np.inf)
            )
        off = off * scale[:, None]

        # apply modifiers for finite faults (if any)
        mod = self._resolve_modifier(G)
        if mod is not None:
            m = mod.forward(x, transform=False)
            if m.ndim == 1:
                m = m[:, None]
            off = off * m

        # return displacement vectors
        return off

    def __repr__(self):
        return (
            f"FaultOffset(contact={self.contact}, offset={self.offset}, width={self.width}, "
            f"shortening={self.shortening}, n_steps={self.n_steps}, dt={self.direction})"
        )

class FoldOffset( OffsetBase ):
    """
    Calculate offsets from a scalar field (SF) representing distance along a fold series.
    """

    def __init__(self, thicker : Union[torch.tensor, np.ndarray], shorter : Union[torch.tensor, np.ndarray], shortening : float, periodic):
        """
        Create a new fold offset object

        Parameters
        ----------
        thicker : torch.tensor | np.ndarray
            The direction of principal stretching (i.e. the direction in which the folds thicken the series)
        shorter : torch.tensor  | np.ndarray
            The direction of principal shortening (i.e. axis along which the folds act)
        shortening : float
            The bulk shortening associated with this folding. Assumed to be constant everywhere.
        periodic : function
            A periodic function that takes an array of scalar values and returns periodically varying offsets.
        """
        super().__init__()
        self.thicker = _tensor( thicker, dev=curlew.device, dt=curlew.dtype)
        self.shorter = _tensor( shorter, dev=curlew.device, dt=curlew.dtype)
        self.shortening = shortening
        self.periodic = periodic
    
    def disp(self, X, G):
        """
        Compute displacement vectors for points `X` based on GeoEvent `G`.
        """
        ds, s = self.dss(X,G,normalize=False) # get gradient direction
        #ds, s = G.field.compute_gradient( X, normalize=False,
        #                              return_value=True,
        #                              transform=False )

        scale = torch.mean( torch.norm(ds, dim=-1) )
        y = self.periodic(s) # compute fold function

        # convert to displacement vectors
        disp = -y[:,None] * self.thicker[None,:] # remove fold amplitude
        disp = disp + s[:,None]*self.shorter[None,:]*(self.shortening / scale) # extend to original length

        # return displacement vectors
        return disp
