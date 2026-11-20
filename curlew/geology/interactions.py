"""How scalar fields combine: overprints, and fault, sheet, and fold offsets.

:class:`~curlew.kinematics.flow.FlowOffset` is re-exported for the restore event.
"""
import numpy as np
import torch
import curlew
from curlew import _tensor
from curlew.core import LearnableBase
from curlew.kinematics.flow import FlowOffset, rk2_step
from typing import Union

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
    

class RK2Offset(OffsetBase):
    """Midpoint RK2 for GeoEvent-derived velocity (sheets and faults)."""

    def __init__(self, *, n_steps=1, direction=-1.0):
        super().__init__()
        n = int(n_steps)
        if n < 1:
            raise ValueError("n_steps must be >= 1")
        self.steps = [n]
        self.direction = float(direction)

    @property
    def n_steps(self):
        return self.steps[0]

    def _velocity_at(self, x, t, *, G=None):
        raise NotImplementedError

    def disp(self, X, G):
        x = X if isinstance(X, torch.Tensor) else _tensor(X)
        sign = 1.0 if self.direction > 0 else -1.0
        duration = abs(self.direction)
        n = self.steps[0]
        dt = duration / n
        for k in range(n):
            t_k = (k / n) if sign > 0 else (1.0 - k / n)
            t_mid = t_k + sign * (0.5 / n)
            v0 = self._velocity_at(x, t_k, G=G)
            x_mid = x + sign * (dt / 2) * v0
            v_mid = self._velocity_at(x_mid, t_mid, G=G)
            x = rk2_step(x, v0, v_mid, sign, dt)
        return x - X


class SheetOffset(RK2Offset):
    """
    Dyke/sill-style opening in the gradient direction of the host scalar field,
    integrated with RK2 (default ``n_steps=1``, ``dt=-1.0``).
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
        super().__init__(n_steps=n_steps, direction=dt)
        assert len(contact) == 2, (
            "Contact must be a list or tuple of length two, representing the lower and upper "
            "surface of this intrusion."
        )
        self.contact = contact
        self.aperture = aperture
        self.polarity = polarity

    def _velocity_at(self, x, t, *, G=None):
        del t
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

class FaultOffset(RK2Offset):
    """
    Fault slip from the GeoEvent implicit surface, integrated with RK2
    (default ``n_steps=1``, ``dt=-1.0``).
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
        super().__init__(n_steps=n_steps, direction=dt)
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

    def _velocity_at(self, x, t, *, G=None):
        del t
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
