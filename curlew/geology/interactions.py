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
from typing import Optional, Union, List

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
    def __init__(self, threshold : Union[str, list] = 0, mode='above'):
        """
        Create a new "overprint" object for applying overprinting stratigraphic (e.g., unconformities) and
        igneous (e.g., dykes, intrusions) events.

        Parameters
        -----------
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
        """
        super().__init__()
        self.threshold = threshold
        self.mode = mode

    def apply(self, parent, child, domain=None ):
        """
        Combine two scalar fields, keeping the parent field where the child field is below a threshold.

        The output will have two dimensions: the first represents the scalar value, and the second 
        represents the ID of the event responsible for this value.

        Parameters
        ----------
        parent : curlew.geofield.Geode
            A `Geode` (output object) from the older GeoField.
        child : curlew.geofield.Geode
            A `Geode` (output object) from the younger GeoField.
        domain : torch.Tensor
            An (N,) array defining the implicit field to use as a domain mask that determines
            which regions are overprinted (as described by `self.mode`). If None (default) `child.scalar`
            will be used, to give a typical e.g., unconformity or intrusion.
        Returns
        -------
        numpy.ndarray
            An updated array of shape (N, 2) containing the updated scalar values and event IDs.
        """
        assert self.thresh is not None, "`self.thresh` must be defined (by e.g. evaluating an isosurface) before calling `overprint`."
        if domain is None: domain = child.scalar # child field determines the domain

        if isinstance(self.thresh, list):
            thresh = float(self.thresh[0]) if len(self.thresh) == 1 else self.thresh
        else:
            thresh = float(self.thresh)

        # apply threshold
        if isinstance(thresh, (float, int)):
            # 1 in areas where child > thresh
            mask = (domain > thresh) #torch.sigmoid(self.sharpness * (domain - thresh))
        else:
            mask = torch.zeros(len(domain), device=curlew.device, dtype=curlew.dtype)
            for i in np.arange(len(thresh), step=2):
                T = thresh[i:(i+2)]
                lower_mask = domain > np.min(T) #torch.sigmoid(self.sharpness * (domain - np.min(T)))
                upper_mask = domain < np.max(T) #torch.sigmoid(self.sharpness * ( np.max(T) - domain))
                mask = torch.logical_or( mask, lower_mask * upper_mask)  # 1 in areas where thresh[0] < child < thresh[1]
        mask = mask.type(curlew.dtype)
        if ('below' in self.mode.lower()) or ('out' in self.mode.lower()):
            mask = 1 - mask # flip mask

        # combine results and return an updated Geode object
        return parent.combine( child, mask )

    def __repr__(self):
        return f"Overprint(mode='{self.mode}', thresh={self.thresh})"

# OFFSETTING RELATIONS - functions used to move things around (kinematic fields; these are the real shakers and movers).
# ------------------------------------------------------------------------------------------------------------------------
class OffsetBase(LearnableBase):
    """
    Class from which all offset classess should inherit. 
    """
    def eval( self, x, G ):
        """
        Get displacement vectors for points `X` based on GeoField `G`. This will be called by the GeoField and return the results of self.disp(...).
        """
        o = self.disp( x, G )
        return o
    
    def learnable(self):
        """ Return true if this offset has learnable parameters (and an optimiser is initialised)."""
        return self.optim is not None
    
    def dss( self, x, G, normalize=False ):
        """
        Evaluate the scalar field gradient (ds) and value (s)  for the points `X` given GeoField `G`. Note that 
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

    def disp( self, X, G ):
        """
        Compute displacement vectors for points `X` based on GeoField `G`. Child classess implementing specific types of offset should implement this function.
        """
        raise NotImplementedError
    
class VFieldOffset(OffsetBase):
    """
    Integrate a "velocity" field to derive displacements. 
    If `field` is a `BaseSF`, its forward pass is treated as the velocity `v(x)`. 
    Subclasses may omit `field` and override `_velocity` instead (see `SheetOffset`, `FaultOffset`).
    """
    def __init__(
        self,
        field: Optional[BaseSF] = None,
        *,
        n_steps: int = 4,
        dt: float = -1.0,
        eval_transform: bool = False,
    ):
        """
        Create a new velocity field offset object.

        Parameters
        ----------
        field : `BaseSF`, optional
            The velocity field to integrate. If None, the subclass must override `_velocity`.
        n_steps : int, optional
            The number of Euler steps to take. Default is 4 (assumes quite a smooth displacement field!).
        dt : float, optional
            The time step size. Default is -1.0, i.e. reconstruct backwards in time. +1 can be used to deform paleo-coordinates forward in time. 
        eval_transform : bool, optional
            Whether to evaluate the field in the local coordinate system of the GeoField. Default is False.
        """
        super().__init__()
        if field is not None and not isinstance(field, BaseSF):
            raise TypeError("field must be a BaseSF instance or None")
        self.field = field
        self.n_steps = int(n_steps)
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.dt = float(dt) / self.n_steps  # Euler sub-step size (total time span is `dt`)
        self._eval_transform = bool(eval_transform)

    def _velocity(self, x: torch.Tensor, G) -> torch.Tensor:
        if self.field is None:
            # this will be implemented in child classes 
            raise NotImplementedError(
                "_velocity must be implemented when no latent field is provided."
            )
        
        # get the velocity vectors from the underlying field
        out = self.field.forward(x, transform=self._eval_transform)
        return _checkVectorShape(out, x)

    def disp(self, X, G):
        x = X # make a copy of position vectors
        u = torch.zeros_like(X) # initialise displacement vectors
        for _ in range(self.n_steps): # integration loop
            v = self._velocity(x, G)
            u = u + v * self.dt # accumulate displacement
            x = x + v * self.dt # update positions (for next velocity evaluation)
        return u # TODO - update so that x' is returned too??

    def __repr__(self):
        frepr = repr(self.field) if self.field is not None else "None"
        return (
            f"VelFieldOffset(field={frepr}, n_steps={self.n_steps}, "
            f"dt={self.dt}, eval_transform={self._eval_transform})" )

class SheetOffset(VFieldOffset):
    """
    Dyke/sill-style opening in the gradient direction of the host scalar field, integrated with
    :class:`VelFieldOffset` (default ``n_steps=1``, ``dt=1``).
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
        super().__init__(field=None, n_steps=n_steps, dt=dt, eval_transform=False)
        assert len(contact) == 2, (
            "Contact must be a list or tuple of length two, representing the lower and upper "
            "surface of this intrusion."
        )
        self.contact = contact
        self.aperture = aperture
        self.polarity = polarity

    def _velocity(self, x, G):
        """Implement infinite dyke displacement"""
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
            f"polarity={self.polarity}, n_steps={self.n_steps}, dt={self.dt})"
        )

class FaultOffset(VFieldOffset):
    """
    Fault-related displacement from the gradient of the GeoField's implicit surface, integrated
    with :class:`VFieldOffset` (default ``n_steps=2``, ``dt=1``). The instantaneous "velocity"
    at each Euler sub-step is the mode-II slip vector constructed from ``dss`` (same construction
    as the historical single-step fault offset). For strongly curved faults, increase ``n_steps``
    (and/or reduce ``dt``) instead of using a separate corrector pass.
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
        modifier : curlew.fields.BaseSF
            An implicit field that is evaluated at all `x` and then used to scale the applied offset. Used to 
            e.g., implement finite faults where offset decays according to some ellipsoidal function.
        polarity : int, optional
            The polarity of the fault offset. If 1 (default), the hangingwall is moved and the footwall is fixed.
            If -1, the footwall is moved and the hangingwall is fixed.
        n_steps : int, optional
            The number of Euler steps to take. Default is 2 (assumes quite a smooth displacement field!).
        dt : float, optional
            The time step size. Default is -1.0 (i.e. reconstruct from modern to paleo-coords), though +1.0 can be useful to move from paleo to modern coords.
        """
        super().__init__(field=None, n_steps=n_steps, dt=dt, eval_transform=False)
        self.shortening = shortening
        self.offset = offset
        self.offsetRange = offsetRange
        self.contact = contact
        self.width = width
        self.polarity = polarity
        self.modifier = modifier

    def _fault_kinematics(self, x, G):
        ds, s = self.dss(x, G, normalize=True)
        contact = self.contact
        if isinstance(contact, str):
            contact = G.getIsovalue(contact)
        s_adj = s - contact

        slip = self.shortening[None, :] - (
            torch.sum(self.shortening * ds, dim=-1, keepdim=True)
        ) * ds
        slip = slip / (torch.norm(slip, dim=1) + 1e-6)[:, None]

        off = self.offset
        if self.offsetRange is not None:
            off = torch.clamp(off, min(self.offsetRange), max(self.offsetRange))

        if self.modifier is not None:
            m = self.modifier.forward(x, transform=False)
            off = (m * off).squeeze()

        off = off * slip

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

        return off * scale[:, None].detach()

    def _velocity(self, x, G):
        return self._fault_kinematics(x, G)

    def __repr__(self):
        return (
            f"FaultOffset(contact={self.contact}, offset={self.offset}, width={self.width}, "
            f"shortening={self.shortening}, n_steps={self.n_steps}, dt={self.dt})"
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
    
    def disp( self, X, G ):
        """
        Compute displacement vectors for points `X` based on GeoField `G`.
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