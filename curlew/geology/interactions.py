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
from curlew.core import LearnableBase
from typing import Union, List

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
    
class SheetOffset( OffsetBase ):
    """
    Calculate offsets for a scalar field representing an infinite sheet intrusion (dyke or sill).
    """

    def __init__(self, contact=(-1,1), aperture=1, polarity=1):
        """
        Initialise a new SheetOffset object.

        Parameters
        ----------
        contact : tuple
            The isosurface values (or names) defining the two sides of this sheet intrusion.
        aperture : float
            Scale factor for the aperture. Default is 1 (Mode I opening).
        polarity : int, optional
            The polarity of the dyke offset. If 1 (default), the hangingwall is moved and the footwall is fixed.
            If -1, the footwall is moved and the hangingwall is fixed.
        """
        super().__init__()
        assert len(contact) == 2, "Contact must be a list or tuple of length two, representing the lower and upper surface of this intrusion."
        self.contact = contact
        self.aperture = aperture
        self.polarity = polarity

    def disp( self, X, G ):
        """
        Compute displacement vectors for points `X` based on GeoField `G`.
        """
        ds, s = self.dss(X,G)
        # m = torch.linalg.norm(ds) # gradient magnitude 

        # get the contact values and use this to define midpoint
        s0, s1 = G.getIsovalues(self.contact)

        # estimate aperture in scalar field units
        a = np.abs(s1-s0)

        # determine mask at which non-zero displacements are applied
        if self.polarity > 0:
            mask = s > min(s0, s1)
        else:
            mask = s < max(s0, s1)
        
        # zero displacement in "footwall"
        ds[~mask] = ds[~mask] * 0

        # offset points by aperture in the gradient direction
        return a * ds * self.aperture
    
    def __repr__(self):
        return f"SheetOffset(contact={self.contact}, aperture={self.aperture})"

class FaultOffset( OffsetBase ):
    """
    Calculate offsets for a scalar field representing an infinite sheet intrusion (dyke or sill).
    """

    def __init__(self, shortening, offset=0, offsetRange=(-np.inf,np.inf), contact=0, width=1e-5, modifier=None, highcurve=False, polarity=1):
        """
        Initialise a new FaultOffset object.

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
            A tuple specifying the minimum and maximum allowed offset. Used if offset is a learnable parameter, 
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
        highcurve : bool, optional
            If True, correct the calculated slip direction vector by re-evaluating the
            gradient of the scalar field at each X+slip and correcting by the difference 
            in scalar value. This can help ensure properly tangential vectors for "highly"
            curved faults, but is computationally more expensive.
        polarity : int, optional
            The polarity of the fault offset. If 1 (default), the hangingwall is moved and the footwall is fixed.
            If -1, the footwall is moved and the hangingwall is fixed.
        """
        super().__init__()
        self.shortening = shortening
        self.offset = offset
        self.offsetRange = offsetRange
        self.contact = contact
        self.width = width
        self.highcurve = highcurve
        self.polarity = polarity
        self.modifier = modifier

    def disp( self, X, G ):
        """
        Compute displacement vectors for points `X` based on GeoField `G`.
        """
        # TODO - only evaluate gradient where slip is significant (faster!)
        ds, s = self.dss(X,G,normalize=True) # get gradient direction
        
        # shift so that the fault surface is at zero
        contact = self.contact
        if isinstance(contact, str):
            contact = G.getIsovalue(contact)
        #s = s - (contact / G.field.mnorm) # force isosurface to be at zero and scale to have ~unit average gradient
        s = s - contact # force isosurface to be at zero

        # get displacement vectors by projecting shortening onto tangent to the scalar field
        # [ project onto tangent plane using: shortening - shortening . gradient ]
        slip = self.shortening[None,:] - (torch.sum(self.shortening * ds, dim=-1, keepdim=True)) * ds
        slip = slip / (torch.norm(slip, dim=1)+1e-6)[:,None] # normalise to length 1

        # handle possibly learnable offset
        offset = self.offset
        if self.offsetRange is not None:
            offset = torch.clamp( offset, min(self.offsetRange), max(self.offsetRange))
        
        # apply modifier (non-uniform offset)
        if self.modifier is not None:
            m = self.modifier.forward( X, transform=False) # N.B. this means the modifier field need to be specified in paleo-coordinates! 
            offset = (m * offset).squeeze()
        
        # compute (mode II) slip vectors
        offset = offset * slip
        if self.polarity < 0:
            s = -s # reverse polarity to move footwall instead of hangingwall
        if isinstance(self.width, tuple):
            s1, s2, p = self.width
            scale = (1-p)*torch.sigmoid(s*4/np.clip(s1,1e-6,np.inf)) + p*torch.sigmoid(s*4/np.clip(s2, 1e-6,np.inf))
        else:
            scale = torch.sigmoid(s*4/np.clip( self.width, 1e-6, np.inf)) # N.B. -4 is approximately where the sigmoid function reaches 0
        
        offset = offset * scale[:, None].detach() # scale displacements to move fault hangingwall only. N.B. we disregard gradients from the scalar field here to help focus on optimising the offset term. 

        # apply correction for non-locally linear scalar field tangents
        # TODO - make this iterative if better approximation is needed?
        if self.highcurve:
            mask = scale > 0 # only apply correction where slip is significant
            ds2, s2 = self.dss( X[mask,:]+offset[mask,:], G )
            #s2 = s2 / G.field.mnorm # normalise so scalar field approximates a distance field
            #s2 = s2 - (contact / G.field.mnorm) # force isosurface to be at zero
            s2 - s2 - contact # force isosurface to be at zero
            delta = s[mask] - s2 # - s
            offset[mask,:] = offset[mask,:] - delta[:,None] * ds2
        return offset # add displacements to get undeformed coordinates
    
    def __repr__(self):
        return f"FaultOffset(contact={self.contact}, offset={self.offset}, width={self.width}, shortening={self.shortening})"
    
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
        self.thicker = torch.tensor( thicker, device=curlew.device, dtype=curlew.dtype)
        self.shorter = torch.tensor( shorter, device=curlew.device, dtype=curlew.dtype)
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