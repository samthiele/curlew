"""
Functions defining how different scalar fields interact with each other to create generative, kinematic and hybrid
events. This includes a function for overprinting scalar fields (generative events) and functions for various
types of kinematics (faults, sheets, etc.). These are the "glue" that bind the different scalar fields in a 
`curlew` model together, allowing for complex geological structures to be represented and manipulated.
"""
import numpy as np
import torch

# OVERPRINTING RELATIONS -- function used to create new rocks (generative fields)
# -----------------------------------------------------------------------------------

def overprint(parent, child, eid=None, thresh=0, sharpness=1e4):
    """
    Combine two scalar fields, keeping the parent field where the child field is below a threshold.

    The output will have two dimensions: the first represents the scalar value, and the second 
    represents the ID of the event responsible for this value.

    Parameters
    ----------
    parent : np.ndarray, torch.Tensor
        The parent scalar field.
    child : np.ndarray, torch.Tensor
        The child scalar field used to overprint the parent one (based on thresh).
    eid : np.ndarray, the unique integer ID of the child event. If None, the previous maximum + 1 is used.
    thresh : float, tuple, str
        The threshold above which the child field will "overprint" the parent one. If a tuple is 
        provided, the child field will "overprint" values where it lies within the specified range. Floating
        point values can be used to define absolute thresholds, while strings can be used to define isosurface
        names (which must exist on the `child` field as per `SF.addIsosurface(...)`).
    sharpness : float
        Multiple used to change the sharpness of the inequality when using differentiable pytorch
        tensors (replacing inequalities with sigmoid functions).
    Returns
    -------
    numpy.ndarray
        An updated array of shape (N, 2) containing the updated scalar values and event IDs.
    """
    child = child.squeeze()
    parent = parent.squeeze()
    if isinstance(thresh, list):
        thresh = thresh[0] if len(thresh) == 1 else thresh
    if len(child.shape) > 1:
        child = child[:,0] # only keep scalar values
    if isinstance(parent, np.ndarray): # numpy -- combination is easy
        if isinstance(thresh, int) or isinstance(thresh, float) or isinstance(thresh, str):
            mask = child > thresh # define mask
        else:
            mask = np.logical_and(child > thresh[0],
                                child < thresh[1] ) # define mask
        if len(parent.squeeze().shape) == 2:
            id = parent[:,1]
            sf = parent[:,0]
        else:
            sf = parent
            id = np.zeros_like(sf)
        if eid is None:
            eid = np.max( id ) + 1
        id[mask] = eid # set eid of material "created" by the child event.
        sf[mask] = child[mask] # overprint scalar value
        return np.array([sf,id]).T
    else: # pytorch -- combine with sigmoid function to ensure differentiability 
        if isinstance(thresh, (float, int)):
            mask = torch.sigmoid(sharpness * (child - thresh))
        else:
            lower_mask = torch.sigmoid(sharpness * (child - thresh[0]))
            upper_mask = torch.sigmoid(sharpness * (thresh[1] - child))
            mask = lower_mask * upper_mask  # Combine masks with logical AND

        if len(parent.squeeze().shape) == 2:
            id = parent[:, 1]
            sf = parent[:, 0]
        else:
            sf = parent
            id = torch.zeros_like(sf, dtype=torch.float32, device=sf.device)

        # Create a differentiable version of id and sf updates
        id_updated = mask * eid + id * (1 - mask)
        sf_updated = mask * child + (1 - mask) * sf  # Weighted combination

        # Concatenate along the second axis
        return torch.cat([sf_updated[:, None], id_updated[:, None]], dim=1)

# OFFSETTING RELATIONS - functions used to move things around (kinematic fields; these are the real shakers and movers).
# ------------------------------------------------------------------------------------------------------------------------

def sheetOffset( X, G, contact=(-1,1), aperture=1 ):
    """
    Calculate offsets for a scalar field representing a dyke.

    Parameters
    ----------
    X : torch.Tensor
        Some (N, ndim) array of points to displace.
    G : SF
        The scalar field (SF) to use for the displacement (e.g., to sample scalar values and/or gradients).
    contact : tuple
        The isosurface values (or names) defining the two sides of this sheet intrusion.
    aperture : float
        Scale factor for the aperture. Default is 1 (Mode I opening).

    Returns
    -------
    torch.Tensor
        The points before dyke opening.
    """
    # get gradient of scalar field at X and associated value
    ds, s = G.field.compute_gradient( X, normalize=False, return_value=True, transform=False )
    # m = torch.linalg.norm(ds) # gradient magnitude 

    # get the contact values and use this to define midpoint
    s0, s1 = G.getIsovalues(contact)
    s = s - 0.5*(s0+s1) # center at mean of the two isosurfaces
    a = np.abs(s1-s0) # aperture, in scalar field units

    # get sign of scalar field at this location
    s[ torch.abs(s) < 1e-6 ] = 1e-6 # ensure s is never 0
    sign = s / torch.abs(s) # TODO -- should this be replaced with a sigmoid function?

    # offset points by aperture in the gradient direction
    return sign[:,None] * a * aperture * 0.5 * ds

def faultOffset( X, G, sigma1, offset=0, contact=0, sharpness=1e5, highcurve=False ):
    """
    Calculate offsets from a scalar field (SF) representing an infinite fault.

    Parameters
    ----------
    X : torch.Tensor
        Some (N, ndim) array of points to displace.
    G : SF
        The scalar field (SF) to use for the displacement (e.g., to sample scalar values and/or gradients).
    sigma1 : torch.tensor
        The principal stress direction used to determine slip direction on the fault, through projection onto 
        the tangent of the fault plane.
    offset : float | tuple
        The mode II shear offset on the fault. Defaults to 0. If a float is passed
        then exactly this offset is used. Otherwise, a tuple should be passed in which 
        the first element is a learnable parameter, and the second two give the allowed
        range of values, such that `offset = torch.clamp( offset[0], offset[1], offset[2] )`.
    contact : float | str
        The isosurface value (or name) defining the value used to define the fault surface. Default is zero.
    sharpness : float | tuple
        The scaling factor for the sigmoid function used to determine the sign of
        the displacement across the fault. Use low values to get shear-zone like 
        ductile deformation, and high values to get sharp "brittle" offsets. Default is 1000.

        A tuple can also be passed to use two sigmoid functions, one for an outer ductile
        deformation (e.g., drag folds) and another for an inner more-brittle deformation. 
        This tuple should contain the following: `(outer_sharpness, inner_sharpness, proportion)`,
        where proportion (0 to 1) defines the strain partioning between the ductile and the brittle parts.
    highcurve : bool, optional
        If True, correct the calculated slip direction vector by re-evaluating the
        gradient of the scalar field at each X+slip and correcting by the difference 
        in scalar value. This can help ensure properly tangential vectors for "highly"
        curved faults, but is computationally more expensive.
    Returns
    -------
    torch.Tensor
        The unfaulted points.
    """

    # get scalar values and gradients
    # N.B. this assumes that X is already in coordiantes of the current event
    ds, s = G.field.compute_gradient( X, normalize=True, return_value=True, transform=False )
    s = s / G.field.mnorm # normalise so scalar field approximates a distance field

    # shift so that the fault surface is at zero
    if isinstance(contact, str):
        contact = G.getIsovalue(contact)
    s = s - (contact / G.field.mnorm) # force isosurface to be at zero

    # get displacement vectors by projecting sigma1 onto tangent to the scalar field
    # [ project onto tangent plane using: sigma1 - sigma1 . gradient ]
    slip = sigma1[None,:] - (torch.sum(sigma1 * ds, dim=-1, keepdim=True)) * ds
    slip = slip / (torch.norm(slip, dim=1)+1e-6)[:,None] # normalise to length 1

    # handle possibly learnable offset
    if isinstance(offset, tuple):
        with torch.no_grad(): # TODO - decide if this is necessary? 
            # keep offset between specified range
            offset[0].clamp(min(offset[1], offset[2]), max(offset[1], offset[2]) )
        offset = offset[0]
    # compute (mode II) slip vectors
    offset = offset * slip
    if isinstance(sharpness, tuple):
        s1, s2, p = sharpness
        sign = (1-p)*torch.sigmoid(s1*s) + p*torch.sigmoid(s2*s) - 0.5
    else:
        sign = torch.sigmoid(sharpness * s)-0.5
    offset = offset * sign[:, None] # get sign of scalar field at this location [determines sign of displacement vector]

    # apply correction for non-locally linear scalar field tangents
    if highcurve:
        ds2, s2 = G.field.compute_gradient( X+offset, normalize=True, return_value=True, transform=False )
        s2 = s2 / G.field.mnorm # normalise so scalar field approximates a distance field
        s2 = s2 - (contact / G.field.mnorm) # force isosurface to be at zero
        delta = s - s2 # - s
        offset = offset - delta[:,None] * ds2
    return offset # add displacements to get undeformed coordinates

def finiteFaultOffset( X, sf, **kwds):
    """
    Function calculate offsets from a scalar field (SF) representing a finite fault.
    """
    pass # TODO! 

def foldOffset( X, G, thicker, shorter, shortening, periodic ):
    """
    Calculate offsets from a scalar field (SF) representing distance along a fold series.

    Parameters
    ----------
    X : torch.Tensor
        Some (N, ndim) array of points to displace.
    G : SF
        The scalar field (SF) to use for the displacement (e.g., to sample scalar values and/or gradients).
    thicker : torch.tensor
        The direction of principal stretching (i.e. the direction in which the folds thicken the series)
    shorter : torch.tensor
        The direction of principal shortening (i.e. axis along which the folds act)
    shortening : float
        The bulk shortening associated with this folding. Assumed to be constant everywhere.
    periodic : float
        A periodic function that takes an array of scalar values and returns periodically varying offsets.
    """
    ds, s = G.field.compute_gradient( X, normalize=False,
                                      return_value=True,
                                      transform=False )
    scale = torch.mean( torch.norm(ds, dim=-1) )
    y = periodic(s) # compute fold function

    # convert to displacement vectors
    disp = -y[:,None] * thicker[None,:] # remove fold amplitude
    disp = disp + s[:,None]*shorter[None,:]*(shortening / scale) # extend to original length

    # apply and return
    return disp