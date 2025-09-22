"""
Functions and objects used for creating and manipulating geological structures in Curlew.
"""
import numpy as np
import torch
from torch import nn
import curlew
from curlew.core import CSet
from curlew.geology.SF import SF
from curlew.geology.interactions import sheetOffset
from curlew.geology.interactions import faultOffset
from curlew.geology.interactions import foldOffset
from curlew.geology.model import _linkF
from curlew.geometry import blended_wave
from curlew.fields.analytical import ALF

# GEOLOGICAL OBJECTS / EVENTS - utility functions for creating SFs for different geological objects, each typically representing a geological event. 
# --------------------------------------------------------------------------------------------------------------------------------------------------
def _initF( name, C, H=None, bound = -np.inf, deformation=None, dargs={}, **kwargs):
    """
    Initialise a SF and handle case where C is a constraint set (`CSet`) or 
    `curlew.fields.NF` or `curlew.fields.analytical.AF` instance.
    """
    if H is None: # init default hyperparams
        H = curlew.core.HSet() 
    if isinstance( C, CSet ): # build a SF
        f = SF( name, H=H, bound=bound, 
               deformation=deformation,
               deformation_args=dargs,
               **kwargs ) # create our SF
        f.field.bind( C ) # bind constraints
    else:  # predifined field
        f = SF( name, H=H, bound=bound, 
               deformation=deformation,
               deformation_args=dargs,
               field=C,
               **kwargs ) # create our SF
    return f

def strati( name, C, H=None, base = -np.inf, **kwargs):
    """
    Create a SF representing a stratigraphic series (base stratigraphy or unconformity).

    Parameters
    ------------
    name : str
        A name for the created stratigraphic series (and SF that represents it).
    C : CSet | curlew.fields.analytical.AF | curlew.fields.NF
        Either a pre-constructed neural field, explicit field or a set of 
        constraints to use to construct a new SF representing this stratigraphic series.
    H : HSet
        Hyperparameters for the created neural field (if C is a CSet). 
    base : float
        Scalar value representing the basement contact of this stratigraphic series. This
        is important for unconformities, as only values greater than `base` are considered
        part of this event. Default is -infinity.

    Keywords
    ----------
    All keywords are passed to `curlew.fields.NF.__init__(...)`.

    Returns
    ---------
    A `curlew.geology.SF` instance for the created structure.
    """
    return _initF( name, C=C, H=H, bound=base, **kwargs)

def sheet(name, C, H=None, contact=(-1,1), aperture=2, **kwargs):
    """
    Create a SF representing a sheet intrusion (dyke, sill or vein).

    Parameters
    -----------
    C : CSet
        The constraints used to constrain this geological structure.
    H : HSet
        The hyperparameters used by the underlying `curlew.fields.NF` interpolator.
    contact : tuple
        A tuple of floats specifying the scalar values for the (upper, lower) sides of 
        the intrusion.
    aperture : float
        The aperture (Mode I opening) of the dyke. Used to displace surrounding rocks.
    Keywords
    -----------
    Keywords are used to set the properties of the underlying `curlew.fields.NF` interpolator
    (i.e. are passed to `curlew.fields.NF.__init__(...)`).

    Returns
    ---------
    A `curlew.geology.SF` instance for the created structure.
    """
    if isinstance(contact, float) or isinstance(contact, int):
        contact = (-contact, contact) # define as upper and lower surface (assuming symmetry)
    dargs = {'contact':contact, 'aperture':aperture}
    return _initF( name, C=C, H=H, bound=contact, 
                  deformation=sheetOffset, dargs=dargs, **kwargs)

def fault(name, C, H=None, sigma1=None, learn_sigma=False, offset=0, contact=0, width=0, highcurve=False, **kwargs):
    """
    Create a SF representing a fault, shear zone or (optionally) dilatant shear vein.

    Parameters
    -----------
    C : CSet
        The constraints used to constrain this geological structure.
    H : HSet
        The hyperparameters used by the underlying `curlew.fields.NF` interpolator.
    sigma1 : np.ndarray
        A numpy array of shape (n,) defining the principal compressive stress vector. This
        is used to resolve the slip direction, by projection onto the tangent of the fault's
        interpolated scalar field. Defaults to vertical.
    learn_sigma : bool
        True if sigma1 should be converted to a learnable parameter. Default is False.
    offset : float | tuple
        The mode-2 slip on this (shear) fault. If a float is passed, 
        the offset will be fixed. If a tuple containing (initial, minimum, maximum) is
        passed then the offset will be learned, by constrained to the specified range.
    contact : float | str
        The value (or name) defining the fault (iso)surface. Default is zero.
    width : float | tuple
        The width of ductile deformation associated with this fault. If a float is passed
        then this will specify the width of ductile deformation (using a sigmoid function),
        such that large widths can be used for shear zones and small values of width
        used for brittle faults. The width is converted to scale factors for a sigmoid fuction using the 
        following formula: `displacementM = slipM x sigmoid(s x (4/width)) x 0.5`.
        
        Optionally, a tuple can also be passed to combine two widths, one for the fault "core"
        and one for broader ductile deformation (e.g., drag folds). This tuple should contain:
        `(width_core, width_ductile, proportion)`, where `width_core` defines the width of the 
        fault core, `width_ductile` defines the (larger) width of surrounding ductile deformation,
        and `proportion` defines the partioning of the total offset between these two deformation
        types.

    Keywords
    -----------
    Keywords are used to set the properties of the underlying `curlew.fields.NF` interpolator
    (i.e. are passed to `curlew.fields.NF.__init__(...)`)

    Returns
    ---------
    A `curlew.geology.SF` instance for the created structure.
    """
    dargs = {'sharpness':1000, 'highcurve': highcurve, 'contact' : contact}

    # handle single or composite width
    if width != 0:
        if isinstance(width, tuple):
            dargs['sharpness'] = (4/width[0], 4/width[1], width[2])
        else:
            dargs['sharpness'] = 4/width
    
    # build field
    f = _initF( name, C=C, H=H, bound=None, 
                  deformation=faultOffset, dargs=dargs, **kwargs)

    # add sigma 1 vector to dargs
    if sigma1 is None: 
        sigma1 = np.zeros( f.field.input_dim )
        sigma1[-1] = -1 # default is vertical vector
    dargs['sigma1'] = torch.tensor( sigma1, device=curlew.device, dtype=curlew.dtype)   

    # handle constant or learnable offsets and/or slip direction
    init=False
    if isinstance( offset, tuple):
        offset, smin, smax = offset # shear (mode II) offset
        f.field.offset = nn.Parameter( torch.tensor( offset, device=curlew.device, dtype=curlew.dtype) ) # will now be changed by optimiser!
        dargs['offset'] = (f.field.offset, smin, smax)
        init=True
    else:
        dargs['offset'] = torch.tensor(offset, device=curlew.device, dtype=curlew.dtype)
    if learn_sigma:
        f.field.sigma1 = nn.Parameter( dargs['sigma1'] )
        dargs['sigma1'] = f.field.sigma1
        init=True
    f.deformation_args = dargs # update dargs
    if init: # re-initialise the optimiser to include the new parameters, if needed
        f.field.init_optim()
    return f

def finiteFault(name, C, H, **kwargs):
    """
    Create a SF representing a finite fault.
    """
    pass

def stock(name, C, H, contact=0, **kwargs):
    """
    Create a SF representing a stock, pluton or batholith. 
    """
    pass

def fold( name, origin, compression, extension, wavelength, amplitude=1.0, sharpness=1.0):
    """
    Create a new SF instance representing a fold structure. This uses an explicitely
    defined scalar field (defining distance along the fold-shortening axis) to compute
    displacement vectors that give the specified fold amplitude and wavelength. 
    """
    # create a fold field and associated deformation function
    compression = compression / np.linalg.norm(compression) # direction of principal compression
    compression *= (2 * np.pi) / wavelength  # scale normal vector to give appropriate wavelength 
    fa = ALF( name, input_dim=len(compression), 
              origin=origin, gradient=compression )
    
    # evaluate fold strain
    x = torch.tensor( np.linspace(0,2,1000), device=curlew.device, dtype=curlew.dtype )
    y = blended_wave(x, f=sharpness, A=amplitude, T=2) # evaluate one waveform
    dx = torch.mean( torch.diff(x) )
    dy = torch.diff( y )
    l0 = torch.sum( torch.sqrt( dx**2 + dy**2 ) ) # line-integral gives initial length
    l1 = x[-1] - x[0] # current length is known
    strain = ((l0-l1) / l1).item() # hence get strain needed to undo folding

    # create a lambda function for evaluating folds from scalar value
    f = lambda x: blended_wave( x, f=sharpness, A=amplitude, T=2)

    # create dargs and set offset function
    extension = extension / np.linalg.norm(extension) # principal stretching direction
    dargs = dict(thicker=torch.tensor(extension, dtype=curlew.dtype, device=curlew.device ),
                 shorter=torch.tensor(compression, dtype=curlew.dtype, device=curlew.device ),
                 shortening=strain, 
                 periodic=f )

    # create and return our SF instance
    return SF( name, None, field=fa, 
        deformation=foldOffset, deformation_args=dargs )

def domainBoundary( name, C, H=None, bound = 0, gt = 0, lt = 1, **kwargs ):
    """
    Create a SF representing a domain boundary, in which different sub-models are modelled on either side of the boundary. 
    This can be very useful for modelling e.g., sedimentary basins, where the basin fill is modelled on one side of the boundary (onlap relations), 
    or for modelling domain boundary faults (in which there is no known or meaningful relationship between the fault's hangingwall and footwall).

    Parameters
    ------------
    name : str
        A name for the created domain boundary.
    C : CSet | curlew.fields.analytical.AF | curlew.fields.NF
        Either a pre-constructed neural field, explicit field or a set of 
        constraints to use to construct a new SF representing this domain boundary.
    H : HSet
        Hyperparameters for the created neural field (if C is a CSet).
    bound : float, str
        A float specifying the value (isosurface value or name) of the interpolated scalar field that represents the domain boundary.
    gt : float | SF | list
        A float or a list of floats or SFs that define the scalar field(s) used to populate the hangingwall (val > bound) of this domain boundary. In essence these
        define the geological sub-model used on the hangingwall side of the domain boundary. If a float is provided, it will be populated with a constant value.
    lt : float | SF | list
        A float or a list of floats or SFs that define the scalar field(s) used to populate the footwall (val < bound) of this domain boundary. In essence these
        define the geological sub-model used on the footwall side of the domain boundary. If a float is provided, it will be populated with a constant value.

    Keywords
    ------------
    All keywords are passed to `curlew.fields.NF.__init__(...)`.

    Returns
    ---------
    A `curlew.geology.SF` instance for the created domain boundary. This can then be included in a GeoModel class. Note that the submodels (i.e. `gt` and `lt`) are now
    associated with this SF instance, so do not need to be (directly) passed to the GeoModel constructor.
    """

    # create field representing domain boundary
    f = _initF( name, C=C, H=H, bound=bound, **kwargs) 

    # define parent / child relationships for domain boundary field
    if not (isinstance(gt, list) or isinstance(gt, list)):
        gt = [gt]
    if not (isinstance(lt, list) or isinstance(lt, list)):
        lt = [lt]
    if isinstance(gt[-1], SF):
        gt[-1].child = f
    if isinstance(lt[-1], SF):
        lt[-1].child = f
    f.parent = gt[-1]
    f.parent2 = lt[-1]

    # build linked list for gt and lt domains
    _linkF( gt + [f])
    _linkF( lt + [f])
    
    return f
