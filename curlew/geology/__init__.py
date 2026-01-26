"""
Functions and objects used for creating and manipulating geological structures in Curlew.
"""
import numpy as np
import torch
from torch import nn
import curlew
from curlew.core import CSet
from curlew.geology.geofield import GeoField
from curlew.geology.interactions import Overprint, SheetOffset, FaultOffset, FoldOffset
from curlew.geology.geomodel import _linkF, GeoModel
from curlew.geometry import blended_wave
from curlew.fields.analytical import LinearField

# GEOLOGICAL OBJECTS / EVENTS - utility functions for creating GeoFields for different geological objects, each typically representing a geological event. 
# --------------------------------------------------------------------------------------------------------------------------------------------------
def _initF( name, C, **kwargs):
    """
    Initialise a GeoField and handle case where C is a constraint set (`CSet`) or 
    `curlew.fields.NF` or `curlew.fields.analytical.AF` instance.
    """
    if isinstance( C, CSet ): # build a GeoField
        f = GeoField( name, **kwargs ) # create our GeoField
        f.field.bind( C ) # bind constraints
    elif C is None: # construct a new field but without constraints
        f = GeoField( name, **kwargs ) # create our GeoField
    else:  # predifined field
        f = GeoField( name, type=type(C), field=C, **kwargs ) # create our GeoField using predefined Field
    return f

def strati( name, *, C, base = -np.inf, sharpness=1e5, mode="above", **kwargs):
    """
    Create a GeoField representing a stratigraphic series (base stratigraphy or unconformity).

    Parameters
    ------------
    name : str
        A name for the created stratigraphic series (and GeoField that represents it).
    C : CSet | curlew.fields.analytical.AF | curlew.fields.NF
        Either a pre-constructed neural field, explicit field or a set of 
        constraints to use to construct a new GeoField representing this stratigraphic series.
    base : float | str
        Scalar value or isosurface name representing the basement contact of this stratigraphic series. This
        is important for unconformities, as only values greater than `base` are considered
        part of this event. Default is -infinity. If a string is provided, an isosurface with the corresponding
        name *must* be added to the returned GeoField.
    mode : str
        The overprinting mode. Options are:
            - `"above"`: replace all regions greater than the provided threshold). Useful for e.g., unconformities.
            - `"below"`: replace all regions less than than the provided threshold). Useful for e.g., intrusions.
    sharpness : float
        Multiple used to change the sharpness of the inequality when using differentiable pytorch
        tensors (as the inequality operator is replaced with a sigmoid functions).

    Keywords
    ----------
    All keywords are passed to `curlew.GeoField.__init__(...)`, many of which are then used to initialise the 
    underlying analytical or neural field. See `curlew.GeoField.__init__(...)` for further details.

    Returns
    ---------
    A `curlew.geology.GeoField` instance for the created structure.
    """
    # build object for unconformable overprinting
    o = Overprint(threshold=base, sharpness=sharpness, mode=mode)
    return _initF( name, C=C, overprint=o, **kwargs)

def sheet(name, *, C, contact=(-1,1), aperture=2, sharpness=1e5, **kwargs):
    """
    Create a GeoField representing a sheet intrusion (dyke, sill or vein).

    Parameters
    -----------
    name : str
        A name for the created stratigraphic series (and GeoField that represents it).
    C : CSet | curlew.fields.analytical.AF | curlew.fields.NF
        The constraints or predefined field used to constrain this geological structure.
    contact : tuple
        A tuple of floats specifying the scalar values for the (upper, lower) sides of 
        the intrusion.
    aperture : float
        The aperture (Mode I opening) of the dyke. Used to displace surrounding rocks.
    sharpness : float
        Multiple used to change the sharpness of the inequality when using differentiable pytorch
        tensors (as the inequality operator is replaced with a sigmoid functions).
    
    Keywords
    ----------
    All keywords are passed to `curlew.GeoField.__init__(...)`, many of which are then used to initialise the 
    underlying analytical or neural field. See `curlew.GeoField.__init__(...)` for further details.

    Returns
    ---------
    A `curlew.geology.GeoField` instance for the created structure.
    """
    if isinstance(contact, float) or isinstance(contact, int):
        contact = (-contact, contact) # define as upper and lower surface (assuming symmetry)
    offset = SheetOffset(contact=contact, aperture=aperture)
    
    o = Overprint(threshold=contact, sharpness=sharpness, mode='in')

    return _initF( name, C=C, deformation=offset, overprint=o, **kwargs)

def fault(name, *, C, sigma1, learn_sigma=False, offset=0, contact=0, width=0, highcurve=False, **kwargs):
    """
    Create a GeoField representing a fault, shear zone or (optionally) dilatant shear vein.

    Parameters
    -----------
    name : str
        A name for the created stratigraphic series (and GeoField that represents it).
    C : CSet | curlew.fields.analytical.AF | curlew.fields.NF
        The constraints or predefined field used to constrain this geological structure.
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
    ----------
    All keywords are passed to `curlew.GeoField.__init__(...)`, many of which are then used to initialise the 
    underlying analytical or neural field. See `curlew.GeoField.__init__(...)` for further details.

    Returns
    ---------
    A `curlew.geology.GeoField` instance for the created structure.
    """
    
    # sigma1
    if sigma1 is None: 
        sigma1 = np.zeros( f.field.input_dim )
        sigma1[-1] = -1 # default is vertical vector
    sigma1 = torch.tensor( sigma1, device=curlew.device, dtype=curlew.dtype) 
    
    # build offset object
    O = FaultOffset(sigma1=sigma1, offset=offset, contact=contact, 
                         width=width, highcurve=highcurve )

    # handle constant or learnable offsets and/or slip direction
    init=False
    if isinstance( offset, tuple):
        offset, smin, smax = offset # shear (mode II) offset
        #f.field.offset = nn.Parameter( torch.tensor( offset, device=curlew.device, dtype=curlew.dtype) ) # will now be changed by optimiser!
        O.offset = nn.Parameter( torch.tensor( offset, device=curlew.device, dtype=curlew.dtype) ) # will now be changed by optimiser!
        O.offset_min = smin
        O.offset_max = smax
        init=True
    else:
        O.offset = torch.tensor(offset, device=curlew.device, dtype=curlew.dtype)
    if learn_sigma:
        O.sigma1 = nn.Parameter( O.sigma1 )
        init=True
    
    if init: # initialise the optimiser to include the new parameters
        O.init_optim(lr=kwargs.get('learning_rate', 1e-1))

    # build field
    f = _initF( name, C=C, deformation=O, **kwargs)

    return f

def finiteFault(name, *, C, H, **kwargs):
    """
    Create a GeoField representing a finite fault.
    """
    pass

def stock(name, *, C, H, contact=0, **kwargs):
    """
    Create a GeoField representing a stock, pluton or batholith. 
    """
    pass

def fold( name, *, origin, compression, extension, wavelength, amplitude=1.0, sharpness=1.0, estStrain=False):
    """
    Create a GeoField representing a fold structure.

    This constructs an explicitly defined scalar field aligned with the
    fold-shortening axis, then computes displacement vectors that reproduce a
    fold geometry with the specified wavelength, amplitude, and sharpness.
    Optionally, an estimated finite strain required to “undo” the folding can
    be computed from the waveform geometry (and applied when transforming from
    model to paleo-coordinates).

    Parameters
    ------------
    name : str
        A name for the created stratigraphic series (and GeoField that represents it).
    origin : array-like
        A point through which the fold scalar field passes; used as the field’s origin.
    compression : array-like
        Vector describing the principal compression direction. This is
        normalized and scaled internally to represent the fold wavelength. This vector
        should be perpendicular to the fold's axial foliation.
    extension : array-like
        Vector describing the principal extension direction, such that the fold axis
        is the cross product between the extension and compression vectors.
    wavelength : float
        The fold wavelength. Used to scale the compression vector into a
        periodic scalar field.
    amplitude : float, default=1.0
        Amplitude of the fold waveform.
    sharpness : float, default=1.0
        Sharpness of the fold waveform, controlling the peakedness of the
        blended wave.
    estStrain : bool, default=False
        If True, numerically estimate the finite strain required to restore the
        folded layer to its unfolded length using a line integral of the
        waveform.

    Keywords
    ----------
    All keywords are passed to `curlew.GeoField.__init__(...)`, many of which are then used to initialise the 
    underlying analytical or neural field. See `curlew.GeoField.__init__(...)` for further details.

    Returns
    ---------
    A `curlew.geology.GeoField` instance representing the fold structure, with an
    associated analytical scalar field and deformation function.
    """
    # create a fold field and associated deformation function
    compression = compression / np.linalg.norm(compression) # direction of principal compression
    compression *= (2 * np.pi) / wavelength  # scale normal vector to give appropriate wavelength 
    fa = LinearField( name, input_dim=len(compression), 
              origin=origin, gradient=compression ) # TODO; allow also interpolated fields here
    
    # evaluate fold strain
    if estStrain:
        x = torch.tensor( np.linspace(0,2,1000), device=curlew.device, dtype=curlew.dtype )
        y = blended_wave(x, f=sharpness, A=amplitude, T=2) # evaluate one waveform
        dx = torch.mean( torch.diff(x) )
        dy = torch.diff( y )
        l0 = torch.sum( torch.sqrt( dx**2 + dy**2 ) ) # line-integral gives initial length
        l1 = x[-1] - x[0] # current length is known
        strain = ((l0-l1) / l1).item() # hence get strain needed to undo folding
    else:
        strain = 0 # ignore shortening
    
    # create a lambda function for evaluating folds from scalar value
    f = lambda x: blended_wave( x, f=sharpness, A=amplitude, T=2)

    # creat fold object
    extension = extension / np.linalg.norm(extension) # principal stretching direction
    defo = FoldOffset( thicker=extension, shorter=compression, shortening=strain, periodic=f )

    # create and return our GeoField instance
    return GeoField( name, None, field=fa, # use pre-existing (analytical) field rather than creating a new one
                    deformation=defo )

def domainBoundary( name, *, C, bound = 0, gt = 0, lt = 1, sharpness=1e4, mode="below", **kwargs ):
    """
    Create a GeoField representing a domain boundary, in which different sub-models are modelled on either side of the boundary. 
    This can be very useful for modelling e.g., sedimentary basins, where the basin fill is modelled on one side of the boundary (onlap relations), 
    or for modelling domain boundary faults (in which there is no known or meaningful relationship between the fault's hangingwall and footwall).

    Parameters
    ------------
    name : str
        A name for the created domain boundary.
    C : CSet | curlew.fields.analytical.AF | curlew.fields.NF
        Either a pre-constructed neural field, explicit field or a set of 
        constraints to use to construct a new GeoField representing this domain boundary.
    bound : float, str
        A float specifying the value (isosurface value or name) of the interpolated scalar field that represents the domain boundary.
    gt : float | GeoField | list
        A float or a list of floats or GeoFields that define the scalar field(s) used to populate the hangingwall (val > bound) of this domain boundary. In essence these
        define the geological sub-model used on the hangingwall side of the domain boundary. If a float is provided, it will be populated with a constant value.
    lt : float | GeoField | list
        A float or a list of floats or GeoFields that define the scalar field(s) used to populate the footwall (val < bound) of this domain boundary. In essence these
        define the geological sub-model used on the footwall side of the domain boundary. If a float is provided, it will be populated with a constant value.
    mode : str
        The overprinting mode. Options are:
            - `"above"`: replace all regions greater than the provided threshold). Useful for e.g., unconformities.
            - `"below"`: replace all regions less than than the provided threshold). Useful for e.g., intrusions.
    sharpness : float
        Multiple used to change the sharpness of the inequality when using differentiable pytorch
        tensors (as the inequality operator is replaced with a sigmoid functions).
    
    Keywords
    ----------
    All keywords are passed to `curlew.GeoField.__init__(...)`, many of which are then used to initialise the 
    underlying analytical or neural field. See `curlew.GeoField.__init__(...)` for further details.

    Returns
    ---------
    A `curlew.geology.GeoField` instance for the created domain boundary. This can then be included in a GeoModel class. Note that the submodels (i.e. `gt` and `lt`) are now
    associated with this GeoField instance, so do not need to be (directly) passed to the GeoModel constructor.
    """

    # create field representing domain boundary
    o = Overprint(threshold=bound, sharpness=sharpness, mode=mode)
    f = _initF( name, C=C, overprint=o, **kwargs)

    # define parent / child relationships for domain boundary field
    if not (isinstance(gt, list) or isinstance(gt, list)):
        gt = [gt]
    if not (isinstance(lt, list) or isinstance(lt, list)):
        lt = [lt]
    if isinstance(gt[-1], GeoField):
        gt[-1].child = f
    if isinstance(lt[-1], GeoField):
        lt[-1].child = f
    f.parent = gt[-1]
    f.parent2 = lt[-1]

    # build linked list for gt and lt domains
    _linkF( gt + [f])
    _linkF( lt + [f])
    
    return f
