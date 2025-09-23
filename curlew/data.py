"""
Generate synthetic datasets and models for testing purposes.
"""
import numpy as np
from curlew.core import CSet
from curlew.geology.model import GeoModel
from curlew.geometry import grid 
from curlew.geology.interactions import overprint
from curlew.visualise import colour
from curlew.fields.analytical import ALF, ASF, ACF
from curlew.geology import strati, fold, fault, domainBoundary, sheet

def sample( pxy, sxy, shape, pv=None, breaks=19, init=100, xstep=300, pval=0.6, cmap='tab20', seed=42 ):
    """
    Sample value, orientation, and property constraints from a scalar field and associated gradients.

    Parameters
    ----------
    pxy : np.ndarray
        A (N, 2) array of position vectors. Note that this function works only with 2D data.
    sxy : np.ndarray
        A (N,) or (N, d) array of scalar values corresponding to the above points, 
        where the last dimension corresponds to the event ID.
    shape : tuple
        A tuple containing the (width, height) of the grid defined by `pxy`.
    pv : np.ndarray or str
        A (N, d) array of n-dimensional property vectors (e.g., color). Can also be 'rgb' to create synthetic colors.
    breaks : list
        A set of scalar values at which contacts (changes in color) should be placed.
    init : int
        The index of the first "drillhole" in the x-direction.
    xstep : int
        The separation between "drillholes" in the x-direction.
    pval : float
        The probability that an observation is a scalar value (rather than just a gradient) constraint.
    cmap : str
        The name of the Matplotlib colormap to use for sampling colors that determine where geological contacts are.
        Must be a discrete colormap.
    seed : int
        Random seed to facilitate reproducible results.

    Returns
    -------
    np.ndarray
        The sampled value, orientation, and property constraints.
    """

    # sample constraints along drillhoes for s0 and s1
    np.random.seed(seed)

    # ensure sf has an "eventID" axis
    if len(sxy.shape) == 1:
        sxy = np.array([sxy, np.zeros_like(sxy) ]).T
    
    # reshape sf to image dims and initialise structure for accumulating constraints
    sf = sxy.reshape( shape + (2, ))
    xy = pxy.reshape( shape + (-1, ))
    constraints = {int(k):{'vp':[], 'vv':[], 'gp':[], 'gv':[], 'gop':[], 'gov':[]} for k in np.unique(sxy[:,1])}

    # compute contact points and gradients (bedding orientation)
    gx = np.diff( sf[...,0], axis= 0 )
    gy = np.diff( sf[...,0], axis= 1 )
    c = colour(sf[...,0], breaks=breaks, cmap=cmap) # discretise resulting scalar field into colours
    contacts = np.sum( np.abs( np.diff(c, axis=1, append=0) ), axis=-1)  > 0 # get contacts
    domains = np.sum( [np.abs( np.diff(sf[...,1], axis=1, append=0) ), # get domain boundaries as gradients will be wrong here
                       np.abs( np.diff(sf[...,1], axis=0, append=0) )], axis=0) > 0 
    if pv == 'rgb':
        pv = c

    # loop through boreholes and collect
    for x in np.arange(init,sf.shape[0],xstep):
        cc = np.argwhere( contacts[x,:] )
        if (len(cc) > 1):
            for y in cc.squeeze()[:-1]: # ignore last as this is the boundary
                i = int( sf[x,y,1] )
                if not domains[x,y]: # ignore domain boundaries as gradients will be wrong
                    constraints[i]['gp'].append( xy[x,y] ) # add gradient constraint
                    constraints[i]['gv'].append( (gx[x,y], gy[x,y]) )
                    constraints[i]['gop'].append( xy[x,y] ) # add identical orientation constraint
                    constraints[i]['gov'].append( (gx[x,y], gy[x,y]) )
                    if np.random.rand() <= pval: # add value constraint
                        constraints[i]['vp'].append( xy[x,y] )
                        constraints[i]['vv'].append( sf[x,y,0] )

        if pv is not None: # add continuous property constraints
            for y in np.arange(sf.shape[1], step=1):
                if 'property' not in constraints:
                    constraints['property'] =  {'pp':[], 'pv':[], 'vp':[], 'vv':[]}
                constraints['property']['pp'].append( xy[x,y] ) # property position
                constraints['property']['pv'].append( pv[x,y] ) # property value
                constraints['property']['vp'].append( xy[x,y] ) # position again
                constraints['property']['vv'].append( sf[x,y,:] ) # also store scalar value -- useful as a reference
        
    # convert to numpy arrays
    for i in constraints.keys():
        for k,v in constraints[i].items():
            if len(v) > 0:
                constraints[i][k] = np.array(v)
                if (k == 'gv') or (k=='gov'):  # normalise gradient constraints
                    constraints[i][k] /= np.linalg.norm( v, axis=-1 )[:,None]
            else:
                constraints[i][k] = None

    return [CSet(**v) for k,v in constraints.items()]

# Geological models
# ------------------------
def steno( shape=(1500,1000), **kwargs ):
    """
    Return a synthetic model with a slightly curved layer-cake stratigraphy.

    Parameters
    ----------
    shape : tuple
        The width and height of the generated data. 
    
    Keywords
    ---------
        All keywords are passed to `curlew.data.sample(...)`
    
    Returns
    --------
    xy : np.ndarray
        A list of (N,d) xy points (from a grid with the specified `shape`)
    s : np.ndarray,
        An array of shape (N,2) containing the scalar values and ID of the "event" that caused them.
    C : list, 
        A list of constraints for each of the (two) events in this synthetic model.
    M : GeoModel
        Geomodel of the synthetic model
    """
    G = grid( shape, step=(1, 1), center=(shape[0]/2,shape[1]/2) ) 
    xy = G.coords()

    s0 = strati('s0', C=ACF( 'f0', input_dim=2, gradient= (0.00001,1), curve=(-0.00005,0), origin = (1000,500) ) )
    M = GeoModel([s0], grid=G)

    s = s0.predict(xy)
    C = sample( xy, s, shape, pv='rgb', **kwargs )

    return C, M

def lehmann( shape=(1500,1000), **kwargs ):
    """
    Return a synthetic model with a folded basement.

    Parameters
    ----------
    shape : tuple
        The width and height of the generated data. 
    
    Keywords
    ---------
        All keywords are passed to `curlew.data.sample(...)`

    Returns
    --------
    xy : np.ndarray
        A list of (N,d) xy points (from a grid with the specified `shape`)
    s : np.ndarray,
        An array of shape (N,2) containing the scalar values and ID of the "event" that caused them.
    C : list, 
        A list of constraints for each of the (two) events in this synthetic model.
    M : GeoModel
        Geomodel of the synthetic model
    """
    G = grid( shape, step=(1,1), center=(shape[0]/2,shape[1]/2) ) 
    xy = G.coords()

    s0 = strati('s0', C=ASF('f0', input_dim=2) ) # Folds
    
    M = GeoModel([s0], grid=G )
    s = M.predict(xy)
    C = sample(xy, s, shape, pv='rgb', **kwargs)
    
    return C, M

def hutton( shape=(1500,1000), **kwargs ):
    """
    Return a synthetic model with a folded basement cut by an unconformity.

    Parameters
    ----------
    shape : tuple
        The width and height of the generated data. 
    
    Keywords
    ---------
        All keywords are passed to `curlew.data.sample(...)`

    Returns
    --------
    C : list, 
        A list of constraints for each of the (two) events in this synthetic model.
    M : GeoModel
        Geomodel of the synthetic model
    """
    G = grid( shape, step=(1,1), center=(shape[0]/2,shape[1]/2) ) 
    xy = G.coords()

    s0 = strati('s0', C=ASF( 'f0', input_dim=2 ) ) # Folds
    s1 = strati('s1', C=ACF( 'f1', input_dim=2, gradient=np.array([0.1,0.9]), origin=(1000,500), curve=(-0.00002,0) ), base=0) # uncorformity surface

    M  = GeoModel( [s0,s1], grid=G )
    s = M.predict(xy)
    C = sample( xy, s, shape, pv='rgb', **kwargs )
    C=[C[1],C[0],C[2]]

    return C, M


def playfair( shape=(1500,1000), width=50, **kwargs ):

    """
    Return a synthetic model with a layer-cake stratigraphy cut
    by a dyke.

    Parameters
    ----------
    shape : tuple
        The width and height of the generated data. 
    width : float
        The half-width of the added dyke.

    Keywords
    ---------
        All keywords are passed to `curlew.data.sample(...)`
    
    Returns
    --------
    C : list, 
        A list of constraints for each of the (two) events in this synthetic model.
    M : GeoModel
        Geomodel of the synthetic model
    """
    G = grid( shape, step=(1,1), center=(shape[0]/2,shape[1]/2) ) 
    xy = G.coords()

    s0 = strati('s0', C=ACF( 'f0', input_dim=2, curve=(-0.00005,0), origin=(1000,500) ) )
    s1 = sheet( 's1', 
           C=ALF( 'f1', input_dim=2, origin = [1000,500], gradient=(0.5,0.5) ), contact=(-width,width) )
    
    M = GeoModel( [s0, s1], grid=G )
    s = M.predict(xy)
    
    kwargs['pval'] = kwargs.get('pval', 1.0) # change default to sample all value constraints
    C = sample( xy, M.predict(xy), shape, pv='rgb', **kwargs )
    C1 = sample( xy, s1.predict(xy), shape, pv='rgb', breaks=[-width,width], **kwargs)

    return [C[1], C1[0], C[-1]], M

def michell( shape=(1500,1000), offset=100, **kwargs ):

    """
    Return a synthetic model with a slightly curved layer-cake stratigraphy cut
    by a thrust fault.

    Parameters
    ----------
    shape : tuple
        The width and height of the generated data. 
    offset : tuple
        The offset of the fault.

    Keywords
    ---------
        All keywords are passed to `curlew.data.sample(...)`
    
    Returns
    --------
    C : list, 
        A list of constraints for each of the (two) events in this synthetic model.
    M : GeoModel
        Geomodel of the synthetic model
    """
    G = grid( shape, step=(1,1), center=(shape[0]/2,shape[1]/2) ) 
    xy = G.coords()

    s0 = strati('s0', C=ACF( 'f0', input_dim=2, curve=(-0.00005,0), origin=(1000,500)  ) )
    s1 = fault( 's1', 
           C=ALF( 'f1', input_dim=2, origin=(1000,500), gradient=(0.5,0.5)  ),
           offset=offset, sigma1 = [-1,0] )

    M = GeoModel( [s0,s1], grid=G )
    s = M.predict(xy)
    C = sample( xy, s, shape, pv='rgb', **kwargs ) # sample unit and bedding constraints
    kwargs['pval'] = kwargs.get('pval', 1.0) # change default to sample all value constraints
    if 'breaks' in kwargs:
        del kwargs['breaks']
    Cf = sample( xy, s1.predict(xy), shape, pv='rgb', breaks=[0.5], **kwargs  )

    C = [C[0], Cf[0], C[-1]] # get stratigraphy, fault and property constraints

    mask = (C[0].gv[:,0] > - 0.68 ) # mask incorrect bebbing constraints near the fault
    C[0].gp = C[0].gp[mask] # mask the gp constraints
    C[0].gv = C[0].gv[mask] # mask the gv constraints
    C[0].gop = C[0].gop[mask] # mask the gop constraints
    C[0].gov = C[0].gov[mask] # mask the gov constraints

    C[1].vv = C[1].vv * 0 # ensure value constraints are exactly zero (fault surface = 0)
    
    return C, M # return

def anderson( shape=(1500,1000), offset1=225, offset2=250, **kwargs ):

    """
    Return a synthetic model with a slightly curved layer-cake stratigraphy cut
    by two intersecting normal faults.

    Parameters
    ----------
    shape : tuple
        The width and height of the generated data. 
    offset1 : tuple
        The offset of the first (older) normal fault.
    offet2 : tuple
        The offset of the second (younger) normal fault.
    
    Keywords
    ---------
        All keywords are passed to `curlew.data.sample(...)`
    
    Returns
    --------
    xy : np.ndarray
        A list of (N,d) xy points (from a grid with the specified `shape`)
    s : np.ndarray,
        An array of shape (N,2) containing the scalar values and ID of the "event" that caused them.
    C : list, 
        A list of constraints for each of the (two) events in this synthetic model.
    M : GeoModel
        Geomodel of the synthetic model
    """
    G = grid( shape, step=(1,1), center=(shape[0]/2,shape[1]/2) ) 
    xy = G.coords()

    s0 = strati('s0', C=ACF( 'f0', input_dim=2, curve=(-0.00005,0), origin=(1000,500)  ) )
    s1 = fault( 's1', 
           C=ALF( 'f1', input_dim=2, origin=(950,550), gradient=(np.cos( np.deg2rad(35) ), np.sin( np.deg2rad(35) ))  ),
           offset=offset1 ) 
    s2 = fault( 's2', 
           C=ALF( 'f2', input_dim=2, origin=(1050,500), gradient=(-np.cos( np.deg2rad(35) ), np.sin( np.deg2rad(35) ))  ),
           offset=offset2 )
    
    M = GeoModel( [s0, s1, s2], grid=G )
    s = M.predict(xy)

    C = sample( xy, s, shape, pv='rgb', **kwargs )
    if 'breaks' in kwargs:
        del kwargs['breaks']
    if 'pv' in kwargs:
        del kwargs['pv']
    kwargs['pval'] = kwargs.get('pval', 1.0) # change default to sample all value constraints
    Cf1 = sample( xy, s1.predict(xy), shape, pv='rgb', breaks=[0.5],xstep=400, **kwargs )
    Cf2 = sample( xy, s2.predict(xy), shape, pv='rgb', breaks=[0.5], **kwargs )

    C = [C[0], Cf1[0], Cf2[0], C[-1]] # combine stratigraphy, fault and property constraints

    mask = (C[0].gv[:,1] >  0.9) & (C[0].gv[:,1] < 1.1) # mask incorrect bebbing constraints near the faults
    C[0].gp = C[0].gp[mask] # mask the gp constraints
    C[0].gv = C[0].gv[mask] # mask the gv constraints
    C[0].gop = C[0].gop[mask] # mask the gop constraints
    C[0].gov = C[0].gov[mask] # mask the gov constraints

    C[1].vv = C[1].vv * 0 # ensure value constraints are exactly zero (fault surface = 0)
    C[2].vv = C[2].vv * 0 # ensure value constraints are exactly zero (fault surface = 0)

    return C, M # return