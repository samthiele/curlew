"""
Generate synthetic datasets and models for testing purposes.
"""
import numpy as np
import curlew
from curlew.core import CSet
from curlew.geology.geomodel import GeoModel
from curlew.geometry import Grid 
from curlew.visualise import colour
from curlew.fields.analytical import LinearField, ListricField, PeriodicField, QuadraticField
from curlew.geology import strati, fold, fault, domainBoundary, sheet
from curlew.geology.geoevent import GeoEvent

EXTENT_2D = (1500, 1000)
EXTENT_3D = (1500, 1500, 1000)
VOXELS_3D = (150, 150, 100)

# helper functions for parsing main function args
def _default_shape():
    """Helper function to get the default grid shape: physical dims in 2D, voxel counts in 3D."""
    if curlew.default_dim == 2:
        return EXTENT_2D
    return VOXELS_3D

def _model_ndim(shape=None):
    """Helper function to get the dimension of model to create"""
    if shape is not None:
        return len(shape)
    return curlew.default_dim

def _physical_extent(shape=None):
    """Get extent based on shape"""
    shape = shape or _default_shape()
    return shape if len(shape) == 2 else EXTENT_3D

def _make_grid(shape=None, center=None):
    """Build grid for synthetic models"""
    shape = shape or _default_shape()
    if len(shape) == 2:
        dims = shape
        step = (1, 1)
        center = center or (dims[0] / 2, dims[1] / 2)
    else:
        dims = EXTENT_3D
        step = tuple(e / n for e, n in zip(EXTENT_3D, shape))
        center = center or tuple(d / 2 for d in dims)
    return Grid(dims, step=step, center=center)

def _p2(pt, ndim, y=None):
    """Promote a 2D (x, z) point to 3D (x, y, z); geology is invariant in y."""
    pt = np.asarray(pt, dtype=float).ravel()
    if ndim == 2:
        return pt[:2]
    if len(pt) >= 3:
        return pt[:3]
    y = EXTENT_3D[1] / 2 if y is None else y
    return np.array([pt[0], y, pt[1]])

def _g2(vec, ndim):
    """Promote a 2D (x, z) vector to 3D with zero y component."""
    vec = np.asarray(vec, dtype=float).ravel()
    if ndim == 2:
        return vec[:2]
    if len(vec) >= 3:
        return vec[:3]
    return np.array([vec[0], 0.0, vec[1]])

def _isosurface_seeds(extent, ndim, n=5):
    # define seed positions for isosurfaces in the models
    if ndim == 2:
        for v in np.linspace(0, extent[1], n):
            yield np.array([extent[0] / 2, v]), f"i{int(v)}"
    else:
        xc, yc = extent[0] / 2, extent[1] / 2
        for v in np.linspace(0, extent[2], n):
            yield np.array([xc, yc, v]), f"i{int(v)}"
            
def _breaks_array(sf, breaks):
    """Scalar break levels used by ``colour()`` for the same ``sf``."""
    if isinstance(breaks, int):
        return np.hstack([np.min(sf), np.linspace(np.min(sf), np.max(sf), breaks)])
    return np.hstack([np.min(sf), np.asarray(breaks, dtype=float), np.max(sf)])


def _finalize_constraints(constraints, M, bind):
    """Build CSet objects from sampled constraint dicts and optionally bind to the model."""
    for i in constraints.keys():
        for k, v in constraints[i].items():
            if k == "eq":
                constraints[i][k] = v if len(v) > 0 else None
                continue
            if len(v) > 0:
                constraints[i][k] = np.array(v)
                if k in ("gv", "gov"):
                    constraints[i][k] /= np.linalg.norm(v, axis=-1)[:, None]
            else:
                constraints[i][k] = None

    out = {}
    for key, v in constraints.items():
        cset = CSet(**v)
        out[key] = cset
        if bind:
            if key == "property":
                M.bind(cset)
            else:
                ev = M[int(key)]
                if ev is not None:
                    ev.field.bind(cset)
    return out


def _csets_in_order(constraints):
    """Return CSets in sorted structure-ID order, then property (if present)."""
    keys = sorted(k for k in constraints if k != "property")
    ordered = [constraints[k] for k in keys]
    if "property" in constraints:
        ordered.append(constraints["property"])
    return ordered


def extract_constraints(M, events=None, include_property=False):
    """
    Return bound constraint sets from a synthetic ``GeoModel`` for use when fitting neural fields.

    Parameters
    ----------
    M : GeoModel
        A synthetic model returned by one of the builders in this module (after ``sample`` has run).
    events : list of str, optional
        Event names to include. Defaults to all events in ``M.events``.
    include_property : bool
        If True, include global property constraints under the key ``'property'``.

    Returns
    -------
    dict
        Constraint sets keyed by event name (and ``'property'`` when requested).
    """
    if events is None:
        events = [e.name for e in M.events]
    out = {}
    for name in events:
        c = M[name].field.C
        if c is not None:
            out[name] = c.numpy()
    if include_property and M.C is not None:
        out["property"] = M.C.numpy()
    return out

def _sample_section(sf, sid, xy, contacts, domains, gx, gy, init, xstep, pval, pv, constraints, eq_accum, breaks):
    # sample constraints along a 2D section
    ndim = xy.shape[-1]
    for x in np.arange(init, sf.shape[0], xstep):
        cc = np.argwhere(contacts[x, :])
        if len(cc) > 0:
            ys = np.atleast_1d(cc.squeeze())
            for y in ys:
                yi = int(np.asarray(y).item())
                i = int(sid[x, yi])
                if domains[x, yi]:
                    continue
                level = int(np.searchsorted(breaks, float(sf[x, yi]), side="right") - 1)
                eq_accum[i].setdefault(level, []).append(np.asarray(xy[x, yi], dtype=float))
            for y in ys[:-1] if len(ys) > 1 else ys:
                yi = int(np.asarray(y).item())
                i = int(sid[x, yi])
                if not domains[x, yi]:
                    grad = (gx[x, yi], gy[x, yi]) if ndim == 2 else (gx[x, yi], 0.0, gy[x, yi])
                    constraints[i]["gp"].append(xy[x, yi])
                    constraints[i]["gv"].append(grad)
                    constraints[i]["gop"].append(xy[x, yi])
                    constraints[i]["gov"].append(grad)
                    if np.random.rand() <= pval:
                        constraints[i]["vp"].append(xy[x, yi])
                        constraints[i]["vv"].append(sf[x, yi])

        if pv is not None:
            for y in np.arange(sf.shape[1], step=1):
                if "property" not in constraints:
                    constraints["property"] = {"pp": [], "pv": [], "vp": [], "vv": []}
                constraints["property"]["pp"].append(xy[x, y])
                constraints["property"]["pv"].append(pv[x, y])
                constraints["property"]["vp"].append(xy[x, y])
                constraints["property"]["vv"].append(sf[x, y])

def sample( G, M, pv=None, breaks=19, init=100, xstep=300, pval=0.6, cmap='tab20', seed=42, bind=True ):
    """
    Sample value, orientation, equality (contact trace), and property constraints from a scalar field and associated gradients.

    Parameters
    ----------
    G : curlew.core.Geode
        A Geode object containing the results from a GeoEvent or GeoModel.
    M : curlew.core.GeoModel
        The geomodel used to construct the geode G. 
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
    bind : bool
        If True, the sampled constraints will be added to their respective fields (and global constraints bound to the
        GeoModel).
        
    Notes
    -----
    For 3D grids, constraints are sampled on x-z sections at y = 25%, 50% and 75%
    of the model y extent (1500 m).

    Returns
    -------
    dict
        Sampled constraints keyed by structure ID (int) and ``'property'`` when present.
        Empty when ``bind=True`` (constraints are attached to ``M`` and its events).
    """

    np.random.seed(seed)

    sf = G.grid.reshape(G.scalar)
    sid = G.grid.reshape(G.structureID.astype(int))
    xy = G.grid.reshape(G.grid.coords())
    constraints = {
        int(k): {"vp": [], "vv": [], "gp": [], "gv": [], "gop": [], "gov": [], "eq": []}
        for k in np.unique(sid)
    }
    eq_accum = {int(k): {} for k in np.unique(sid)}

    if G.grid.ndim == 3:
        y_global = xy[0, :, 0, 1]
        for y_frac in (0.25, 0.5, 0.75):
            j = int(np.argmin(np.abs(y_global - G.grid.dims[1] * y_frac)))
            sf2 = sf[:, j, :]
            sid2 = sid[:, j, :]
            xy2 = xy[:, j, :, :]
            gx = np.diff(sf2, axis=0)
            gz = np.diff(sf2, axis=1)
            c = colour(sf2, breaks=breaks, cmap=cmap)
            breaks_arr = _breaks_array(sf2, breaks)
            contacts = np.sum(np.abs(np.diff(c, axis=1, append=0)), axis=-1) > 0
            domains = np.sum(
                [np.abs(np.diff(sid2, axis=1, append=0)), np.abs(np.diff(sid2, axis=0, append=0))], axis=0
            ) > 0
            pv2 = c if pv == "rgb" else pv
            _sample_section(
                sf2, sid2, xy2, contacts, domains, gx, gz, init, xstep, pval, pv2,
                constraints, eq_accum, breaks_arr,
            )
    else:
        gx = np.diff(sf, axis=0)
        gy = np.diff(sf, axis=1)
        c = colour(sf, breaks=breaks, cmap=cmap)
        breaks_arr = _breaks_array(sf, breaks)
        contacts = np.sum(np.abs(np.diff(c, axis=1, append=0)), axis=-1) > 0
        domains = np.sum(
            [np.abs(np.diff(sid, axis=1, append=0)), np.abs(np.diff(sid, axis=0, append=0))], axis=0
        ) > 0
        if pv == "rgb":
            pv = c
        _sample_section(
            sf, sid, xy, contacts, domains, gx, gy, init, xstep, pval, pv,
            constraints, eq_accum, breaks_arr,
        )

    for i, levels in eq_accum.items():
        traces = [np.array(pts) for pts in levels.values() if len(pts) >= 2]
        if traces:
            constraints[i]["eq"].extend(traces)

    out = _finalize_constraints(constraints, M, bind=bind)
    return {} if bind else out

# Geological models
# ------------------------
def steno( shape=None, **kwargs ):
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
    M : GeoModel
        Geomodel of the synthetic model with sampled constraints bound to events.
    """
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)

    s0 = strati('s0', C=QuadraticField( 'f0', input_dim=ndim, gradient=_g2((0.00001, 1), ndim), curve=_g2((-0.00005, 0), ndim), origin=_p2((1000, 500), ndim) ) )

    for seed, name in _isosurface_seeds(extent, ndim, 5):
        s0.addIsosurface(name, seed=seed)
    
    M = GeoModel([s0], grid=G, name="steno")

    s = s0.predict(G)
    sample(s, M, pv='rgb', **kwargs)

    return M

def lehmann( shape=None, **kwargs ):
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
    M : GeoModel
        Geomodel of the synthetic model with sampled constraints bound to events.
    """
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)

    if ndim == 3:
        s0 = strati('s0', C=PeriodicField('f0', input_dim=ndim, gradient=np.array([0., 0., 1.]), axialPlane=np.array([1., 0., 0.])))
    else:
        s0 = strati('s0', C=PeriodicField('f0', input_dim=ndim))
    
    M = GeoModel([s0], grid=G, name='lehmann' )

    for seed, name in _isosurface_seeds(extent, ndim, 5):
        s0.addIsosurface(name, seed=seed)
    
    s = M.predict(G)
    sample(s, M, pv='rgb', **kwargs)
    
    return M

def hutton( shape=None, **kwargs ):
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
    M : GeoModel
        Geomodel of the synthetic model with sampled constraints bound to events.
    """
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)

    s0 = strati('s0', C=QuadraticField( 'f1', input_dim=ndim, gradient=_g2((0, 1), ndim), origin=_p2((0, 0), ndim), curve=_g2((-0.00002, 0), ndim) ))
    d1 = fold('d1', origin=_p2((0, 0), ndim),
                    extension=_g2((0, 1), ndim),
                    compression=_g2((1, 0.3), ndim),
                    wavelength=2000,
                    amplitude=250, sharpness=0.7)
    s1 = strati('s1', C=QuadraticField( 'f1', input_dim=ndim, gradient=_g2((0.1, 0.9), ndim), origin=_p2((1000, 500), ndim), curve=_g2((-0.00002, 0), ndim) ), base=0)

    for seed, name in _isosurface_seeds(extent, ndim, 5):
        s0.addIsosurface(name, seed=seed)
    for seed, name in _isosurface_seeds(extent, ndim, 10):
        s1.addIsosurface(name, seed=seed)

    M  = GeoModel( [s0,d1,s1], grid=G, name='hutton' )
    s = M.predict(G)
    Cs = sample(s, M, pv='rgb', bind=False, **kwargs)
    ordered = _csets_in_order(Cs)
    M['s0'].field.bind(ordered[1])
    M['s1'].field.bind(ordered[0])
    if len(ordered) > 2:
        M.bind(ordered[2])

    return M

def playfair( shape=None, width=50, addFault=False, **kwargs ):

    """
    Return a synthetic model with a layer-cake stratigraphy cut
    by a dyke.

    Parameters
    ----------
    shape : tuple
        The width and height of the generated data. 
    width : float
        The half-width of the added dyke.
    addFault : bool
        If True, add a fault crosscutting the dykes and stratigraphy. Default is False. 

    Keywords
    ---------
        All keywords are passed to `curlew.data.sample(...)`
    
    Returns
    --------
    M : GeoModel
        Geomodel of the synthetic model with sampled constraints bound to events.
    """
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)
    o = _p2((1000, 500), ndim)

    s0 = strati('s0', C=QuadraticField( 'f0', input_dim=ndim, curve=_g2((-0.00005, 0), ndim), origin=o ) )
    s1 = sheet( 's1',
           C=LinearField( 'f1', input_dim=ndim, origin=o, gradient=_g2((0.5, 0.5), ndim) ), contact=(-width, width) )

    for seed, name in _isosurface_seeds(extent, ndim, 5):
        s0.addIsosurface(name, seed=seed)
    
    if addFault:
        s2 = fault( 's2',
           C=LinearField( 'f2', input_dim=ndim, origin=_p2((1050, 500), ndim), gradient=_g2((-np.cos( np.deg2rad(35) ), np.sin( np.deg2rad(35) )), ndim)  ),
           offset=100, shortening=_g2((0, -1), ndim) )
        M = GeoModel( [s0, s1, s2], grid=G, name='newcastle' )
    else:
        M = GeoModel( [s0, s1], grid=G, name='playfair' )
    
    kwargs['pval'] = kwargs.get('pval', 1.0) # change default to sample all value constraints
    Cs = sample(M.predict(G), M, pv='rgb', bind=False, **kwargs)
    C1 = sample(s1.predict(G), M, pv='rgb', breaks=[-width, width], bind=False, **kwargs)
    ordered = _csets_in_order(Cs)
    dyke = _csets_in_order(C1)[0]
    M['s0'].field.bind(ordered[1])
    M['s1'].field.bind(dyke)
    if len(ordered) > 2:
        M.bind(ordered[2])

    return M

def walker( shape=None, width=[60,50,40,50,40], pos=[0,100,200,400,600], addFault=True, **kwargs ):

    """
    Return a synthetic model with a layer-cake stratigraphy cut
    by several parallel dykes.

    Parameters
    ----------
    shape : tuple
        The width and height of the generated data. 
    width : float
        A list specifying the dyke widths.
    pos : float
        A list specifying the dyke positions in the x-axis. Must have the same length as width.
    addFault : bool
        If True, add a fault crosscutting the dykes and stratigraphy. Default is False.
    Keywords
    ---------
        All keywords are passed to `curlew.data.sample(...)`
    
    Returns
    --------
    M : GeoModel
        Geomodel of the synthetic model with sampled constraints bound to events.
    """
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)
    o = _p2((0, extent[1] / 2 if ndim == 2 else extent[2] / 2), ndim)
    s0 = strati('s0', C=QuadraticField( 'f0', input_dim=ndim, curve=_g2((-0.00005, 0), ndim), origin=o ) )
    contact = [(pos[i], pos[i]+width[i]) for i in range(len(pos))]
    s1 = sheet( 's1',
           C=LinearField( 'f1', input_dim=ndim, origin=o, gradient=_g2((0.5, 0.5), ndim) ), contact=contact )

    for seed, name in _isosurface_seeds(extent, ndim, 5):
        s0.addIsosurface(name, seed=seed)
    
    if addFault:
        s2 = fault( 's2',
           C=LinearField( 'f2', input_dim=ndim, origin=_p2((1050, 500), ndim), gradient=_g2((-np.cos( np.deg2rad(35) ), np.sin( np.deg2rad(35) )), ndim)  ),
           offset=100, shortening=_g2((0, -1), ndim) )
        M = GeoModel( [s0, s1, s2], grid=G, name='newcastle' )
    else:
        M = GeoModel( [s0, s1], grid=G, name='walker' )
    
    kwargs['pval'] = kwargs.get('pval', 1.0) # change default to sample all value constraints
    Cs = sample(M.predict(G), M, pv='rgb', bind=False, **kwargs)
    C1 = sample(s1.predict(G), M, pv='rgb', breaks=np.hstack(contact), bind=False, **kwargs)
    ordered = _csets_in_order(Cs)
    dyke = _csets_in_order(C1)[0]
    M['s0'].field.bind(ordered[1])
    M['s1'].field.bind(dyke)
    if len(ordered) > 2:
        M.bind(ordered[2])

    return M

def michell( shape=None, offset=100, **kwargs ):

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
    M : GeoModel
        Geomodel of the synthetic model with sampled constraints bound to events.
    """
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)
    o = _p2((1000, 500), ndim)

    s0 = strati('s0', C=QuadraticField( 'f0', input_dim=ndim, curve=_g2((-0.00005, 0), ndim), origin=o ) )
    s1 = fault( 's1',
           C=LinearField( 'f1', input_dim=ndim, origin=o, gradient=_g2((0.5, 0.5), ndim)  ),
           offset=offset, shortening=_g2((-1, 0), ndim) )

    for seed, name in _isosurface_seeds(extent, ndim, 5):
        s0.addIsosurface(name, seed=seed)

    M = GeoModel( [s0,s1], grid=G, name='michell' )
    s = M.predict(G)
    Cs = sample(s, M, pv='rgb', bind=False, **kwargs)
    kwargs['pval'] = kwargs.get('pval', 1.0)
    kw_fault = dict(kwargs)
    kw_fault.pop('breaks', None)
    Cf = sample(s1.predict(G), M, pv='rgb', breaks=[0.5], bind=False, **kw_fault)

    ordered = _csets_in_order(Cs)
    strat = ordered[0]
    fault_c = _csets_in_order(Cf)[0]
    prop = ordered[-1] if 'property' in Cs else None

    gv = strat.gv
    if gv is not None:
        mask = gv[:, 0] > -0.68
        strat.gp = strat.gp[mask]
        strat.gv = strat.gv[mask]
        strat.gop = strat.gop[mask]
        strat.gov = strat.gov[mask]

    if fault_c.vv is not None:
        fault_c.vv = fault_c.vv * 0

    M['s0'].field.bind(strat)
    M['s1'].field.bind(fault_c)
    if prop is not None:
        M.bind(prop)

    return M

def anderson( shape=None, offset1=225, offset2=250, **kwargs ):

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
    M : GeoModel
        Geomodel of the synthetic model with sampled constraints bound to events.
    """
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)
    vert = 1 if ndim == 2 else 2

    s0 = strati('s0', C=QuadraticField( 'f0', input_dim=ndim, curve=_g2((-0.00005, 0), ndim), origin=_p2((1000, 500), ndim)  ) )
    s1 = fault( 's1',
           C=LinearField( 'f1', input_dim=ndim, origin=_p2((950, 550), ndim), gradient=_g2((np.cos( np.deg2rad(35) ), np.sin( np.deg2rad(35) )), ndim)  ),
           offset=offset1, shortening=_g2((0, -1), ndim) )
    s2 = fault( 's2',
           C=LinearField( 'f2', input_dim=ndim, origin=_p2((1050, 500), ndim), gradient=_g2((-np.cos( np.deg2rad(35) ), np.sin( np.deg2rad(35) )), ndim)  ),
           offset=offset2, shortening=_g2((0, -1), ndim) )
    
    for seed, name in _isosurface_seeds(extent, ndim, 5):
        s0.addIsosurface(name, seed=seed)
        
    M = GeoModel( [s0, s1, s2], grid=G, name='anderson' )
    s = M.predict(G)

    Cs = sample(s, M, pv='rgb', bind=False, **kwargs)
    kw_fault = dict(kwargs)
    kw_fault.pop('breaks', None)
    kw_fault.pop('pv', None)
    kw_fault['pval'] = kw_fault.get('pval', 1.0)
    Cf1 = sample(s1.predict(G), M, pv='rgb', breaks=[0.5], xstep=600, bind=False, **kw_fault)
    Cf2 = sample(s2.predict(G), M, pv='rgb', breaks=[0.5], bind=False, **kw_fault)

    ordered = _csets_in_order(Cs)
    strat = ordered[0]
    prop = ordered[-1] if 'property' in Cs else None
    f1_c = _csets_in_order(Cf1)[0]
    f2_c = _csets_in_order(Cf2)[0]

    gv = strat.gv
    if gv is not None:
        mask = (gv[:, vert] > 0.9) & (gv[:, vert] < 1.1)
        strat.gp = strat.gp[mask]
        strat.gv = strat.gv[mask]
        strat.gop = strat.gop[mask]
        strat.gov = strat.gov[mask]

    for fc in (f1_c, f2_c):
        if fc.vv is not None:
            fc.vv = fc.vv * 0

    M['s0'].field.bind(strat)
    M['s1'].field.bind(f1_c)
    M['s2'].field.bind(f2_c)
    if prop is not None:
        M.bind(prop)

    return M

def seuss(shape=None, nlayers=6, **kwargs):
    """
    Return a synthetic model with layered stratigraphy, a dyke, an intrusion
    domain boundary, and two listric faults (one older, one younger), matching
    the "Seuss" GeoModel from the Building Analytical Models tutorial.

    Parameters
    ----------
    shape : tuple
        The width and height of the generated data. Default (1500, 700) matches
        the tutorial notebook.
    nlayers : int
        Number of stratigraphic layers (isosurfaces) in each package. Default 6.

    Keywords
    ---------
        All keywords are passed to `curlew.synthetic.sample(...)` when
        generating constraints.

    Returns
    --------
    M : GeoModel
        Geomodel with listric faults and domain boundaries (name='Seuss').
    """
    shape = shape or EXTENT_2D
    G = _make_grid(shape)
    dims = _physical_extent(shape)
    
    # First stratigraphic package (layer-cake)
    s0 = strati(
        "s0",
        C=LinearField("f0", input_dim=2, gradient=(0.1, 0.9)),
    )
    sy1 = np.linspace(0, dims[1], nlayers)
    sx1 = np.full_like(sy1, dims[0] / 10)
    for i, (x, y) in enumerate(zip(sx1, sy1)):
        s0.addIsosurface("S%d" % (i + 1), seed=np.array([x, y]))

    # Dyke
    s1 = sheet(
        "s1",
        C=LinearField(
            "f1",
            input_dim=2,
            origin=np.array([750.0, 300.0]),
            gradient=np.array([-0.5, 0.5]),
        ),
        contact=(-20, 20),
        aperture=2,
    )

    # Domain boundary: intrusion (constant -2) below, s0 and s1 above
    intrusion = GeoEvent("i0", field=-2, type=int) # quick and easy way to define a constant scalar field
    d1 = domainBoundary(
        "d1",
        C=QuadraticField(
            "d1f",
            input_dim=2,
            origin=np.array([400.0, 50.0]),
            gradient=np.array([0.4, 1.0]),
            curve=(0, 0.002),
        ),
        bound=0,
        lt=[intrusion],
        gt=[s0, s1],
    )

    # Older listric fault
    f1 = fault(
        "f1",
        C=ListricField(
            "f1f",
            input_dim=2,
            origin=np.array([650.0, 700.0]),
            fault_floor=0.0,
            fault_ceil=700.0,
            curvature_rate=0.006,
        ),
        width=1e-9,
        n_steps=3,
        offset=100,
        shortening=np.array([1, -1]),
    )

    # Second stratigraphic package (unconformable cover)
    s2 = strati(
        "s2",
        C=LinearField(
            "f2",
            input_dim=2,
            origin=np.array([0.0, 500.0]),
            gradient=np.array([0.0, 1.0]),
        ),
    )
    sy2 = np.linspace(350, dims[1], nlayers)
    sx2 = np.full_like(sy2, 9 * dims[0] / 10)
    for i, (x, y) in enumerate(zip(sx2, sy2)):
        s2.addIsosurface("S%d" % (i + 1), seed=np.array([x, y]))

    # Unconformity domain boundary: d1 and f1 below, s2 above
    d2 = domainBoundary(
        "d2",
        C=LinearField(
            "d2f",
            input_dim=2,
            origin=np.array([700.0, 500.0]),
            gradient=np.array([0.2, 0.9]),
        ),
        bound=0,
        lt=[d1, f1],
        gt=[s2],
    )
    
    f2 = fault('f2', type=ListricField, C=None,
           contact=0.0, 
           offset=100, # The displacement magnitude
           n_steps=3, # apply displacement in several steps, due to high curvature
           shortening=[1, -1],
           input_dim=2,
           origin=np.array([250.0, 700.0]),
           fault_ceil=700., # Defines the ceiling of the fault
           fault_floor=0., # Defines the floor of the fault
           width=1e-9,
           curvature_rate=0.006 # Defines the steepness of the decay
           
        )
    
    M = GeoModel([d2, f2], grid=G, name="Seuss")
    s = M.predict(G)
    sample(s, M, pv="rgb", **kwargs)
    return M