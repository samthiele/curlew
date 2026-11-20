"""
Generate synthetic datasets and models for testing purposes.
"""
import itertools
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

import curlew
from curlew.core import CSet
from curlew.geology.geomodel import GeoModel
from curlew.geometry import Grid 
from curlew.visualise import colour
from curlew.fields.analytical import LinearField, ListricField, PeriodicField, QuadraticField, EllipsoidalField
from curlew.geology import strati, fold, fault, domainBoundary, sheet, FaultOffset
from curlew import _tensor
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
    Return bound constraint sets from a synthetic ``curlew.geology.geomodel.GeoModel`` for use when fitting neural fields.

    Parameters
    ----------
    M : curlew.geology.geomodel.GeoModel
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
    M : curlew.geology.geomodel.GeoModel
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

    sf = G.grid.reshape(curlew._numpy(G.scalar))
    sid = G.grid.reshape(curlew._numpy(G.structureID).astype(int))
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
        All keywords are passed to `curlew.synthetic.sample(...)`
    
    Returns
    --------
    M : curlew.geology.geomodel.GeoModel
        Geomodel of the synthetic model with sampled constraints bound to events.
    """
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)

    s0 = strati('s0', C=QuadraticField( 'f0', 
                                       input_dim=ndim, 
                                       gradient=_g2((0.00001, 1), ndim), 
                                       curve=_g2((-0.00005, 0), ndim), 
                                       origin=_p2((1000, 500), ndim) ) )

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
        All keywords are passed to `curlew.synthetic.sample(...)`

    Returns
    --------
    M : curlew.geology.geomodel.GeoModel
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
        All keywords are passed to `curlew.synthetic.sample(...)`

    Returns
    --------
    M : curlew.geology.geomodel.GeoModel
        Geomodel of the synthetic model with sampled constraints bound to events.
    """
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)

    s0 = strati('s0', C=QuadraticField( 'f1', input_dim=ndim, gradient=_g2((0, 1), ndim), 
                                       origin=_p2((0, 0), ndim), curve=_g2((-0.00002, 0), ndim) ))
    d1 = fold('d1', origin=_p2((0, 0), ndim),
                    extension=_g2((0, 1), ndim),
                    compression=_g2((1, 0.3), ndim),
                    wavelength=2000,
                    amplitude=250, sharpness=0.7)
    s1 = strati('s1', C=QuadraticField( 'f1', input_dim=ndim, gradient=_g2((0.1, 0.9), ndim), 
                                       origin=_p2((1000, 500), ndim), curve=_g2((-0.00002, 0), ndim) ), base=0)

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
        All keywords are passed to `curlew.synthetic.sample(...)`
    
    Returns
    --------
    M : curlew.geology.geomodel.GeoModel
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
        All keywords are passed to `curlew.synthetic.sample(...)`
    
    Returns
    --------
    M : curlew.geology.geomodel.GeoModel
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
        All keywords are passed to `curlew.synthetic.sample(...)`
    
    Returns
    --------
    M : curlew.geology.geomodel.GeoModel
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
        All keywords are passed to `curlew.synthetic.sample(...)`
    
    Returns
    --------
    M : curlew.geology.geomodel.GeoModel
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
    M : curlew.geology.geomodel.GeoModel
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
    
    f2 = fault('f2', type=ListricField,
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

def goguel(shape=None, zstep=300, dip=60.0, offset=0.2, n_faults=3, **kwargs):
    """
    Return a synthetic model with horizontal stratigraphy and several finite faults.

    Each fault has planar geometry (shared implicit field), constant slip scaled by a
    cosine-tapered ellipsoidal patch (finite fault extent).

    Parameters
    ----------
    shape : tuple, optional
        The width and height of the generated data. 
    zstep : float
        Vertical spacing between fault seed positions.
    dip : float
        Fault dip in degrees (from horizontal).
    offset : float
        Slip magnitude. Values in ``(0, 1)`` are interpreted as a fraction of the
        ellipsoid major-axis radius; values ``>= 1`` are absolute slip (same units as extent).
    n_faults : int
        Number of finite faults (default 3).

    Keywords
    --------
        Passed to :func:`curlew.synthetic.sample`.

    Returns
    -------
    M : curlew.geology.geomodel.GeoModel.
    """
    # Grid, extent, and model centre (fault/strat seeds are placed relative to these).
    G = _make_grid(shape)
    extent = _physical_extent(shape)
    ndim = _model_ndim(shape)
    vert = 1 if ndim == 2 else 2
    origin = np.asarray(G.center, dtype=float)

    # Ellipsoid major axis and fault spacing scale with model height; slip from offset arg.
    patch_radius = 2.5 * extent[vert] / (n_faults + 1)
    xstep = 1.5 * extent[0] / (n_faults + 1)
    if isinstance(offset, (int, np.integer)) or (isinstance(offset, float) and offset >= 1):
        slip_mag = float(offset)
    else:
        slip_mag = offset * patch_radius
    slip = [slip_mag] * n_faults

    # Stagger fault centres in x and z, then re-centre the cluster on the grid midpoint.
    cx = np.array([xstep * (i + 1) for i in range(n_faults)])
    cz = np.array([extent[vert] - zstep * (i + 1) for i in range(n_faults)])
    cx = cx - np.mean(cx) + origin[0]
    cz = cz - np.mean(cz) + origin[vert]

    # Layer-cake basement: horizontal scalar field and regularly spaced isosurface seeds.
    s0 = strati(
        "s0",
        type=LinearField,
        origin=_g2(origin, ndim),
        gradient=_g2(np.array([0.0, 1.0]), ndim),
    )
    for seed, name in _isosurface_seeds(extent, ndim, 8):
        s0.addIsosurface(name, seed=seed)

    # One fault GeoEvent: shared plane dipping toward +x; vertical shortening gives normal (top-down) slip.
    fault_grad = np.array([np.sin(np.deg2rad(dip)), np.cos(np.deg2rad(dip))])
    contacts = [f"ff{i}" for i in range(n_faults)]
    modifiers = [f"fe{i}" for i in range(n_faults)]
    s1 = fault(
        "s1",
        type=LinearField,
        origin=_g2(origin, ndim),
        gradient=_g2(fault_grad, ndim),
        shortening=_g2((0, -1), ndim),
        offset=slip,
        contact=contacts,
        width=0,
        modifier=modifiers,
        n_steps=1,
    )

    # Ellipse frame: tangent along strike, normal across the fault plane; axes set patch size.
    fn = _g2(s1["s1"].grad.detach().cpu().numpy(), ndim)
    ft = _g2(np.array([-fn[1], fn[0]]), ndim)
    axes = _g2(np.array([patch_radius, 0.75 * patch_radius]), ndim)

    # Per fault: isosurface seed on the shared plane + cosine ellipsoid that tapers slip to zero.
    for i, (_x, _z) in enumerate(zip(cx, cz)):
        seed = _g2(np.array([_x, _z]), ndim)
        s1.addIsosurface(
            f"ff{i}",
            seed=seed.copy(),
            field="s1",
        )
        if ndim == 2:
            directions = np.array([ft, fn])
        else:
            directions = np.stack([ft, np.cross(ft, fn), fn])
        s1.addField(
            f"fe{i}",
            field=EllipsoidalField(
                f"fe{i}",
                input_dim=ndim,
                origin=seed.copy(),
                axes=axes,
                directions=directions,
                decay="cosine",
            ),
        )

    # Assemble model and sample constraints (bedding on s0, fault plane on s1, global property).
    M = GeoModel([s0, s1], grid=G, name="goguel")
    s = M.predict(G)

    Cs = sample(s, M, pv="rgb", bind=False, **kwargs)
    kw_fault = dict(kwargs)
    kw_fault.pop("breaks", None)
    kw_fault["pval"] = kw_fault.get("pval", 1.0)
    fault_breaks = [s1.getIsovalue(f"ff{i}") for i in range(n_faults)]
    Cf = sample(
        s1.predict(G),
        M,
        pv="rgb",
        breaks=fault_breaks,
        bind=False,
        **kw_fault,
    )

    ordered = _csets_in_order(Cs)
    strat = ordered[0]
    prop = ordered[-1] if "property" in Cs else None
    fault_c = _csets_in_order(Cf)[0]

    # Keep near-horizontal bedding orientations only (s0 gradient is vertical).
    gv = strat.gv
    if gv is not None:
        mask = (gv[:, vert] > 0.9) & (gv[:, vert] < 1.1)
        strat.gp = strat.gp[mask]
        strat.gv = strat.gv[mask]
        strat.gop = strat.gop[mask]
        strat.gov = strat.gov[mask]

    if fault_c.vv is not None:
        fault_c.vv = fault_c.vv * 0

    M["s0"].field.bind(strat)
    M["s1"]["s1"].bind(fault_c)
    if prop is not None:
        M.bind(prop)

    return M


# ── Periclinal recumbent fold (GemPy-bubble bait) ─────────────────────────────


class _PericlinalVortex(nn.Module):
    """
    Divergence-free vortex in the (x, z) plane, amplitude dying along y.

    ``v = A(y) B exp(-r²) (-p_z/s_z², 0, p_x/s_x²)`` with
    ``A = exp(-(y-y_c)²/s_y²)``. Each (x, z) slice is a Gaussian vortex;
    ``A(y)`` makes the recumbent core periclinal. ``div v = 0`` identically.
    """

    def __init__(self, centre, sigma, B):
        super().__init__()
        self.dim = 3
        c = torch.as_tensor(centre, dtype=curlew.dtype, device=curlew.device)
        s = torch.as_tensor(sigma, dtype=curlew.dtype, device=curlew.device)
        self.register_buffer("centre", c.reshape(3))
        self.register_buffer("sigma", s.reshape(3))
        self.B = float(B)

    def _envelopes(self, x):
        p = x - self.centre
        sx, sy, sz = self.sigma
        A = torch.exp(-(p[:, 1] / sy).square())
        E = torch.exp(-(p[:, 0] / sx).square() - (p[:, 2] / sz).square())
        return p, A, E

    def forward(self, x, w=None, t=None):
        del w, t
        p, A, E = self._envelopes(x)
        sx, _, sz = self.sigma
        amp = self.B * A * E
        v = torch.zeros_like(x)
        v[:, 0] = -amp * p[:, 2] / sz.square()
        v[:, 2] = amp * p[:, 0] / sx.square()
        return v

    def forward_and_jacobian(self, x, w=None, t=None):
        v = self.forward(x, w, t)
        p, A, E = self._envelopes(x)
        sx, sy, sz = self.sigma
        px, py, pz = p[:, 0], p[:, 1], p[:, 2]
        sx2, sy2, sz2 = sx.square(), sy.square(), sz.square()
        Ex = -2.0 * px / sx2
        Ez = -2.0 * pz / sz2
        Ay = A * (-2.0 * py / sy2)
        n = x.shape[0]
        J = v.new_zeros(n, 3, 3)
        # vx = -B A E pz / sz²
        J[:, 0, 0] = -self.B * A * E * Ex * pz / sz2
        J[:, 0, 1] = -self.B * Ay * E * pz / sz2
        J[:, 0, 2] = -self.B * A * ((E * Ez) * pz + E) / sz2
        # vz =  B A E px / sx²
        J[:, 2, 0] = self.B * A * ((E * Ex) * px + E) / sx2
        J[:, 2, 1] = self.B * Ay * E * px / sx2
        J[:, 2, 2] = self.B * A * E * Ez * px / sx2
        return v, J


class _VortexSum(nn.Module):
    """Sum of :class:`_PericlinalVortex` velocities (still divergence-free)."""

    def __init__(self, vortices):
        super().__init__()
        self.dim = 3
        self.vortices = nn.ModuleList(vortices)

    def forward(self, x, w=None, t=None):
        return sum(v(x, w, t) for v in self.vortices)

    def forward_and_jacobian(self, x, w=None, t=None):
        vs, Js = zip(*(v.forward_and_jacobian(x, w, t) for v in self.vortices))
        return sum(vs), sum(Js)


def _z_crossings(z, s, h):
    """Linear interpolated z where column ``s(z)`` crosses ``h``."""
    out = []
    ds = s[1:] - s[:-1]
    hit = (s[:-1] - h) * (s[1:] - h) <= 0.0
    hit &= ds != 0.0
    for k in np.nonzero(hit)[0]:
        t = (h - s[k]) / ds[k]
        if 0.0 <= t <= 1.0:
            out.append(float(z[k] + t * (z[k + 1] - z[k])))
    return out


def _batch_inverse_map(flow, pts, *, jac=False, batch=8192):
    chunks_x, chunks_j = [], []
    x = _tensor(pts, dev=curlew.device, dt=curlew.dtype)
    for i in range(0, len(x), batch):
        sl = x[i : i + batch]
        if jac:
            xi, ji = flow.inverse_map_with_jacobian(sl)
            chunks_x.append(xi)
            chunks_j.append(ji)
        else:
            chunks_x.append(flow.inverse_map(sl))
    x0 = torch.cat(chunks_x, dim=0)
    if not jac:
        return x0
    return x0, torch.cat(chunks_j, dim=0)


def recumbent_mushroom(
    shape=None,
    *,
    n_steps=24,
    overturn=1.15 * np.pi,
    sigma=None,
    centre=None,
    n_horizons=4,
    n_sections=5,
    x_stride=8,
    core_stride=1,
    younging_cut=0.3,
    seed=0,
):
    """
    Sparse 3-D periclinal recumbent fold.

    A volume-preserving vortex in ``(x, z)``, dying along strike. Every
    contact is the image of a plane. The central section is overturned; along
    strike the same horizon is an open fold. Sampling keeps the limbs and
    drops the hinge (``younging_cut``), which is where a scalar interpolator
    closes an isosurface.

    Parameters
    ----------
    shape : tuple, optional
        Voxel counts ``(nx, ny, nz)``. Physical extent is :data:`EXTENT_3D`.
    overturn : float or sequence of float
        Rotation (radians) of the vortex at the strike centre. Default is a
        little over π, enough to invert the lower limb.
    centre, sigma : array-like, optional
        Centre and widths (x, y, z) of the vortex. Pass ``(n, 3)`` arrays (and
        ``n`` overturns, or one for all) for ``n`` vortices: their velocities
        add, so the flow stays volume-preserving and every contact is still
        the image of a plane. Each vortex makes its own recumbent fold.
    younging_cut : float
        Keep only samples with ``|n · e_z| >= younging_cut`` (drop the hinge).
    n_sections : int
        Strike sections to sample, including the recumbent core and lower-
        amplitude flanks.
    x_stride, core_stride : int
        Sample every ``x_stride``-th column in the far field and every
        ``core_stride``-th column inside ``2 σ_x`` of the vortex, so the
        overturned limb is not starved relative to the upright far field.

    Returns
    -------
    types.SimpleNamespace
        ``grid``, ``flow``, ``phi`` (restored depth on the grid), ``C``
        (:class:`~curlew.core.CSet` for :func:`~curlew.geology.restore`),
        ``interfaces`` / ``orientations`` (GemPy-style records), ``horizons``
        (restored-depth isovalues, shallow → deep), ``centre``.
    """
    from curlew.kinematics.flow import FlowOffset

    if curlew.default_dim != 3:
        raise ValueError(
            "recumbent_mushroom is 3-D; set curlew.default_dim = 3 first"
        )
    if shape is None:
        shape = (64, 48, 40)
    G = _make_grid(shape)
    extent = np.asarray(_physical_extent(shape), dtype=float)
    c = np.asarray(centre if centre is not None else extent * 0.5, dtype=float)
    # slightly below mid-depth so the mushroom sits in the section
    if centre is None:
        c[2] = extent[2] * 0.45
    if sigma is None:
        sigma = np.array(
            [0.18 * extent[0], 0.22 * extent[1], 0.22 * extent[2]], dtype=float
        )
    sigma = np.asarray(sigma, dtype=float)
    centres = np.atleast_2d(c)
    n_vortices = len(centres)
    sigmas = np.broadcast_to(np.atleast_2d(sigma), (n_vortices, 3)).astype(float)
    overturns = np.broadcast_to(np.atleast_1d(np.asarray(overturn, dtype=float)), (n_vortices,))
    # peak angular rate B / (sx sz) ; for sx = sz this is B / σ²
    vortices = [
        _PericlinalVortex(ci, si, float(oi) * float(si[0] * si[2]))
        for ci, si, oi in zip(centres, sigmas, overturns)
    ]
    vel = vortices[0] if n_vortices == 1 else _VortexSum(vortices)
    flow = FlowOffset(vel, n_steps=n_steps, integrator="rk4")

    pts = G.coords()
    x0 = _batch_inverse_map(flow, pts, jac=False)
    phi = G.reshape(curlew._numpy(x0[:, 2]))
    xyz = G.reshape(pts)
    z_axis = xyz[0, 0, :, 2]

    # pick restored-depth isovalues that actually overturn on the core sections
    j_cores = sorted({int(np.argmin(np.abs(xyz[0, :, 0, 1] - ci[1]))) for ci in centres})
    lo, hi = float(np.nanmin(phi)), float(np.nanmax(phi))
    probe = np.linspace(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo), 24)
    multi = []
    for h in probe:
        n_multi = 0
        for j_c in j_cores:
            for i in range(phi.shape[0]):
                if len(_z_crossings(z_axis, phi[i, j_c, :], h)) >= 2:
                    n_multi += 1
        multi.append(n_multi)
    h_core = float(probe[int(np.argmax(multi))])
    offsets = np.linspace(-0.12 * (hi - lo), 0.12 * (hi - lo), n_horizons)
    horizons = np.clip(h_core + offsets, lo + 1e-3, hi - 1e-3)
    horizons = np.unique(np.round(horizons, 5))[::-1]  # shallow (high φ) first

    y_targets = np.linspace(
        float(np.min(centres[:, 1] - 1.15 * sigmas[:, 1])),
        float(np.max(centres[:, 1] + 1.15 * sigmas[:, 1])),
        n_sections,
    )
    y_idx = [
        int(np.argmin(np.abs(xyz[0, :, 0, 1] - y))) for y in y_targets
    ]

    names = [f"H{k + 1}" for k in range(len(horizons))]
    contacts = {n: [] for n in names}
    x_axis = xyz[:, 0, 0, 0]
    in_core = np.any(
        np.abs(x_axis[:, None] - centres[:, 0]) <= 2.0 * sigmas[:, 0], axis=1
    )
    i_far = np.arange(0, phi.shape[0], max(1, int(x_stride)))
    i_core = np.where(in_core)[0][:: max(1, int(core_stride))]
    i_samples = np.unique(np.concatenate([i_far, i_core]))
    for name, h in zip(names, horizons):
        for j in y_idx:
            for i in i_samples:
                for zc in _z_crossings(z_axis, phi[i, j, :], h):
                    # bilinear-ish: nearest x,y, interpolated z
                    x_pt = float(xyz[i, j, 0, 0])
                    y_pt = float(xyz[i, j, 0, 1])
                    contacts[name].append((x_pt, y_pt, zc))

    all_pts = []
    all_form = []
    for name, rows in contacts.items():
        if not rows:
            continue
        p = np.asarray(rows, dtype=float)
        all_pts.append(p)
        all_form.extend([name] * len(p))
    if not all_pts:
        raise RuntimeError(
            "recumbent_mushroom found no horizon crossings; increase overturn "
            "or refine `shape`"
        )
    contact_xyz = np.vstack(all_pts)
    _, J = _batch_inverse_map(flow, contact_xyz, jac=True)
    g = torch.bmm(
        J.transpose(-1, -2),
        torch.tensor(
            [0.0, 0.0, 1.0], dtype=J.dtype, device=J.device
        ).expand(J.shape[0], 3).unsqueeze(-1),
    ).squeeze(-1)
    g = torch.nn.functional.normalize(g, dim=-1)
    younging = curlew._numpy(g)
    nz = younging[:, 2]
    keep = np.abs(nz) >= float(younging_cut)

    contact_xyz = contact_xyz[keep]
    younging = younging[keep]
    forms = np.array(all_form)[keep]

    # jitter nothing; drop exact duplicates from stride/section coincidence
    key = np.round(contact_xyz, 4)
    _, uniq = np.unique(key, axis=0, return_index=True)
    uniq = np.sort(uniq)
    contact_xyz = contact_xyz[uniq]
    younging = younging[uniq]
    forms = forms[uniq]

    interfaces = np.rec.fromarrays(
        [contact_xyz[:, 0], contact_xyz[:, 1], contact_xyz[:, 2], forms],
        names=("X", "Y", "Z", "formation"),
    )
    dip = np.degrees(np.arccos(np.clip(np.abs(younging[:, 2]), 0.0, 1.0)))
    azimuth = (np.degrees(np.arctan2(younging[:, 0], younging[:, 1])) + 360.0) % 360.0
    polarity = np.where(younging[:, 2] >= 0.0, 1, -1).astype(int)
    orientations = np.rec.fromarrays(
        [
            contact_xyz[:, 0],
            contact_xyz[:, 1],
            contact_xyz[:, 2],
            younging[:, 0],
            younging[:, 1],
            younging[:, 2],
            dip,
            azimuth,
            polarity,
            forms,
        ],
        names=(
            "X", "Y", "Z", "G_x", "G_y", "G_z",
            "dip", "azimuth", "polarity", "formation",
        ),
    )

    eq = []
    gp, gv = [], []
    for name in names:
        m = forms == name
        if m.sum() < 2:
            continue
        eq.append(contact_xyz[m])
        gp.append(contact_xyz[m])
        gv.append(younging[m])
    C = CSet(
        eq=eq or None,
        gp=np.vstack(gp) if gp else None,
        gv=np.vstack(gv) if gv else None,
        gop=np.vstack(gp) if gp else None,
        gov=np.vstack(gv) if gv else None,
    )
    return SimpleNamespace(
        grid=G,
        flow=flow,
        phi=phi,
        C=C,
        interfaces=interfaces,
        orientations=orientations,
        horizons=horizons,
        names=names,
        centre=c,
        sigma=sigma,
        overturn=float(overturns[0]) if n_vortices == 1 else overturns.copy(),
        centres=centres.copy(),
        sigmas=sigmas.copy(),
        overturns=overturns.copy(),
        n_inverted=int((polarity < 0).sum()),
        n_upright=int((polarity > 0).sum()),
    )


def write_gempy_tables(dataset, directory):
    """
    Write GemPy-style ``interfaces.csv`` and ``orientations.csv``.

    Orientations carry both pole vectors ``G_x, G_y, G_z`` (younging of the
    restored-depth potential) and ``dip, azimuth, polarity``.
    """
    import csv
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    iface = dataset.interfaces
    with open(directory / "interfaces.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y", "Z", "formation"])
        for row in iface:
            w.writerow([row["X"], row["Y"], row["Z"], row["formation"]])
    ori = dataset.orientations
    with open(directory / "orientations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["X", "Y", "Z", "G_x", "G_y", "G_z", "dip", "azimuth", "polarity", "formation"]
        )
        for row in ori:
            w.writerow(
                [
                    row["X"], row["Y"], row["Z"],
                    row["G_x"], row["G_y"], row["G_z"],
                    row["dip"], row["azimuth"], int(row["polarity"]),
                    row["formation"],
                ]
            )
    return directory / "interfaces.csv", directory / "orientations.csv"


def count_interior_extrema(phi):
    """
    Count fully-connected interior local maxima and minima of a 2-D or 3-D scalar.

    A pixel/voxel is an extremum only if it is strictly above (below) every
    neighbour in its full :math:`3^d` neighbourhood (8-connected in 2-D,
    26-connected in 3-D), not merely the face-adjacent ones. A bubble of an
    implicit potential is an interior extremum: a closed isosurface (2-D:
    isoline) around it. The true restored-depth field of a diffeomorphism of
    a layer cake should return ``(0, 0)``.
    """
    a = np.asarray(phi)
    if a.ndim not in (2, 3):
        raise ValueError(f"phi must be a 2-D or 3-D array, got ndim={a.ndim}")
    d = a.ndim
    shape = a.shape
    core = a[(slice(1, -1),) * d]
    is_max = np.ones(core.shape, dtype=bool)
    is_min = np.ones(core.shape, dtype=bool)
    for offset in itertools.product((-1, 0, 1), repeat=d):
        if all(o == 0 for o in offset):
            continue
        n = a[tuple(
            slice(1 + o, s - 1 + o) for o, s in zip(offset, shape)
        )]
        is_max &= core > n
        is_min &= core < n
    return int(is_max.sum()), int(is_min.sum())