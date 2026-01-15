import numpy as np

def _checkCSet( C, dims=2, val=True, grad=True, ori=True, prop=True ):
    if val: # check value constraints exist and have the correct shape
        assert len(C.vp) > 0
        assert C.vp.shape[-1] == dims
        assert len(C.vp) > 0
    if grad: # check value constraints exist and have the correct shape
        assert len(C.gp) > 0
        assert C.gp.shape[-1] == dims
        assert len(C.gv) > 0
        assert C.gv.shape[-1] == dims
    if ori: # check value constraints exist and have the correct shape
        assert len(C.gop) > 0
        assert C.gop.shape[-1] == dims
        assert len(C.gov) > 0
        assert C.gov.shape[-1] == dims
    if prop:
        assert len(C.pp) > 0
        assert C.pp.shape[-1] == dims
        assert len(C.pv) > 0

def test_synthetic():
    from curlew.synthetic import steno, hutton, michell, playfair, anderson, lehmann, suess
    dims = (2001,1001)  # dimensions used for each of our 2D models
    for f in [steno, hutton, michell, playfair, anderson, lehmann]: # suess
        C, M = f(dims) # create the synthetic "hutton" dataset
        xy = M.grid.coords() # get associated grid points
        g = M.predict(xy) # evaluate model
        assert len(np.unique(g.lithoID)) > 1 # check we have more than one lithology
        d, c = M.drill( [0,0], dims, step=10) # evaluate model along diagonal drillhole
        assert len( np.unique(d.lithoID) ) > 1 # assert we intersected some contacts
        assert len(c.x) > 0 # check we have some contacts
        assert c.gradient is not None # check gradients were also computed for the contacts
            
        for _C in C[:-1]: _checkCSet(_C, prop=False) # check sampled constraints are valid
        assert len(xy) == np.prod(dims)
        assert len(g.scalar) == np.prod(dims) # check scalar field is at least the correct size...

        # check we have enough structures
        n = len(np.unique(g.structureID)) # check there are three structure IDs
        assert len(g.structureLookup) == n # check there are three structures
        for k in np.unique(g.structureID): # check structures are named
            assert k in g.structureLookup
        
        # check the evaluated scalar values match 
        for sid,n in g.structureLookup.items():
            mask = (g.structureID == sid)
            assert np.percentile( np.abs( g.scalar[mask] - g.fields[n][mask] ), 99) < 1e-6 # almost all values should match

        # check stackValues function
        gs = g.stackValues( mn=0, mx=1 )
        assert np.max(gs.scalar) == len(gs.structureLookup) # max value should be equal to number of structures