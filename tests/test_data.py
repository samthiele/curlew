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
    from curlew.data import steno, hutton, michell, playfair, anderson
    dims = (2001,1001)  # dimensions used for each of our 2D models
    for f in [steno, hutton, michell, playfair, anderson]:
        C, M = f(dims) # create the synthetic "hutton" dataset
        xy = M.grid.coords() # get associated grid points
        s = M.predict(xy) # evaluate model

        for _C in C[:-1]: _checkCSet(_C, prop=False) # check sampled constraints are valid
        assert len(xy) == np.prod(dims)
        assert len(s) == np.prod(dims)