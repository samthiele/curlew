import numpy as np
from curlew.fields import BaseNF

def test_basic():
    # test colour map
    from curlew import ccmap
    assert len(ccmap(0.5)) == 4 # check it runs and returns a RGBA colour

    from curlew import CSet,HSet

    H = HSet()
    assert H.value_loss != 0
    H = HSet().zero()
    assert H.value_loss == 0
    H = H.copy(value_loss=10)
    assert H.value_loss == 10

    # check CSet can be cast to and from torch tensors
    C = CSet()
    C.vp = np.random.rand(10,3)
    C.vv = np.random.rand(10)
    C.gp = np.random.rand(10,3)
    C.gv = np.random.rand(10,3)
    C.iq = (10, [([np.random.rand(10,3), np.random.rand(10,3), '=']) for i in range(3)])
    C2 = C.torch()
    C3 = C2.numpy()

    # check inequality constraints were properly cast
    for i in range( len(C2.iq[1]) ):
        assert np.mean( np.abs( C.iq[1][i][0] - C3.iq[1][i][0])) < 1e-6 # left-hand points
        assert np.mean( np.abs(C.iq[1][i][1] - C3.iq[1][i][1])) < 1e-6 # right-hand points
        assert C.iq[1][i][2] == C3.iq[1][i][2] # inequality character
    
    # check we can transform points
    def t(p):
        return p+1
    for C in [C2, C3]:
        C0 = C.transform( t )
        for a0,a1 in zip([C0.vp, C0.gp, C0.iq[1][i][0], C0.iq[1][i][1]], [C.vp, C.gp, C.iq[1][i][0], C.iq[1][i][1]]):
            assert np.median(a0 - a1) == 1
    
        def f(p):
            mask = np.full( len(p), True)
            mask[0] = False
            return mask
        C0 = C.filter( f )
        for a0,a1 in zip([C0.vp, C0.gp, C0.iq[1][i][0], C0.iq[1][i][1]], [C.vp, C.gp, C.iq[1][i][0], C.iq[1][i][1]]):
            assert len(a1) - len(a0) == 1

def test_transform():
    from curlew.core import Transform
    import numpy as np
    import torch

    # check initialisation with identity matrix
    T_id_np = Transform(2)
    pts2_np = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
    ])
    assert np.allclose(pts2_np, T_id_np(pts2_np)) # should be no change!

    T_id_np = Transform(3)
    pts3_np = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0],
    ])
    assert np.allclose(pts3_np, T_id_np(pts3_np)) # should be no change!
    assert np.allclose(pts3_np, T_id_np.inverse()(T_id_np(pts3_np)))

    # -------------------------------------------------
    # NumPy 2D – translation
    # -------------------------------------------------
    T2_np = Transform(
        np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0,  5.0],
            [0.0, 0.0,  1.0],
        ])
    )

    pts2_np = np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    expected2_np = np.array([
        [10.0,  5.0],
        [11.0,  7.0],
        [13.0,  9.0],
    ])

    out2_np = T2_np(pts2_np)
    assert np.allclose(out2_np, expected2_np)
    assert np.allclose(pts2_np, T2_np.inverse()(out2_np))

    # -------------------------------------------------
    # NumPy 3D – scaling
    # -------------------------------------------------
    T3_np = Transform(
        np.array([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    )
    pts3_np = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0],
    ])
    expected3_np = np.array([
        [2.0,  3.0,  4.0],
        [4.0,  9.0, 16.0],
    ])
    out3_np = T3_np.apply(pts3_np)
    assert np.allclose(out3_np, expected3_np)
    assert np.allclose(pts3_np, T3_np.inverse()(out3_np))

    # Torch 2D – translation
    T2_t = Transform(
        torch.tensor([
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 1.0],
        ])
    )
    pts2_t = torch.tensor([
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    expected2_t = torch.tensor([
        [3.0, 4.0],
        [4.0, 5.0],
    ])
    out2_t = T2_t(pts2_t)
    assert torch.allclose(out2_t, expected2_t)
    assert np.allclose(pts2_t, T2_t.inverse()(out2_t))

    # Torch 3D – translation
    T3_t = Transform(
        torch.tensor([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    )
    pts3_t = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ])
    expected3_t = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ])
    out3_t = T3_t.apply(pts3_t)
    assert torch.allclose(out3_t, expected3_t)
    assert np.allclose(pts3_t, T3_t.inverse()(out3_t))

def test_geometry():
    from curlew.geometry import grid, section, _extrude_array

    # test 2D grid creation code
    dims = (200,100) 
    G = grid( dims, step=(1,1), center=(dims[0]/2,dims[1]/2) ) 
    cxy = G.coords(transform=False)
    assert cxy.shape[0] == np.prod( dims )
    assert cxy.shape[-1] == 2
    for i in range(G.ndim):
        assert np.max(cxy[:,i]) == np.max(G.axes[i])
    block = G.reshape(cxy[:,0]) # reshape x-coordinate into 2D block
    assert (block[:,0] == G.axes[0]).all() # ensure reshaping worked 
    block = G.reshape(cxy) # reshape all coordinates into block
    assert (block[:,0,0] == G.axes[0]).all() # ensure reshaping worked 
    assert (block[0,:,1] == G.axes[1]).all() # ensure reshaping worked 

    # test 3D grid creation
    dims = (200,100,50) 
    G = grid( dims, step=(1,1,1), center=(dims[0]/2,dims[1]/2,dims[2]/2) ) 
    cxyz = G.coords(transform=False)
    assert cxyz.shape[0] == np.prod( dims )
    assert cxyz.shape[-1] == 3
    for i in range(G.ndim):
        assert np.max(cxyz[:,i]) == np.max(G.axes[i])
    block = G.reshape(cxyz[:,0]) # reshape x-coordinate into 3D block
    assert (block[:,0,0] == G.axes[0]).all() # ensure reshaping worked 
    block = G.reshape(cxyz) # reshape all coordinates into block
    assert (block[:,0,0,0] == G.axes[0]).all() # ensure reshaping worked 
    assert (block[0,:,0,1] == G.axes[1]).all() # ensure reshaping worked 
    assert (block[0,0,:,2] == G.axes[2]).all() # ensure reshaping worked 

    # test 3D section creation
    dims = (200,100) 
    for i in range(3):
        norm = [0,0,0]
        norm[i] = 1
        _, sxyz = section( dims, (0,0,0), normal=norm, step=(1,1,1))
        _, sxyz2 = section( dims, (0,0,0), normal=norm, step=1)
        assert (sxyz == sxyz2).all()
        assert sxyz.shape[0] == np.prod(dims)
        assert sxyz.shape[1] == 3
        assert (sxyz[:,i] == 0).all()

    # test extrusion code (only used for synthetic 3D examples)
    for n,step,dir in [(5,[0,100,0],"up"), (5,[0,0,100],"north")]:
        xyz = _extrude_array(cxy, step=step, n=n, y=dir)
        assert xyz.shape[0] == cxy.shape[0]*n
        assert xyz.shape[1] == 3
        assert np.max(xyz[:,0]) == np.max(cxy[:,0])
        if dir == "up":
            assert np.max(xyz[:,2]) == np.max(cxy[:,1])
            assert np.max(xyz[:,1]) == (n-1)*np.max(step)
        else:
            assert np.max(xyz[:,1]) == np.max(cxy[:,1])
            assert np.max(xyz[:,2]) == (n-1)*np.max(step)

    # test Grid object
    from curlew.geometry import Grid
    G1 = Grid([4600,4000,2500], step=30, center=[200,100,50] )
    G2 = Grid([4600,4000], step=30, center=[200,100] )
    for G in [G1, G2]: # Test 2D and 3D grids
        # check coords() function works with transform = False
        points = G.coords(transform=False)
        for i in range(G.ndim):
            assert np.min(points[:,i]) == np.min(G.axes[i])
            assert np.max(points[:,i]) == np.max(G.axes[i])

            # also check in reshaped form
            g = G.reshape( points[:,i] )
            assert np.min(g) == np.min(G.axes[i])
            assert np.max(g) == np.max(G.axes[i])

        # check coords() function works with origin offset
        points = G.coords(transform=True)
        for i in range(G.ndim):
            assert np.min(points[:,i]) == np.min(G.axes[i]) + G.center[i]
            assert np.max(points[:,i]) == np.max(G.axes[i]) + G.center[i]

            # also check in reshaped form
            g = G.reshape( points[:,i] )
            assert np.min(g) == np.min(G.axes[i]) + G.center[i]
            assert np.max(g) == np.max(G.axes[i]) + G.center[i]

def test_extrude():
    from curlew.geometry import extrude
    from curlew.synthetic import steno
    C, _ = steno((1000,500))
    n=3
    C3D = extrude(C, step=(0,100,0), n=n, y="up")
    for i,_C in enumerate(C3D):
        if _C.gp is not None:
            assert _C.vp.shape[1] == 3
            assert _C.vv.shape[0] == _C.vp.shape[0]
            assert len(_C.vv.shape) == 1
            assert _C.vp.shape[0] == n*C[i].vp.shape[0]
            assert _C.vv.shape[0] == n*C[i].vv.shape[0]
            assert (_C.vv[0:C[i].vv.shape[0]] == C[i].vv).all()
            assert (_C.vp[0:C[i].vv.shape[0],0] == C[i].vp[:,0]).all()
            assert (_C.vp[0:C[i].vv.shape[0],2] == C[i].vp[:,1]).all()
            assert (_C.gv.shape[0] == _C.gp.shape[0]) # 3D gradients
            assert (_C.gv.shape[-1] == 3) # 3D gradients
            assert ( (_C.gv[:,1] == 0).all() ) # no out-of-plane gradients
        else:
            assert _C.pp.shape[1] == 3
            assert _C.pv.shape[0] == _C.pv.shape[0]
            assert len(_C.pv.shape) == 2
            assert _C.pp.shape[0] == n*C[i].pp.shape[0]
            assert _C.pv.shape[0] == n*C[i].pv.shape[0]
            assert (_C.pv[0:C[i].pv.shape[0]] == C[i].pv).all()
            assert (_C.pp[0:C[i].pv.shape[0],0] == C[i].pp[:,0]).all()
            assert (_C.pp[0:C[i].pv.shape[0],2] == C[i].pp[:,1]).all()
    

