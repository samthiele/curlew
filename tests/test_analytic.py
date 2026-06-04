import numpy as np
import torch

def test_EllipsoidalField():
    from curlew.fields.analytical import EllipsoidalField
    from curlew.geometry import Grid

    # Default: distance-like mode (unit circle)
    for dim in [2, 3]:
        G = Grid([12] * dim, step=[2.0] * dim, center=[0.0] * dim)
        x = G.coords()
        field = EllipsoidalField(name="ell", input_dim=dim)
        out = field.forward(torch.tensor(x, dtype=torch.float32)).squeeze()
        assert out.shape == (x.shape[0],)
        assert (out >= 0).all().item()

        # Center should be ~0
        center_pt = np.zeros((1, dim))
        center_val = field.forward(torch.tensor(center_pt, dtype=torch.float32)).squeeze().item()
        assert abs(center_val - 0.0) < 1e-5, f"center value {center_val} not 0"

        # Boundary at r == 1 in canonical coordinates
        boundary = np.zeros((1, dim))
        boundary[0, 0] = 1.0
        boundary_val = field.forward(torch.tensor(boundary, dtype=torch.float32)).squeeze().item()
        assert abs(boundary_val - 1.0) < 1e-5, f"boundary value {boundary_val} not 1"

        # Far outside should be > 1
        far = np.ones((1, dim)) * 1e6
        far_val = field.forward(torch.tensor(far, dtype=torch.float32)).squeeze().item()
        assert far_val > 1.0, f"far value {far_val} not > 1"

        # Decay mode: 1 at center, 0 at/after boundary
        decay_field = EllipsoidalField(name="ell_decay", input_dim=dim, decay=True)
        out_d = decay_field.forward(torch.tensor(x, dtype=torch.float32)).squeeze()
        assert (out_d >= 0).all().item() and (out_d <= 1).all().item()
        center_val_d = decay_field.forward(torch.tensor(center_pt, dtype=torch.float32)).squeeze().item()
        assert abs(center_val_d - 1.0) < 1e-5, f"center value {center_val_d} not 1"
        far_val_d = decay_field.forward(torch.tensor(far, dtype=torch.float32)).squeeze().item()
        assert far_val_d == 0.0, f"far value {far_val_d} not 0"

        # String alias matches bool decay=True
        ellipse_field = EllipsoidalField(name="ell_ellipse", input_dim=dim, decay="ellipse")
        assert torch.allclose(
            ellipse_field.forward(torch.tensor(x, dtype=torch.float32)).squeeze(),
            out_d,
        )

        # Gaussian decay: 1 at center, smooth tail (not exactly 0 at r=1)
        normal_field = EllipsoidalField(name="ell_normal", input_dim=dim, decay="normal")
        center_val_n = normal_field.forward(torch.tensor(center_pt, dtype=torch.float32)).squeeze().item()
        assert abs(center_val_n - 1.0) < 1e-5
        boundary_val_n = normal_field.forward(torch.tensor(boundary, dtype=torch.float32)).squeeze().item()
        assert abs(boundary_val_n - np.exp(-0.5)) < 1e-5
        far_val_n = normal_field.forward(torch.tensor(far, dtype=torch.float32)).squeeze().item()
        assert far_val_n < 1e-10

        for decay_name, boundary_expected, far_max in [
            ("cosine", 0.0, 0.0),
            ("wendland", 0.0, 0.0),
            ("inverse_quadratic", 0.5, None),
        ]:
            f = EllipsoidalField(name=f"ell_{decay_name}", input_dim=dim, decay=decay_name)
            assert abs(
                f.forward(torch.tensor(center_pt, dtype=torch.float32)).squeeze().item() - 1.0
            ) < 1e-5
            assert abs(
                f.forward(torch.tensor(boundary, dtype=torch.float32)).squeeze().item()
                - boundary_expected
            ) < 1e-5
            if far_max is not None:
                assert (
                    f.forward(torch.tensor(far, dtype=torch.float32)).squeeze().item() <= far_max
                )

        sig = EllipsoidalField(name="ell_sig", input_dim=dim, decay="sigmoid", sigmoid_scale=20.0)
        assert abs(sig.forward(torch.tensor(center_pt, dtype=torch.float32)).squeeze().item() - 1.0) < 1e-5
        mid = np.zeros((1, dim))
        mid[0, 0] = 0.5
        assert sig.forward(torch.tensor(mid, dtype=torch.float32)).squeeze().item() > 0.9

    # Custom decay callable: matches built-in normal; u is zero at centre
    captured = {}

    def custom_normal(u, r):
        captured["u"] = u.detach().cpu().numpy()
        return torch.exp(-0.5 * r**2)

    origin = np.array([10.0, 20.0])
    axes = np.array([5.0, 2.0])
    custom = EllipsoidalField(
        name="ell_custom", input_dim=2, origin=origin, axes=axes, decay=custom_normal
    )
    ref = EllipsoidalField(name="ell_ref", input_dim=2, origin=origin, axes=axes, decay="normal")
    pts = torch.tensor([[10.0, 20.0], [15.0, 20.0], [10.0, 22.0]], dtype=torch.float32)
    assert torch.allclose(custom.forward(pts).squeeze(), ref.forward(pts).squeeze(), atol=1e-5)
    assert np.abs(captured["u"][0]).max() < 1e-5
    assert np.allclose(captured["u"][1], [5.0, 0.0], atol=1e-5)
    assert np.allclose(captured["u"][2], [0.0, 2.0], atol=1e-5)

    # 2D with custom origin and axes (ellipse)
    axes = np.array([5.0, 2.0])  # semi-axes
    E = EllipsoidalField(name="e2", input_dim=2, origin=origin, axes=axes, decay=True)
    G2 = Grid([20, 20], step=[1.0, 1.0], center=origin)
    x2 = G2.coords()
    out2 = E.forward(torch.tensor(x2, dtype=torch.float32)).squeeze()
    assert out2.shape == (x2.shape[0],)
    center_val2 = E.forward(torch.tensor(origin.reshape(1, -1), dtype=torch.float32)).squeeze().item()
    assert abs(center_val2 - 1.0) < 1e-5

    # 3D sphere (equal axes)
    axes3 = np.array([3.0, 3.0, 3.0])
    E3 = EllipsoidalField(name="e3", input_dim=3, origin=np.zeros(3), axes=axes3, decay=True)
    c3 = np.zeros((1, 3))
    assert abs(E3.forward(torch.tensor(c3, dtype=torch.float32)).squeeze().item() - 1.0) < 1e-5

    # Directions (rotation): 2D identity directions
    directions = np.eye(2)
    E_rot = EllipsoidalField(name="er", input_dim=2, origin=np.zeros(2), axes=np.ones(2), directions=directions, decay=True)
    assert abs(E_rot.forward(torch.tensor(np.zeros((1, 2)), dtype=torch.float32)).squeeze().item() - 1.0) < 1e-5

def test_ALF():
    from curlew.fields.analytical import LinearField
    from curlew.geometry import Grid
    
    for i in [2,3]:
        G = Grid( [10 for _i in range(i)],
                  step = [2 for _i in range(i)] )
        x = G.coords()
        s0 = LinearField(name='f0', input_dim=i, gradient=np.ones(i), normalise=False )
        field = s0.forward( torch.tensor(x) ).squeeze()

        # gradient = 1,1,1 -- so we know the answer
        # (is the sum of the coordinates)
        assert (field.numpy() == np.sum(x, axis=-1)).all()

        # also test axis-aligned fields
        for j in range(i):
            grad = np.zeros(i)
            orig = np.zeros(i) - 1
            grad[j] = 2
            s0 = LinearField(name='f0', input_dim=i, 
                     gradient=grad, origin=orig )
            field = s0.forward( torch.tensor(x) ).squeeze()
            assert (field.numpy() == 2*(x[:,j] + 1)).all()

def test_learnable():
    import torch
    from curlew import CSet, HSet
    from curlew.fields.analytical import LinearField

    true_origin = np.array([1.0, 2.0])
    true_grad = np.array([0.5, -0.3])
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(64, 2))
    vals = np.sum((pts - true_origin) * true_grad, axis=-1)

    field = LinearField(
        name='learnable_plane',
        input_dim=2,
        H=HSet().zero(value_loss=1.0),
        origin=np.zeros(2),
        gradient=np.array([1.0, 0.0]),
        learnable=True,
        normalise=False,
    )
    field.bind(CSet(
        vp=np.vstack([pts, true_origin.reshape(1, -1)]),
        vv=np.hstack([vals, 0.0]),
    ))

    assert isinstance(field.origin, torch.nn.Parameter)
    assert isinstance(field.grad, torch.nn.Parameter)

    loss1, _ = field.fit(epochs=1, vb=False, early_stop=None)
    loss2, pebble2 = field.fit(epochs=300, vb=False, early_stop=None)

    pred = field(torch.tensor(pts, dtype=torch.float64)).squeeze().detach().numpy()
    assert loss2 < loss1
    assert pebble2.total() < 1e-3
    assert np.mean((pred - vals) ** 2) < 1e-4
    assert np.allclose(field.grad.detach().cpu().numpy(), true_grad, atol=1e-3)
    # origin is only defined up to shift perpendicular to grad; o·g fixes the plane
    fitted_o = field.origin.detach().cpu().numpy()
    assert np.allclose(fitted_o @ true_grad, true_origin @ true_grad, atol=1e-3)

def test_multi():
    from curlew import GeoModel
    from curlew.geology import strati, fault, sheet
    from curlew.geometry import Grid
    from curlew.fields.analytical import LinearField

    # define a grid covering our model domain
    dims = (1000,500)
    G = Grid( dims, step=(1,1), center=(dims[0]/2,dims[1]/2) ) 
    cxy = G.coords()
    
    # create a model
    s0 = strati('s0', C=LinearField( 'f0', input_dim=2, 
                         origin=np.array([0,0]),
                         gradient=np.array([0.4,0.7]) ) )

    s1 = strati('s1', C=LinearField( 'f1', input_dim=2, 
                            origin=np.array([0,0]),
                            gradient=np.array([0,1]) ), base=300 )
    
    s2 = sheet( 'dyke', C=LinearField( 'f2', input_dim=2, 
                        origin=np.array([300,0]),
                        gradient=np.array([1.5,-1]), normalise=True ),
                contact=(-50,50) )

    s3 = fault( 'fault', C=LinearField( 'fault', input_dim=2, 
                                origin=np.array([600,0]),
                                gradient=np.array([-1,-1]), normalise=True ),
                shortening=np.array([0,1]), 
                offset=60.0, 
                width=(1, 1/50, 0.4) )
    
    M = GeoModel( [s0, s1, s2, s3 ] )

    # train and check that runs
    loss, _ = M.fit(100)
    assert loss == 0

    # evaluate the model
    g = M.predict(cxy)
    
    # check we have enough structures
    assert len(np.unique(g.structureID)) == 3 # check there are three structure IDs
    assert len(g.structureLookup) == 3 # check there are three structures
    for k in np.unique(g.structureID): # check structures are named
        assert k in g.structureLookup
    
    # check the evaluated scalar values match 
    for sid, ename in g.structureLookup.items():
        fname = M.eidLookup[sid].getField(0).name
        mask = (g.structureID == sid)
        assert np.percentile( np.abs( g.scalar[mask] - g.fields[fname][mask] ), 99) < 1e-6

    # check stackValues function
    gs = g.stackValues( mn=0, mx=1 )
    assert np.max(gs.scalar) == len(gs.structureLookup) # max value should be equal to number of structures
    
    import matplotlib.pyplot as plt
    sf = G.reshape( gs.scalar)
    
    #dx = s3.deformation.eval( cxy, s3)

    #plt.imshow(sf.T)

def test_fold():

    from curlew import GeoModel
    from curlew.geology import strati, fold
    from curlew.geology import sheet, fault
    from curlew.fields.analytical import LinearField

    s0 = strati('s0', C=LinearField( 'f0', input_dim=2, 
                            origin=np.array([0,0]),
                            gradient=np.array([0.0,0.1]) ) )

    s1 = fold('s1', origin=np.array([0,0]), 
                    extension=np.array([0,1]), 
                    compression=np.array([1,0]), 
                    wavelength=150, 
                    amplitude=20, sharpness=0.7 )

    s2 = fold('s2', origin=np.array([0,0]), 
                extension=np.array([0,1]), 
                compression=np.array([1,0]), 
                wavelength=1000, 
                amplitude=100, sharpness=0.7 )

    s3 = sheet( 'dyke', C=LinearField( 'f2', input_dim=2, 
                        origin=np.array([300,0]),
                        gradient=np.array([1.5,-1]), normalise=True ),
                contact=(-50,50) )

    s4 = fault( 'fault', C=LinearField( 'fault', input_dim=2, 
                                origin=np.array([600,0]),
                                gradient=np.array([-1,-1]), normalise=True ),
                shortening=np.array([0,1]), 
                offset=250.0, 
                width=(1, 50, 0.4) )

    s5 = fold('s4', origin=np.array([20,0]), 
                    extension=np.array([0,1]), 
                    compression=np.array([1,0]), 
                    wavelength=1000, 
                    amplitude=30, sharpness=0.6 )

    for z in [0,50,75,100,200,250,300,400,500,750]:
            s0.addIsosurface(f"layer{z}", seed=np.array([[200,z]]))
    s3.addIsosurface("dyke", value=-np.inf) # add isosurface so dyke is filled with material

    M = GeoModel( [s0, s1, s2, s3, s4, s5 ] )

    # evaluate it. Not sure how to check if it "worked"...
    from curlew.geometry import Grid
    dims = (1000,500)
    G = Grid( dims, step=(1,1), center=(dims[0]/2,dims[1]/2) ) 
    geo = M.predict(G)
    assert np.isfinite(geo.scalar).all() # scalar values should all be finite
    assert len( np.unique( geo.lithoID ) ) > 9 # should have lots of layers
    assert len( np.unique( geo.structureID )) >= 2 # two different structures; stratigraphy and dyke.