import numpy as np
from curlew.geology.model import GeoModel
from curlew.geometry import grid

def test_hutton():
    """
    Run the hutton model as a test.
    """
    from curlew.data import hutton
    from curlew import HSet
    from curlew.geology import strati

    dims = (2000,1000)  # dimensions of our 2D section
    C, Ms = hutton(dims, breaks=10, cmap='prism', pval=1.0) 

    # initialise random sampling for global constraints
    G = grid( dims, step=(10,10), center=(dims[0]/2,dims[1]/2), sampleArgs=dict(N=1024) ) 
    for _c in C:
        _c.grid = G # add a random grid for each of our constraints
        _c.delta = 10
    
    # define interpolator for basement field
    H = HSet( value_loss='1.0', mono_loss='0.01', thick_loss='1.0')
    s0 = strati('basement', # name for this scalar field
                C[0], # constraints for this field
                H, # interpolator hyperparameters
                base=-np.inf, # basal surface (important for unconformities)
                input_dim=2, # field input coordinate dimensions (2D in our case)
                hidden_layers=[32,], # hidden layers in the multi-layer perceptron that parameterises our field
                rff_features=64, # number of random sin and cos features to create for each scale 
                length_scales=[500,]) # the length scales in our model

    # define interpolator for unconformity field
    s1 = strati('unconformity', # name of created geological neural field (GNF)
                C[1], # constraints for this field
                H.copy(mono_loss="1.0", thick_loss=1.0), # change some hyperparams
                base="base", # basal surface (important for unconformities). In this case these have a value of 0.
                input_dim=2, # field input coordinate dimensions (2D in our case)
                hidden_layers=[32], # hidden layers in the multi-layer perceptron that parameterises our field
                rff_features=64, # number of random sin and cos features to create for each scale 
                length_scales=[2000,]) # the length scales in our model
    
    # define isosurfaces
    s1.addIsosurface("base", seed=Ms.fields[1].field.origin) # layer near the base of the unconformity
    s1.addIsosurface("layer1", seed=(1500,800)) # layer higher up
    s0.addIsosurface("layer1", seed=(1000,500)) # layer in basement
    s0.addIsosurface("layer2", seed=(1000,200)) # deeper layer in basement

    # also add a basement isosurface with multiple seed points
    round_val = np.round( C[0].vv ) # round value constraints (to reduce numerical differences)
    vals, counts = np.unique( round_val, return_counts=True ) # find unique layers
    vv = vals[ np.argmax(counts) ] # pick the one with the most points
    pp = C[0].vp[ round_val == vv, : ] # get corresponding positions
    s0.addIsosurface("layer3", seed=pp )

    # combine into a geomodel
    M = GeoModel([s0,s1]) 

    # fit scalar fields independently
    loss1 = M.prefit( epochs=5, best=True, vb=False )
    loss2 = M.prefit( epochs=500, best=True, vb=False )

    # check model is converging
    for k, v in loss1.items():
        assert loss1[k][0] > loss2[k][0]
    
    # get isosurface values
    isovals = s0.getIsovalues()
    assert len(isovals) == 3

    isovals = s1.getIsovalues()
    assert len(isovals) == 2

    # create a grid (section) to evaluate our model on
    #sdims, sxy = grid( dims, step=(10,10), origin=(0,0) ) 
    G2 = grid( dims, step=(20,20), center=(dims[0]/2,dims[1]/2), sampleArgs=dict(N=1024) )
    sxy = G2.coords()

    # evaluate scalar field
    from curlew.utils import batchEval
    pred = batchEval(sxy, M.predict, batch_size=10000) # just use this to check it works
    assert (pred[:,1] == 2).any() # check some basement is present
    assert (pred[:,1] == 1).any() # check some unconformity is present

    # evalutate gradients
    grad, pred2 = M.gradient( sxy, normalize=True, return_vals=True )
    assert np.max( np.abs(pred2[:,0] - pred[:,0]) ) < 1e-6 # check predictions are the same
    assert np.max( np.abs(1-np.linalg.norm(grad, axis=1)) ) < 1e-6 # check vectors are unit vectors

    # add a forward model and check that these can be trained together
    from curlew.fields import NF
    from torch import nn

    M.forward = NF( HSet().zero(prop_loss=1.0),
                name = 'forward', 
                input_dim=2,
                output_dim=3,
                hidden_layers=[64,64,64], 
                activation=nn.ReLU(),
                loss=nn.SmoothL1Loss(), 
                rff_features=0 ) # don't use fourier features

    M.forward.bind(C[-1]) # add property constraints 

    # check model is converging
    L1, loss1 = M.fit( epochs=2, best=True, vb=False )
    L2, loss2 = M.fit( epochs=500, best=True, vb=False, early_stop=None )

    #assert loss1['basement'][0] > loss2['basement'][0] # loss should be better
    #assert loss1['unconformity'][0] > loss2['unconformity'][0] # loss should be better
    assert loss1['forward'][0] > loss2['forward'][0] 

def test_playfair():
    from curlew.data import playfair
    dims = (2000,1000)  # dimensions of our 2D section
    C, _ = playfair(dims) # create the synthetic "hutton" dataset

    from curlew import HSet
    from curlew.geology import strati, sheet

    # initialise random sampling for global constraints
    G = grid( dims, step=(10,10), center=(dims[0]/2,dims[1]/2), sampleArgs=dict(N=1024) ) 
    for _c in C:
        _c.grid = G # add a random grid for each of our constraints
        _c.delta = 10

    # define interpolator for basement field
    H = HSet( value_loss=1, # strength of penalty for mismatch between value constraints and field outputs
            grad_loss=1,  # strength of penalty for mismatch between gradient constraints and field gradients
            mono_loss='0.01', thick_loss='0.1') # disable these for now
    s0 = strati('basement', # name of created geological neural field (GNF)
                C[0], # constraints for this field
                H, # interpolator hyperparameters
                base=-np.inf, # basal surface (important for unconformities)
                input_dim=2, # field input coordinate dimensions (2D in our case)
                hidden_layers=[8,], # hidden layers in the multi-layer perceptron that parameterises our field
                rff_features=32, # number of random sin and cos features to create for each scale 
                length_scales=[2000]) # the length scales in our model


    # define interpolator for unconformity field
    H = HSet( value_loss=1, # strength of penalty for mismatch between value constraints and field outputs
            grad_loss=1,  # strength of penalty for mismatch between gradient constraints and field gradients
            mono_loss="0.01", 
            thick_loss=1.0) # constant thickness is relatively important for dyke scalar fields as this is linked to offset
    s1 = sheet('dyke', # name of created geological neural field (GNF)
                C[1], # constraints for this field
                H, # interpolator hyperparameters
                contact=("upper","lower"), # Lower and upper surface of our dyke (which in this case is 100 m thick).
                input_dim=2, # field input coordinate dimensions (2D in our case)
                hidden_layers=[8,], # hidden layers in the multi-layer perceptron that parameterises our field
                rff_features=32, # number of random sin and cos features to create for each scale 
                length_scales=[2000,]) # the length scales in our model
    s1.addIsosurface("upper", seed=np.array([dims[0]/2 - 25, 0]) ) # create isosurfaces used to determine where the dyke is
    s1.addIsosurface("lower", seed=np.array([dims[0]/2 + 25, 0]) )

    # combine into a geomodel
    M = GeoModel([s0,s1]) 

    # fit scalar fields
    _, loss1 = M.fit( epochs=25, best=True, vb=False )
    _, loss2 = M.fit( epochs=200, best=True, vb=False, early_stop=None )

    # check model is converging
    for k, v in loss1.items():
        assert loss1[k][0] - loss2[k][0] > 10

def test_michell():
    # load an example containing a fault
    from curlew.data import michell
    dims = (2000,1000)  # dimensions of our 2D section
    C, _ = michell(dims, offset=225) # create the synthetic "hutton" dataset
    C = C[:-1] # drop value constraints as they're not needed

    from curlew import HSet
    from curlew.geology import strati, fault

    # initialise random sampling for global constraints
    G = grid( dims, step=(10,10), center=(dims[0]/2,dims[1]/2), sampleArgs=dict(N=1024) ) 
    for _c in C:
        _c.grid = G # add a random grid for each of our constraints
        _c.delta = 10

    # define interpolator for basement field
    H = HSet( value_loss=1, grad_loss=1,
            mono_loss='0.1', thick_loss="1.0")
    s0 = strati('basement', # name of created geological neural field (GNF)
                C[0], # constraints for this field
                H, # interpolator hyperparameters
                base=-np.inf, # basal surface (important for unconformities)
                input_dim=2, # field input coordinate dimensions (2D in our case)
                hidden_layers=[8,], # hidden layers in the multi-layer perceptron that parameterises our field
                rff_features=32, # number of random sin and cos features to create for each scale 
                length_scales=[2000]) # the length scales in our model

    # define interpolator for unconformity field
    H = HSet( value_loss=1, # strength of penalty for mismatch between value constraints and field outputs
            grad_loss=1,  # strength of penalty for mismatch between gradient constraints and field gradients
            mono_loss="0.01") # constant thickness is relatively important for dyke scalar fields as this is linked to offset
    s1 = fault('fault', # name of created geological neural field (GNF)
                C[1], # constraints for this field
                H, # interpolator hyperparameters
                sigma1=(-1,0), # horizontal stress
                offset=(100,0,200), # Initial slip estimate, minimum slip, maximum slip
                width=(5,100,0.5), # add a funky drag fold, coz we can and it's pretty.
                input_dim=2, # field input coordinate dimensions (2D in our case)
                hidden_layers=[8,], # hidden layers in the multi-layer perceptron that parameterises our field
                rff_features=32, # number of random sin and cos features to create for each scale 
                length_scales=[2000,]) # the length scales in our model
    
    # combine into a geomodel
    M = GeoModel([s0, s1]) 

    # fit scalar fields independently [ helps get fault surface sorted out first ]
    loss1 = M.prefit( epochs=25, best=True, vb=False, early_stop=None )
    loss2 = M.prefit( epochs=200, best=True, vb=False, early_stop=None )

    # check model is converging
    for k, v in loss1.items():
        if isinstance(v, tuple):
            assert loss1[k][0] / loss2[k][0] > 2 # loss should be better than half the inital

    # optimise slip
    M.freeze( s1, geometry=True, params=False)
    _, loss1 = M.fit( epochs=25, learning_rate=0.1 , early_stop=None ) # and now optimise only fault slip (and the stratigraphic field)
    _, loss2 = M.fit( epochs=200, learning_rate=0.1 , early_stop=None ) # and now optimise only fault slip (and the stratigraphic field)

    # check model is converging
    assert loss1['basement'][0] / loss2['basement'][0] > 1.5 # loss should be better
    assert abs( s1.field.offset.item() - 200 ) > 10 # more than 10 m difference in offset

    # check training at least runs for single-field fitting
    _, loss3 = s0.fit( 100, cache=True, faultBuffer=20)

    # check fault buffer function
    b = s1.buffer(G.coords(), 0, width=50 )
    assert np.sum(b) > 1000 # some points should be on the fault
    b2 = s1.buffer(G.coords(), 0, width=25 )
    assert np.sum(b2) < np.sum(b) # smaller buffer gives fewer points!

    # check we can undeform a CSet to get a paleo-deformed CSet
    C1 = C[0].copy()
    C1.iq = (1024, [(C1.vp, C1.vp, '=')]) # add a fake equality to check this is also undeformed
    C0 = C1.transform( s0.undeform )
    assert (C1.grid != C0.grid) # check grid instances are a copy
    assert ((C1.grid.coords() != C0.grid.coords()).all()) # check grid coords have been updated
    for p0, p1 in zip([C0.vp, C0.gp, C0.iq[1][0][0], C0.iq[1][0][1]],
                      [C1.vp, C1.gp, C1.iq[1][0][0], C1.iq[1][0][1]]):
        diff = np.median( np.linalg.norm( p1-p0, axis=1) )
        assert abs( diff - s1.field.offset.item()/2 ) < 1 # TODO - why is the measure applied displacement half of the specified??

    # check isolated training works with this undeformed CSet
    _, loss3 = s0.field.fit(1, C=C0, transform=False)
    assert loss3['basement'][0] / loss2['basement'][0] < 1.1 # loss should be similar as we didn't train much

    # check loss explodes if we don't transform!
    _, loss4 = s0.field.fit(1, C=C0, transform=True)
    assert loss4['basement'][0] / loss2['basement'][0] > 5

def test_anderson():
    # load an example containing a fault
    from curlew.data import anderson
    dims = (2000,1000)  # dimensions of our 2D section
    C, _ = anderson(dims) # create the synthetic "hutton" dataset
    C = C[:-1] # drop value constraints as they're not needed

    from curlew import HSet
    from curlew.geology import strati, fault

    # initialise random sampling for global constraints
    G = grid( dims, step=(10,10), center=(dims[0]/2,dims[1]/2), sampleArgs=dict(N=1024) ) 
    for _c in C:
        _c.grid = G # add a random grid for each of our constraints
        _c.delta = 10

    # define interpolator for basement field
    # define generic parameters first
    H = HSet( value_loss=1, grad_loss=1,
            mono_loss='0.1', thick_loss="1.0")
    params = dict(
        input_dim=2, # field input coordinate dimensions
        hidden_layers=[8,], # hidden layers
        rff_features=32, # number of fourier features
        length_scales=[4000]
    )

    s0 = strati('basement', # basement stratigraphy field
            C[0], # constraints
            H, # hyperparameters
            **params)
    s1 = fault('fault1', # older fault field
                C[1], # constraints
                H, # hyperparameters
                sigma1=(0,1), # vertical sigma 1
                learn_sigma=False,
                offset=(-200,-100,-400),
                width=1e-6,
                **params)
    s2 = fault('fault2', # younger fault field
                C[2], # constraints
                H, # hyperparameters
                sigma1=(0,1), # vertical sigma 1
                learn_sigma=False,
                offset=(-200,-100,-400),
                width=1e-6,
                **params)
    M = GeoModel([s0,s1,s2]) # combine into a geomodel

    # check model is converging
    loss1 = M.prefit( epochs=5, best=True, vb=False )
    loss2 = M.prefit( epochs=500, best=True, vb=False )
    for k, v in loss1.items():
        assert loss1[k][0] / loss2[k][0] > 2 # loss should be better than half the inital

    # optimise slip
    M.freeze( [s1, s2], geometry=True, params=False)
    _, loss1 = M.fit( epochs=25, learning_rate=1e-1 ) # and now optimise only fault slip (and the stratigraphic field)
    _, loss2 = M.fit( epochs=500, learning_rate=1e-1 ) # and now optimise only fault slip (and the stratigraphic field)

    # check model is converging
    assert loss1['basement'][0] > loss2['basement'][0] # loss should be better
    assert abs( s1.field.offset.item() - (-200) ) > 1 # more than 1 m difference in offset
    assert abs( s2.field.offset.item() - (-200) ) > 1 # more than 1 m difference in offset

def test_anderson3D():
        # load an example containing a fault
    from curlew.data import anderson
    dims = (2000,1000)  # dimensions of our 2D section
    C, _ = anderson(dims) # create the synthetic "hutton" dataset
    C = C[:-1] # drop value constraints as they're not needed


    # extrude to make 3D test dataset
    from curlew.geometry import extrude
    C3D = extrude( C, step=(0,200,0), n=3 )

    from curlew import HSet
    from curlew.geology import strati, fault

    # initialise random sampling for global constraints
    dims = ( dims[0], 600, dims[1] ) 
    G = grid( dims, step=(25,25,25), center=(dims[0]/2,dims[1]/2,dims[2]/2), sampleArgs=dict(N=4096) ) 
    for _c in C3D:
        _c.grid = G # add a random grid for each of our constraints
        _c.delta = 10

    # define interpolator for basement field
    # define generic parameters first
    H = HSet( value_loss=1, grad_loss=1,
            mono_loss='0.1', thick_loss="1.0")
    params = dict(
        input_dim=3, # field input coordinate dimensions
        hidden_layers=[8,], # hidden layers
        rff_features=32, # number of fourier features
        length_scales=[4000]
    )

    s0 = strati('basement', # basement stratigraphy field
            C3D[0], # constraints
            H, # hyperparameters
            **params)
    s1 = fault('fault1', # older fault field
                C3D[1], # constraints
                H, # hyperparameters
                sigma1=(0,0,1), # vertical sigma 1
                learn_sigma=False,
                offset=(-200,-100,-400),
                width=1e-6,
                **params)
    s2 = fault('fault2', # younger fault field
                C3D[2], # constraints
                H, # hyperparameters
                sigma1=(0,0,1), # vertical sigma 1
                learn_sigma=False,
                offset=(-200,-100,-400),
                width=1e-6,
                **params)
    M = GeoModel([s0,s1,s2]) # combine into a geomodel

    # check model is converging
    loss1 = M.prefit( epochs=25, best=True, vb=False )
    loss2 = M.prefit( epochs=200, best=True, vb=False )
    for k, v in loss1.items():
        assert loss1[k][0] / loss2[k][0] > 2 # loss should be better than half the inital

    # optimise slip
    M.freeze( [s1, s2], geometry=True, params=False)
    _, loss1 = M.fit( epochs=5, learning_rate=1e-1 ) # and now optimise only fault slip (and the stratigraphic field)
    _, loss2 = M.fit( epochs=500, learning_rate=1e-1 ) # and now optimise only fault slip (and the stratigraphic field)

    # check model is converging
    assert loss1['basement'][0] > loss2['basement'][0] # loss should be better
    assert abs( s1.field.offset.item() - (-200) ) > 1 # more than 1 m difference in offset
    assert abs( s2.field.offset.item() - (-200) ) > 1 # more than 1 m difference in offset

    tcont = False # test contouring (requires skimage)
    try:
        import skimage
        tcont = True
    except:
        pass
    if tcont:
        G = grid( dims, step=(100,100,100), center=(dims[0]/2,dims[1]/2,dims[2]/2) )
        cxy = G.coords()
        gdim = G.shape
        from curlew.utils import batchEval
        pred = batchEval( cxy, M.fields[-1].predict, batch_size=10000) # predict in a RAM-safe way
        verts, faces = G.contour( pred[:,0], 0) # fit contours
        vals = np.mean( np.abs( M.fields[-1].predict(verts)[:,0] ) ) # check values 
        assert np.mean(vals) < 0.1 # should be small

        # also test evaluate function (at least, that it runs)
        out = M.evaluate( G, topology=True, buffer=10., surfaces=True, batch_size=10000)
        assert 'topology' in out
        assert 'surfaces' in out
        assert 'buffer' in out
        assert np.max( out['buffer'] ) > 0 # should be some values
        assert np.max( out['topology'] ) == 1 # some hangingwall
        assert np.min( out['topology'] ) == -1 # footwall too

        out = M.evaluate( cxy, topology=True, buffer=10., surfaces=None, batch_size=10000)
        assert 'topology' in out
        assert 'surfaces' not in out
        assert 'buffer' in out
        assert np.max( out['buffer'] ) > 0 # should be some values
        assert np.max( out['topology'] ) == 1 # some hangingwall
        assert np.min( out['topology'] ) == -1 # footwall too

def test_isosurfaces():
    # use analytic implicit field to test isosurface calculation code
    import curlew
    from curlew.data import michell

    dims = (2000,1000)  # dimensions of our 2D section
    M = michell(dims, pval=1.0)[-1] # create the synthetic "hutton" dataset

    # add isosurface
    f1 = M.fields[-1]

    # test different isosurface definitions and evaluations
    f1.addIsosurface(name='fault', value=0)
    assert f1.getIsovalue('fault') == 0
    assert f1.getIsovalue(0) == 0
    for iso in [f1.field.origin, [f1.field.origin, f1.field.origin]]:
        f1.addIsosurface(name='fault', seed=iso) # add an isosurface on the fault surface
        assert f1.getIsovalue('fault') == 0

    # test isovalue offset
    assert (f1.getIsovalue('fault', offset=1.0) - np.linalg.norm( f1.field.gradient )) < 1e-6
    assert (-f1.getIsovalue('fault', offset=-1.0) - np.linalg.norm( f1.field.gradient )) < 1e-6
