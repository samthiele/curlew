import numpy as np
import torch
from curlew.geology.geomodel import GeoModel
from curlew.geometry import grid
from curlew.fields.fourier import NFF
from curlew.fields.series import FSF
import curlew

curlew.default_dim = 2 # specify that these tests run in 2D by default

def test_hutton():
    """
    Run the hutton model as a test.
    """
    from curlew.synthetic import hutton
    from curlew import HSet
    from curlew.geology import strati
    curlew.default_dim = 2

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
                C=C[0], # constraints for this field
                H=H, # interpolator hyperparameters
                type=NFF,
                base=-np.inf, # basal surface (important for unconformities)
                hidden_layers=[], # hidden layers in the multi-layer perceptron that parameterises our field
                activation=None,
                rff_features=128, # number of random sin and cos features to create for each scale 
                length_scales=[500/2*np.pi,]) # the length scales in our model

    # define interpolator for unconformity field
    s1 = strati('unconformity', # name of created geological neural field (GNF)
                C=C[1], # constraints for this field
                H=H.copy(mono_loss="1.0", thick_loss=1.0), # change some hyperparams
                type=NFF,
                base="base", # basal surface (important for unconformities). In this case these have a value of 0.
                hidden_layers=[], # no need for hidden layers!
                activation=None,
                rff_features=128, # number of random sin and cos features to create for each scale 
                length_scales=[2000/2*np.pi,]) # the length scales in our model
    
    # define isosurfaces
    s1.isosurfaces = Ms['s1'].isosurfaces
    s0.isosurfaces = Ms['s0'].isosurfaces
    s1.addIsosurface("base", seed=Ms.fields[1].field.origin) # layer near the base of the unconformity

    # combine into a geomodel
    M = GeoModel([s0,s1])

    # fit scalar fields independently
    loss1 = M.prefit( epochs=1, best=True, vb=False)
    loss2 = M.prefit( epochs=200, best=True, vb=False)

    # check model is converging
    for k, v in loss1.items():
        assert loss1[k][0] > loss2[k][0]
    
    # get isosurface values
    isovals = s0.getIsovalues()
    assert len(isovals) > 3

    isovals = s1.getIsovalues()
    assert len(isovals) > 3

    # create a grid (section) to evaluate our model on
    G2 = grid( dims, step=(20,20), center=(dims[0]/2,dims[1]/2), sampleArgs=dict(N=1024) )
    sxy = G2.coords()

    # evaluate scalar field
    from curlew.utils import batchEval
    pred = batchEval(sxy, M.predict, batch_size=1000) # just use this to check it works
    assert (pred.structureID == 2).any() # check some basement is present
    assert (pred.structureID == 1).any() # check some unconformity is present

    # check lithoIDs were assigned correctly
    lithoNames = set( pred.lithoLookup.values() )
    assert 'basement' in lithoNames
    assert 'basement_i0' in lithoNames
    assert 'unconformity' in lithoNames
    assert 'unconformity_i0' in lithoNames
    assert len( np.unique(pred.lithoID) ) > 4 # should be at least 4 different lithologies...
    
    # check evaluate gradients function works
    grad, pred2 = s0.gradient( G2.coords(), normalize=True, return_vals=True )
    assert np.max( np.abs(1-np.linalg.norm(grad, axis=1)) ) < 1e-6 # check vectors are unit vectors

    # add a forward model and check that these can be trained together
    if False:
        from curlew.fields import BaseNF
        from torch import nn

        M.forward = BaseNF( HSet().zero(prop_loss=1.0),
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

def test_hutton_FSF():
    """
    Run the hutton model as a test. Use Fourier Series Fields (FSF)
    instead of Neural Fields (NFF) to check that these work as well.
    """
    from curlew.synthetic import hutton
    from curlew import HSet
    from curlew.geology import strati
    curlew.default_dim = 2

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
                C=C[0], # constraints for this field
                H=H, # interpolator hyperparameters
                type=FSF,
                base=-np.inf, # basal surface (important for unconformities)
                rff_features=128, # number of random sin and cos features to create for each scale 
                length_scale_range=[500, 500]) # the length scales in our model

    # define interpolator for unconformity field
    s1 = strati('unconformity', # name of created geological neural field (GNF)
                C=C[1], # constraints for this field
                H=H.copy(mono_loss="1.0", thick_loss=1.0), # change some hyperparams
                type=FSF,
                base="base", # basal surface (important for unconformities). In this case these have a value of 0.
                rff_features=128, # number of random sin and cos features to create for each scale 
                length_scale_range=[2000, 2000]) # the length scales in our model
    
    # define isosurfaces
    s1.isosurfaces = Ms['s1'].isosurfaces
    s0.isosurfaces = Ms['s0'].isosurfaces
    s1.addIsosurface("base", seed=Ms.fields[1].field.origin) # layer near the base of the unconformity

    # combine into a geomodel
    M = GeoModel([s0,s1])

    # fit scalar fields independently
    loss1 = M.prefit( epochs=1, best=True, vb=False)
    loss2 = M.prefit( epochs=200, best=True, vb=False)

    # check model is converging
    for k, v in loss1.items():
        assert loss1[k][0] > loss2[k][0]
    
    # get isosurface values
    isovals = s0.getIsovalues()
    assert len(isovals) > 3

    isovals = s1.getIsovalues()
    assert len(isovals) > 3

    # create a grid (section) to evaluate our model on
    G2 = grid( dims, step=(20,20), center=(dims[0]/2,dims[1]/2), sampleArgs=dict(N=1024) )
    sxy = G2.coords()

    # evaluate scalar field
    from curlew.utils import batchEval
    pred = batchEval(sxy, M.predict, batch_size=1000) # just use this to check it works
    assert (pred.structureID == 2).any() # check some basement is present
    assert (pred.structureID == 1).any() # check some unconformity is present

    # check lithoIDs were assigned correctly
    lithoNames = set( pred.lithoLookup.values() )
    assert 'basement' in lithoNames
    assert 'basement_i0' in lithoNames
    assert 'unconformity' in lithoNames
    assert 'unconformity_i0' in lithoNames
    assert len( np.unique(pred.lithoID) ) > 4 # should be at least 4 different lithologies...
    
    # check evaluate gradients function works
    grad, pred2 = s0.gradient( G2.coords(), normalize=True, return_vals=True )
    assert np.max( np.abs(1-np.linalg.norm(grad, axis=1)) ) < 1e-6 # check vectors are unit vectors

def test_playfair():
    from curlew.synthetic import playfair
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
                C=C[0], # constraints for this field
                H=H, # interpolator hyperparameters
                type=NFF,
                base=-np.inf, # basal surface (important for unconformities)
                hidden_layers=[], # no need for hidden layers!
                activation=None,
                rff_features=32, # number of random sin and cos features to create for each scale 
                length_scales=[2000/2*np.pi]) # the length scales in our model
    
    # define interpolator for unconformity field
    H = HSet( value_loss=1, # strength of penalty for mismatch between value constraints and field outputs
            grad_loss=1,  # strength of penalty for mismatch between gradient constraints and field gradients
            mono_loss="0.01", 
            thick_loss=1.0) # constant thickness is relatively important for dyke scalar fields as this is linked to offset
    s1 = sheet('dyke', # name of created geological neural field (GNF)
                C=C[1], # constraints for this field
                H=H, # interpolator hyperparameters
                type=NFF,
                contact=("upper","lower"), # Lower and upper surface of our dyke (which in this case is 100 m thick).
                hidden_layers=[], # no need for hidden layers!
                activation=None,
                rff_features=32, # number of random sin and cos features to create for each scale 
                length_scales=[2000/2*np.pi,]) # the length scales in our model
    s1.addIsosurface("upper", seed=np.array([dims[0]/2 - 25, 0]) ) # create isosurfaces used to determine where the dyke is
    s1.addIsosurface("lower", seed=np.array([dims[0]/2 + 25, 0]) )

    # combine into a geomodel
    M = GeoModel([s0,s1]) 

    # fit scalar fields
    _, loss1 = M.fit( epochs=1, best=True, vb=False )
    _, loss2 = M.fit( epochs=100, best=True, vb=False, early_stop=None )

    # check model is converging
    for k, v in loss1.items():
        assert loss1[k][0] - loss2[k][0] > 10

def test_michell():
    # load an example containing a fault
    from curlew.synthetic import michell
    dims = (2000,1000)  # dimensions of our 2D section
    C, _ = michell(dims, offset=225) # create the synthetic "hutton" dataset
    C = C[:-1] # drop value constraints as they're not needed

    # expand constraints for fault to get more value constraints
    #n = 100
    #C[1].vp = np.vstack([ C[1].vp, C[1].vp+n*C[1].gv, C[1].vp-n*C[1].gv])
    #C[1].vv = np.hstack([ C[1].vv, C[1].vv+1, C[1].vv-1])

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
                C=C[0], # constraints for this field
                H=H, # interpolator hyperparameters
                type=NFF,
                base=-np.inf, # basal surface (important for unconformities)
                hidden_layers=[],
                activation=None,
                rff_features=32, # number of random sin and cos features to create for each scale 
                length_scales=[2000]) # the length scales in our model

    # define interpolator for unconformity field
    H = HSet( value_loss=1, # strength of penalty for mismatch between value constraints and field outputs
            grad_loss=1,  # strength of penalty for mismatch between gradient constraints and field gradients
            mono_loss="0.01") # constant thickness is relatively important for dyke scalar fields as this is linked to offset
    s1 = fault('fault', # name of created geological neural field (GNF)
                C=C[1], # constraints for this field
                H=H, # interpolator hyperparameters
                type=NFF,
                shortening=(-1,0), # horizontal stress
                offset=(250,0,300), # Initial slip estimate, minimum slip, maximum slip
                width=0, # brittle fault
                hidden_layers=[], 
                activation=None,
                rff_features=32, # number of random sin and cos features to create for each scale 
                length_scales=[6000,], # the length scales in our model
                n_steps=1,  # default n_steps=2 ~ doubles integrated slip vs this test’s offset check
    )
    
    # combine into a geomodel
    M = GeoModel([s0, s1]) 

    # fit scalar fields independently [ helps get fault surface sorted out first ]
    loss1 = M.prefit( epochs=1, best=True, vb=False, early_stop=None )
    loss2 = M.prefit( epochs=100, best=True, vb=False, early_stop=None )

    # check model is converging
    for k, v in loss1.items():
        if isinstance(v, tuple):
            assert loss1[k][0] / loss2[k][0] > 2 # loss should be better than half the inital

    # optimise slip
    M.freeze( s1, geometry=True, params=False)
    _, loss1 = M.fit( epochs=1, learning_rate=0.1 , early_stop=None ) # and now optimise only fault slip (and the stratigraphic field)
    _, loss2 = M.fit( epochs=100, learning_rate=0.1 , early_stop=None ) # and now optimise only fault slip (and the stratigraphic field)

    # check model is converging
    assert loss1['basement'][0] / loss2['basement'][0] > 1 # loss should be better (if only a bit)
    assert abs( s1.deformation.offset.item() - 100 ) > 5 # more than 10 m difference in offset

    # check training at least runs for single-field fitting
    _, loss3 = s0.fit( 50, cache=True, faultBuffer=20)

    # check fault buffer function
    #b = s1.buffer(G.coords(), 0, width=50 )
    #assert np.sum(b) > 1000 # some points should be on the fault
    #b2 = s1.buffer(G.coords(), 0, width=25 )
    #assert np.sum(b2) < np.sum(b) # smaller buffer gives fewer points!

    # check we can undeform a CSet to get a paleo-deformed CSet
    C1 = C[0].copy()
    C1.iq = (1024, [(C1.vp, C1.vp, '=')]) # add a fake equality to check this is also undeformed
    C0 = C1.transform( s0.undeform )
    assert (C1.grid != C0.grid) # check grid instances are a copy
    assert (np.sum(C1.grid.coords() == C0.grid.coords()) > 1000) # check some grid coords remain stationary
    assert ( np.sum(C1.grid.coords() != C0.grid.coords()) > 1000 ) # check some grid coords have moved

    for p0, p1 in zip([C0.vp, C0.gp, C0.iq[1][0][0], C0.iq[1][0][1]],
                        [C1.vp, C1.gp, C1.iq[1][0][0], C1.iq[1][0][1]]):
        diff = np.linalg.norm( p1-p0, axis=1) # get offset vectors
        diff = np.median( diff[diff > 1]) # take median of points that have moved
        assert abs( diff - s1.deformation.offset.item() ) < 1 # check that median offset matches offset on fault

    # check isolated training works with this undeformed CSet
    _, loss4 = s0.field.fit(1, C=C0, transform=False) # Transform = False as C0 is in paleo-coordinates
    assert loss3['basement'][0] / loss3['basement'][0] < 1.1 # loss should be similar as we didn't train much

    # check loss explodes if we don't transform!
    _, loss4 = s0.field.fit(1, C=C0, transform=True) # If Transform=True, constraints should end up in incorrect locations
    assert loss4['basement'][0] / loss3['basement'][0] > 1.5

def test_anderson():
    # load an example containing a fault
    from curlew.synthetic import anderson
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
        hidden_layers=[], # no need for hidden layers!
        activation=None,
        rff_features=32, # number of fourier features
        length_scales=[4000]
    )

    s0 = strati('basement', # basement stratigraphy field
            type=NFF,
            C=C[0], # constraints
            H=H, # hyperparameters
            **params)
    s1 = fault('fault1', # older fault field
                type=NFF,
                C=C[1], # constraints
                H=H, # hyperparameters
                shortening=(0,1), # vertical sigma 1
                learn_sigma=False,
                offset=(-200,-100,-400),
                width=1e-6,
                **params)
    s2 = fault('fault2', # younger fault field
                type=NFF,
                C=C[2], # constraints
                H=H, # hyperparameters
                shortening=(0,1), # vertical sigma 1
                learn_sigma=False,
                offset=(-200,-100,-400),
                width=1e-6,
                **params)
    M = GeoModel([s0,s1,s2]) # combine into a geomodel

    # check model is converging
    loss1 = M.prefit( epochs=1, best=True, vb=False )
    loss2 = M.prefit( epochs=250, best=True, vb=False )
    for k, v in loss1.items():
        assert loss1[k][0] / loss2[k][0] > 2 # loss should be better than half the inital

    # optimise slip
    M.freeze( [s1, s2], geometry=True, params=False)
    _, loss1 = M.fit( epochs=1, learning_rate=1e-1 ) # and now optimise only fault slip (and the stratigraphic field)
    _, loss2 = M.fit( epochs=250, learning_rate=1e-1 ) # and now optimise only fault slip (and the stratigraphic field)

    # check model is converging
    assert loss1['basement'][0] > loss2['basement'][0] # loss should be better
    assert abs( s1.deformation.offset.item() - (-200) ) > 1 # more than 1 m difference in offset
    assert abs( s2.deformation.offset.item() - (-200) ) > 1 # more than 1 m difference in offset

def test_anderson3D():
        # load an example containing a fault
    from curlew.synthetic import anderson
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
        type=NFF,
        input_dim=3, # explicitly make these fields 3D [ we could also just change `curlew.default_dim` ]
        hidden_layers=[], # no need for hidden layers!
        activation=None,
        rff_features=32, # number of fourier features
        length_scales=[4000],
    )

    s0 = strati('basement', # basement stratigraphy field
            C=C3D[0], # constraints
            H=H, # hyperparameters
            **params)
    s1 = fault('fault1', # older fault field
                C=C3D[1], # constraints
                H=H, # hyperparameters
                shortening=(0,0,1), # vertical sigma 1
                learn_sigma=False,
                offset=(-200,-100,-400),
                width=1e-6,
                **params)
    s2 = fault('fault2', # younger fault field
                C=C3D[2], # constraints
                H=H, # hyperparameters
                shortening=(0,0,1), # vertical sigma 1
                learn_sigma=False,
                offset=(-200,-100,-400),
                width=1e-6,
                **params)
    M = GeoModel([s0,s1,s2]) # combine into a geomodel

    # check model is converging
    loss1 = M.prefit( epochs=1, best=True, vb=False )
    loss2 = M.prefit( epochs=250, best=True, vb=False )
    for k, v in loss1.items():
        assert loss1[k][0] / loss2[k][0] > 2 # loss should be better than half the inital

    # optimise slip
    M.freeze( [s1, s2], geometry=True, params=False)
    _, loss1 = M.fit( epochs=1, learning_rate=1e-1 ) # and now optimise only fault slip (and the stratigraphic field)
    _, loss2 = M.fit( epochs=250, learning_rate=1e-1 ) # and now optimise only fault slip (and the stratigraphic field)

    # check model is converging
    assert loss1['basement'][0] > loss2['basement'][0] # loss should be better
    assert abs( s1.deformation.offset.item() - (-200) ) > 1 # more than 1 m difference in offset
    assert abs( s2.deformation.offset.item() - (-200) ) > 1 # more than 1 m difference in offset

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
        verts, faces = G.contour( pred.scalar, 0) # fit contours
        #vals = np.mean( np.abs( M.fields[-1].predict(verts).scalar ) ) # check values 
        #assert np.mean(vals) < 0.1 # should be small

        if False:
            out = M.evaluate( cxy, topology=True, buffer=10., surfaces=None)
            assert 'topology' in out
            assert 'surfaces' not in out
            assert 'buffer' in out
            assert np.max( out['buffer'] ) > 0 # should be some values
            assert np.max( out['topology'] ) == 1 # some hangingwall
            assert np.min( out['topology'] ) == -1 # footwall too

def test_isosurfaces():
    # use analytic implicit field to test isosurface calculation code
    import curlew
    from curlew.synthetic import michell

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
    assert (f1.getIsovalue('fault', offset=1.0) - np.linalg.norm( f1.field.grad )) < 1e-6
    assert (-f1.getIsovalue('fault', offset=-1.0) - np.linalg.norm( f1.field.grad )) < 1e-6


def test_anchors():
    """Check that anchors are stored, transformed to paleo via getAnchor, and injected into the field during evaluation."""
    from curlew.synthetic import michell

    dims = (2000, 1000)
    C, _ = michell(dims, offset=100)
    C = C[:-1]

    from curlew import HSet
    from curlew.geology import strati, fault

    G = grid(dims, step=(50, 50), center=(dims[0] / 2, dims[1] / 2), sampleArgs=dict(N=256))
    for _c in C:
        _c.grid = G
        _c.delta = 10

    H = HSet(value_loss=1, grad_loss=1, mono_loss="0.1", thick_loss="1.0")
    s0 = strati(
        "basement",
        C=C[0],
        H=H,
        type=NFF,
        base=-np.inf,
        hidden_layers=[8],
        rff_features=32,
        length_scales=[2000],
    )
    s1 = fault(
        "fault",
        C=C[1],
        H=H,
        type=NFF,
        shortening=(-1, 0),
        offset=(100, 0, 300),
        width=0,
        hidden_layers=[8],
        rff_features=32,
        length_scales=[6000],
    )
    M = GeoModel([s0, s1])
    M.prefit(epochs=10, best=True, vb=False)

    # Add anchors in modern-day coordinates
    pt_modern = np.array([dims[0] / 2, 400.0])
    s1.addAnchor("centre", pt_modern)
    s1.addAnchor("other", [100.0, 200.0])

    assert "centre" in s1.anchors
    assert "other" in s1.anchors
    np.testing.assert_array_almost_equal(s1.anchors["centre"][1], pt_modern)

    # getAnchor returns numpy by default; for s1 (youngest, child is None) it returns the point unchanged
    paleo_centre = s1.getAnchor("centre")[0]
    assert hasattr(paleo_centre, "shape")
    assert paleo_centre.shape == (1, 2)
    paleo_other = s1.getAnchor("other")[0]
    assert paleo_other.shape == (1, 2)
    np.testing.assert_allclose(
        np.asarray(paleo_centre),
        pt_modern[None, :],
        rtol=1e-5,
        atol=1e-5,
    )

    # For s0 (has child s1), getAnchor uses child.undeform so it should match s0.undeform
    s0.addAnchor("centre", pt_modern)
    pt_batch = np.asarray(pt_modern)
    if pt_batch.ndim == 1:
        pt_batch = pt_batch[None, :]
    pt_t = torch.tensor(pt_batch, device=curlew.device, dtype=curlew.dtype)
    np.testing.assert_allclose(
        s0.getAnchor("centre")[0],
        s0.undeform(pt_t).detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    # During evaluation, field should receive anchor_{name} as tensors (to_numpy=False)
    x = np.array([[dims[0] / 2, 500.0]])
    _ = s1.predict(x, combine=False, to_numpy=False)
    assert hasattr(s1.field, "centre"), "field should have anchor_centre after evaluation"
    assert hasattr(s1.field, "other"), "field should have anchor_other after evaluation"
    np.testing.assert_allclose(
        getattr(s1.field, "centre").detach().cpu().numpy(),
        s1.getAnchor("centre")[0],
        rtol=1e-5,
        atol=1e-5,
    )

    # Direction anchor: position + direction (normalised in reconstructed space)
    s1.addAnchor("axis", position=[500.0, 300.0], direction=[3.0, 4.0])
    assert isinstance(s1.anchors["axis"][1], dict)
    assert s1.anchors["axis"][1]["normalize"] is True
    pos_axis, dir_axis = s1.getAnchor("axis")
    assert dir_axis is not None
    np.testing.assert_allclose(np.linalg.norm(dir_axis), 1.0, rtol=1e-5, atol=1e-5)
    # Stored end = position + direction = [503, 304]; for s1 (no child) recon = same
    np.testing.assert_allclose(pos_axis, [[500.0, 300.0]], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dir_axis, [[3.0 / 5.0, 4.0 / 5.0]], rtol=1e-5, atol=1e-5)

    # Direction anchor: start + end (not normalised)
    s1.addAnchor("segment", start=[0.0, 0.0], end=[100.0, 0.0])
    assert s1.anchors["segment"][1]["normalize"] is False
    pos_seg, dir_seg = s1.getAnchor("segment")
    assert dir_seg is not None
    np.testing.assert_allclose(dir_seg, [[100.0, 0.0]], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pos_seg, [[0.0, 0.0]], rtol=1e-5, atol=1e-5)

    # During evaluation, direction anchors inject name and name_direction
    _ = s1.predict(x, combine=False, to_numpy=False)
    assert hasattr(s1.field, "axis") and hasattr(s1.field, "axis_direction")
    assert hasattr(s1.field, "segment") and hasattr(s1.field, "segment_direction")

    # Invalid addAnchor combinations raise
    with np.testing.assert_raises(ValueError):
        s1.addAnchor("bad", position=[0, 0], start=[0, 0], end=[1, 1])
    with np.testing.assert_raises(ValueError):
        s1.addAnchor("bad2")


def test_isosurfaces_and_volumes():
    """
    Notebook-derived test for multi-field GeoField:
    - isosurfaces evaluate correctly across underlying fields
    - volumes evaluate correctly as boolean functional domains (half-interior of ellipse)
    """
    import curlew
    from curlew.geology.geofield import GeoField
    from curlew.fields.analytical import LinearField, EllipsoidalField

    curlew.default_dim = 2

    # Field 0: LinearField (y)
    G = GeoField(
        name="G",
        type=LinearField,
        input_dim=2,
        origin=np.array([0.0, 0.0]),
        gradient=np.array([0.0, 1.0]),
        normalise=False,
    )

    # Field 1: ellipse
    # (Note: GeoField.addField currently forwards **kwargs to the underlying field constructor,
    # so we attach anchors/isosurfaces explicitly via addAnchor/addIsosurface.)
    G.addField(
        "ellipse",
        type=EllipsoidalField,
        input_dim=2,
        origin=np.array([0.0, 0.0]),
        axes=np.array([60.0, 30.0]),
    )
    G.addIsosurface("ellipse_boundary", seed=np.array([60.0, 0.0]), field="ellipse")
    # Linear isosurface at y=0
    G.addIsosurface("linear_y0", seed=np.array([0.0, 0.0]), field=0)

    # --- Isosurfaces ---
    iso = G.getIsovalues()
    assert "ellipse_boundary" in iso
    assert "linear_y0" in iso
    assert abs(iso["linear_y0"]) < 1e-6
    assert abs(iso["ellipse_boundary"]) < 1e-6

    # --- Volume: half-interior of ellipse (inside ellipse AND below y=0 plane) ---
    G.addVolume("halfEllipse", "(ellipse > ellipse_boundary) & (G < linear_y0)")

    GR = grid((250, 180), step=(1, 1), center=(0, 0))
    R = G.predict(GR, combine=False, to_numpy=True, isosurfaces=False, litho=False, props=False)
    mask = G.getVolume("halfEllipse", GR, to_numpy=True)

    assert mask.dtype == bool
    assert mask.ndim == 1
    assert mask.shape[0] == GR.coords().shape[0]
    assert mask.any()
    assert (~mask).any()

    # Mask should satisfy both inequalities
    ell = np.asarray(R.fields["ellipse"])
    lin = np.asarray(R.fields["G"])
    np.testing.assert_array_less(iso["ellipse_boundary"] - 1e-9, ell[mask])
    np.testing.assert_array_less(lin[mask], iso["linear_y0"] + 1e-9)

    # Extract points and check they're in the "lower half"
    pts = GR.coords()[mask]
    assert pts.shape[1] == 2
    assert pts.shape[0] > 10
    assert np.max(pts[:, 1]) <= 1e-6
