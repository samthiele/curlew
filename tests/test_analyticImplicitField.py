import numpy as np

def test_ALF():
    from curlew.fields.analytical import ALF
    from curlew.geometry import grid
    
    for i in [2,3]:
        G = grid( [10 for _i in range(i)],
                  step = [2 for _i in range(i)] )
        x = G.coords()
        s0 = ALF(name='f0', input_dim=i, gradient=np.ones(i), normalise=False )
        field = s0.predict( x ).squeeze()

        # gradient = 1,1,1 -- so we know the answer
        # (is the sum of the coordinates)
        assert (field == np.sum(x, axis=-1)).all()

        # also test axis-aligned fields
        for j in range(i):
            grad = np.zeros(i)
            orig = np.zeros(i) - 1
            grad[j] = 2
            s0 = ALF(name='f0', input_dim=i, 
                     gradient=grad, origin=orig )
            field = s0.predict( x ).squeeze()
            assert (field == 2*(x[:,j] + 1)).all()

def test_multi():
    from curlew import GeoModel
    from curlew.geology import strati, fault, sheet
    from curlew.geometry import grid
    from curlew.fields.analytical import ALF

    # define a grid covering our model domain
    dims = (1000,500)
    G = grid( dims, step=(1,1), origin=(dims[0]/2,dims[1]/2) ) 
    cxy = G.coords()
    # create a model
    s0 = strati('s0', C=ALF( 'f0', input_dim=2, 
                         origin=np.array([0,0]),
                         gradient=np.array([0.4,0.7]) ) )

    s1 = strati('s1', C=ALF( 'f1', input_dim=2, 
                            origin=np.array([0,0]),
                            gradient=np.array([0,1]) ), base=300 )
    
    s2 = sheet( 'dyke', C=ALF( 'f2', input_dim=2, 
                        origin=np.array([300,0]),
                        gradient=np.array([1.5,-1]), normalise=True ),
                contact=(-50,50) )

    s3 = fault( 'fault', C=ALF( 'fault', input_dim=2, 
                                origin=np.array([600,0]),
                                gradient=np.array([-1,-1]), normalise=True ),
                sigma1=np.array([0,1]), 
                offset=60.0, 
                width=(1, 50, 0.4) )
    
    M = GeoModel( [s0, s1, s2, s3 ] )

    # train and check that runs
    loss, _ = M.fit(100)
    assert loss == 0

    # evaluate the model
    sf = M.predict(cxy)
    
    # check we have enough structures
    assert len(np.unique(sf[:,1].astype(int))) == 3 

    # check stackValues function
    from curlew.utils import stackValues
    sfs = stackValues( sf )
    assert np.max(sfs[:,0]) == len(np.unique(sf[:,1])) # max value should be equal to number of structures
    
def test_anderson():
    from curlew import GeoModel
    from curlew.geology import strati, fault, sheet
    from curlew.geometry import grid
    from curlew.data import anderson
    from curlew import HSet
    from curlew.fields.analytical import ALF

    dims = (2000,1000)  # dimensions of our 2D section
    C,M = anderson(dims) # create the synthetic "hutton" dataset
    C = C[0] # we only need the stratigraphic constraints

    G = grid( dims, step=(25,25), origin=(dims[0]/2,dims[1]/2), sampleArgs=dict(N=1024))
    C.grid = G # add grid to our stratigraphy constraints
    C.trend = np.array([0,1]) # prefer flat stratigraphy (important for optimising fault offset)

    # create a model
    # define generic parameters first
    H = HSet( value_loss=1, grad_loss=1,
            mono_loss='0.1', thick_loss="1.0", flat_loss="0.1")
    params = dict(
        input_dim=2, # field input coordinate dimensions
        hidden_layers=[8,], # hidden layers
        rff_features=32, # number of fourier features
        length_scales=[4000]
    )
    s0 = strati('basement', # basement stratigraphy field
                C, # constraints
                H, # hyperparameters
                **params)
    w,h = np.array(dims) / 2
    gradient = np.array([ np.cos( np.deg2rad(35) ), 
                        np.sin( np.deg2rad(35) ) ] )
    s1 = fault( 'fault1', C=ALF( 'fault', input_dim=2, 
                                origin=np.array([w+50,h-50]),
                                gradient=gradient, normalise=True ),
                sigma1=np.array([0,-1]), 
                offset=(200,100,400) )

    gradient = np.array([ -np.cos( np.deg2rad(35) ), 
                        np.sin( np.deg2rad(35) ) ] )
    s2 = fault( 'fault2', C=ALF( 'fault', input_dim=2, 
                                origin=np.array([w-50, h+50]),
                                gradient=gradient, normalise=True ),
                sigma1=np.array([0,-1]), 
                offset=(200,100,400) )
    
    # construct a model and fit it
    M = GeoModel( [s0, s1, s2 ] )
    loss = M.fit(400)

    # check new slip is different to original one
    assert abs( s1.field.offset.item() - (-200) ) > 10 # more than 10 m difference in offset
    assert abs( s2.field.offset.item() - (-200) ) > 10 # more than 10 m difference in offset

def test_fold():
    from curlew import GeoModel
    from curlew.geology import strati, fold
    from curlew.geology import sheet, fault
    from curlew.fields.analytical import ALF

    s0 = strati('s0', C=ALF( 'f0', input_dim=2, 
                         origin=np.array([0,0]),
                         gradient=np.array([0.0,0.1]) ) )
    
    s1 = fold('s1', origin=np.array([0,0]), 
                extension=np.array([0,1]), 
                compression=np.array([1,0]), 
                wavelength=100, 
                amplitude=30, sharpness=0.7 )

    s2 = fold('s2', origin=np.array([0,0]), 
                extension=np.array([0,1]), 
                compression=np.array([1,0]), 
                wavelength=2000, 
                amplitude=100, sharpness=0.7 )
    
    s3 = sheet( 'dyke', C=ALF( 'f2', input_dim=2, 
                        origin=np.array([300,0]),
                        gradient=np.array([1.5,-1]), normalise=True ),
                contact=(-50,50) )

    s4 = fault( 'fault', C=ALF( 'fault', input_dim=2, 
                                origin=np.array([600,0]),
                                gradient=np.array([-1,-1]), normalise=True ),
                sigma1=np.array([0,1]), 
                offset=250.0, 
                width=(1, 50, 0.4) )

    s5 = fold('s4', origin=np.array([20,0]), 
                    extension=np.array([0,1]), 
                    compression=np.array([1,0]), 
                    wavelength=1000, 
                    amplitude=30, sharpness=0.6 )

    M = GeoModel( [s0, s1, s2, s3, s4, s5 ] )
    
    # evaluate it. Not sure how to check if it "worked"...
    from curlew.geometry import grid
    dims = (1000,500)
    G = grid( dims, step=(1,1), origin=(dims[0]/2,dims[1]/2) ) 
    cxy = G.coords()
    sf = M.predict(cxy)
    assert np.isfinite(sf).all()