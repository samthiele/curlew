import numpy as np

def test_ALF():
    from curlew.fields.analytical import LinearField
    from curlew.geometry import grid
    
    for i in [2,3]:
        G = grid( [10 for _i in range(i)],
                  step = [2 for _i in range(i)] )
        x = G.coords()
        s0 = LinearField(name='f0', input_dim=i, gradient=np.ones(i), normalise=False )
        field = s0.predict( x ).squeeze()

        # gradient = 1,1,1 -- so we know the answer
        # (is the sum of the coordinates)
        assert (field == np.sum(x, axis=-1)).all()

        # also test axis-aligned fields
        for j in range(i):
            grad = np.zeros(i)
            orig = np.zeros(i) - 1
            grad[j] = 2
            s0 = LinearField(name='f0', input_dim=i, 
                     gradient=grad, origin=orig )
            field = s0.predict( x ).squeeze()
            assert (field == 2*(x[:,j] + 1)).all()

def test_multi():
    from curlew import GeoModel
    from curlew.geology import strati, fault, sheet
    from curlew.geometry import grid
    from curlew.fields.analytical import LinearField

    # define a grid covering our model domain
    dims = (1000,500)
    G = grid( dims, step=(1,1), center=(dims[0]/2,dims[1]/2) ) 
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
                sigma1=np.array([0,1]), 
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
    for sid,n in g.structureLookup.items():
        mask = (g.structureID == sid)
        assert np.percentile( np.abs( g.scalar[mask] - g.fields[n][mask] ), 99) < 1e-6 # almost all values should match

    # check stackValues function
    gs = g.stackValues( mn=0, mx=1 )
    assert np.max(gs.scalar) == len(gs.structureLookup) # max value should be equal to number of structures
    
    import matplotlib.pyplot as plt
    sf = G.reshape( gs.scalar)
    
    #dx = s3.deformation.eval( cxy, s3)

    #plt.imshow(sf.T)

def test_fold():
    return # disable test; TODO - fix and re-enable

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
                wavelength=100, 
                amplitude=30, sharpness=0.7 )

    s2 = fold('s2', origin=np.array([0,0]), 
                extension=np.array([0,1]), 
                compression=np.array([1,0]), 
                wavelength=2000, 
                amplitude=100, sharpness=0.7 )
    
    s3 = sheet( 'dyke', C=LinearField( 'f2', input_dim=2, 
                        origin=np.array([300,0]),
                        gradient=np.array([1.5,-1]), normalise=True ),
                contact=(-50,50) )

    s4 = fault( 'fault', C=LinearField( 'fault', input_dim=2, 
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
    G = grid( dims, step=(1,1), center=(dims[0]/2,dims[1]/2) ) 
    cxy = G.coords()
    sf = M.predict(cxy)
    assert np.isfinite(sf).all()