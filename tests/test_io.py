import pytest
import numpy as np
import os

def test_PLY(tmp_path):
    try:
        import plyfile # only run test if plyfile is installed
    except:
        return
    
    from curlew.io import savePLY, loadPLY
    xyz = np.random.rand(100,3)*100
    normals = np.random.rand(100,3)
    normals = normals / np.linalg.norm(normals, axis=1)[:,None]
    rgb = np.clip( (np.random.rand(100,3) * 255), 0, 255 ).astype(np.uint8)
    data = np.random.rand(100,5)
    names = ['Field%d'%i for i in range(5)]
    for n in [None, normals]:
        for c in [None, rgb]:
            for d in [None, data]:
                for nm in [None, names]:
                    file_path = tmp_path / "test.ply"
                    savePLY( file_path, xyz, rgb=c, normals=n, attr=d, names=nm)
                    p = loadPLY( file_path )
                    assert np.max( np.abs(xyz - p['xyz'] ) ) < 1e-6 # check positions match
                    if n is not None:
                        assert np.max( np.abs(normals - p['normals'] ) ) < 1e-6 # check positions match
                    if c is not None:
                        assert np.max( np.abs(rgb - p['rgb'] ) ) < 1e-6 # check positions match
                    if d is not None:
                        assert np.max( np.abs(data - p['attr'] ) ) < 1e-6 # check positions match
                        if nm is not None:
                            assert np.all([names[i] == p['names'][i] for i in range(len(names))])

def test_OBJ(tmp_path):
    from curlew.io import saveOBJ
    xyz = np.random.rand(100,3)*100 # make some random points
    faces = [ np.random.choice(len(xyz), 3) for i in range(100) ] # make some random faces
    rgb = np.clip( (np.random.rand(100,3) * 255), 0, 255 ).astype(np.uint8)
    saveOBJ( tmp_path / "test.obj", xyz=xyz, rgb=rgb, faces=faces ) # check OBJ writes
    assert os.path.exists( tmp_path / "test.obj" ) # not the robust test; but better than nothing...
    

