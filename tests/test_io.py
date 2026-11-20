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

def test_model_io(tmp_path):
    import curlew
    from curlew.io import saveModel, loadModel
    from curlew.synthetic import steno
    
    # build synthetic model and check it saves / loads
    curlew.default_dim = 2
    M = steno()
    path = tmp_path / "steno.pt"
    saveModel(path, M)
    M2 = loadModel(path)
    assert M2.name == M.name
    assert len(M2.events) == len(M.events)
    xy = M.grid.coords()[:50]
    g1 = M.predict(xy)
    g2 = M2.predict(xy)
    assert np.max(np.abs(g1.scalar - g2.scalar)) < 1e-6

    # extract one event and check it also saves / loads
    path_evt = tmp_path / "s0.pt"
    saveModel(path_evt, M.events[0])
    E2 = loadModel(path_evt)
    assert E2.name == M.events[0].name


@pytest.mark.parametrize("dim", [2, 3])
def test_fitted_restore_model_io(tmp_path, dim):
    """
    A fitted restore event holds caches keyed by weak references (shear
    groups, ridge geometry, equality strata), which pickle cannot handle.
    They are dropped on save and rebuilt on use: the reloaded model must
    predict the same values and lithology, and keep training.
    """
    import torch
    import curlew
    from curlew.core import CSet, HSet
    from curlew.geology import restore
    from curlew.kinematics.euler import euler_velocity
    from curlew.io import saveModel, loadModel

    prev = curlew.device, curlew.dtype, curlew.default_dim
    curlew.device, curlew.dtype, curlew.default_dim = "cpu", torch.float32, dim
    try:
        rng = np.random.default_rng(0)
        up = np.eye(dim)[-1]
        h1 = rng.uniform(-500, 500, (10, dim))
        h1[:, -1] = 0.0
        h2 = h1 - 100.0 * up
        C = CSet(gp=h1[:3], gv=np.tile(up, (3, 1)), eq=[h1, h2])
        bank = {"n_features": 32, "n_directions": 8, "length_scale_range": (1000.0, 4000.0)}
        if dim == 3:
            vel = euler_velocity(3, seed=1, stream="alpha", activation="cubic",
                                 alpha=bank, hinge=[0.0, 1.0, 0.0])
        else:
            vel = euler_velocity(2, seed=1, stream="alpha", activation="cubic", **{
                "n_features": 32, "n_directions": 8, "length_scale_range": (1000.0, 4000.0)})
        event = restore("r", C=C, H=HSet().zero(grad_loss="1.", eq_loss="0.1"), velocity=vel,
                        n_steps=1, integrator="shear", depth_axis=dim - 1,
                        signed_normals=True, learning_rate=1e-1, pin=torch.zeros(dim))
        event.addIsosurface("H1", seed=h1)
        event.addIsosurface("H2", seed=h2)
        M = curlew.GeoModel([event])
        M.prefit(epochs=5, vb=False, early_stop=None)

        x = torch.tensor(rng.uniform(-400, 400, (50, dim)), dtype=torch.float32)
        with torch.no_grad():
            g1 = M.predict(x, litho=True)
        path = tmp_path / "restore.pt"
        saveModel(path, M)
        M2 = loadModel(path)
        with torch.no_grad():
            g2 = M2.predict(x, litho=True)
        assert np.max(np.abs(np.asarray(g1.scalar) - np.asarray(g2.scalar))) < 1e-6
        assert np.array_equal(np.asarray(g1.lithoID), np.asarray(g2.lithoID))
        M2.prefit(epochs=2, vb=False, early_stop=None)
    finally:
        curlew.device, curlew.dtype, curlew.default_dim = prev
