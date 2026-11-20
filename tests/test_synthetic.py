import numpy as np
import curlew

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

def _checkGeodeCRS(g, M):
    """Check Geode.x defines global, model, and each GeoEvent coordinate frame."""
    assert "global" in g.x
    assert "model" in g.x
    for F in M.events:
        assert F.name in g.x
    n = len(g)
    for key in ("global", "model", *[F.name for F in M.events]):
        assert len(g.x[key]) == n

def test_synthetic():
    from curlew.synthetic import steno, hutton, michell, playfair, anderson, lehmann, walker, goguel

    models = [steno, hutton, michell, walker, playfair, anderson, lehmann, goguel]
    for ndim in (2, 3):
        curlew.default_dim = ndim
        for f in models:
            if f is goguel and ndim == 3:
                continue  # sandbox example is 2D; 3D build is supported but not regression-tested here
            M = f()
            xy = M.grid.coords()
            g = M.predict(xy)
            _checkGeodeCRS(g, M)
            assert len(np.unique(g.lithoID)) > 1

            start = np.zeros(ndim)
            end = np.array(M.grid.dims, dtype=float)
            d, c = M.drill(start, end, step=10)
            _checkGeodeCRS(d, M)
            if c is not None:
                _checkGeodeCRS(c, M)
            assert len(np.unique(d.lithoID)) > 1
            assert len(c.coords()) > 0
            assert c.gradient is not None

            for ev in M.events:
                fields = ev.field if isinstance(ev.field, list) else [ev.field]
                for field in fields:
                    if getattr(field, "C", None) is not None:
                        _checkCSet(field.C, dims=ndim, prop=False)
                        if f in (steno, hutton, playfair, lehmann):
                            assert field.C.eq is not None
                            assert all(len(t) >= 2 for t in field.C.eq)
            if M.C is not None:
                _checkCSet(M.C, dims=ndim, val=False, grad=False, ori=False)
            assert len(xy) == np.prod(M.grid.shape)
            assert len(g.scalar) == np.prod(M.grid.shape)

            k, counts = np.unique(g.structureID, return_counts=True)
            assert len(g.structureLookup) == np.sum(counts > 10)
            for _k, _c in zip(k, counts):
                if _c > 10:
                    assert _k in g.structureLookup

        for sid, ename in g.structureLookup.items():
            fname = M.eidLookup[sid].getField(0).name
            mask = (g.structureID == sid)
            assert np.percentile(np.abs(g.scalar[mask] - g.fields[fname][mask]), 99) < 1e-6

            g.stackValues(mn=0, mx=1)


def test_count_interior_extrema_26():
    from curlew.synthetic import count_interior_extrema

    peak = np.zeros((5, 5, 5))
    peak[2, 2, 2] = 1.0
    assert count_interior_extrema(peak) == (1, 0)

    pit = np.zeros((5, 5, 5))
    pit[2, 2, 2] = -1.0
    assert count_interior_extrema(pit) == (0, 1)

    # Higher than all six faces, lower than a corner: not a 26-max.
    saddle = np.zeros((5, 5, 5))
    saddle[2, 2, 2] = 1.0
    saddle[1, 1, 1] = 2.0
    n_max, n_min = count_interior_extrema(saddle)
    assert n_max == 1  # the corner voxel
    assert n_min == 0

    x = np.linspace(-1.0, 1.0, 9)
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    assert count_interior_extrema(xx) == (0, 0)


def test_recumbent_mushroom_opposing_limbs(tmp_path):
    """
    Periclinal recumbent fold: through-going truth, inverted + upright limbs
    of the same horizon (the GemPy-bubble bait).
    """
    import torch
    from curlew.synthetic import (
        count_interior_extrema,
        recumbent_mushroom,
        write_gempy_tables,
    )

    curlew.default_dim = 3
    curlew.device = "cpu"
    curlew.dtype = torch.float64
    ds = recumbent_mushroom(shape=(48, 40, 36), n_steps=16, seed=0)
    assert ds.n_inverted >= 8
    assert ds.n_upright > ds.n_inverted
    both = False
    for name in np.unique(ds.orientations["formation"]):
        pol = ds.orientations["polarity"][ds.orientations["formation"] == name]
        if (pol < 0).any() and (pol > 0).any():
            both = True
            break
    assert both, "at least one horizon must show inverted and upright younging"
    n_max, n_min = count_interior_extrema(ds.phi)
    assert (n_max, n_min) == (0, 0)
    assert ds.C.eq is not None and len(ds.C.eq) >= 2
    assert ds.C.gp.shape[0] == len(ds.interfaces)

    out = write_gempy_tables(ds, tmp_path)
    assert out[0].exists() and out[1].exists()

def test_recumbent_mushroom_two_vortices():
    """Two vortices: one fold each, still through-going; one vortex is unchanged."""
    import torch
    from curlew.synthetic import EXTENT_3D, count_interior_extrema, recumbent_mushroom

    curlew.default_dim = 3
    curlew.device = "cpu"
    curlew.dtype = torch.float64
    E = np.asarray(EXTENT_3D, dtype=float)
    centres = np.array([[0.34, 0.42, 0.45], [0.66, 0.58, 0.45]]) * E
    sigma = np.array([0.11, 0.10, 0.20]) * E
    ds = recumbent_mushroom(shape=(48, 40, 36), n_steps=16, centre=centres, sigma=sigma,
                            overturn=1.35 * np.pi)
    assert ds.centres.shape == (2, 3) and ds.sigmas.shape == (2, 3) and len(ds.overturns) == 2
    assert count_interior_extrema(ds.phi) == (0, 0)
    inv = ds.orientations["polarity"] < 0
    x = ds.orientations["X"]
    for cx in centres[:, 0]:                     # each fold has overturned samples
        assert np.any(inv & (np.abs(x - cx) <= 2.0 * sigma[0]))

    one = recumbent_mushroom(shape=(24, 20, 18), n_steps=8, overturn=1.2 * np.pi)
    same = recumbent_mushroom(shape=(24, 20, 18), n_steps=8, overturn=[1.2 * np.pi],
                              centre=[one.centre], sigma=[one.sigma])
    assert np.allclose(one.phi, same.phi)
    assert one.centres.shape == (1, 3) and isinstance(one.overturn, float)
