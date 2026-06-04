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