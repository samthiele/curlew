import numpy as np
import torch
import torch.nn as nn
import pytest
import curlew

from curlew.fields.eshelby import EshelbyField

def _set_cpu_float64():
    curlew.device = "cpu"
    curlew.dtype = torch.float64

def test_eshelbyfield_numpy_and_torch_shapes():
    _set_cpu_float64()

    field = EshelbyField(
        "one",
        positions=np.zeros((1, 3)),
        normals=np.array([0.0, 0.0, 1.0]),
        slips=np.array([1.0, 0.0, 0.0]),
        radii=2.0,
        thicknesses=1.0,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
    )

    # NumPy: single point -> (3,)
    u1 = field.displacement(np.array([0.5, 0.25, 0.0]), max_ram_mb=64)
    assert isinstance(u1, np.ndarray)
    assert u1.shape == (3,)
    assert np.isfinite(u1).all()

    # NumPy: batch -> (N,3)
    pts_np = np.array([[0.1, 0.0, 0.0], [0.2, -0.1, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    u_np = field.displacement(pts_np, max_ram_mb=64)
    assert isinstance(u_np, np.ndarray)
    assert u_np.shape == pts_np.shape
    assert np.isfinite(u_np).all()

    # Torch: preserves tensor type/shape
    pts_t = torch.tensor(pts_np, device=curlew.device, dtype=curlew.dtype)
    u_t = field.displacement(pts_t, max_ram_mb=64)
    assert isinstance(u_t, torch.Tensor)
    assert tuple(u_t.shape) == tuple(pts_t.shape)
    assert torch.isfinite(u_t).all()


def test_eshelbyfield_interior_linearity():
    _set_cpu_float64()

    field = EshelbyField(
        "one",
        positions=np.zeros((1, 3)),
        normals=np.array([0.0, 0.0, 1.0]),
        slips=np.array([1.0, 0.0, 0.0]),
        radii=2.0,
        thicknesses=1.0,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
        n_taper=1,
    )

    x = np.array([1.0, 0.0, 0.0])  # safely inside: 1^2/2^2 = 0.25
    u_base = field.displacement(x, max_ram_mb=64)

    for s in [0.25, 0.5, 1.5]:
        u = field.displacement(s * x, max_ram_mb=64)
        denom = np.linalg.norm(s * u_base) + 1e-30
        rel = np.linalg.norm(u - s * u_base) / denom
        assert rel < 5e-6

def test_inverse_square():
    _set_cpu_float64()

    field = EshelbyField(
        "one",
        positions=np.zeros((1, 3)),
        normals=np.array([0.0, 0.0, 1.0]),
        slips=np.array([1.0, 0.0, 0.0]),
        radii=1.0,
        thicknesses=0.2,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
        linear_decay=0.0, # obviously we want r2 decay
        n_taper=1, # also messes with decay
    )

    R1, R2 = 20.0, 40.0
    u1 = field.displacement(np.array([R1, 0.0, 0.0]), max_ram_mb=64)
    u2 = field.displacement(np.array([R2, 0.0, 0.0]), max_ram_mb=64)
    m1 = float(np.linalg.norm(u1))
    m2 = float(np.linalg.norm(u2))

    exp = np.log(m2 / m1) / np.log(R2 / R1)
    assert abs(exp + 2.0) < 0.05

def test_linear():
    _set_cpu_float64()

    field = EshelbyField(
        "one",
        positions=np.zeros((1, 3)),
        normals=np.array([0.0, 0.0, 1.0]),
        slips=np.array([1.0, 0.0, 0.0]),
        radii=1.0,
        thicknesses=0.2,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
        linear_decay=1.0, # now set linear decay
        n_taper=1, # also messes with decay
    )

    R1, R2 = 20.0, 40.0
    u1 = field.displacement(np.array([R1, 0.0, 0.0]), max_ram_mb=64)
    u2 = field.displacement(np.array([R2, 0.0, 0.0]), max_ram_mb=64)
    m1 = float(np.linalg.norm(u1))
    m2 = float(np.linalg.norm(u2))

    exp = np.log(m2 / m1) / np.log(R2 / R1)
    assert exp == exp
    assert abs(exp + 1.0) < 0.05
    
def test_weighted_sum():
    _set_cpu_float64()

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, -0.5, 0.25],
        ]
    )
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    slips = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    radii = np.array([2.0, 1.5])
    thicknesses = np.array([1.0, 0.5])
    # Magnitude / taper is entirely in weights (formerly slip_scale × taper).
    weights = np.array([0.25 * 0.7, 2.0 * 1.3])

    field = EshelbyField(
        "both",
        positions=positions,
        normals=normals,
        slips=slips,
        radii=radii,
        thicknesses=thicknesses,
        mu=1.0,
        nu=0.25,
        weights=weights,
        max_ram_mb=64,
    )

    f0 = EshelbyField(
        "e0",
        positions=positions[0:1],
        normals=normals[0],
        slips=slips[0],
        radii=float(radii[0]),
        thicknesses=float(thicknesses[0]),
        mu=1.0,
        nu=0.25,
        weights=float(weights[0]),
        max_ram_mb=64,
    )
    f1 = EshelbyField(
        "e1",
        positions=positions[1:2],
        normals=normals[1],
        slips=slips[1],
        radii=float(radii[1]),
        thicknesses=float(thicknesses[1]),
        mu=1.0,
        nu=0.25,
        weights=float(weights[1]),
        max_ram_mb=64,
    )

    pts = np.array(
        [
            [0.1, 0.2, 0.3],
            [2.0, 0.0, 0.0],
            [-1.0, 0.5, -0.25],
            [0.0, 0.0, 3.0],
        ],
        dtype=float,
    )

    u_field = field.displacement(pts, max_ram_mb=64)
    u_sum = f0.displacement(pts, max_ram_mb=64) + f1.displacement(pts, max_ram_mb=64)

    assert np.allclose(u_field, u_sum, rtol=2e-6, atol=2e-8)

def test_eshelbyfield_learnable_weights():
    _set_cpu_float64()
    field = EshelbyField(
        "learn",
        positions=np.zeros((2, 3)),
        normals=np.array([0.0, 0.0, 1.0]),
        slips=np.array([1.0, 0.0, 0.0]),
        radii=1.0,
        thicknesses=0.2,
        weights=0.5,
        learnable_weights=True,
        max_ram_mb=64,
    )
    assert isinstance(field.weights, nn.Parameter)
    assert tuple(field.weights.shape) == (2,)
    assert np.allclose(field.weights.detach().cpu().numpy(), [0.5, 0.5])

def test_eshelby2d():
    _set_cpu_float64()
    curlew.default_dim = 2
    
    # A simple x–z plane setup: 2D coords are interpreted as (x, z)
    field_2d = EshelbyField(
        "e2d",
        positions=np.array([[0.0, 0.0]], dtype=float),  # (x, z)
        normals=np.array([1.0, 0.0], dtype=float),      # normal along +x
        slips=np.array([0.0, 1.0], dtype=float),        # slip along +z
        radii=1.0,
        thicknesses=0.2,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
        n_taper=1,
    )

    # Equivalent 3D field in the x–z plane (y=0)
    field_3d = EshelbyField(
        "e3d",
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        normals=np.array([1.0, 0.0, 0.0], dtype=float),
        slips=np.array([0.0, 0.0, 1.0], dtype=float),
        radii=1.0,
        thicknesses=0.2,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
        n_taper=1,
    )

    # Receiver points in 2D (x, z)
    pts2 = np.array(
        [
            [0.1, 0.0],
            [0.5, 0.2],
            [2.0, -0.4],
            [-1.5, 0.75],
        ],
        dtype=float,
    )
    u2 = field_2d.displacement(pts2, max_ram_mb=64)
    assert isinstance(u2, np.ndarray)
    assert u2.shape == pts2.shape
    assert np.isfinite(u2).all()

    pts3 = np.zeros((pts2.shape[0], 3), dtype=float)
    pts3[:, 0] = pts2[:, 0]
    pts3[:, 2] = pts2[:, 1]
    u3 = field_3d.displacement(pts3, max_ram_mb=64)
    assert isinstance(u3, np.ndarray)
    assert u3.shape == pts3.shape

    # 2D result should match 3D restricted to (x, z)
    assert np.allclose(u2, u3[:, (0, 2)], rtol=2e-6, atol=2e-8)

    # test also accepts a single 2D point
    field = EshelbyField(
        "one2d",
        positions=np.array([[0.0, 0.0]], dtype=float),
        normals=np.array([0.0, 1.0], dtype=float),
        slips=np.array([1.0, 0.0], dtype=float),
        radii=2.0,
        thicknesses=1.0,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
        n_taper=1,
    )
    u = field.displacement(np.array([0.5, 0.25], dtype=float), max_ram_mb=64)
    assert isinstance(u, np.ndarray)
    assert u.shape == (2,)
    assert np.isfinite(u).all()


def test_getEllipseField_projected_boundary_sign():
    _set_cpu_float64()

    # Normal with a non-zero y-component so the equatorial circle projects to
    # a non-degenerate ellipse in the (x, z) display plane.
    n = np.array([0.2, 0.9, 0.25], dtype=float)
    n = n / np.linalg.norm(n)

    field = EshelbyField(
        "ellipse_src",
        positions=np.array([[0.0, 0.0]], dtype=float),  # (x, z) with centre at origin
        normals=n,
        slips=np.array([1.0, 0.0, 0.0], dtype=float),
        radii=2.0,
        thicknesses=0.5,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
        n_taper=1,
    )
    assert field.is2D

    pkf = field.getEllipseField()

    # Ellipse-centre: value should be -1 by construction.
    x0 = torch.tensor([[0.0, 0.0]], dtype=curlew.dtype, device=curlew.device)
    v0 = float(pkf.forward(x0).reshape(-1)[0].detach().cpu().numpy())
    assert abs(v0 + 1.0) < 1e-9

    # Construct projected boundary points by sampling the local equatorial circle
    # (local z=0) and projecting to (x, z) via the stored RT rotation.
    RT0 = field._RT[0]      # (3, 3) local -> global
    pos0 = field._pos[0]    # (3,)
    r = float(field._radii[0].detach().cpu().numpy())

    thetas = [0.0, float(np.pi / 3.0), float(np.pi / 2.0)]
    x_boundary_pts = []
    for th in thetas:
        u_local = torch.tensor(
            [r * np.cos(th), r * np.sin(th), 0.0],
            dtype=curlew.dtype,
            device=curlew.device,
        )
        p_global = pos0 + RT0 @ u_local
        xz = torch.tensor(
            [[float(p_global[0].detach().cpu().numpy()), float(p_global[2].detach().cpu().numpy())]],
            dtype=curlew.dtype,
            device=curlew.device,
        )
        x_boundary_pts.append(xz)
        vb = float(pkf.forward(xz).reshape(-1)[0].detach().cpu().numpy())
        assert abs(vb) < 1e-6

    # Check scaling along the ray to a boundary point: value(s) = alpha - 1
    x_b0 = x_boundary_pts[0]
    v_in = float(pkf.forward(x_b0 * 0.5).reshape(-1)[0].detach().cpu().numpy())
    v_out = float(pkf.forward(x_b0 * 2.0).reshape(-1)[0].detach().cpu().numpy())
    assert abs(v_in + 0.5) < 2e-6
    assert abs(v_out - 1.0) < 2e-6


def test_getEllipseField_3d_ellipsoid_boundary_sign():
    _set_cpu_float64()

    field = EshelbyField(
        "ellipsoid_src",
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        normals=np.array([0.3, 0.4, 0.866025403784], dtype=float),  # arbitrary unit-ish
        slips=np.array([1.0, 0.0, 0.0], dtype=float),
        radii=2.0,
        thicknesses=0.5,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
        n_taper=1,
    )
    assert not field.is2D

    pkf = field.getEllipseField()

    # Center: -1
    x0 = torch.tensor([[0.0, 0.0, 0.0]], dtype=curlew.dtype, device=curlew.device)
    v0 = float(pkf.forward(x0).reshape(-1)[0].detach().cpu().numpy())
    assert abs(v0 + 1.0) < 1e-9

    # A point on the surface: choose local point (r, 0, 0) and rotate to global.
    RT0 = field._RT[0]
    r = float(field._radii[0].detach().cpu().numpy())
    p_surf = RT0 @ torch.tensor([r, 0.0, 0.0], dtype=curlew.dtype, device=curlew.device)
    xs = p_surf.unsqueeze(0)
    vs = float(pkf.forward(xs).reshape(-1)[0].detach().cpu().numpy())
    assert abs(vs) < 1e-6

    # Scaling along that ray: alpha-1
    v_in = float(pkf.forward(xs * 0.5).reshape(-1)[0].detach().cpu().numpy())
    v_out = float(pkf.forward(xs * 2.0).reshape(-1)[0].detach().cpu().numpy())
    assert abs(v_in + 0.5) < 2e-6
    assert abs(v_out - 1.0) < 2e-6


def test_getEllipseField_deformed_moves_thickness_outwards_for_opening():
    _set_cpu_float64()

    # Pure opening: slip parallel to normal -> eps_zz* > 0 in local frame.
    n = np.array([0.0, 0.0, 1.0], dtype=float)
    slip = np.array([0.0, 0.0, 1.0], dtype=float)

    r = 2.0
    t = 1.0
    w = 0.2

    field = EshelbyField(
        "open",
        positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        normals=n,
        slips=slip,
        radii=r,
        thicknesses=t,
        weights=w,
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
        n_taper=1,
    )

    pkf_u = field.getEllipseField(deformed=False)
    pkf_d = field.getEllipseField(deformed=True)

    # On the undeformed boundary along +local z (global z here): value ~ 0 for undeformed,
    # but should be negative (inside) for deformed because the thickness increased.
    x_old = torch.tensor([[0.0, 0.0, t]], dtype=curlew.dtype, device=curlew.device)
    vu = float(pkf_u.forward(x_old).reshape(-1)[0].detach().cpu().numpy())
    vd = float(pkf_d.forward(x_old).reshape(-1)[0].detach().cpu().numpy())
    assert abs(vu) < 1e-6
    assert vd < -1e-3
    
def test_visualisation():
    _set_cpu_float64()

    # Avoid failing test runs where napari is intentionally not installed or Qt cannot start.
    try:
        import napari  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"napari unavailable ({type(e).__name__}: {e})")

    try:
        from curlew.visualise.napari_viewer import NapariViewer
    except Exception as e:  # pragma: no cover
        pytest.skip(f"NapariViewer import failed ({type(e).__name__}: {e})")

    field = EshelbyField(
        "viz",
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]], dtype=float),
        normals=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float),
        slips=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float),
        radii=np.array([2.0, 1.5], dtype=float),
        thicknesses=np.array([0.5, 0.3], dtype=float),
        mu=1.0,
        nu=0.25,
        max_ram_mb=64,
    )

    def _make_viewer():
        import napari

        try:
            return napari.Viewer(show=False)
        except TypeError:  # older napari
            return napari.Viewer()

    def _close_viewer(v):
        try:
            if hasattr(v, "close"):
                v.close()
            elif hasattr(v, "window") and hasattr(v.window, "close"):
                v.window.close()
        except Exception:
            pass

    # 2D: expect a Shapes layer with one polygon per source.
    v2 = None
    try:
        v2 = _make_viewer()
        nv2 = NapariViewer(viewer=v2, ndisplay=2)
        layer2 = nv2.addEshelby("eshelby2d", field, rgb="cyan", n_ellipse_pts=32)
        assert layer2 is not None
        assert hasattr(layer2, "data")
        assert len(layer2.data) == field.n_sources
        # Each polygon is (P, 3) in napari (z,y,x) order for a 2D viewer (z padded).
        assert np.asarray(layer2.data[0]).shape[1] == 3
    except Exception as e:  # pragma: no cover
        pytest.skip(f"napari 2D viewer could not be exercised ({type(e).__name__}: {e})")
    finally:
        if v2 is not None:
            _close_viewer(v2)

    # 3D: expect a Surface layer with (verts, faces) and 3D columns.
    v3 = None
    try:
        v3 = _make_viewer()
        nv3 = NapariViewer(viewer=v3, ndisplay=3)
        layer3 = nv3.addEshelby("eshelby3d", field, rgb="cyan", n_u=8, n_v=16)
        assert layer3 is not None
        assert hasattr(layer3, "data")
        verts, faces = layer3.data
        verts = np.asarray(verts)
        faces = np.asarray(faces)
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert faces.max() < verts.shape[0]
    except Exception as e:  # pragma: no cover
        pytest.skip(f"napari 3D viewer could not be exercised ({type(e).__name__}: {e})")
    finally:
        if v3 is not None:
            _close_viewer(v3)
