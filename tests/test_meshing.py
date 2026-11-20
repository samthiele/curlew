"""Octree dual contouring: planes, spheres, and GeoEvent.mesh."""
import numpy as np
import torch
import curlew
from curlew.fields.analytical import EllipsoidalField, LinearField
from curlew.geometry.meshing import dual_contour


def _linear_oracle(origin, gradient):
    field = LinearField(
        name="plane", input_dim=len(origin), origin=origin, gradient=gradient,
    )

    def value_and_grad(x):
        g, v = field.gradient(
            x, normalize=False, return_value=True, accumulate=False,
        )
        return v, g

    return value_and_grad


def _sphere_oracle(dim=3):
    field = EllipsoidalField(name="sphere", input_dim=dim)

    def value_and_grad(x):
        g, v = field.gradient(
            x, normalize=False, return_value=True, accumulate=False,
        )
        return v, g

    return value_and_grad


def _counting(fn):
    calls = {"n": 0}

    def wrapped(x):
        calls["n"] += int(x.shape[0])
        return fn(x)

    wrapped.calls = calls
    return wrapped


def test_plane_3d_vertices_lie_on_the_plane():
    curlew.default_dim = 3
    vg = _linear_oracle([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    verts, faces = dual_contour(vg, bounds, iso=0.0, max_depth=5)
    assert len(faces) > 0 and faces.shape[1] == 3
    assert verts.shape[1] == 3
    assert np.abs(verts[:, 2]).max() < 1e-8
    # Open rectangle filling the x–y extent of the box.
    assert verts[:, 0].min() < -0.8 and verts[:, 0].max() > 0.8
    assert verts[:, 1].min() < -0.8 and verts[:, 1].max() > 0.8


def test_plane_2d_segments_lie_on_the_line():
    curlew.default_dim = 2
    vg = _linear_oracle([0.0, 0.0], [0.0, 1.0])
    bounds = np.array([[-1.0, -1.0], [1.0, 1.0]])
    verts, edges = dual_contour(vg, bounds, iso=0.0, max_depth=5)
    assert len(edges) > 0 and edges.shape[1] == 2
    assert np.abs(verts[:, 1]).max() < 1e-8
    assert verts[:, 0].min() < -0.8 and verts[:, 0].max() > 0.8


def test_sphere_is_closed_and_on_the_isosurface():
    curlew.default_dim = 3
    vg = _sphere_oracle(3)
    bounds = np.array([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
    verts, faces = dual_contour(vg, bounds, iso=1.0, max_depth=5)
    r = np.linalg.norm(verts, axis=1)
    # Hermite QEF on a sphere sits on the surface well inside a cell.
    assert np.abs(r - 1.0).max() < 0.05
    # Closed manifold: every edge shared by two triangles, Euler χ = 2.
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()
    e = len(counts)
    v = len(verts)
    f = len(faces)
    assert v - e + f == 2


def test_empty_iso_returns_no_mesh():
    curlew.default_dim = 3
    vg = _linear_oracle([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    verts, faces = dual_contour(vg, bounds, iso=10.0, max_depth=4)
    assert len(verts) == 0 and len(faces) == 0


def test_adaptive_samples_far_fewer_than_a_dense_grid():
    curlew.default_dim = 3
    vg = _counting(_linear_oracle([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    depth = 6
    dual_contour(vg, bounds, iso=0.0, max_depth=depth)
    dense = (2 ** depth + 1) ** 3
    assert vg.calls["n"] < dense / 4, (vg.calls["n"], dense)


def test_geoevent_mesh_plane():
    from curlew.geology import strati

    curlew.default_dim = 3
    field = LinearField(
        name="s", input_dim=3, origin=[0.0, 0.0, 0.0], gradient=[0.0, 0.0, 1.0],
    )
    ev = strati("s", C=field, base=-np.inf)
    ev.addIsosurface("contact", value=0.0)
    verts, faces = ev.mesh(
        "contact",
        bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        max_depth=4,
    )
    assert len(faces) > 0
    assert np.abs(verts[:, 2]).max() < 1e-7
    by_value, faces_value = ev.mesh(
        0.0,
        bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        max_depth=4,
    )
    by_kw, faces_kw = ev.mesh(
        bounds=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]),
        iso=0.0,
        max_depth=4,
    )
    assert len(faces_value) > 0 and len(faces_kw) > 0
    assert np.abs(by_value[:, 2]).max() < 1e-7
    assert np.abs(by_kw[:, 2]).max() < 1e-7
