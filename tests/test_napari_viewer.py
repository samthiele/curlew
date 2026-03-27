"""
Tests for ``image_affine_from_grid`` (no napari GUI required).
"""
from __future__ import annotations

import numpy as np
import pytest

from curlew.geometry import Grid
from curlew.core import CSet, Geode
from curlew.utils.napari_viewer import (
    NapariViewer,
    image_affine_from_grid,
    image_affine_napari_row_column_image_2d,
)


@pytest.fixture(params=[2, 3])
def napari_viewer(request):
    """Create a NapariViewer using napari's non-GUI ViewerModel (headless-safe)."""
    napari = pytest.importorskip("napari")
    _ = napari
    from napari.components import ViewerModel

    viewer = ViewerModel(title="test")
    yield NapariViewer(title="test", ndisplay=int(request.param), viewer=viewer)


def _xyz(n: int, ndim: int) -> np.ndarray:
    if ndim == 2:
        return np.array([[0.0, 0.0], [1.0, 2.0], [-0.5, 0.3]], dtype=np.float64)[:n]
    return np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [-0.5, 0.3, 0.2]], dtype=np.float64)[:n]


def test_add_points_2d_and_3d(napari_viewer):
    nd = napari_viewer._ndisplay
    layer = napari_viewer.addPoints("pts", _xyz(3, nd), rgb="red", size=5.0)
    assert layer is not None
    assert "pts" in napari_viewer._layers


def test_add_vectors_2d_and_3d(napari_viewer):
    nd = napari_viewer._ndisplay
    x0 = _xyz(2, nd)
    d = np.ones_like(x0) * 0.1
    layer = napari_viewer.addVectors("vec", x0, d, rgb="green", length=2.0, width=1.0)
    assert layer is not None
    assert "vec" in napari_viewer._layers


def test_add_paths_2d_and_3d(napari_viewer):
    nd = napari_viewer._ndisplay
    p = _xyz(3, nd)
    layer = napari_viewer.addPaths("paths", [p], edge_color="white", edge_width=2.0)
    assert layer is not None
    assert "paths" in napari_viewer._layers


def test_add_lines_2d_and_3d(napari_viewer):
    nd = napari_viewer._ndisplay
    a = _xyz(2, nd)
    seg = np.asarray([a[0], a[1]], dtype=np.float64)
    layer = napari_viewer.addLines("lines", [seg], edge_color="yellow", edge_width=1.5)
    assert layer is not None
    assert "lines" in napari_viewer._layers


def test_add_mesh_2d_and_3d(napari_viewer):
    verts = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    layer = napari_viewer.addMesh("mesh", verts, faces, rgb="white", shading="flat")
    assert layer is not None
    assert "mesh" in napari_viewer._layers


def test_add_volume_with_grid_2d_and_3d(napari_viewer):
    nd = napari_viewer._ndisplay
    if nd == 2:
        vol = np.random.default_rng(0).random((6, 5))
        layer = napari_viewer.addVolume("vol", vol, scale=(2.0, 1.0), rendering="iso")
    else:
        vol = np.random.default_rng(0).random((6, 5, 4))
        layer = napari_viewer.addVolume("vol", vol, scale=(3.0, 2.0, 1.0), rendering="iso")
    assert layer is not None
    assert "vol" in napari_viewer._layers


def test_add_drillhole_2d_and_3d(napari_viewer):
    from types import SimpleNamespace

    nd = napari_viewer._ndisplay
    x = _xyz(4, nd)
    litho = np.array([0, 1, 2, 3], dtype=np.float64)
    hole = SimpleNamespace(x=x, lithoID=litho)
    layer = napari_viewer.addDrillHole("dh", hole, vmn=0.0, vmx=3.0, linewidth=2.0)
    assert layer is not None
    assert "dh" in napari_viewer._layers


def test_add_geode_points_mode_2d_and_3d(napari_viewer):
    nd = napari_viewer._ndisplay
    x = _xyz(5, nd)
    litho = np.arange(x.shape[0], dtype=np.float64)
    scalar = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float64)
    g = Geode(x=x, grid=None, lithoID=litho, scalar=scalar, fields={})
    layers = napari_viewer.addGeode(
        g,
        lithoID=True,
        scalar=True,
        surfaces=False,
        displacement=False,
    )
    assert isinstance(layers, dict)
    assert "lithoID" in layers or "scalar" in layers


def test_add_cset_2d_and_3d(napari_viewer):
    nd = napari_viewer._ndisplay
    vp = _xyz(3, nd)
    vv = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)

    gp = _xyz(2, nd)
    gv = np.ones_like(gp) * 0.01

    gop = _xyz(2, nd)
    gov = np.ones_like(gop) * 0.02

    # inequalities: one '=', one '>'
    p1_eq = _xyz(2, nd)
    p2_eq = _xyz(2, nd) + 0.1
    p1_gt = _xyz(2, nd) + 0.2
    p2_gt = _xyz(2, nd) - 0.2
    iq = (2, [(p1_eq, p2_eq, "="), (p1_gt, p2_gt, ">")])

    C = CSet(vp=vp, vv=vv, gp=gp, gv=gv, gop=gop, gov=gov, iq=iq)
    layers = napari_viewer.addCSet(
        "constraints",
        C,
        bedding_length=1.2,
        bedding_width=2.0,
        grad_length=2.0,
        grad_width=1.0,
        orient_length=2.0,
        orient_width=1.0,
        value_size=5.0,
        iq_size=4.0,
    )
    assert isinstance(layers, dict)
    # iq layers are hidden by default; just ensure they were created
    assert any(k.startswith("constraints_iq") for k in layers.keys()) or ("constraints_eq" in layers)
