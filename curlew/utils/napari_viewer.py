"""
2D and 3D visualisation using Napari, with an add-layer API similar in spirit to
`curlew.utils.datascreen.DataScreen` (real-world coordinates).

Requires `napari` (and a Qt backend) to be installed separately.

The affine helpers ``image_affine_from_grid`` and ``image_affine_napari_row_column_image_2d`` are implemented here.
:func:`image_affine_from_grid` etc. (now implemented directly here.)
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import curlew


if TYPE_CHECKING:
    from curlew.core import CSet, Geode
    from curlew.fields.eshelby import EshelbyField
    from matplotlib.colors import Colormap as MplColormap

# try loading napari
try:
    import napari
except ImportError as e:  # pragma: no cover - optional dependency
    assert False, "Please install napari (`pip install napari[all]`) to use napariViewer"


def resolve_cmap(cmap=None):
    """
    Resolve a colormap for Napari layers or for Matplotlib-only use (e.g. ``Normalize``).

    * ``None`` — use :attr:`curlew.ccramp` when available; otherwise matplotlib ``viridis``.
    * ``str`` — matplotlib registered name.
    * :class:`matplotlib.colors.Colormap`;
      otherwise converted to a napari :class:`napari.utils.colormaps.Colormap`.
    """
    from matplotlib import colormaps as mpl_colormaps
    from matplotlib.colors import Colormap as MplColormap

    if cmap is not None:
        from napari.utils.colormaps import Colormap as NapariColormap
        if isinstance(cmap, NapariColormap):
            return cmap
    if isinstance(cmap, str):
        mpl_c = mpl_colormaps[cmap]
    elif isinstance(cmap, MplColormap):
        mpl_c = cmap
    elif cmap is None:
        mpl_c = curlew.ccramp if curlew.ccramp is not None else mpl_colormaps["viridis"]
    else:
        raise TypeError(
            "cmap must be None, str, or matplotlib.colors.Colormap"
            + (" or napari Colormap"
            + f"; got {type(cmap).__name__}"
        ))

    from napari.utils.colormaps import Colormap as NapariColormap

    colors = mpl_c(np.linspace(0, 1, 256))
    cname = getattr(mpl_c, "name", None) or "curlew_colormap"
    return NapariColormap(colors, name=str(cname))

def image_affine_from_grid(grid : curlew.geometry.Grid) -> np.ndarray:
    """
    Compute the napari affine matrix needed to align a 2D or 3D image with curlew world coordinates defined
    in a `curlew.geometry.Grid` instance.
    """
    if grid.ndim not in (2, 3):
        raise ValueError( f"image_affine_from_grid expects a 2D or 3D grid; got ndim={grid.ndim}" )
    
    M = np.asarray(grid.matrix, dtype=np.float64)

    if grid.ndim == 3:
        sx, sy, sz = (float(s) for s in grid.step)
        dx, dy, dz = (float(d) for d in grid.dims)
        index_to_local = np.array(
            [
                [sx, 0.0, 0.0, -dx / 2.0],
                [0.0, sy, 0.0, -dy / 2.0],
                [0.0, 0.0, sz, -dz / 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        if M.shape != (4, 4):
            raise ValueError(
                f"Expected grid.matrix shape (4, 4) for 3D grid; got {M.shape}"
            )
        return M @ index_to_local

    sx, sy = (float(s) for s in grid.step)
    dx, dy = (float(d) for d in grid.dims)
    index_to_local = np.array(
        [
            [sx, 0.0, -dx / 2.0],
            [0.0, sy, -dy / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if M.shape != (3, 3):
        raise ValueError(
            f"Expected grid.matrix shape (3, 3) for 2D grid; got {M.shape}"
        )
    A3 = M @ index_to_local
    
    A4 = np.eye(4, dtype=np.float64)
    A4[0, 0] = A3[0, 0]
    A4[0, 1] = A3[0, 1]
    A4[1, 0] = A3[1, 0]
    A4[1, 1] = A3[1, 1]
    A4[0, 3] = A3[0, 2]
    A4[1, 3] = A3[1, 2]
    A4[2, 2] = 1.0
    return A4
    
def image_affine_napari_row_column_image_2d(grid) -> np.ndarray:
    """
    4×4 affine for a 2D Napari ``Image`` built like matplotlib ``plot2D``:

    * ``data = numpy.flipud(numpy.transpose(curlew_vol))`` with ``curlew_vol.shape == grid.shape``.

    Then ``A_nap @ [row, col, 0, 1] == A_cur @ [col, n₁ - 1 - row, 0, 1]`` where
    ``A_cur = image_affine_from_grid(grid)`` and ``grid.shape == (n₀, n₁)``.
    """
    for attr in ("matrix", "dims", "step", "ndim"):
        if not hasattr(grid, attr):
            raise TypeError(
                f"grid must provide .{attr} (e.g. a curlew.geometry.Grid instance)"
            )
    if grid.ndim != 2:
        raise ValueError(
            "image_affine_napari_row_column_image_2d expects a 2D grid; "
            f"got ndim={grid.ndim}"
        )
    _, n1 = grid.shape
    
    # flip
    nn = float(n1 - 1)
    nn = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, nn],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    
    return image_affine_from_grid(grid) @ nn

def _get_colors(n_shapes: int) -> np.ndarray:
    """RGBA per shape for Shapes layers: fully transparent faces (no polygon fill)."""
    t = np.zeros((n_shapes, 4), dtype=np.float32)
    t[:, 3] = 0.0
    return t

def _geode_ravel_values(data) -> np.ndarray:
    """Geode per-point array (numpy) → 1D (N,) for ``Grid.reshape`` or point scalars."""
    a = np.asarray(data, dtype=np.float64)
    if a.ndim == 0:
        return a.reshape(1)
    return a.ravel()

class NapariViewer:
    """
    Open a Napari viewer (2D or 3D) and add meshes, point clouds, images / volumes, and shapes.

    **Coordinate conventions (Curlew → Napari)**

    Napari uses dimension order **(z, y, x)** in layer data: slice/depth first, then row,
    then column. Curlew 2D points are **(x, y)** (first model axis, second). They are mapped
    to ``(0, y, x)``. Curlew 3D positions **(x, y, z)** map to ``(z, y, x)``. The padded
    slice coordinate is **always the first component**, not the last.

    **2D images:** Curlew rasters from :meth:`~curlew.geometry.Grid.reshape` are indexed
    ``(i₀, i₁)`` like **(x-axis, y-axis)**. Napari ``Image`` arrays are **row-major**
    ``(row, col)`` with ``row = i₁``, ``col = i₀``; pass ``volume.T`` and use
    :func:`~curlew.utils.grid_napari_affine.image_affine_napari_row_column_image_2d` — this
    is applied automatically when you call :meth:`addVolume` with a 2D ``grid``.

    Parameters
    ----------
    title : str, optional
        Viewer window title.
    ndisplay : int, optional
        ``2`` or ``3`` (default). Use ``2`` for planar models and ``matplotlib``-style sections.
    **viewer_kwds
        Extra keyword arguments passed to `napari.Viewer()`.
    """

    def __init__(
        self,
        title: str = "curlew",
        *,
        ndisplay: int = None,
        viewer=None,
        **viewer_kwds,
    ):
        if ndisplay is None:
            ndisplay = curlew.default_dim
        if ndisplay not in (2, 3):
            raise ValueError("ndisplay must be 2 or 3")

        # Configure Napari's 3D axis orientation preferences (when supported) so that
        # Curlew's convention matches the viewer defaults:
        # - depth axis: towards
        # - vertical axis: up
        # - horizontal axis: right
        try: # can differ between versions....
            from napari.settings import get_settings
            _settings = get_settings()
            _app = getattr(_settings, "application", None)
            if _app is not None:
                # Newer napari versions may expose a single tuple-valued field.
                if hasattr(_app, "camera_orientation"):
                    _app.camera_orientation = ("towards", "up", "right")
                # Fallbacks for split fields (name variants differ across releases).
                if hasattr(_app, "depth_axis_orientation"):
                    _app.depth_axis_orientation = "towards"
                if hasattr(_app, "vertical_axis_orientation"):
                    _app.vertical_axis_orientation = "up"
                if hasattr(_app, "horizontal_axis_orientation"):
                    _app.horizontal_axis_orientation = "right"
        except Exception:
            pass

        self.viewer = viewer if viewer is not None else napari.Viewer(title=title, **viewer_kwds)
        self._ndisplay = int(ndisplay)
        self.viewer.dims.ndisplay = self._ndisplay

        # Ensure the active viewer matches the requested orientation, independent of
        # whether global settings took effect.
        try:  # can differ between versions...
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "orientation"):
                self.viewer.camera.orientation = ("towards", "up", "right")
        except Exception:
            pass

        # Match Curlew's axis convention in the Napari GUI.
        # Napari's displayed axes are controlled by `viewer.axes`, while `dims.axis_labels`
        # controls the text shown next to the sliders.
        if hasattr(self.viewer, "axes"):
            self.viewer.axes.visible = True
        if hasattr(self.viewer, "scale_bar"):
            self.viewer.scale_bar.visible = True
        self._layers: dict[str, napari.layers.Layer] = {}
        self.viewer.dims.axis_labels = ("z", "y", "x") if self._ndisplay == 3 else ("y", "x")
        self._cnt = 0

        self._lighting = (self._ndisplay == 3)
        self._lighting_connected = False
        if self._lighting:
            self._connect_lighting()

    # Lighting control (3D only)
    def enable_light_tracking(self, enabled: bool = True) -> None:
        """Enable/disable aligning surface lighting with the camera direction."""
        self._lighting = bool(enabled)
        if self._lighting:
            self._connect_lighting()
            self._on_camera_change()
        else:
            self._disconnect_lighting_events()

    def _connect_lighting(self) -> None:
        if self._lighting_connected:
            return
        try:
            self.viewer.camera.events.angles.connect(self._on_camera_change)
            self._lighting_connected = True
        except Exception:
            # Headless ViewerModel (no vispy canvas) or older napari versions.
            self._lighting_connected = False

    def _disconnect_lighting_events(self) -> None:
        if not self._lighting_connected:
            return
        try:
            self.viewer.camera.events.angles.disconnect(self._on_camera_change)
        except Exception:
            pass # no woman, no cry!
        self._lighting_connected = False

    def _get_dims_displayed(self, layer) -> np.ndarray:
        """Best-effort equivalent of napari-threedee `get_dims_displayed`."""
        try:
            displayed = getattr(getattr(layer, "_slice_input", None), "displayed", None)
            if displayed is not None:
                return np.asarray(displayed, dtype=int)
        except Exception:
            pass
        try:
            ndim = int(getattr(layer, "ndim", 3))
        except Exception:
            ndim = 3
        return np.arange(ndim, dtype=int)

    def _get_napari_visual(self, layer):
        """
        Return the vispy visual for a layer when running with a Qt viewer.
        In headless ViewerModel this returns None.
        """
        viewer = self.viewer
        # ViewerModel has no .window; only napari.Viewer (Qt) does.
        win = getattr(viewer, "window", None)
        if win is None:
            return None

        # New/old attribute names across napari versions.
        qt_viewer = getattr(win, "_qt_viewer", None) or getattr(win, "qt_viewer", None)
        if qt_viewer is None:
            return None

        # Common mapping: layer_to_visual (dict-like).
        layer_to_visual = getattr(qt_viewer, "layer_to_visual", None)
        if layer_to_visual is not None:
            try:
                return layer_to_visual[layer]
            except Exception:
                pass

        # Fallback: try a callable accessor if present.
        get_visual = getattr(qt_viewer, "get_visual", None)
        if callable(get_visual):
            try:
                return get_visual(layer)
            except Exception:
                pass

        return None

    def _on_camera_change(self, event=None) -> None:
        if not self._lighting:
            return

        # If there's no camera/view_direction (or no Qt canvas), this should be a no-op.
        try:
            view_direction = np.asarray(self.viewer.camera.view_direction, dtype=np.float64)
        except Exception:
            return
        try:
            all_layers = list(self.viewer.layers)
        except Exception:
            return

        for layer in all_layers:
            if layer is None:
                continue
            try:
                import napari as _napari_mod

                if not isinstance(layer, _napari_mod.layers.Surface):
                    continue
            except Exception:
                if layer.__class__.__name__ != "Surface":
                    continue

            visual = self._get_napari_visual(layer)
            if visual is None:
                continue

            try:
                # Convert camera direction to this layer's data coords, then select displayed dims.
                ray = (
                    np.asarray(layer._world_to_data_ray(view_direction), dtype=np.float64)
                    if hasattr(layer, "_world_to_data_ray")
                    else view_direction
                )
                dims_displayed = self._get_dims_displayed(layer)
                layer_view_dir = np.asarray(ray, dtype=np.float64)[dims_displayed]

                node = getattr(visual, "node", None)
                shading_filter = getattr(node, "shading_filter", None) if node is not None else None
                if shading_filter is None or not hasattr(shading_filter, "light_dir"):
                    continue
                shading_filter.light_dir = layer_view_dir[::-1] # Napari/vispy use reversed xyz convention vs our (z,y,x) direction.
            except Exception:
                continue # Never let lighting updates break rendering or tests.

    def _remove_layer_if_present(self, name: str) -> None:
        # hack; but put this here as axis labels are not properly set in the constructor for some reason. 
        self.viewer.dims.axis_labels = ("z", "y", "x") if self._ndisplay == 3 else ("y", "x")
        
        # now check if the layer exists and remove if need be. 
        if name in self._layers:
            try:
                self.viewer.layers.remove(self._layers[name])
            except ValueError:
                pass
            del self._layers[name]

    def _to_napari_xyz(self, arr) -> np.ndarray:
        """
        Curlew positions or directions ``(x, y)`` / ``(x, y, z)`` → napari **(z, y, x)**.
        """
        a = np.asarray(arr, dtype=np.float64)
        if a.size == 0:
            return a.reshape(0, 3)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[1] == 2:
            # Pad to (z, y, x) with z=0 for 2D inputs.
            # n.b. napari defaults to y-axis-down, we want y-axis-up.
            return np.column_stack([np.zeros(a.shape[0]), a[:, 1], a[:, 0]])
        if a.shape[1] == 3:
            return np.column_stack([a[:, 2], a[:, 1], a[:, 0]])
        raise ValueError(
            f"Expected 2 or 3 columns (Curlew x,y or x,y,z); got shape {a.shape}"
        )

    def addMesh(
        self,
        name: str,
        verts: np.ndarray,
        faces: np.ndarray,
        normals: np.ndarray | None = None,
        rgb: str | np.ndarray = "green",
        shading: str = "smooth",
        *,
        curlew_coords: bool = True,
        **kwargs,
    ):
        """
        Add a triangular surface mesh.

        Parameters
        ----------
        name : str
            Layer name (replaces an existing layer with the same name).
        verts : array, shape (N, 3)
        faces : array, shape (M, 3)
            Triangle vertex indices.
        normals : array, optional
            Per-vertex normals (N, 3). Not passed to napari when incompatible with the
            installed napari version; use ``vertex_colors`` / shading only.
        rgb : str or array, optional
            Matplotlib colour name if `str`. If array, per-vertex colours with shape (N, 3) RGB
            or (N, 4) RGBA in [0, 1].
        shading : str, optional
            Passed to `viewer.add_surface` (e.g. ``'smooth'``, ``'flat'``).
        curlew_coords : bool, optional
            If True (default), treat ``verts`` / ``normals`` as Curlew **(x, y, z)** and map to
            napari **(z, y, x)**. If False, vertices are already in napari order.
        **kwargs
            Additional arguments for `viewer.viewer.add_surface`.
        """
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(faces)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("verts must have shape (N, 3)")
        if curlew_coords:
            verts = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]])

        if normals is not None:
            normals = np.asarray(normals, dtype=np.float64)
            if normals.ndim != 2 or normals.shape[1] != 3:
                raise ValueError("normals must have shape (N, 3)")
            if normals.shape[0] != verts.shape[0]:
                raise ValueError(
                    "normals shape mismatch: expected per-vertex normals with the same "
                    f"length as verts (got normals.shape[0]={normals.shape[0]} vs verts.shape[0]={verts.shape[0]})"
                )
            if curlew_coords:
                normals = np.column_stack(
                    [normals[:, 2], normals[:, 1], normals[:, 0]]
                )

        from matplotlib.colors import to_rgba

        if isinstance(rgb, str):
            rgba = np.asarray(to_rgba(rgb), dtype=np.float64)
            vertex_colors = np.tile(rgba, (len(verts), 1))
        else:
            vc = np.asarray(rgb, dtype=np.float64)
            # allow passing a single colour as (3,) or (4,)
            if vc.ndim == 1 and vc.shape[0] in (3, 4):
                rgba = vc if vc.shape[0] == 4 else np.r_[vc, 1.0]
                vertex_colors = np.tile(rgba, (len(verts), 1))
            else:
                if vc.ndim != 2 or vc.shape[0] != len(verts):
                    raise ValueError(
                        "rgb array must be a single (3,)/(4,) colour or have one row per vertex"
                    )
                if vc.shape[1] == 3:
                    vertex_colors = np.c_[vc, np.ones(len(verts))]
                elif vc.shape[1] == 4:
                    vertex_colors = vc
                else:
                    raise ValueError("rgb must be (3,), (4,), (N, 3) or (N, 4)")

        self._remove_layer_if_present(name)
        
        kwargs = dict(kwargs)
        kwargs.pop("normals", None)
        layer = self.viewer.add_surface(
            (verts, faces),
            name=name,
            vertex_colors=vertex_colors,
            shading=shading,
            **kwargs,
        )
        self._layers[name] = layer
        self._cnt += 1
        return layer

    def addPoints(
        self,
        name: str,
        xyz: np.ndarray,
        rgb: str | np.ndarray | None = None,
        scalar: np.ndarray | None = None,
        colormap=None,
        size: float = 10.0,
        border_color: str | None = None,
        symbol: str = "o",
        *,
        curlew_coords: bool = True,
        **kwargs,
    ):
        """
        Add a point cloud.

        By default ``xyz`` uses Curlew **(x, y)** or **(x, y, z)** — same as :class:`~curlew.core.Geode`
        ``x`` / constraint positions. Set ``curlew_coords=False`` only if you already permuted to
        napari **(z, y, x)**.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] not in (2, 3):
            raise ValueError("xyz must have shape (N, 2) or (N, 3)")
        if curlew_coords:
            xyz = self._to_napari_xyz(xyz)
        elif xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3) when curlew_coords=False")

        if scalar is not None and rgb is not None:
            raise ValueError("Pass either scalar= or rgb=, not both.")

        self._remove_layer_if_present(name)

        n = xyz.shape[0]

        layer_opts: dict = {
            "out_of_slice_display": True,
            "symbol": symbol,
        }
        if border_color is None:
            if "border_width" not in kwargs:
                layer_opts["border_width"] = 0
        else:
            layer_opts["border_color"] = border_color
            if "border_width" not in kwargs:
                layer_opts["border_width"] = 0.05

        if scalar is not None:
            v = np.asarray(scalar)
            if v.ndim != 1 or v.shape[0] != n:
                raise ValueError("scalar must be a 1D array of length N")
            feat_key = "_curlew_scalar_points"
            cmap = resolve_cmap(colormap)
            layer = self.viewer.add_points(
                xyz,
                name=name,
                features={feat_key: v},
                face_color=feat_key,
                face_colormap=cmap,
                size=size,
                **layer_opts,
                **kwargs,
            )
        elif rgb is None:
            layer = self.viewer.add_points(
                xyz,
                name=name,
                face_color="red",
                size=size,
                **layer_opts,
                **kwargs,
            )
        elif isinstance(rgb, str):
            layer = self.viewer.add_points(
                xyz,
                name=name,
                face_color=rgb,
                size=size,
                **layer_opts,
                **kwargs,
            )
        else:
            v = np.asarray(rgb)
            if v.ndim == 1 and v.shape[0] in (3, 4):
                rgba = v if v.shape[0] == 4 else np.r_[v, 1.0]
                face_color = np.tile(rgba, (n, 1))
                layer = self.viewer.add_points(
                    xyz,
                    name=name,
                    face_color=face_color,
                    size=size,
                    **layer_opts,
                    **kwargs,
                )
            elif v.ndim == 2 and v.shape[0] == n and v.shape[1] in (3, 4):
                layer = self.viewer.add_points(
                    xyz,
                    name=name,
                    face_color=v,
                    size=size,
                    **layer_opts,
                    **kwargs,
                )
            else:
                raise ValueError(
                    "rgb must be str, a single (3,)/(4,) colour, or an array of shape (N,3)/(N,4) for per-point colours"
                )

        self._layers[name] = layer
        self._cnt += 1
        return layer

    def addVolume(
        self,
        name: str,
        volume: np.ndarray,
        *,
        grid=None,
        scale=None,
        translate=None,
        affine=None,
        rendering: str = "iso",
        colormap=None,
        blending: str = "opaque",
        interpolation2d: str = "nearest",
        interpolation3d: str = "nearest",
        **kwargs,
    ):
        """
        Add a 2D image or 3D volume (scalar field on a regular grid).

        When ``grid`` is **2D**, ``volume`` must have shape ``grid.shape`` (Curlew
        ``(i₀, i₁)``). It is passed to napari as ``flipud(transpose(volume))`` so the
        vertical axis matches :func:`curlew.visualise.plot2D` (matplotlib
        ``imshow(..., origin='lower')``). Napari's default row order would otherwise flip
        the section upside down relative to drillholes and constraints. When ``grid`` is
        **3D**, ``volume`` must match ``grid.shape`` (Curlew ``(x, y, z)``); it is
        internally transposed to Napari axes ``(z, y, x)`` when an affine is inferred.

        For 2D map views, set ``ndisplay=2`` on the viewer.
        """
        volume = np.asarray(volume)
        if volume.ndim not in (2, 3):
            raise ValueError("volume must be a 2D or 3D array")

        if grid is not None and (scale is not None or translate is not None):
            raise ValueError(
                "addVolume: use grid=... for affine, or scale=/translate=, not both."
            )
        if grid is not None and grid.ndim != volume.ndim:
            raise ValueError(
                f"grid.ndim ({grid.ndim}) must match volume.ndim ({volume.ndim})"
            )

        colormap = resolve_cmap(colormap)

        affine_use = affine
        if affine_use is None and grid is not None:
            if grid.ndim == 2:
                volume = np.flipud(np.transpose(volume))
                affine_use = image_affine_napari_row_column_image_2d(grid)
            else:
                # Curlew grids are sampled in (x, y, z) order, while Napari image
                # axes (and the viewer coordinate convention we use for points/surfaces)
                # are (z, y, x). Transpose the volume data accordingly, and permute the
                # affine so voxel indices (z, y, x) map to world coordinates (z, y, x).
                volume = np.transpose(volume, (2, 1, 0))
                A_cur = image_affine_from_grid(grid)  # maps (x,y,z indices) -> (x,y,z world)
                # Swap axes 0 and 2 (x <-> z) in both input indices and output world coords.
                P = np.array(
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                affine_use = P @ A_cur @ P

        legacy_interp = kwargs.pop("interpolation", None)
        if legacy_interp is not None:
            import warnings

            warnings.warn(
                "napari add_image does not accept `interpolation=`; use "
                "`interpolation2d=` and `interpolation3d=` on addVolume instead. "
                f"Applying {legacy_interp!r} to both.",
                UserWarning,
                stacklevel=2,
            )
            interpolation2d = interpolation3d = legacy_interp

        opts: dict = {
            "name": name,
            "rendering": rendering,
            "colormap": colormap,
            "blending": blending,
            "interpolation2d": interpolation2d,
            "interpolation3d": interpolation3d,
        }
        if scale is not None:
            opts["scale"] = scale
        if translate is not None:
            opts["translate"] = translate
        if affine_use is not None:
            opts["affine"] = affine_use
        opts.update(kwargs)

        self._remove_layer_if_present(name)
        layer = self.viewer.add_image(volume, **opts)
        self._layers[name] = layer
        self._cnt += 1
        return layer

    def addVectors(
        self,
        name: str,
        origins: np.ndarray,
        directions: np.ndarray,
        *,
        rgb: str | np.ndarray | None = None,
        length: float = 1.0,
        width: float = 1,
        colormap=None,
        curlew_coords: bool = True,
        **kwargs,
    ):
        """
        Add a vector field as napari ``Vectors`` (2D and 3D viewers).

        By default ``origins`` and ``directions`` use Curlew **(x, y)** / **(x, y, z)**. Pass
        ``curlew_coords=False`` if they are already in napari **(z, y, x)**.
        """
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] not in (2, 3):
            raise ValueError("origins must have shape (N, 2) or (N, 3)")
        if directions.shape != origins.shape:
            raise ValueError("directions must have the same shape as origins")
        if curlew_coords:
            origins = self._to_napari_xyz(origins)
            directions = self._to_napari_xyz(directions)
        elif origins.shape[1] != 3:
            raise ValueError("origins must have shape (N, 3) when curlew_coords=False")
        n = origins.shape[0]
        data = np.stack([origins, directions], axis=1)

        opts: dict = {
            "name": name,
            "length": length,
            "edge_width": width,
            "vector_style": kwargs.get("vector_style", "arrow"),
            "out_of_slice_display": kwargs.get("out_of_slice_display", True),
        }
        if isinstance(rgb, str):
            opts["edge_color"] = rgb
        elif rgb is not None:
            v = np.asarray(rgb)
            if v.ndim != 1 or v.shape[0] != n:
                raise ValueError(
                    "rgb must be str or a 1D array of length N (scalar per vector)"
                )
            feat_key = "_curlew_scalar"
            opts["features"] = {feat_key: v}
            opts["edge_color"] = feat_key
            opts["edge_colormap"] = resolve_cmap(colormap)
        else:
            opts["edge_color"] = "red"

        opts.update(kwargs)

        self._remove_layer_if_present(name)
        layer = self.viewer.add_vectors(data, **opts)
        self._layers[name] = layer
        self._cnt += 1
        return layer

    def addPaths(
        self,
        name: str,
        paths: list[np.ndarray],
        *,
        edge_color: str | np.ndarray = "white",
        edge_width: float = 2.0,
        curlew_coords: bool = True,
        **kwargs,
    ):
        """
        Add open polylines as a ``Shapes`` layer. Vertex rows use Curlew **(x, y)** / **(x, y, z)**
        by default (same as contour / model world coordinates). Use ``curlew_coords=False`` for
        napari **(z, y, x)**.
        """
        if not paths:
            return None
        if curlew_coords:
            paths = [self._to_napari_xyz(p) for p in paths]
        shape_type = ["path"] * len(paths)
        self._remove_layer_if_present(name)
        layer = self.viewer.add_shapes(
            paths,
            shape_type=shape_type,
            name=name,
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=_get_colors(len(paths)),
            **kwargs,
        )
        self._layers[name] = layer
        self._cnt += 1
        return layer

    def addDrillHole(
        self,
        name: str,
        hole,
        *,
        vmn: float = 0.0,
        vmx: float = 20.0,
        cmap: str | MplColormap | None = None,
        linewidth: float = 4.0,
        **kwargs,
    ):
        """
        Draw a drillhole trace as napari ``Shapes`` **line** segments (cf. ``plotDrill2D``).

        Works with ``ndisplay=2`` or ``3``. Uses transparent faces so GUI **edge width**
        controls line thickness.

        ``hole`` may be a :class:`~curlew.core.Geode` (or any object with ``.numpy()``);
        positions **(x, y)** or **(x, y, z)** are mapped to napari **(z, y, x)**.

        Each segment ``x[i]``→``x[i+1]`` is coloured by ``lithoID[i]``.

        Parameters
        ----------
        cmap
            Matplotlib **name**, :class:`matplotlib.colors.Colormap`, or ``None`` (uses
            :attr:`curlew.ccramp` if set, else ``tab20b``) for the segment colours.
        """
        from matplotlib.colors import Normalize
        from matplotlib import colormaps as mpl_colormaps
        from matplotlib.colors import Colormap as MplColormap

        if hasattr(hole, "numpy") and callable(hole.numpy):
            hole = hole.numpy()
        x = np.asarray(hole.x, dtype=np.float64)
        lid = np.asarray(hole.lithoID).ravel()
        if x.shape[0] < 2:
            return None
        if x.shape[1] not in (2, 3):
            raise ValueError("hole.x must have shape (N, 2) or (N, 3)")
        x = self._to_napari_xyz(x)

        # Use a matplotlib colormap to compute per-segment RGBA.
        if cmap is None:
            mpl_c = curlew.ccramp if curlew.ccramp is not None else mpl_colormaps["tab20b"]
        elif isinstance(cmap, str):
            mpl_c = mpl_colormaps[cmap]
        elif isinstance(cmap, MplColormap):
            mpl_c = cmap
        else:
            raise TypeError("cmap must be None, str, or matplotlib.colors.Colormap")
        norm = Normalize(vmin=vmn, vmax=vmx)

        lines: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        for i in range(len(x) - 1):
            seg = np.asarray([x[i], x[i + 1]], dtype=np.float64)
            lines.append(seg)
            rgba = np.asarray(mpl_c(norm(float(lid[i]))), dtype=np.float64)
            colors.append(rgba)

        edge_rgba = np.stack(colors, axis=0).astype(np.float32)
        return self.addLines(
            name,
            lines,
            edge_color=edge_rgba,
            edge_width=linewidth,
            curlew_coords=False,
            **kwargs,
        )

    def addLines(
        self,
        name: str,
        segments: list[np.ndarray],
        *,
        edge_color: str | np.ndarray = "white",
        edge_width: float = 2.0,
        curlew_coords: bool = True,
        **kwargs,
    ):
        """
        Add 2-point **line** segments (``shape_type='line'``) with transparent faces so the
        GUI **edge width** controls line thickness. Vertex rows use Curlew coordinates by default.
        """
        if not segments:
            return None
        if curlew_coords:
            segments = [self._to_napari_xyz(s) for s in segments]
        shape_type = ["line"] * len(segments)
        self._remove_layer_if_present(name)
        layer = self.viewer.add_shapes(
            segments,
            shape_type=shape_type,
            name=name,
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=_get_colors(len(segments)),
            **kwargs,
        )
        self._layers[name] = layer
        self._cnt += 1
        return layer

    def addGeode(
        self,
        geode: Geode,
        *,
        lithoID: bool = True,
        scalar: bool = True,
        surfaces: bool = True,
        displacement: bool = True,
        offset_length: float = 1.0,
        offset_width: float = 20.0,
    ) -> dict[str, napari.layers.Layer]:
        """
        Visualise a :class:`curlew.core.Geode` (model prediction) in the viewer.

        For a 2D :class:`~curlew.geometry.Grid`, scalar fields are 2D images; isosurfaces are
        drawn as contour polylines (``Shapes``). For 3D grids, behaviour matches the previous
        ``Napari3D`` implementation (volumes + triangle meshes).
        """
        from curlew.core import Geode as GeodeType

        if not isinstance(geode, GeodeType):
            raise TypeError("geode must be a curlew.core.Geode instance")

        geode = geode.numpy()
        cmap = resolve_cmap()
        layers: dict[str, napari.layers.Layer] = {}
        grid = geode.grid

        def add_scalar_field(
            suffix: str,
            data,
            *,
            opacity=0.8,
            visible: bool,
        ) -> None:
            if data is None:
                return
            nm = f"{suffix}"
            flat = _geode_ravel_values(data)
            if grid is not None:
                vol = np.asarray(grid.reshape(flat), dtype=np.float64)
                if vol.ndim not in (2, 3):
                    raise ValueError(
                        f"{suffix}: grid.reshape produced shape {vol.shape}; expected 2D or 3D"
                    )
                layers[nm] = self.addVolume(
                    nm,
                    vol,
                    grid=grid,
                    colormap=cmap,
                    blending="additive" if suffix == "lithoID" else "opaque",
                    rendering="additive" if suffix == "lithoID" else "iso",
                    visible=visible,
                    opacity=opacity,
                )
            else:
                if geode.x is None:
                    raise ValueError(
                        f"Cannot add {suffix}: geode has no grid and no x coordinates"
                    )
                gx = np.asarray(geode.x, dtype=np.float64)
                if flat.shape[0] != gx.shape[0]:
                    raise ValueError(
                        f"{suffix}: length {flat.shape[0]} does not match x rows {gx.shape[0]}"
                    )
                layers[nm] = self.addPoints(
                    nm,
                    gx,
                    scalar=flat.astype(np.float64),
                    colormap=cmap,
                    visible=visible,
                    opacity=opacity,
                )

        if lithoID and geode.lithoID is not None:
            add_scalar_field("lithoID", geode.lithoID, visible=True, opacity=0.35)

        if scalar and geode.scalar is not None:
            add_scalar_field("scalar", geode.scalar, visible=False)

        if scalar and geode.fields:
            for key, arr in geode.fields.items():
                add_scalar_field(f"field_{key}", arr, visible=False)

        if surfaces and geode.isosurfaces and geode.grid is not None:
            if geode.grid.ndim == 2:
                path_specs: list[tuple[str, np.ndarray]] = []
                path_idx = 0
                for field in geode.fields:
                    for surf_name in geode.isosurfaces[field]:
                        polylines = geode.getSurface(field, surf_name, normals=False)
                        if not isinstance(polylines, list):
                            continue
                        for poly in polylines:
                            if poly is None or len(poly) < 2:
                                continue
                            n = f"{field}_{surf_name}_c{path_idx}"
                            path_idx += 1
                            path_specs.append((n, poly))
                n_paths = max(1, len(path_specs))
                for i, (n, poly) in enumerate(path_specs):
                    col = np.asarray(curlew.ccramp(i / n_paths), dtype=np.float32)
                    edge_rgba = np.tile(col, (1, 1))
                    layers[n] = self.addPaths(
                        n,
                        [poly],
                        edge_color=edge_rgba,
                        edge_width=2.0,
                    )
            else:
                names = []
                for field in geode.fields:
                    for surf_name in geode.isosurfaces[field]:
                        n = f"{field}_{surf_name}"
                        verts, faces, normals = geode.getSurface(
                            field, surf_name, normals=True
                        )
                        layers[n] = self.addMesh(
                            n,
                            verts,
                            faces,
                            normals=normals,
                            rgb="white",
                            shading="smooth",
                        )
                        names.append(n)

                for i, n in enumerate(names):
                    verts = layers[n].data[0]
                    col = curlew.ccramp(i / len(names))
                    vertex_colors = np.tile(col, (len(verts), 1))
                    layers[n].vertex_colors = vertex_colors

        if displacement and geode.offsets:
            if geode.x is None:
                raise ValueError("geode.offsets requires geode.x for vector origins")
            x0 = np.asarray(geode.x, dtype=np.float64)
            for key, disp in geode.offsets.items():
                nm = f"offset_{key}"
                d = np.asarray(disp, dtype=np.float64)
                if d.shape != x0.shape:
                    raise ValueError(
                        f"offsets[{key!r}] shape {d.shape} does not match x shape {x0.shape}"
                    )
                if x0.shape[0] == 0:
                    continue
                mag = np.linalg.norm(d, axis=1)
                layers[nm] = self.addVectors(
                    nm,
                    x0,
                    d,
                    rgb=mag,
                    length=offset_length,
                    width=offset_width,
                    vector_style="arrow",
                    visible=False,
                )

        return layers

    def addCSet(
        self,
        name: str,
        C: CSet,
        *,
        lithoID: bool = True,
        scalar: bool = True,
        displacement: bool = True,
        H=None,
        grad_length: float = 50.0,
        grad_width: float = 20.0,
        orient_length: float = 50.0,
        orient_width: float = 10.0,
        bedding_length: float = 0.0,
        bedding_width: float = 4.0,
        bedding_edge_color: str = "black",
        value_size: float = 30.0,
        value_border_color: str | None = "black",
        iq_size: float = 10.0,
    ) -> dict[str, napari.layers.Layer]:
        """
        Visualise a :class:`curlew.core.CSet` in the viewer.

        Optional ``H`` (e.g. :class:`~curlew.core.HSet`) gates layers like
        :func:`curlew.visualise.plotConstraints2D`. When ``bedding_length > 0`` and gradient
        constraints are shown, adds perpendicular "bedding" tick lines (2D-style).
        """
        from curlew.core import CSet as CSetType

        if not isinstance(C, CSetType):
            raise TypeError("C must be a curlew.core.CSet instance")

        C = C.numpy()
        _ = (lithoID, displacement)

        layers: dict[str, napari.layers.Layer] = {}
        cmap_pts = resolve_cmap()

        if C.gp is not None and C.gv is not None:
            gp = np.asarray(C.gp, dtype=np.float64)
            gv = np.asarray(C.gv, dtype=np.float64)
            if gp.shape[0] > 0:
                nm = f"{name}_gradient"
                layers[nm] = self.addVectors(
                    nm,
                    gp,
                    gv,
                    rgb="red",
                    length=grad_length,
                    width=grad_width,
                    vector_style="arrow",
                )
                if bedding_length > 0 and self._ndisplay == 2:
                    gv_n = self._to_napari_xyz(C.gv)
                    gvn = np.linalg.norm(gv_n, axis=1, keepdims=True)
                    gu = gv_n / (gvn + 1e-8)
                    # In napari (z,y,x), gradient in the display plane uses axes 1 and 2.
                    perp = np.column_stack(
                        [np.zeros(len(gu)), -gu[:, 2], gu[:, 1]]
                    )
                    gp_n = self._to_napari_xyz(C.gp)
                    nm_b = f"{name}_gradient_bedding"
                    layers[nm_b] = self.addVectors(
                        nm_b,
                        gp_n,
                        perp,
                        rgb=bedding_edge_color,
                        length=bedding_length,
                        width=bedding_width,
                        vector_style="line",
                        curlew_coords=False,
                    )

        if C.gop is not None and C.gov is not None:
            gop = np.asarray(C.gop, dtype=np.float64)
            gov = np.asarray(C.gov, dtype=np.float64)
            if gop.shape[0] > 0:
                nm = f"{name}_orientation"
                layers[nm] = self.addVectors(
                    nm,
                    gop,
                    gov,
                    rgb="green",
                    length=orient_length,
                    width=orient_width,
                    vector_style="line",
                )

        if scalar and C.vp is not None and C.vv is not None and C.pp is None:
            vp = np.asarray(C.vp, dtype=np.float64)
            vv = np.asarray(C.vv, dtype=np.float64)
            if vp.shape[0] > 0:
                scalars = vv.squeeze()
                if scalars.shape[0] != vp.shape[0]:
                    raise ValueError(
                        "vp and vv must have the same number of rows for value constraints"
                    )
                nm = f"{name}_value"
                pt_kw: dict = {}
                if value_border_color is not None:
                    pt_kw["border_color"] = value_border_color
                    pt_kw["border_width"] = 0.05
                layers[nm] = self.addPoints(
                    nm,
                    vp,
                    scalar=scalars,
                    colormap=cmap_pts,
                    size=value_size,
                    symbol="clobber",
                    **pt_kw,
                )

        if C.pp is not None and C.pv is not None:
            pp = np.asarray(C.pp, dtype=np.float64)
            pv = np.asarray(C.pv, dtype=np.float64)
            if pp.shape[0] > 0:
                nm = f"{name}_property"
                if pv.shape[-1] == 1:
                    layers[nm] = self.addPoints(
                        nm,
                        pp,
                        scalar=pv[:, 0],
                        colormap=cmap_pts,
                        size=value_size * 0.5,
                    )
                elif pv.shape[-1] >= 3:
                    rgb = pv[:, [0, 1, 2]].astype(float)
                    rgb -= np.min(rgb, axis=0)[None, :]
                    denom = np.max(rgb, axis=0)[None, :] - np.min(rgb, axis=0)[None, :]
                    denom = np.where(denom < 1e-12, 1.0, denom)
                    rgb = rgb / denom
                    rgba = np.c_[rgb, np.ones(len(rgb))]
                    layers[nm] = self.addPoints(
                        nm,
                        pp,
                        rgb=rgba.astype(np.float64),
                        size=value_size * 0.5,
                    )

        if C.iq is not None:
            iq_list = C.iq[1]
            from matplotlib import colormaps as mpl_colormaps

            cmap_lhs = mpl_colormaps["Reds"]
            cmap_rhs = mpl_colormaps["Blues"]
            cmap_eq = mpl_colormaps["viridis"]

            eq_pts_parts: list[np.ndarray] = []
            eq_rgba_parts: list[np.ndarray] = []
            eq_sym_parts: list[str] = []
            n_iq = max(1, len(iq_list))
            for i, entry in enumerate(iq_list):
                P1, P2, rel = entry
                rel_s = (rel if isinstance(rel, str) else str(rel)).strip()
                p1 = np.asarray(P1, dtype=np.float64)
                p2 = np.asarray(P2, dtype=np.float64)

                t = 0.0 if n_iq <= 1 else (i / (n_iq - 1))
                col_lhs = np.asarray(cmap_lhs(t), dtype=np.float32)
                col_rhs = np.asarray(cmap_rhs(t), dtype=np.float32)
                col_eq = np.asarray(cmap_eq(t), dtype=np.float32)

                if rel_s == "=":
                    if p1.shape[0] > 0:
                        p1n = self._to_napari_xyz(p1)
                        eq_pts_parts.append(p1n)
                        eq_rgba_parts.append(np.tile(col_eq, (p1n.shape[0], 1)))
                        eq_sym_parts.extend(["diamond"] * p1n.shape[0])
                    continue

                # inequality: choose symbols by relationship and side
                # '>' : LHS cross, RHS hbar; '<' : LHS hbar, RHS cross
                if rel_s == ">":
                    lhs_sym, rhs_sym = ("cross", "hbar")
                elif rel_s == "<":
                    lhs_sym, rhs_sym = ("hbar", "cross")
                else:
                    raise ValueError(
                        f"Unsupported inequality relation {rel_s!r}; expected '=', '<', or '>'"
                    )

                pts_parts: list[np.ndarray] = []
                sym_parts: list[str] = []
                rgba_parts: list[np.ndarray] = []
                n_pts = 0

                if p1.shape[0] > 0:
                    p1n = self._to_napari_xyz(p1)
                    pts_parts.append(p1n)
                    sym_parts.extend([lhs_sym] * p1n.shape[0])
                    rgba_parts.append(np.tile(col_lhs, (p1n.shape[0], 1)))
                    n_pts += p1n.shape[0]

                if p2.shape[0] > 0:
                    p2n = self._to_napari_xyz(p2)
                    pts_parts.append(p2n)
                    sym_parts.extend([rhs_sym] * p2n.shape[0])
                    rgba_parts.append(np.tile(col_rhs, (p2n.shape[0], 1)))
                    n_pts += p2n.shape[0]

                if pts_parts:
                    nm = f"{name}_iq{i}"
                    self._remove_layer_if_present(nm)
                    xyz = np.concatenate(pts_parts, axis=0)
                    rgba = np.concatenate(rgba_parts, axis=0)
                    layer = self.viewer.add_points(
                        xyz,
                        name=nm,
                        face_color=rgba,
                        symbol=np.asarray(sym_parts, dtype=object),
                        size=iq_size,
                        out_of_slice_display=True,
                    )
                    layer.visible = False # hidden by default.
                    self._layers[nm] = layer
                    self._cnt += 1
                    layers[nm] = layer

            if eq_pts_parts:
                nm_eq = f"{name}_eq"
                self._remove_layer_if_present(nm_eq)
                eq_xyz = np.concatenate(eq_pts_parts, axis=0)
                eq_rgba = np.concatenate(eq_rgba_parts, axis=0)
                eq_layer = self.viewer.add_points(
                    eq_xyz,
                    name=nm_eq,
                    face_color=eq_rgba,
                    symbol=np.asarray(eq_sym_parts, dtype=object),
                    size=iq_size,
                    out_of_slice_display=True,
                )
                eq_layer.visible = False # hidden by default.
                self._layers[nm_eq] = eq_layer
                self._cnt += 1
                layers[nm_eq] = eq_layer

        return layers
    
    def addTracks(
        self,
        name: str,
        tracks,
        *,
        features: dict | None = None,
        colormap=None,
        tail_width: float = 2.0,
        tail_length: int = 30,
        head_length: int = 0,
        **kwargs,
    ):
        """
        Add 3D particle trajectories as a napari ``Tracks`` layer.

        Parameters
        ----------
        name : str
            Layer name (replaces an existing layer with the same name).
        tracks : array-like
            One of three formats:

            - **Napari format** ``(N, 5)`` — columns are ``[track_id, t, x0, x1, x2]``
                (passed through directly).
            - **Stacked trajectories** ``(N_tracks, T, 3)`` — ``N_tracks`` particle paths,
                each sampled at ``T`` equally-spaced time steps, coordinates ``(x, y, z)``.
                Track IDs are assigned as ``0 … N_tracks-1`` and time as ``0 … T-1``.
            - **List of arrays** — each element is ``(T_i, 3)`` for variable-length paths.
                Track IDs and per-track time indices are assigned automatically.

        features : dict, optional
            Per-point scalar attributes (length ``N``), passed to napari as ``features``.
            Example: ``{'speed': speed_array}``. Can be used for colour-by-feature.
        colormap : optional
            Napari colormap (or name) for ``color_by`` features. Defaults to Curlew
            ``ccramp`` when available.
        tail_width : float, optional
            Track tail line width.
        tail_length : int, optional
            Number of previous time-steps to render as a tail.
        head_length : int, optional
            Number of future time-steps to render ahead of the current frame.
        **kwargs
            Additional arguments passed to ``viewer.add_tracks`` (e.g. ``color_by``,
            ``blending``, ``opacity``).

        Returns
        -------
        napari.layers.Tracks
        """
        _require_napari()

        tracks_arr = np.asarray(tracks, dtype=np.float64) if not isinstance(tracks, list) else None

        if tracks_arr is not None and tracks_arr.ndim == 2 and tracks_arr.shape[1] == 5:
            # Already in napari format [track_id, t, x0, x1, x2]
            data = tracks_arr
        elif tracks_arr is not None and tracks_arr.ndim == 3 and tracks_arr.shape[2] == 3:
            # (N_tracks, T, 3) → (N_tracks*T, 5)
            n_tracks, T, _ = tracks_arr.shape
            ids = np.repeat(np.arange(n_tracks, dtype=np.float64), T)
            times = np.tile(np.arange(T, dtype=np.float64), n_tracks)
            coords = tracks_arr.reshape(-1, 3)
            data = np.column_stack([ids, times, coords])
        elif isinstance(tracks, list) or (tracks_arr is not None and tracks_arr.ndim == 1):
            # List of variable-length (T_i, 3) arrays
            rows = []
            for track_id, traj in enumerate(tracks):
                t = np.asarray(traj, dtype=np.float64)
                if t.ndim != 2 or t.shape[1] != 3:
                    raise ValueError(
                        f"Track {track_id}: expected shape (T, 3), got {t.shape}"
                    )
                ids = np.full(len(t), track_id, dtype=np.float64)
                times = np.arange(len(t), dtype=np.float64)
                rows.append(np.column_stack([ids, times, t]))
            data = np.vstack(rows)
        else:
            raise ValueError(
                "tracks must be (N, 5) napari format, (N_tracks, T, 3), or a list of (T_i, 3) arrays"
            )

        opts: dict = {
            "name": name,
            "tail_width": tail_width,
            "tail_length": tail_length,
            "head_length": head_length,
        }
        if features is not None:
            opts["features"] = features
        if colormap is not None:
            opts["colormap"] = colormap
        elif features:
            opts["colormap"] = _curlew_napari_colormap()
        opts.update(kwargs)

        self._remove_layer_if_present(name)
        layer = self.viewer.add_tracks(data, **opts)
        self._layers[name] = layer
        self._cnt += 1
        return layer

    def addEshelby(
        self,
        name: str,
        field: "EshelbyField",
        *,
        rgb: str = "cyan",
        edge_width: float = 2.0,
        n_ellipse_pts: int = 96,
        n_u: int = 24,
        n_v: int = 48,
        shading: str = "smooth",
        visible: bool = True,
        opacity: float = 0.8,
    ):
        """
        Add Eshelby inclusions as ellipses (2D viewer) or ellipsoids (3D viewer).

        2D: each inclusion is drawn as the projection of its equatorial circle (local z=0)
        into the (x, y) display plane, rendered as a closed polygon in a ``Shapes`` layer.

        3D: inclusions are rendered as triangle meshes (one combined ``Surface`` layer).
        """
        # Import here to keep napari optional for the rest of curlew.
        from curlew.fields.eshelby import EshelbyField as EshelbyFieldType

        if not isinstance(field, EshelbyFieldType):
            raise TypeError("field must be a curlew.fields.eshelby.EshelbyField instance")

        def _to_np(a) -> np.ndarray:
            if hasattr(a, "detach"):
                a = a.detach()
            if hasattr(a, "cpu"):
                a = a.cpu()
            if hasattr(a, "numpy"):
                return np.asarray(a.numpy(), dtype=np.float64)
            return np.asarray(a, dtype=np.float64)

        pos = _to_np(field._pos)  # (n,3)
        RT = _to_np(field._RT)    # (n,3,3) local->global
        r = _to_np(field._radii).reshape(-1)  # (n,)
        t = np.sqrt(_to_np(field._t2).reshape(-1))  # (n,)
        n_src = int(pos.shape[0])
        if n_src == 0:
            return None

        if self._ndisplay == 2:
            n_pts = int(max(12, n_ellipse_pts))
            theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
            circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])  # (P,3)

            polys: list[np.ndarray] = []
            for i in range(n_src):
                # Local equatorial circle at z_loc=0, radius r[i]
                loc = circle * float(r[i])
                glob = pos[i][None, :] + (RT[i] @ loc.T).T  # (P,3) in curlew xyz
                polys.append(glob)

            # Render as closed polygons with transparent faces.
            self._remove_layer_if_present(name)
            edge_color = rgb
            layer = self.viewer.add_shapes(
                [self._to_napari_xyz(p) for p in polys],
                shape_type=["polygon"] * len(polys),
                name=name,
                edge_color=edge_color,
                edge_width=edge_width,
                face_color=_get_colors(len(polys)),
                visible=visible,
                opacity=opacity,
            )
            self._layers[name] = layer
            self._cnt += 1
            return layer

        # 3D mesh: triangulated ellipsoid surfaces, combined into one layer.
        n_u = int(max(6, n_u))
        n_v = int(max(12, n_v))
        uu = np.linspace(0.0, np.pi, n_u)          # polar
        vv = np.linspace(0.0, 2.0 * np.pi, n_v, endpoint=False)  # azimuth
        U, V = np.meshgrid(uu, vv, indexing="ij")  # (n_u, n_v)
        sx = np.sin(U) * np.cos(V)
        sy = np.sin(U) * np.sin(V)
        sz = np.cos(U)
        sphere = np.stack([sx, sy, sz], axis=-1).reshape(-1, 3)  # (n_u*n_v, 3)

        # Faces for the (n_u x n_v) grid with wrap in v.
        faces: list[list[int]] = []
        for iu in range(n_u - 1):
            for iv in range(n_v):
                iv2 = (iv + 1) % n_v
                a = iu * n_v + iv
                b = iu * n_v + iv2
                c = (iu + 1) * n_v + iv
                d = (iu + 1) * n_v + iv2
                #faces.append([a, c, b])
                #faces.append([b, c, d])
                faces.append([b, c, a])
                faces.append([d, c, b])
        faces0 = np.asarray(faces, dtype=np.int64)

        verts_all: list[np.ndarray] = []
        faces_all: list[np.ndarray] = []
        v_off = 0
        for i in range(n_src):
            # Scale unit sphere to ellipsoid in local coords, then rotate + translate.
            scale = np.array([float(r[i]), float(r[i]), float(t[i])], dtype=np.float64)
            loc = sphere * scale[None, :]
            glob = pos[i][None, :] + (RT[i] @ loc.T).T  # (N,3) in curlew xyz
            verts_all.append(glob)
            faces_all.append(faces0 + v_off)
            v_off += glob.shape[0]

        verts = np.vstack(verts_all)
        faces = np.vstack(faces_all)
        layer = self.addMesh(
            name,
            verts,
            faces,
            rgb=rgb,
            shading=shading,
            visible=visible,
            opacity=opacity,
        )
        return layer

    def show(self, *, block: bool = False):
        """
        Show the viewer window (Qt). In a script you may need ``napari.run()`` afterwards
        unless ``block=True``.
        """
        self.viewer.show(block=block)
        return self.viewer


__all__ = [
    "NapariViewer",
]