"""
3D visualisation using Napari, with an add-layer API similar in spirit to
`curlew.utils.datascreen.DataScreen` (but using real-world coordinates, not normalised space).

Requires `napari` (and a Qt backend) to be installed separately.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from curlew.core import CSet, Geode

try:
    import napari
except ImportError as e:  # pragma: no cover - optional dependency
    napari = None  # type: ignore
    _NAPARI_IMPORT_ERROR = e
else:
    _NAPARI_IMPORT_ERROR = None


def _require_napari() -> None:
    if napari is None:
        raise ImportError(
            "Please install `napari` (and a Qt backend) to use NapariViewer."
        ) from _NAPARI_IMPORT_ERROR


def _curlew_napari_colormap():
    """Build a Napari Colormap from `curlew.ccramp`, or fall back to a named matplotlib cmap."""
    _require_napari()
    from napari.utils.colormaps import Colormap

    import curlew

    if curlew.ccramp is not None:
        return Colormap(curlew.ccramp(np.linspace(0, 1, 256)), name="curlew_ccramp")
    return "viridis"


def image_affine_from_grid(grid) -> np.ndarray:
    """
    Napari image affine (4×4): voxel indices → world coordinates for a Curlew ``Grid``.

    Uses ``grid.matrix @ index_to_local`` where ``index_to_local`` scales indices by
    ``grid.step`` and translates by ``-grid.dims / 2`` per axis, matching
    ``curlew.geometry.Grid`` axis construction.

    Parameters
    ----------
    grid
        A ``curlew.geometry.Grid`` (or duck-typed object with ``ndim``, ``dims``,
        ``step``, and ``matrix``).

    Returns
    -------
    np.ndarray
        Shape ``(4, 4)``, suitable for ``napari`` ``Image`` ``affine=``.
    """
    for attr in ("matrix", "dims", "step", "ndim"):
        if not hasattr(grid, attr):
            raise TypeError(
                f"grid must provide .{attr} (e.g. a curlew.geometry.Grid instance)"
            )
    if grid.ndim != 3:
        raise ValueError(
            f"image_affine_from_grid expects a 3D grid; got ndim={grid.ndim}"
        )
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
    M = np.asarray(grid.matrix, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(
            f"Expected grid.matrix shape (4, 4) for 3D grid; got {M.shape}"
        )
    return M @ index_to_local


def _cset_array_to_numpy(x):
    """Convert constraint array (numpy, list, or torch tensor) to float64 ndarray."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _cset_positions_to_xyz3(arr) -> np.ndarray:
    """(N, 2) or (N, 3) positions → (N, 3) with z=0 for 2D."""
    a = _cset_array_to_numpy(arr)
    if a.size == 0:
        return a.reshape(0, 3)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.shape[1] == 2:
        z = np.zeros((a.shape[0], 1), dtype=np.float64)
        a = np.hstack([a, z])
    elif a.shape[1] != 3:
        raise ValueError(
            f"Expected constraint positions with 2 or 3 columns; got shape {a.shape}"
        )
    return a


def _cset_value_scalars(vv: np.ndarray) -> np.ndarray:
    """Scalar per value constraint for colouring (first output dim if multivariate)."""
    if vv.ndim == 1:
        return vv
    if vv.shape[1] == 1:
        return vv[:, 0]
    return vv[:, 0]


def _geode_ravel_values(data) -> np.ndarray:
    """Geode per-point array → 1D (N,) for ``Grid.reshape`` or point scalars."""
    a = _cset_array_to_numpy(data)
    if a.ndim == 0:
        return a.reshape(1)
    return a.ravel()


def _sanitize_geode_key(key: str) -> str:
    """Safe substring for napari layer names."""
    s = str(key)
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)


class NapariViewer:
    """
    Open a Napari viewer in 3D and add meshes, point clouds, and volumetric images.

    Coordinates are passed through unchanged so layers share a common world space (metres,
    etc.). For volumes, pass ``grid=G`` to ``addVolume`` or set ``scale`` / ``translate`` /
    ``affine`` so image layers align with meshes and points (see napari ``Image`` docs).

    Parameters
    ----------
    title : str, optional
        Viewer window title.
    **viewer_kwds
        Extra keyword arguments passed to `napari.Viewer()`.
    """

    def __init__(self, title: str = "curlew", **viewer_kwds):
        _require_napari()
        self.viewer = napari.Viewer(title=title, **viewer_kwds)
        self.viewer.dims.ndisplay = 3
        self._layers: dict[str, napari.layers.Layer] = {}
        self._cnt = 0

    def _remove_layer_if_present(self, name: str) -> None:
        if name in self._layers:
            try:
                self.viewer.layers.remove(self._layers[name])
            except ValueError:
                pass
            del self._layers[name]

    def addMesh(
        self,
        name: str,
        verts: np.ndarray,
        faces: np.ndarray,
        rgb: str | np.ndarray = "green",
        shading: str = "smooth",
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
        rgb : str or array, optional
            Matplotlib colour name if `str`. If array, per-vertex colours with shape (N, 3) RGB
            or (N, 4) RGBA in [0, 1].
        shading : str, optional
            Passed to `viewer.add_surface` (e.g. ``'smooth'``, ``'flat'``).
        **kwargs
            Additional arguments for `napari.viewer.add_surface`.
        """
        _require_napari()
        verts = np.asarray(verts, dtype=np.float64)
        faces = np.asarray(faces)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("verts must have shape (N, 3)")

        if isinstance(rgb, str):
            from matplotlib.colors import to_rgba

            rgba = np.asarray(to_rgba(rgb), dtype=np.float64)
            vertex_colors = np.tile(rgba, (len(verts), 1))
        else:
            vc = np.asarray(rgb, dtype=np.float64)
            if vc.shape[0] != len(verts):
                raise ValueError("rgb array must have one row per vertex")
            if vc.shape[1] == 3:
                vertex_colors = np.c_[vc, np.ones(len(verts))]
            elif vc.shape[1] == 4:
                vertex_colors = vc
            else:
                raise ValueError("rgb must be (N, 3) or (N, 4)")

        self._remove_layer_if_present(name)
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
        **kwargs,
    ):
        """
        Add a 3D point cloud.

        Parameters
        ----------
        name : str
            Layer name.
        xyz : array, shape (N, 3)
        rgb : str, array, or None, optional
            Backwards-compatible colour control:
            - ``None`` and ``scalar`` is ``None``: default red points.
            - ``str``: uniform matplotlib colour name.
            - 2D array (N, 3) or (N, 4): per-point RGB or RGBA.
        scalar : array, optional
            1D array of length ``N`` giving a scalar attribute per point. When provided,
            napari's colormapping is used via ``features`` and ``face_colormap``. If
            both ``scalar`` and ``rgb`` are given, a ``ValueError`` is raised.
        colormap : optional
            Napari colormap or name to use with ``scalar``. Defaults to Curlew
            ``ccramp`` when available.
        size : float, optional
            Point display size.
        border_color : str or None, optional
            Outline colour for each marker. If ``None`` (default), ``border_width`` is set
            to 0 so no outline is drawn (unless you pass ``border_width`` in ``kwargs``).
            If set, ``border_width`` defaults to ``0.05`` when not given in ``kwargs``.
        symbol : str, optional
            Napari marker name, e.g. ``\"o\"`` / ``\"disc\"``, ``\"square\"``, ``\"star\"``, …
        **kwargs
            Passed to ``viewer.add_points`` (e.g. ``border_width``, ``opacity``). Overrides
            defaults above.

        Notes
        -----
        ``out_of_slice_display`` defaults to ``True`` so points remain visible when slicing.
        """
        _require_napari()
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")

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
            cmap = colormap if colormap is not None else _curlew_napari_colormap()
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
            if v.ndim == 2 and v.shape[0] == n and v.shape[1] in (3, 4):
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
                    "rgb must be str or an array of shape (N,3)/(N,4) for per-point colours"
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
        Add a 3D scalar image (volume).

        Default rendering is isosurface (``rendering='iso'``), blending defaults to
        ``'opaque'``, sampling uses **nearest** interpolation in 2D and 3D, and the
        colormap is ``curlew.ccramp`` when matplotlib is available.

        Parameters
        ----------
        name : str
            Layer name.
        volume : array
            3D array; leading dimensions are napari image axes (same as `numpy` shape).
        grid : optional
            A ``curlew.geometry.Grid`` (3D). When given and ``affine`` is not passed,
            sets ``affine`` via :func:`image_affine_from_grid` (same as
            ``grid.matrix @ index_to_local`` in the napari notebook). Cannot be combined
            with ``scale`` or ``translate``.
        scale, translate, affine : optional
            Physical mapping for the image layer (see napari `Image` layer docs).
            Explicit ``affine`` overrides ``grid``.
        rendering : str, optional
            Default ``'iso'``; override via this argument or ``**kwargs``.
        colormap : optional
            Default Curlew ramp; pass a napari colormap or name to override.
        blending : str, optional
            Napari image blending mode (see napari ``Image``).
        interpolation2d, interpolation3d : str, optional
            Passed to ``viewer.add_image``. Napari does **not** accept a single
            ``interpolation=`` argument; use these (e.g. ``\"nearest\"``, ``\"linear\"``).
        **kwargs
            Additional arguments for ``viewer.add_image`` (can override ``rendering``,
            ``colormap``, ``blending``, ``interpolation2d``, ``interpolation3d``, etc.).
        """
        _require_napari()
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError("volume must be a 3D array")

        if grid is not None and (scale is not None or translate is not None):
            raise ValueError(
                "addVolume: use grid=... for affine, or scale=/translate=, not both."
            )

        if colormap is None:
            colormap = _curlew_napari_colormap()

        affine_use = affine
        if affine_use is None and grid is not None:
            affine_use = image_affine_from_grid(grid)

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
        **kwargs,
    ):
        """
        Add a 3D vector field as napari ``Vectors`` (arrows from origins along directions).

        Parameters
        ----------
        name : str
            Layer name (replaces an existing layer with the same name).
        origins : array, shape (N, 3)
            Tail / start position of each vector.
        directions : array, shape (N, 3)
            Vector components (napari draws from ``origin`` along ``direction``; scaled by
            ``length``).
        rgb : str, array, or None, optional
            - ``None``: uniform ``edge_color='red'``.
            - ``str``: matplotlib colour name passed as ``edge_color``.
            - 1D array of length ``N``: scalar attribute per vector; uses napari
              ``features`` + ``edge_colormap`` (default Curlew ``ccramp``).
        length : float, optional
            Napari vector length scale.
        edge_width : float, optional
            Line / arrow width.
        **kwargs
            Passed to ``viewer.add_vectors`` (e.g. ``edge_colormap``, ``edge_contrast_limits``,
            ``blending``, ``opacity``, ``scale``, ``translate``, ``affine``). Overrides
            defaults from ``rgb`` when applicable.

        Notes
        -----
        Do not pass ``features`` in ``kwargs`` when using a scalar ``rgb`` array, or the
        feature column used for colouring will be overwritten.
        """
        _require_napari()
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins must have shape (N, 3)")
        if directions.shape != origins.shape:
            raise ValueError("directions must have the same shape as origins (N, 3)")
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
            opts["edge_colormap"] = _curlew_napari_colormap()
        else:
            opts["edge_color"] = "red"

        opts.update(kwargs)

        self._remove_layer_if_present(name)
        layer = self.viewer.add_vectors(data, **opts)
        self._layers[name] = layer
        self._cnt += 1
        return layer

    def addGeode(
        self,
        geode: Geode,
        *,
        lithoID: bool = True,
        scalar: bool = True,
        displacement: bool = True,
        offset_length: float = 1.0,
        offset_width: float = 20.0,
    ) -> dict[str, napari.layers.Layer]:
        """
        Visualise a :class:`curlew.core.Geode` (model prediction) in the viewer.

        If ``geode.grid`` is set, ``lithoID`` and ``scalar`` (and each entry in
        ``geode.fields``) are shown as volumes via :meth:`addVolume` with
        ``grid=geode.grid`` and ``curlew.ccramp``. Otherwise they are shown as point
        clouds at ``geode.x`` via :meth:`addPoints` with ``scalar=`` and the same
        colormap.

        The main ``geode.scalar`` layer is **hidden** by default (``visible=False``).
        Each ``geode.fields`` layer is also hidden by default.

        Displacements are read from ``geode.offsets`` (dict of (N,3) vectors at ``x``).
        Each offset field is drawn with :meth:`addVectors` (arrow style), coloured by
        vector magnitude, with default ``length`` and ``edge_width`` from
        ``offset_length`` and ``offset_width``.

        Parameters
        ----------
        name : str
            Prefix for layer names (e.g. ``\"pred\"`` → ``\"pred_lithoID\"``).
        geode : curlew.core.Geode
            Prediction result (numpy or torch arrays accepted).
        lithoID, scalar, displacement : bool, optional
            If ``False``, skip lithology IDs, the main scalar field, or offset vector
            layers respectively.
        offset_length, offset_width : float
            Passed to :meth:`addVectors` for each ``geode.offsets`` entry.

        Returns
        -------
        dict
            Map layer name → napari layer for each layer created.
        """
        from curlew.core import Geode as GeodeType

        if not isinstance(geode, GeodeType):
            raise TypeError("geode must be a curlew.core.Geode instance")

        _require_napari()
        cmap = _curlew_napari_colormap()
        layers: dict[str, napari.layers.Layer] = {}
        grid = geode.grid

        def add_scalar_field(
            suffix: str,
            data,
            *,
            visible: bool,
        ) -> None:
            if data is None:
                return
            nm = f"{suffix}"
            flat = _geode_ravel_values(data)
            if grid is not None:
                vol = np.asarray(grid.reshape(flat), dtype=np.float64)
                if vol.ndim != 3:
                    raise ValueError(
                        f"{suffix}: grid.reshape produced shape {vol.shape}; expected 3D"
                    )
                layers[nm] = self.addVolume(
                    nm,
                    vol,
                    grid=grid,
                    colormap=cmap,
                    visible=visible,
                )
            else:
                if geode.x is None:
                    raise ValueError(
                        f"Cannot add {suffix}: geode has no grid and no x coordinates"
                    )
                xyz = _cset_positions_to_xyz3(geode.x)
                if flat.shape[0] != xyz.shape[0]:
                    raise ValueError(
                        f"{suffix}: length {flat.shape[0]} does not match x rows {xyz.shape[0]}"
                    )
                layers[nm] = self.addPoints(
                    nm,
                    xyz,
                    scalar=flat.astype(np.float64),
                    colormap=cmap,
                    visible=visible,
                )

        if lithoID and geode.lithoID is not None:
            add_scalar_field("lithoID", geode.lithoID, visible=True)

        if scalar and geode.scalar is not None:
            add_scalar_field("scalar", geode.scalar, visible=False)

        if scalar and geode.fields:
            for key, arr in geode.fields.items():
                sk = _sanitize_geode_key(key)
                add_scalar_field(f"field_{sk}", arr, visible=False)

        if displacement and geode.offsets:
            if geode.x is None:
                raise ValueError("geode.offsets requires geode.x for vector origins")
            x0 = _cset_positions_to_xyz3(geode.x)
            for key, disp in geode.offsets.items():
                sk = _sanitize_geode_key(key)
                nm = f"offset_{sk}"
                d = _cset_positions_to_xyz3(disp)
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
        grad_length: float = 50.0,
        grad_width: float = 20.0,
        orient_length: float = 50.0,
        orient_width: float = 10.0,
        value_size: float = 30.0,
        iq_size: float = 10.0,
    ) -> dict[str, napari.layers.Layer]:
        """
        Visualise a :class:`curlew.core.CSet` in the viewer.

        - **Gradient** (``gp``, ``gv``): arrows, red, via :meth:`addVectors`.
        - **Orientation** (``gop``, ``gov``): lines (no arrowheads), green, via
          :meth:`addVectors`.
        - **Value** (``vp``, ``vv``): points coloured by value with ``curlew.ccramp`` via
          :meth:`addPoints` (``scalar=``).
        - **Inequalities** (``iq``): one logical constraint per entry in ``iq[1]``. For
          ``'='`` only the left-hand side is shown in white. For ``'<'`` or ``'>'`` the
          LHS is cyan and the RHS orange.

        Parameters
        ----------
        C : curlew.core.CSet
            Constraint set (torch or numpy arrays are accepted).
        name : str
            Prefix for layer names (e.g. ``\"c\"`` → ``\"c_gradient\"``).
        lithoID, displacement : bool, optional
            Accepted for API symmetry with :meth:`addGeode`; **ignored** for ``CSet``
            (no lithology or displacement data).
        scalar : bool, optional
            If ``False``, value constraints (``vp`` / ``vv``) are not added.
        grad_length, grad_width : float
            Vector length and edge width for gradient constraints.
        orient_length, orient_width : float
            Same for orientation constraints.
        value_size : float
            Point size for value constraints.
        iq_size : float
            Point size for inequality LHS/RHS points.

        Returns
        -------
        dict
            Map layer name → napari layer for each layer created.
        """
        from curlew.core import CSet as CSetType

        if not isinstance(C, CSetType):
            raise TypeError("C must be a curlew.core.CSet instance")

        # lithoID / displacement: API parity with addGeode only (no CSet data).
        _ = (lithoID, displacement)

        _require_napari()
        layers: dict[str, napari.layers.Layer] = {}

        if C.gp is not None and C.gv is not None:
            gp = _cset_positions_to_xyz3(C.gp)
            gv = _cset_positions_to_xyz3(C.gv)
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

        if C.gop is not None and C.gov is not None:
            gop = _cset_positions_to_xyz3(C.gop)
            gov = _cset_positions_to_xyz3(C.gov)
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

        if scalar and C.vp is not None and C.vv is not None:
            vp = _cset_positions_to_xyz3(C.vp)
            vv = _cset_array_to_numpy(C.vv)
            if vp.shape[0] > 0:
                scalars = _cset_value_scalars(vv)
                if scalars.shape[0] != vp.shape[0]:
                    raise ValueError(
                        "vp and vv must have the same number of rows for value constraints"
                    )
                nm = f"{name}_value"
                layers[nm] = self.addPoints(
                    nm,
                    vp,
                    scalar=scalars,
                    colormap=_curlew_napari_colormap(),
                    size=value_size,
                    symbol="clobber",
                )

        if C.iq is not None:
            iq_list = C.iq[1]
            for i, entry in enumerate(iq_list):
                P1, P2, rel = entry
                rel_s = rel if isinstance(rel, str) else str(rel)
                rel_s = rel_s.strip()
                p1 = _cset_positions_to_xyz3(P1)
                p2 = _cset_positions_to_xyz3(P2)
                base = f"{name}_iq{i}"
                if rel_s == "=":
                    if p1.shape[0] > 0:
                        nm = f"{base}_eq"
                        layers[nm] = self.addPoints(
                            nm, p1, rgb="white", size=iq_size
                        )
                else:
                    if p1.shape[0] > 0:
                        nm = f"{base}_lhs"
                        layers[nm] = self.addPoints(
                            nm, p1, rgb="cyan", size=iq_size
                        )
                    if p2.shape[0] > 0:
                        nm = f"{base}_rhs"
                        layers[nm] = self.addPoints(
                            nm, p2, rgb="orange", size=iq_size
                        )

        return layers

    def show(self, *, block: bool = False):
        """
        Show the viewer window (Qt). In a script you may need ``napari.run()`` afterwards
        unless ``block=True``.
        """
        _require_napari()
        self.viewer.show(block=block)
        return self.viewer
