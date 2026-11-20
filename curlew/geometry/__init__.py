"""Grids, sections, and homogeneous transforms.

Winding numbers live in :mod:`curlew.geometry.gwn`. Isosurface extraction
lives in :mod:`curlew.geometry.meshing`.
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import curlew
from curlew import _numpy, _tensor

# TODO - add functions for converting angle pairs (e.g., strike, dip) to vectors

from typing import Optional, List, Tuple, Union
ArrayLike = Union[np.ndarray, "torch.Tensor"]

def poisson_disk_indices_2d(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    max_points: int,
    seed: Optional[int] = None,
    ) -> np.ndarray:
    """
    Indices into ``(x, y)`` of a subset with separation at least ``radius``.

    Cell size ``radius / √2`` makes the 8-neighbourhood enough. Returns fewer
    than ``max_points`` when the set cannot supply them. ``seed`` fixes the
    visit order.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have identical shapes.")
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if max_points <= 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    coords = np.c_[x, y]
    n = coords.shape[0]

    cell = radius / np.sqrt(2.0)  # r/√2 -> 8-neighborhood suffices
    xmin, ymin = coords.min(axis=0)
    ix = np.floor((coords[:, 0] - xmin) / cell).astype(np.int32)
    iy = np.floor((coords[:, 1] - ymin) / cell).astype(np.int32)

    order = rng.permutation(n)
    grid: dict[int, dict[int, int]] = {}
    chosen: List[int] = []
    r2 = radius * radius

    neighbors = [(-1,-1),(-1,0),(-1,1),
                 ( 0,-1),( 0,0),( 0,1),
                 ( 1,-1),( 1,0),( 1,1)]

    for p in order:
        if len(chosen) >= max_points:
            break
        cx, cy = int(ix[p]), int(iy[p])
        ok = True
        for dx, dy in neighbors:
            gx, gy = cx + dx, cy + dy
            col = grid.get(gx)
            if col is None:
                continue
            q = col.get(gy)
            if q is None:
                continue
            dxv = coords[p, 0] - coords[q, 0]
            dyv = coords[p, 1] - coords[q, 1]
            if dxv * dxv + dyv * dyv < r2:
                ok = False
                break
        if ok:
            grid.setdefault(cx, {})[cy] = p
            chosen.append(p)

    return np.asarray(chosen, dtype=np.int64)

def poisson_disk_indices_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    radius: float,
    max_points: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Indices into ``(x, y, z)`` of a subset with separation at least ``radius``.

    Cell size ``radius / √3`` makes the 27-neighbourhood enough. Returns fewer
    than ``max_points`` when the set cannot supply them. ``seed`` fixes the
    visit order.
    """
    if not (x.shape == y.shape == z.shape):
        raise ValueError("x, y, and z must have identical shapes.")
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if max_points <= 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    coords = np.c_[x, y, z]
    n = coords.shape[0]

    cell = radius / np.sqrt(3.0)  # cube diagonal equals radius

    xmin, ymin, zmin = coords.min(axis=0)
    ix = np.floor((coords[:, 0] - xmin) / cell).astype(np.int32)
    iy = np.floor((coords[:, 1] - ymin) / cell).astype(np.int32)
    iz = np.floor((coords[:, 2] - zmin) / cell).astype(np.int32)

    order = rng.permutation(n)
    grid: dict[Tuple[int, int, int], int] = {}
    chosen: List[int] = []
    r2 = radius * radius

    for p in order:
        if len(chosen) >= max_points:
            break

        cx, cy, cz = int(ix[p]), int(iy[p]), int(iz[p])
        ok = True

        # Check 27 neighboring cells: (dx, dy, dz) in {-1, 0, 1}
        for dx in (-1, 0, 1):
            if not ok:
                break
            for dy in (-1, 0, 1):
                if not ok:
                    break
                for dz in (-1, 0, 1):
                    q = grid.get((cx + dx, cy + dy, cz + dz))
                    if q is None:
                        continue
                    dxv = coords[p, 0] - coords[q, 0]
                    dyv = coords[p, 1] - coords[q, 1]
                    dzv = coords[p, 2] - coords[q, 2]
                    if dxv * dxv + dyv * dyv + dzv * dzv < r2:
                        ok = False
                        break

        if ok:
            grid[(cx, cy, cz)] = p
            chosen.append(p)

    return np.asarray(chosen, dtype=np.int64)

def grid( extent : tuple | np.ndarray, size : tuple, **kwds ):
    """
    A :class:`Grid` spanning ``extent``.

    ``extent`` is ``((xmin, ...), (xmax, ...))``, or a point array whose
    per-axis min and max are used. ``size`` is a spacing (float) or a cell
    count (int), one value or one per axis. :attr:`Grid.dims` is that
    coordinate span. Extra keywords go to :class:`Grid`.
    """
    # get grid bounds if not passed directly
    if isinstance(extent, np.ndarray):
        extent = (extent.min(axis=0), extent.max(axis=0))

    ndim = len(extent[0])
    if isinstance(size, (float, int)):
        size = tuple([size] * ndim)

    # Grid.dims is the coordinate span along each axis (not the number of cells).
    span = tuple(extent[1][i] - extent[0][i] for i in range(ndim))
    step = tuple(
        span[i] / s if isinstance(s, int) else s
        for i, s in enumerate(size)
    )

    # define grid centerpoint
    kwds['center'] = tuple((extent[0][i] + extent[1][i]) / 2 for i in range(ndim))

    return Grid(span, step, **kwds)

class Grid(object):
    """Sampling grid, axis-aligned or rotated, and the reshape back to an image."""
    def __init__( self, dims, step, center=None, rotation=None, sampleArgs={} ):
        """
        ``dims`` is the coordinate span along each axis. ``step`` is the
        spacing, one value or one per axis. ``center`` defaults to the origin.
        ``rotation`` is ``(ndim, ndim)``. ``sampleArgs`` is passed to
        :meth:`sample` by :meth:`draw`.
        """

        self.dims = tuple([i for i in dims]) # store dims
        if isinstance(step, float) or isinstance(step, int):  # expand to step each dimension if a float or int is passed
            step = tuple([step for i in dims])
        self.step = step
        self.ndim = len(self.dims) # number of dimensions in grid (2- or 3)
        self.sampleArgs = sampleArgs

        # build axes arrays
        self.axes = [np.arange(d, step=s) - d/2 for d,s in zip(self.dims, self.step)]
        self.shape = tuple([len(a) for a in self.axes]) # store grid shape
        
        # build matrix
        self.matrix = np.eye(self.ndim+1)
        if center is not None: # define grid origin
            self.matrix[:self.ndim, self.ndim] = np.array(center)
            self.tformed_axes = [self.axes[i] + center[i] for i in range(len(center))]
        if rotation is not None: # define grid rotation matrix
            self.matrix[:self.ndim, :self.ndim] = rotation
        self.center = self.matrix[:self.ndim, self.ndim] # store for convenience
        self._clearCache()

    def __str__(self):
        from curlew.text import grid_str
        return grid_str(self)

    def coords(self, transform=True):
        """Cell centres as ``(N, ndim)``. ``transform=False`` skips the rigid frame."""
        if self._cache is not None:
            assert transform, "Transform cannot be False if altCoords have been defined."
            return self._cache

        coords = np.meshgrid(*self.axes, indexing='ij')
        grid = np.array([c.T for c in coords[::-1]]).T
        grid = grid[...,::-1]
        points = grid.reshape((-1, self.ndim))

        # apply transform matrix
        if transform:
            points = np.hstack([points, np.ones( (len(points),1) )]) # concatentate additional dimension
            points = (points @ self.matrix.T)[:,:-1] # drop fake dimension
        return points
    
    def reshape(self, values):
        """``(N,)`` or ``(N, d)`` values back to ``shape`` or ``shape + (d,)``."""
        if len(values.shape) == 1:
            return values.reshape(self.shape) # easy
        else:
            return values.reshape( self.shape + (values.shape[-1], ) )
    
    def _setCache(self, coords):
        """Cache transformed cell centres so a field fit does not rebuild them."""
        assert (np.array( coords.shape ) == np.array( self.coords().shape )).all()
        self._cache = coords
    
    def _clearCache(self):
        """Drop the cached coordinates."""
        self._cache = None
    
    def draw(self, transform=None):
        """One draw from :meth:`sample` using ``sampleArgs``, as a tensor."""
        coords = self.sample(**self.sampleArgs, tensor=True)
        if transform and (self._cache is not None):
            # The cache holds the transformed centres, so the field frame applies here.
            coords = transform(coords, end=None)
        return coords

    def sample(self, N=4096, poissonDisk=None, tensor=False):
        """
        ``N`` cell centres, without replacement.

        ``poissonDisk=(radius, max_points, seed)`` enforces a minimum
        separation instead. ``tensor=True`` returns a torch tensor.
        """

        # get grid of points to sample from
        grid = self.coords()

        # do random sampling
        if poissonDisk is not None: # Do poisson sampling to ensure evenly spaced points
            if grid.shape[1] == 2: # 2D Grid
                ix = poisson_disk_indices_2d(grid[:, 0], grid[:, 1],
                                              radius=poissonDisk[0], max_points=poissonDisk[1],
                                              seed=poissonDisk[2])
            else: # 3D Grid
                ix = poisson_disk_indices_3d(grid[:, 0], grid[:, 1], grid[:, 2],
                                              radius=poissonDisk[0], max_points=poissonDisk[1],
                                              seed=poissonDisk[2])

        else:
            ix = np.random.choice(len(grid), N, replace=False ) # draw random points from the grid (without replacement)
        
        # get chosen points
        out = grid[ix,:]

        # return in desired format
        if tensor:
            from curlew.core import _tensor
            return _tensor(out)
        else:
            return out
    
    def contour(self, values, iso, normals=False, transform=True, mask=None, erodeMask=0 ):
      """
      Isosurface of ``values`` (length ``len(coords())``) at ``iso``.

      3-D returns ``(verts, faces)``, plus vertex normals when ``normals``.
      2-D returns a list of polylines. ``transform`` (default) maps them
      through this grid's frame. ``mask`` is a boolean array, ravelled or
      shaped like the grid. ``erodeMask > 0`` erodes it; a negative value
      dilates by that many iterations.
      """
      try:
          import skimage
      except:
          assert False, "Please install scikit-image using `pip install scikit-image`"
      
      vol = self.reshape(values)
      contour_mask = np.isfinite(vol)
      if mask is not None:
          m = np.asarray(mask)
          if m.dtype != bool:
              m = m.astype(bool)
          if m.shape == (len(values),):
              mask = self.reshape(m)
          elif m.shape == self.shape:
              mask = m
          else:
              raise ValueError(
                  "mask must be 1D (len(values),) or shaped like the grid (self.shape). "
                  f"Got mask.shape={m.shape}, expected {(len(values),)} or {self.shape}."
              )
          contour_mask = contour_mask & mask
          if erodeMask:
              if erodeMask > 0:
                  from scipy.ndimage import binary_erosion
                  contour_mask = binary_erosion(contour_mask, iterations=int(erodeMask))
              else:
                  from scipy.ndimage import binary_dilation
                  contour_mask = binary_dilation(contour_mask, iterations=int(-erodeMask))

      if self.ndim == 3: # marching cubes
          from skimage.measure import marching_cubes
          vix, faces, norm, _ = marching_cubes( vol, level=iso, mask=contour_mask ) # find isosurface using marching cubes 

          # build interpolators that convert indices to positions
          from scipy.interpolate import interp1d
          ix = [ interp1d(np.arange(self.shape[i]), self.axes[i]) for i in range(len(self.shape)) ]
          verts = np.vstack( [ix[i]( vix[:, i] ) for i in range(len(self.shape))] ).T

          # apply rotation matrix
          if transform:
              verts = np.hstack([verts, np.ones( (len(verts),1) )]) # concatentate fourth dimension
              verts = (verts @ self.matrix.T)[:,:-1] # drop fake dimension

          # remove invalid vertices
          if np.isnan(verts).any():
              vmask = ~np.isnan(verts).any(axis=-1)
              verts = verts[vmask]
              norm = norm[vmask]
                
              # encode faces again...
              old2New=np.arange(len(vmask), dtype=float)
              old2New[vmask] = np.arange(np.sum(vmask))
              old2New[~vmask] = np.nan
              faces = np.array( [(old2New[i], old2New[j], old2New[k]) for i,j,k in faces] )
              faces = faces[np.isfinite(faces).all(axis=-1)]

          if normals:
              return verts, faces, norm
          else:
              return verts, faces
      elif self.ndim == 2: # marching squares
          from skimage.measure import find_contours
          contours = find_contours( vol, iso, mask=contour_mask)

          # build interpolators that convert indices to positions
          from scipy.interpolate import interp1d
          for i,vix in enumerate(contours):
              ix = [ interp1d(np.arange(self.shape[i]), self.axes[i], fill_value='extrapolate') for i in range(len(self.shape)) ]
              contours[i] = np.vstack( [ix[i]( vix[:, i] ) for i in range(len(self.shape))] ).T

          if transform:
              for i,c in enumerate(contours):
                points = np.hstack([c, np.ones( (len(c),1) )]) # concatentate additional dimension
                contours[i] = (points @ self.matrix.T)[:,:-1] # apply and drop fake dimension

          # TODO; mask NaN areas from 2D contours

          return contours
        
    def copy(self):
        """A new grid with the same frame, and the cache if one is set."""
        new_grid = Grid(self.dims, self.step, center=self.center, rotation=self.matrix[:self.ndim, :self.ndim], sampleArgs=self.sampleArgs)
        if self._cache is not None:
            new_grid._setCache(self._cache)
        return new_grid


def compute_topology_adjacency(soft_masks, symmetrize=True, connection="conservative"):
    """
    Class-contact areas from soft masks ``(B, C, H, W)`` or ``(B, C, D, H, W)``.

    A depthwise neighbourhood sums the weight of class ``i`` against the
    neighbours of class ``j``. The diagonal is zero. ``symmetrize`` (default)
    returns ``(A + Aᵀ) / 2``. ``connection="conservative"`` counts shared faces
    (4- or 6-connected); ``"inclusive"`` also counts edges and vertices
    (8- or 26-connected).
    """
    # (B, C, H, W) or (B, C, D, H, W).
    if soft_masks.ndim not in (4, 5):
        raise ValueError(
            "soft_masks must have shape (B, C, H, W) or (B, C, D, H, W); "
            f"got {tuple(soft_masks.shape)}"
        )
    B, C = soft_masks.shape[:2]
    spatial = soft_masks.shape[2:]
    n_vox = int(np.prod(spatial))
    device, dtype = soft_masks.device, soft_masks.dtype

    connection = connection.lower()
    if connection not in ("conservative", "inclusive"):
        raise ValueError(
            f"connection must be 'conservative' or 'inclusive', not {connection!r}"
        )

    if len(spatial) == 2:
        if connection == "conservative":
            kernel = torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
                device=device,
                dtype=dtype,
            )
        else:
            kernel = torch.ones((3, 3), device=device, dtype=dtype)
            kernel[1, 1] = 0.0
        kernel = kernel.view(1, 1, 3, 3).expand(C, 1, 3, 3)
        dilated = F.conv2d(soft_masks, kernel, padding=1, groups=C)
    elif len(spatial) == 3:
        kernel = torch.zeros((3, 3, 3), device=device, dtype=dtype)
        if connection == "conservative":
            kernel[1, 1, 0] = kernel[1, 1, 2] = 1.0
            kernel[1, 0, 1] = kernel[1, 2, 1] = 1.0
            kernel[0, 1, 1] = kernel[2, 1, 1] = 1.0
        else:
            kernel[:] = 1.0
            kernel[1, 1, 1] = 0.0
        kernel = kernel.view(1, 1, 3, 3, 3).expand(C, 1, 3, 3, 3)
        dilated = F.conv3d(soft_masks, kernel, padding=1, groups=C)
    else:
        raise ValueError(f"topology supports 2D and 3D grids only; spatial shape {spatial}")

    flat_masks = soft_masks.reshape(B, C, n_vox)
    flat_dilated = dilated.reshape(B, C, n_vox).transpose(1, 2)
    adjacency = torch.bmm(flat_masks, flat_dilated)
    
    adjacency = adjacency * (1.0 - torch.eye(C, device=device, dtype=dtype).unsqueeze(0))
    if symmetrize:
        adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
    return adjacency


# TODO - make this return a Grid object?
def section(dims : tuple, origin : np.ndarray, normal : np.ndarray, width : float = None, height : float = None, step=None):
    """
    Points on a plane through ``origin`` with normal ``normal``.

    ``dims`` is ``(nx, ny)``. Without ``step``, the grid spans ``width`` and
    ``height``. With ``step`` (a float or ``(sx, sy)``), those extents are
    ignored. Returns ``(shape, points)``: ``shape`` is ``(nx, ny, 3)`` and
    ``points`` is ``(nx * ny, 3)``.
    """

    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)
    n = normal / np.linalg.norm(normal)

    # A vector not parallel to n. Vertical n uses north as the in-plane "up".
    if np.allclose(n, [0, 0, 1]) or np.allclose(n, [0, 0, -1]):
        arbitrary = np.array([0, 1, 0], dtype=float)
    else:
        arbitrary = np.array([0, 0, 1], dtype=float)

    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)  # (u, v, n) right-handed
    v /= np.linalg.norm(v)
    
    nx, ny = dims
    
    if step is None:
        # Endpoints sit on ±width/2, so the spacing is width/(nx-1).
        u_coords = np.linspace(-width/2, width/2, nx)
        v_coords = np.linspace(-height/2, height/2, ny)
    else:
        if isinstance(step, tuple) or isinstance(step, list):
            sx = step[0]
            sy = step[1]
        else:
            sx = sy = step
        u_coords = -((nx - 1) / 2) * sx + np.arange(nx) * sx
        v_coords = -((ny - 1) / 2) * sy + np.arange(ny) * sy
    
    # Create the grid points.
    grid = np.empty((nx, ny, 3))
    for i, u_val in enumerate(u_coords):
        for j, v_val in enumerate(v_coords):
            grid[i, j] = origin + u_val * u + v_val * v
    
    return grid.shape, grid.reshape((-1,3))

def triangle_wave(x, A=1, T=2*np.pi, n_terms=11):
    """Fourier triangle wave of amplitude ``A`` and period ``T``, ``n_terms`` odd harmonics."""
    f_x = x * 0
    isTorch = isinstance(f_x, torch.Tensor)
    for n in range(1, n_terms + 1, 2):  # Only odd harmonics
        if isTorch:
            f_x = f_x + ((-1) ** ((n-1)//2)*(A/n**2))*torch.sin((2*torch.pi*n/T)*x)
        else:
            f_x = f_x + ((-1) ** ((n-1)//2)*(A/n**2))*np.sin((2*np.pi*n/T)*x)
    return (8/np.pi**2)*f_x  # Normalizing factor for amplitude scaling

def blended_wave( x, f=0.5, A=1, T=2*np.pi ):
    """``(1 - f)`` times a sinusoid plus ``f`` times a triangle wave, both of amplitude ``A`` and period ``T``."""
    y1 = triangle_wave(x, A=A, T=T, n_terms=1) # sinusoid wave 
    y2 = triangle_wave(x, A=A, T=T, n_terms=11) # triangle wave 
    return (1-f)*y1 + f*y2

@dataclass
class Transform:
    """Homogeneous transform: 3×3 for 2-D points, 4×4 for 3-D. An int ``n`` is the identity in ``n`` dimensions."""
    matrix: ArrayLike

    def __post_init__(self):
        if isinstance(self.matrix, int):
            self.matrix = np.eye(self.matrix + 1)

        # If handed a CUDA tensor, pull it to CPU before storing as numpy
        if isinstance(self.matrix, torch.Tensor):
            self.matrix = self.matrix.detach().cpu().numpy()

        self.matrix = _numpy(self.matrix)

        if self.matrix.shape not in [(3, 3), (4, 4)]:
            raise ValueError("Transform matrix must be 3x3 or 4x4")

    def isIdentity(self) -> bool:
        """Return True if this transform is the identity matrix."""
        return np.allclose(self.matrix, np.eye(self.matrix.shape[0]))

    def set(self, mat: ArrayLike):
        """Set the transform matrix."""
        self.matrix = mat
        self.__post_init__()

    def apply(self, points: ArrayLike) -> ArrayLike:
        """Apply to ``(N, 2)`` or ``(N, 3)`` points. NumPy stays NumPy; torch stays torch."""
        if isinstance(points, torch.Tensor):
            return self._apply_torch(points)
        elif isinstance(points, np.ndarray):
            return self._apply_numpy(points)
        else:
            raise TypeError("points must be a numpy array or torch tensor")

    def __call__(self, points: ArrayLike) -> ArrayLike:
        """Alias for apply()."""
        return self.apply(points)

    def inverse(self) -> "Transform":
        """Return a Transform representing the inverse operation."""
        # Invert on CPU. cuSOLVER is not initialised for a one-off inverse.
        mat = torch.as_tensor(self.matrix)

        try:
            inv = torch.linalg.inv(mat)
        except torch.linalg.LinAlgError as e:
            raise ValueError("Transform matrix is not invertible") from e
        # Any other RuntimeError (CUDA, dtype, driver) propagates naturally

        return Transform(inv.numpy())

    def _apply_numpy(self, points: np.ndarray) -> np.ndarray:
        """Apply the transform to a numpy (N, D) array."""
        if points.ndim != 2:
            raise ValueError("points must have shape (N, D)")

        n, d = points.shape
        mat = self.matrix

        if mat.shape == (3, 3) and d != 2:
            raise ValueError("3x3 transform requires (N,2) points")
        if mat.shape == (4, 4) and d != 3:
            raise ValueError("4x4 transform requires (N,3) points")

        ones = np.ones((n, 1), dtype=points.dtype)
        homo = np.concatenate([points, ones], axis=1)   # (N, D+1)
        transformed = (mat @ homo.T).T                   # (N, D+1)
        w = transformed[:, -1:]
        return transformed[:, :-1] / w

    def _apply_torch(self, points: torch.Tensor) -> torch.Tensor:
        """Apply the transform to a torch (N, D) tensor."""
        if points.ndim != 2:
            raise ValueError("points must have shape (N, D)")

        n, d = points.shape

        mat = torch.as_tensor(self.matrix, dtype=points.dtype, device=points.device)

        if mat.shape == (3, 3) and d != 2:
            raise ValueError("3x3 transform requires (N,2) points")
        if mat.shape == (4, 4) and d != 3:
            raise ValueError("4x4 transform requires (N,3) points")

        # Mirror device and dtype of points exactly
        ones = torch.ones((n, 1), dtype=points.dtype, device=points.device)
        homo = torch.cat([points, ones], dim=1)           # (N, D+1)
        transformed = (mat @ homo.T).T                    # (N, D+1)
        w = transformed[:, -1:].clamp(min=1e-12)
        return transformed[:, :-1] / w


