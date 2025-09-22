"""
Utility functions for generating basic geometries (grids, sections, etc.) and
performing other simple geometric tasks. 
"""

import numpy as np
import torch
import curlew

# TODO - add functions for converting angle pairs (e.g., strike, dip) to vectors

from typing import Optional, List, Tuple

def poisson_disk_indices_2d(
    x: np.ndarray,
    y: np.ndarray,
    radius: float,
    max_points: int,
    seed: Optional[int] = None,
    ) -> np.ndarray:
    """
    Fast Poisson-disk sampling on an arbitrary set of (x,y) points.

    A uniform grid-hashing approach ensures that any two chosen points are
    at least ``radius`` apart by checking only the 8 neighboring cells.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays of the same length N (flattened coordinates).
    radius : float
        Minimum separation between sampled points.
    max_points : int
        Maximum number of points to return (may return fewer if not feasible).
    seed : int | None, default=None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Indices (into ``x``/``y``) of selected points, dtype=int64.
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
    Fast Poisson-disk sampling on an arbitrary set of (x,y,z) points.

    Uses a uniform 3D grid-hashing approach so that any two chosen points are
    at least `radius` apart by checking only the 27 neighboring cells.

    Parameters
    ----------
    x, y, z : np.ndarray
        1D arrays of the same length N (flattened coordinates).
    radius : float
        Minimum separation between sampled points.
    max_points : int
        Maximum number of points to return (may return fewer if not feasible).
    seed : int | None, default=None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Indices (into x/y/z) of selected points, dtype=int64.
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

    # In 3D, choose cube side so the space diagonal equals `radius`:
    # s * sqrt(3) = radius  ->  s = radius / sqrt(3)
    cell = radius / np.sqrt(3.0)

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

import numpy as np

class Grid(object):
    """
    Class encapsulating more advanced grids (e.g., non-axis aligned grids) and facilitating array reshaping.
    """
    def __init__( self, dims, step, origin=None, rotation=None, sampleArgs={} ):
        """
        Initialize a grid object with specified dimensions, step size, origin, and rotation.
        Parameters:
            dims (tuple): Dimensions of the grid. A tuple of integers specifying the size along each axis.
            step (float or int or tuple): Step size for the grid. If a single float or int is provided, 
                it is applied uniformly to all dimensions. If a tuple is provided, it specifies the step 
                size for each dimension.
            origin (tuple, optional): The origin of the grid in the coordinate space. Defaults to None.
            rotation (numpy.ndarray, optional): A rotation matrix defining the orientation of the grid. 
                Defaults to None.
            sampleArgs (dict, optional): Additional arguments for sampling with draw(). Defaults to an empty dictionary.
        Attributes:
            dims (tuple): Stored dimensions of the grid.
            step (tuple): Step size for each dimension.
            ndim (int): Number of dimensions in the grid.
            sampleArgs (dict): Additional arguments for sampling.
            axes (list): List of numpy arrays representing the axes of the grid.
            shape (tuple): Shape of the grid based on the axes.
            matrix (numpy.ndarray): Transformation matrix combining origin and rotation.
            origin (numpy.ndarray): Stored origin of the grid for convenience.
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
        if origin is not None: # define grid origin
            self.matrix[:self.ndim, self.ndim] = np.array(origin)
        if rotation is not None: # define grid rotation matrix
            self.matrix[:self.ndim, :self.ndim] = rotation
        self.origin = self.matrix[:self.ndim, self.ndim] # store for convenience
        self._clearCache()

    def coords(self, transform=True):
        """
        Get an (N,2) or (N,3) array of coordinates representing the position of each cell in this grid. 
        """
        if self._cache is not None:
            assert transform, "Transform cannot be False if altCoords have been defined."
            return self._cache # easy!

        # create ravelled list of points
        #points = np.array( np.meshgrid( *self.axes, indexing='xy' ) ).T.reshape((-1,self.ndim))
        #points = np.array( np.meshgrid( *self.axes, indexing='xy' ) ).reshape((-1,self.ndim))
        coords = np.meshgrid(*self.axes, indexing='ij')
        grid = np.array([c.T for c in coords[::-1]]).T
        grid = grid[...,::-1]
        points = grid.reshape((-1, self.ndim) )

        # apply transform matrix
        if transform:
            points = np.hstack([points, np.ones( (len(points),1) )]) # concatentate fourth dimension
            points = (points @ self.matrix.T)[:,:-1] # drop fake dimension
        return points
    
    def reshape(self, values):
        """
        Reshape an (N,) or (N,d) array of points to an N-d grid (image or voxel) array.
        """
        if len(values.shape) == 1:
            return values.reshape(self.shape) # easy
        else:
            return values.reshape( self.shape + (values.shape[-1], ) )
    
    def _setCache(self, coords):
        """
        Store a cached transformed grid. Used during training of individual fields, to avoid lots of recomputations 
        of transformed grid coords.
        """
        assert (np.array( coords.shape ) == np.array( self.coords().shape )).all()
        self._cache = coords
    
    def _clearCache(self):
        """ Remove any defined cache. """
        self._cache = None
    
    def draw(self, transform=None):
        """
        Draw a random sample according to `self.sampleArgs`. This is used to draw
        random samples (using `self.sample`) when fitting global constraints.
        """
        coords = self.sample(**self.sampleArgs, tensor=True)
        if transform and (self._cache is not None): # we need to apply a transform?
            # N.B. assumes that if self._cache is defined then this contains
            # pre-computed transformed grid point positions
            coords = transform(coords)
        return coords

    def sample(self, N=4096, poissonDisk=None, tensor=False):
        """
        Sample random points from this grid. Used for e.g., evaluating "global" constraints. 

        Parameters
        ----------
        N : int
            The number of random samples to select.
        poissonDisk : tuple | optional (defaults to None)
            If not None, should be given as a tuple (r, k, seed), where r is the minimum distance between
            the points and k is the maximum number of points allowed, and seed ensures reproducibility
            (can be None to use previous random seed).
            If None, draws randomly from the grid.
        tensor : bool | optional
            True if sampled points should be returned as a torch.Tensor. Default is False. 
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
            return torch.tensor( out, device=curlew.device, dtype=curlew.dtype)
        else:
            return out
    
    def contour(self, values, iso, normals=False, transform=True ):
        """
        Use values computed for this grid to extract 2D (lines) or 3D (surfaces) contours. Requires scikit-image.

        Parameters
        -----------
            values : np.ndarray
                A 1D array of values to extract contours from. Length must be the same as `self.coords()`.
            iso : float
                The isovalue to extract. Must be within the range of `values`.
            normals : bool
                If True, an array of vertex normals will be returned in addition to the vertices and faces.
            transform : bool | optional
                True (default) if this grids transorm matrix should be applied to return coordinates in world coords. Otherwise
                internal (local) grid coordinates will be returned.
        
        Returns
        --------
            An array of `vert`, `faces` for 3D grids, or a list of contour polylines (lists of coordinates) for 2D grids.
        """
        try:
            import skimage
        except:
            assert False, "Please install scikit-image using `pip install scikit-image`"
        
        if self.ndim == 3: # marching cubes
            from skimage.measure import marching_cubes
            vol = self.reshape(values) # predicted value as (3D) grid
            vix, faces, norm, _ = marching_cubes( vol, level=iso, mask=np.isfinite(vol) ) # find isosurface using marching cubes 

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
            img = self.reshape(values)
            contours = find_contours( img, iso, mask=np.isfinite(vol))

            # build interpolators that convert indices to positions
            from scipy.interpolate import interp1d
            for i,vix in enumerate(contours):
                ix = [ interp1d(np.arange(self.shape[i]), self.axes[i]) for i in range(len(self.shape)) ]
                contours[i] = np.vstack( [ix[i]( vix[:, i] ) for i in range(len(self.shape))] ).T

                if transform:
                    verts = np.hstack([contours[i], np.ones( (len(verts),1) )]) # concatentate fourth dimension
                    contours[i] = (verts @ self.matrix.T)[:,:-1] # apply and drop fake dimension

            # TODO; mask NaN areas from 2D contours

            return contours
        
    def copy(self):
        """
        Create a copy of the current Grid instance.
        """
        new_grid = Grid(self.dims, self.step, origin=self.origin, rotation=self.matrix[:self.ndim, :self.ndim], sampleArgs=self.sampleArgs)
        if self._cache is not None:
            new_grid._setCache(self._cache)
        return new_grid
    
def grid( dims : tuple, step : tuple, origin=None, sampleArgs={} ):
    """
    Utility function to quickly generate and return an n-dimensional grid of points.

    Parameters
    ----------
    dims : tuple
        A tuple of the form (xdim, ydim, ...) defining the extent of the grid in each dimension. Note that this is in 
        units (e.g. meters) rather than number of grid cells.
    step : tuple
        A tuple of the form (xstep, ystep, ...) defining the size of each grid cell in each dimension.
    origin : np.ndarray or None
        A NumPy array containing the origin (the first point) of the grid. If None, the origin is set to 0.
    sampleArgs : dict, optional
        Additional arguments for sampling with `Grid.draw()`. Defaults to an empty dictionary. Can be used to 
        e.g., set the number of random grid points (`N`) drawn for global constraints while training a neural field.

    Returns
    -------
    A Grid instance for this grid.
    """
    return Grid( dims, step, origin=origin, sampleArgs=sampleArgs)

# TODO - make this return a Grid object
def section(dims : tuple, origin : np.ndarray, normal : np.ndarray, width : float = None, height : float = None, step=None):
    """
    Create a grid of 3D points on a plane. Useful for evaluating and plotting
    cross sections through 3D models.

    Parameters
    ----------
    dims : tuple of int
        A tuple (nx, ny) specifying the number of grid points in the horizontal and vertical directions.
    origin : array-like, shape (3,)
        The center point of the grid.
    normal : array-like, shape (3,)
        The normal vector defining the plane (need not be unit length).
    width : float
        The total extent of the grid in the horizontal direction. If None, step must be defined.
    height : float
        The total extent of the grid in the vertical direction. If None, step must be defined.
    step : float, optional
        Spacing between grid points. If provided, the grid points will be spaced by this 
        amount in both in-plane directions, and the extents will be determined by `dims` and `step` 
        rather than by the given `width` and `height`.

    Returns
    -------
    grid : np.ndarray
        An array of shape (nx, ny, 3) representing the grid of 3D points.
    """

    # Ensure origin and normal are numpy arrays of type float.
    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)
    
    # Normalize the normal vector.
    n = normal / np.linalg.norm(normal)
    
    # To construct an orthonormal basis for the plane, we need a vector that is
    # not parallel to n. We choose an arbitrary vector; for example, if n is close
    # to [0,0,1] we can choose [1,0,0], otherwise we choose [0,0,1].
    if np.allclose(n, [0, 0, 1]) or np.allclose(n, [0, 0, -1]):
        arbitrary = np.array([0, 1, 0], dtype=float) # height axis is N-S (map)
    else:
        arbitrary = np.array([0, 0, 1], dtype=float) # height axis is up-down (section)
    
    # First in-plane axis: u = n x arbitrary
    u = np.cross(n, arbitrary)
    u /= np.linalg.norm(u)
    
    # Second in-plane axis: v = n x u (this ensures u, v, n is a right-handed frame)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    
    nx, ny = dims
    
    if step is None:
        # If step is not provided, we want the grid to exactly span the given width and height.
        # For nx>1, the spacing along the u-direction is width/(nx-1) so that the points
        # go from -width/2 to +width/2.
        u_coords = np.linspace(-width/2, width/2, nx)
        v_coords = np.linspace(-height/2, height/2, ny)
    else:
        # If step is provided, we ignore width and height.
        # We center the grid so that the middle point is at zero.
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

def _extrude_array(arr, step=(0,100,0), n=3, y="up" ):
    """
    Extrude a 2D array into 3D by inserting a new coordinate.

    Given an array of shape (n,2), returns an array of shape (n,3) where a new coordinate, 
    is calculated as multiples of the step vector.

    Parameters
    ----------
    arr : array-like of shape (n, 2)
        The input 2D array to be extruded.
    step : array-like of shape (n, 3)
        The offset by which each entry is extruded for each step.
    n : int
        The number of extrusions (duplications and offsets) to make.
    y : str
        Defines how the 2D y-axis is interpreted. Options are:
        - "north": Input data is treated as 2D map information.
        - "up": Input data is treated as 2D section information.

    Returns
    -------
    extruded : np.ndarray of shape (n, 3)
        Each row is structured as [original[0], new_coordinate, original[1]].
    """
    arr = np.asarray(arr)
    if arr.shape[1] != 2:
        raise ValueError("Input array must have shape (n,2)")
    
    step = np.asarray(step)
    if len(step) != 3:
        raise ValueError("Step vector must have shape (3,)")
    
    # add new dimension
    d = np.zeros( len(arr), arr.dtype)
    if 'up' in y.lower():
        arr = np.array([(arr[i,0], d[i], arr[i,1]) for i in range(len(d))])
    else:
        arr = np.array([(arr[i,0], arr[i,1], d[i]) for i in range(len(d))])
    
    # do extrusion
    extruded = np.vstack([arr + i*step[None,:] for i in range(n)])
    return extruded

def extrude( C, step=(0,100,0), n=3, y="up" ):
    """
    Dulicate and extrude constraints in the passed CSet instance
    to fake 3D data.

    Parameters
    ----------
    C : CSet, list
        The input CSet(s), containing 2D data to be extruded.
    step : array-like of shape (n, 3)
        The offset by which each entry is extruded for each step.
    n : int
        The number of extrusions (duplications and offsets) to make.
    y : str
        Defines how the 2D y-axis is interpreted. Options are:
        - "north": Input data is treated as 2D map information.
        - "up": Input data is treated as 2D section information.

    Returns
    -------
    extruded : CSet
        A CSet (or list thereof) with added 3D constraints.
    """
    if isinstance(C, list) or isinstance(C, tuple):
        return [extrude(c, step=step, n=n, y=y) for c in C]
    
    out = C.copy()
    out._offset = None # needs to be recalculated
    for k in dir(C):
        if '_' in k:
            continue # ignore
        if callable( C.__getattribute__(k) ):
            continue # ignore
        if (k[-1] == 'p') or (k == 'grid'):
            xy = C.__getattribute__(k)
            if xy is not None:
                assert xy.shape[-1] == 2, "Error, constraint %s is not 2D"%k
                xyz = _extrude_array( xy, step, n, y)
                out.__setattr__(k, xyz)
        else:
            val = C.__getattribute__(k)
            if val is not None:
                if len( val.shape ) == 1: # scalar constraints
                    valE = np.hstack([val for i in range(n)] )
                elif 'g' in k: # gv and gov gradient constraints
                    valE = []
                    for i in range(n):
                        for v in val:
                            if y == "up":
                                valE.append([v[0], 0, v[1]])
                            else:
                                valE.append([v[0], v[1], 0])
                    valE = np.array(valE)
                else: # other N-D vector constraints
                    valE = np.vstack([val for i in range(n)] )
                out.__setattr__(k, valE)
    return out

def clip( points, thresh, width, height, normal, origin):
    """
    Clip a numpy array of points or a CSet instance to within the specified distance
    of the defined section.
    """
    pass

def triangle_wave(x, A=1, T=2*np.pi, n_terms=11):
    """
    Approximates a triangle wave using a Fourier series.

    Parameters:
    x : array-like
        Input values (time or position).
    A : float
        Amplitude of the triangle wave.
    T : float
        Period of the triangle wave.
    n_terms : int
        Number of terms in the Fourier series approximation.

    Returns:
    f_x : array-like
        Approximated triangle wave values.
    """
    f_x = x * 0
    isTorch = isinstance(f_x, torch.Tensor)
    for n in range(1, n_terms + 1, 2):  # Only odd harmonics
        if isTorch:
            f_x = f_x + ((-1) ** ((n-1)//2)*(A/n**2))*torch.sin((2*torch.pi*n/T)*x)
        else:
            f_x = f_x + ((-1) ** ((n-1)//2)*(A/n**2))*np.sin((2*np.pi*n/T)*x)
    return (8/np.pi**2)*f_x  # Normalizing factor for amplitude scaling

def blended_wave( x, f=0.5, A=1, T=2*np.pi ):
    """
    Blend a triangle and a sinusoid wave to get varying degrees of sharpness in the fold geometry. 

    Parameters:
    x : array-like
        Input values (time or position).
    f : float
        The blending factor to use. 0 gives a perfectly sinusoidal wave, while 1 gives a perfectly triangular wave.
    A : float
        Amplitude of the triangle wave.
    T : float
        Period of the triangle wave.
    n_terms : int
        Number of terms in the Fourier series approximation.

    Returns:
    f_x : array-like
        Approximated blend between a triangle wave and a sine wave. 
    """
    y1 = triangle_wave(x, A=A, T=T, n_terms=1) # sinusoid wave 
    y2 = triangle_wave(x, A=A, T=T, n_terms=11) # triangle wave 
    return (1-f)*y1 + f*y2


