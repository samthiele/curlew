"""
Classes for defining analytical (rather than interpolated) implicit fields, as 
these can also be used to build geological models. 
"""

import numpy as np
import curlew
import torch
from curlew.geometry import blended_wave, Transform
from curlew.fields import BaseAF

class LinearField( BaseAF ):
    """
    Analytical linear field used to represent planar geometries.

    Parameters (passed as keyword arguments through constructor)
    ---------------------------------------------------------------
    origin : np.ndarray, optional
        The point at which this function equals 0. Should be a torch tensor or 
        numpy array of length `input_dim`. If None (default), this will be initialised
        as all zeros.
    gradient : np.ndarray, optional
        The gradient vector of this linear function. Should be a torch tensor or 
        numpy array of length `input_dim`. If None (default), this will be initialised
        as all zeros and one in the last dimensions (vertical).
    normalise : bool, optional
        Normalise the gradient vector to have a length of one (such that the resulting
        field is a distance field). Default is False.
    """
    def initField( self,
                   origin: np.ndarray = None,
                   gradient: np.ndarray = None,
                   normalise: bool = False ):

        # store origin and gradient as torch tensors
        if origin is None:
            origin = np.zeros( self.input_dim )
        if gradient is None:
            gradient = np.zeros( self.input_dim )
            gradient[-1] = 1
        self.origin = torch.tensor( origin, dtype=curlew.dtype, device=curlew.device )
        self.normalise = normalise
        self.mnorm = np.linalg.norm(gradient)
        if normalise:
            gradient = gradient / self.mnorm
            self.mnorm = 1.0
        self.grad = torch.tensor( gradient, dtype=curlew.dtype, device=curlew.device )

    def evaluate( self, x: torch.Tensor ):
        """
        Evaluate the linear function determining the scalar field values.
        """
        return torch.sum( (x - self.origin[None,:]) * self.grad[None,:], axis=-1 )

class QuadraticField( BaseAF ):
    """
    Analytical quadratic field used to represent curved geometries.

    Parameters (passed as keyword arguments through constructor)
    ---------------------------------------------------------------
    origin : np.ndarray, optional
        The point at which this function equals 0. Should be a torch tensor or 
        numpy array of length `self.input_dim`. If None (default), this will be initialised
        as all zeros.
    gradient : np.ndarray, optional
        The gradient vector of this linear function. Should be a torch tensor or 
        numpy array of length `self.input_dim`. If None (default), this will be initialised
        as all zeros and one in the last dimensions (vertical).
    curve : np.ndarray, optional
        The curve vector of this linear function. Should be a torch tensor or 
        numpy array of length `self.input_dim`. If None (default), this will be initialised
        as all zeros.
    normalise : bool, optional
        Normalise the gradient vector to have a length of one (such that the resulting
        field is a distance field). Default is False.
    """
    def initField( self,
                 origin: np.ndarray = None,
                 gradient: np.ndarray = None,
                 curve: np.ndarray = None,
                 normalise: bool = False,
                 ):
        
        # store origin and gradient as torch tensors
        if origin is None:
            origin = np.zeros( self.input_dim )
        if gradient is None:
            gradient = np.zeros( self.input_dim )
            gradient[-1] = 1
        self.curve = torch.tensor( np.zeros ( self.input_dim ), dtype=curlew.dtype, device=curlew.device )
        if curve is not None:
            self.curve = torch.tensor( curve, dtype=curlew.dtype, device=curlew.device )
        self.origin = torch.tensor( origin, dtype=curlew.dtype, device=curlew.device )
        self.normalise = normalise
        self.mnorm = np.linalg.norm(gradient)
        if normalise:
            gradient = gradient / self.mnorm
        self.grad = torch.tensor( gradient, dtype=curlew.dtype, device=curlew.device )

    def evaluate( self, x: torch.Tensor ):
        """
        Evaluate the linear function determining the scalar field values.
        """
        return torch.sum( (x - self.origin[None,:]) * self.grad[None,:] + self.curve[None,:] * ((x - self.origin[None,:]))**2, axis=-1 )

class PeriodicField(BaseAF):
    """
    Analytical periodic field, used to represent e.g., folded geometries.

    Parameters (passed as keyword arguments through constructor)
    ---------------------------------------------------------------
    origin : np.ndarray, optional
        The point at which this function equals 0. Should be a torch tensor or 
        numpy array of length `self.input_dim`. If None (default), this will be initialised
        as all zeros.
    gradient : np.ndarray, optional
        The gradient vector of a linear function to which the periodic function is added. Should be a torch tensor or 
        numpy array of length `self.input_dim`. If None (default), this will be initialised as all zeros and 1 in the 
        last dimensions (vertical). It will also be normalised as otherwise the gradient magnitude interferes with
        the `amplitude` argument.
    axialPlane : float, optional
        A vector indicating the normal vector to the axial foliation of the sinusoids / folds. Note that this
        will be normalised to length one (use the wavelength parameter to adjust the wavelength).
    wavelength : float, optional
        Wavelength of the evaluated periodic function. Default is 800.
    amplitude : float, optional
        Amplitude of the evaluated periodic function. Default is 150.
    sharpness : float, optional
        A value between 0 and 1 determining the shape of the periodic function, where 0 gives a sinusoid and 1 gives a triangle-wave.
    """

    def initField( self,
                 origin: np.ndarray = None,
                 gradient : np.ndarray = None,
                 axialPlane : np.ndarray = None,
                 wavelength : float = 800,
                 amplitude : float = 150,
                 sharpness: float = 0):

        # store origin and gradient as torch tensors
        if origin is None:
            origin = np.zeros( self.input_dim )
        if axialPlane is None:
            axialPlane = np.zeros( self.input_dim )
            axialPlane[0] = 4
            axialPlane[1] = 1.5
        if gradient is None:
            gradient = np.zeros( self.input_dim )
            gradient[1] = 1
            
        self.origin = origin
        self.grad = gradient
        self.grad = -self.grad / np.linalg.norm(self.grad) * np.pi / 4
        self.axialPlane = np.array(axialPlane)
        self.axialPlane /= np.linalg.norm(self.axialPlane) # normalise to length 1
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.sharpness = sharpness
        
        # convert to torch tensors
        self.origin = torch.tensor( self.origin, dtype=curlew.dtype, device=curlew.device )
        self.grad = torch.tensor( self.grad, dtype=curlew.dtype, device=curlew.device )
        self.axialPlane = torch.tensor( self.axialPlane, dtype=curlew.dtype, device=curlew.device )

    def evaluate( self, x: torch.Tensor ):
        """
        Evaluate the sinusoidal function determining the scalar field values.
        """
        # evaluate linear component
        linear = torch.sum( (x - self.origin[None,:]) * self.grad[None,:], axis=-1 )

        # project x onto line perpendicular to axial foliation
        proj = (x-self.origin[None,:]) @ self.axialPlane
        
        return -(linear + blended_wave( proj, f=self.sharpness, A=self.amplitude, T=self.wavelength))
    
# Softramp listric faults
class ListricField(BaseAF):
    """
    Analytical softramp field to represent listric fault geometries. The geometry is defined using a modified hyperbolic
    tangent function to control curvature and asymptotic behavior.

    It inherits from analytical fields implementing specific geometrical functions.

    Parameters (passed as keyword arguments through constructor)
    ---------------------------------------------------------------
    origin : np.ndarray
        Origin point where the fault starts (e.g., [x0, y0, z0]).
    fault_depth_scale : float, optional
        Scaling factor controlling the overall vertical extent (depth) of the fault. Default is 1500.0.
    curvature_rate : float, optional
        Controls the horizontal curvature of the fault. A higher value means tighter curvature. Default is 0.001.
    asymptote_factor : float, optional
        Controls how the fault curve approaches its asymptote (flattens out). Values between 0 and 1 recommended. Default is 1.0.
    """
    def initField(self,
                  origin: np.ndarray = None,
                  fault_floor: float = 0.,
                  curvature_rate: float = 0.001,
                  fault_ceil: float = 700. ):
        if origin is None:
            origin = np.zeros(self.input_dim)
        self.origin = torch.tensor(origin, dtype=torch.float32)

        # store origin and gradient as torch tensors
        self.field = self.listric()
        self.grad = self.listric_grad()
        self.k = curvature_rate
        self.y_f = fault_floor
        self.y_s = fault_ceil

    def evaluate(self, x: torch.Tensor):
        """
        Evaluate the scalar field representing a listric fault.
        """
        return self.field(x) / torch.linalg.norm(self.grad(x), dim=-1)
    
    def listric(self):
        """
        Return the listric fault scalar field.
        """
        def listric_func(coords):
            return coords[:, 1] - self.y_f - (self.y_s - self.y_f)*(torch.log2(1 + 2**(-self.k * (coords[:, 0] - self.origin[0]))))
        return listric_func
    
    def listric_grad(self):
        """
        Return the gradient of the scalar field.
        """
        def listric_grad_func(coords):
            num = (self.y_s - self.y_f) * self.k
            den = np.log(2) * (1 + 2**(-self.k * (coords[:, 0] - self.origin[0])))
            return torch.cat([num/den, torch.ones_like(coords[:, 1])], dim=-1)
        return listric_grad_func

class EllipsoidalField(BaseAF):
    """
    A dimension-agnostic analytical field for N-dimensional hyper-ellipsoids.
    Using the Transform function within the BaseSF class, applies an affine transform
    to the coordinates to morph the distance field into an ellipsoidal field such that
    the value at the center is always 1 and falls off to zero.

    The zero isosurface exists at a distance of 1 in the pre-transformed coordinates; hence
    it exists at the ellipsoid described by the axes and directions.
    """
    def initField(self,
                  origin: np.ndarray = None,
                  axes: np.ndarray = None,
                  directions: np.ndarray = None):
        
        # 1. Determine dimensionality from input parameters or default
        if origin is not None:
            dim = len(origin)
        elif axes is not None:
            dim = len(axes)
        else:
            dim = getattr(self, "input_dim", 3) # Fallback to 3D

        # Store dimension
        self.dim = dim

        # Setup Center (Vector v)
        if origin is None:
            origin = np.zeros(dim)
        self.origin = torch.tensor(origin, dtype=curlew.dtype, device=curlew.device)

        # Setup Scaling (Diagonal Matrix D)
        if axes is None:
            axes = np.ones(dim)
        axes_t = torch.tensor(axes, dtype=curlew.dtype, device=curlew.device)
        D = torch.diag(axes_t)

        # Setup Rotation/Orientation (Matrix R)
        if directions is None:
            R = torch.eye(dim, dtype=curlew.dtype, device=curlew.device)
        else:
            # directions should be (dim, dim)
            R = torch.tensor(directions, dtype=curlew.dtype, device=curlew.device)
            # Ensure the basis vectors are unit length (normalization)
            R = R / torch.linalg.norm(R, dim=-1, keepdim=True)

        # The Shape Matrix M = R @ D @ R.T
        M = R @ D @ R.T # This is the top left section of the affine transform
        # The translation is concatenated based on the origin
        T_matrix = torch.cat([M, self.origin.unsqueeze(1)], dim=1)
        T_matrix = torch.cat([T_matrix, torch.cat(
                                                 [torch.zeros(self.dim, dtype=curlew.dtype, device=curlew.device),
                                                 torch.ones(1, device=curlew.device, dtype=curlew.dtype)]).unsqueeze(0)], dim=0)
        # We need to invert the transform matrix (as we move from world to field coordinates)
        self.T = Transform(matrix=T_matrix).inverse()

    def evaluate(self, x: torch.Tensor):
        """
        """
        # Return the norm of the transformed position (the distance from the origin morphed to the ellipse)
        # We subtract it from 1 to make it fall off to zero at the boundary.
        # N.B. We need the clamp to avoid negatives
        return torch.clamp(1 - torch.linalg.norm(x, dim=1), min=0)

class RectangularPrismField(BaseAF):
    """
    A dimension-agnostic analytical field for N-dimensional Rectangular Prisms (Hyper-rectangles).
    
    Using the Transform function within the BaseSF class, it applies an affine transform
    to map world coordinates to a canonical 'unit cube' space.
    
    The zero isosurface exists at the boundary of the prism defined by the axes (half-extents)
    and directions. Ideally, 'axes' should define the half-width, half-height, etc.
    """
    def initField(self,
                  origin: np.ndarray = None,
                  axes: np.ndarray = None,
                  directions: np.ndarray = None,
                  mask: bool = False
                  ):
        
        # 1. Determine dimensionality
        if origin is not None:
            dim = len(origin)
        elif axes is not None:
            dim = len(axes)
        else:
            dim = getattr(self, "input_dim", 3)

        self.dim = dim
        self.mask = mask

        # Setup Center
        if origin is None:
            origin = np.zeros(dim)
        self.origin = torch.tensor(origin, dtype=curlew.dtype, device=curlew.device)

        # Setup Scaling (Half-extents)
        if axes is None:
            axes = np.ones(dim)
        axes_t = torch.tensor(axes, dtype=curlew.dtype, device=curlew.device)
        D = torch.diag(axes_t)

        # Setup Rotation/Orientation
        if directions is None:
            R = torch.eye(dim, dtype=curlew.dtype, device=curlew.device)
        else:
            if directions.shape == (dim - 1, dim):
                # If we have only dim-1 directions, we assume the last one is the cross product of the others
                directions = np.vstack([directions, np.cross(directions[0], directions[1])])
            elif directions.shape != (dim, dim):
                raise ValueError(f"Directions should be of shape ({dim}, {dim}) or ({dim-1}, {dim}), but got {directions.shape}")
            R = torch.tensor(directions, dtype=curlew.dtype, device=curlew.device)
            # Normalize basis vectors
            R = R / torch.linalg.norm(R, dim=-1, keepdim=True)

        # Construct the Shape Matrix M = R @ D @ R.T
        # This scales the space along the specific direction vectors
        M = R @ D
        
        # Construct full Affine Transform
        T_matrix = torch.cat([M, self.origin.unsqueeze(1)], dim=1)
        
        # Add the homogenous row [0, 0, ... 1]
        homogenous_row = torch.cat(
            [torch.zeros(self.dim, dtype=curlew.dtype, device=curlew.device),
             torch.ones(1, device=curlew.device, dtype=curlew.dtype)]
        ).unsqueeze(0)
        
        T_matrix = torch.cat([T_matrix, homogenous_row], dim=0)

        # Invert to get World -> Field transform
        self.T = Transform(matrix=T_matrix).inverse()

    def evaluate(self, x: torch.Tensor):
        """
        Evaluates the field intensity using the L-Infinity norm (max absolute difference).
        
        In the canonical space, the prism is a unit hypercube [-1, 1].
        We calculate the maximum distance along any axis from the center.
        """
        # 1. Calculate absolute values of the coordinates (in canonical space)
        abs_x = torch.abs(x)
        
        # 2. Find the maximum value along the dimension axis (L_infinity norm)
        # torch.max returns a named tuple (values, indices), we just want values.
        # This represents the distance from center to the nearest face in unit space.
        chebyshev_dist = torch.max(abs_x, dim=1).values
        
        # 3. Calculate falloff
        # Inside the box, dist < 1.0, so result is positive.
        # On the boundary, dist = 1.0, result is 0.
        if self.mask:
            return torch.clamp(1 - chebyshev_dist, min=0)
        else:
            return 1 - chebyshev_dist