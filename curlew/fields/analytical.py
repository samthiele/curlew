"""
Classes for defining analytical (rather than interpolated) implicit fields, as 
these can also be used to build geological models. 
"""

import numpy as np
import curlew
import torch
import torch.nn as nn
from collections.abc import Callable
from curlew import _tensor
from curlew.geometry import blended_wave, Transform
from curlew.fields import BaseAF


class _FastAffineTransform:
    """Cached row-vector affine map used by ``EllipsoidalField`` (no homogeneous divide)."""

    __slots__ = ("_field",)

    def __init__(self, field: "EllipsoidalField"):
        self._field = field

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        f = self._field
        M_inv = f._M_inv.to(dtype=points.dtype, device=points.device)
        origin = f.origin.to(dtype=points.dtype, device=points.device)
        return (points - origin) @ M_inv.T

    def isIdentity(self) -> bool:
        return self._field._affine_is_identity


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
    learnable : bool, optional
        If True, ``origin`` and ``gradient`` are registered as ``nn.Parameter`` instances
        and can be optimised with ``field.fit(...)``. Default is False.
    learning_rate : float, optional
        Optimiser learning rate used when ``learnable=True``. Default is 1e-1.
    """
    def initField( self,
                   origin: np.ndarray = None,
                   gradient: np.ndarray = None,
                   normalise: bool = False,
                   learnable: bool = False,
                   learning_rate: float = 1e-1 ):

        # store origin and gradient as torch tensors
        if origin is None:
            origin = np.zeros( self.input_dim )
        if gradient is None:
            gradient = np.zeros( self.input_dim )
            gradient[-1] = 1
        self.normalise = normalise
        self.mnorm = np.linalg.norm(gradient)
        grad_arr = np.array(gradient, dtype=float)
        if normalise:
            grad_arr = grad_arr / self.mnorm
            self.mnorm = 1.0
        if learnable:
            self.origin = nn.Parameter(_tensor(origin, dt=curlew.dtype, dev=curlew.device))
            self.grad = nn.Parameter(_tensor(grad_arr, dt=curlew.dtype, dev=curlew.device))
            self.init_optim(lr=learning_rate)
        else:
            self.origin = _tensor(origin, dt=curlew.dtype, dev=curlew.device)
            self.grad = _tensor(grad_arr, dt=curlew.dtype, dev=curlew.device)

    def evaluate( self, x: torch.Tensor ):
        """
        Evaluate the linear function determining the scalar field values.
        """
        grad = self.grad
        if self.normalise:
            grad = grad / (torch.norm(grad) + 1e-8)
        return torch.sum( (x - self.origin[None,:]) * grad[None,:], axis=-1 )

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
        self.curve = _tensor( np.zeros ( self.input_dim ), dt=curlew.dtype, dev=curlew.device )
        if curve is not None:
            self.curve = _tensor( curve, dt=curlew.dtype, dev=curlew.device )
        self.origin = _tensor( origin, dt=curlew.dtype, dev=curlew.device )
        self.normalise = normalise
        self.mnorm = np.linalg.norm(gradient)
        if normalise:
            gradient = gradient / self.mnorm
        self.grad = _tensor( gradient, dt=curlew.dtype, dev=curlew.device )

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
        self.origin = _tensor( self.origin, dt=curlew.dtype, dev=curlew.device )
        self.grad = _tensor( self.grad, dt=curlew.dtype, dev=curlew.device )
        self.axialPlane = _tensor( self.axialPlane, dt=curlew.dtype, dev=curlew.device )

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
        self.origin = _tensor(origin, dt=torch.float32, dev=curlew.device)

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
            # stack to (N, 2); torch.cat on two 1D tensors would concatenate to length 2N
            return torch.stack([num / den, torch.ones_like(coords[:, 1])], dim=-1)
        return listric_grad_func

class EllipsoidalField(BaseAF):
    """
    A dimension-agnostic analytical field for N-dimensional hyper-ellipsoids.
    Using the Transform function within the BaseSF class, applies an affine transform
    to the coordinates to morph the canonical field into an ellipsoidal field.

    The zero isosurface exists at a distance of 1 in the pre-transformed coordinates; hence
    it exists at the ellipsoid described by the axes and directions.

    decay
    -----
    False (default):
        Elliptical distance in canonical coordinates: 0 at the center, 1 on the
        unit ellipsoid boundary, increasing outside.
    ``'ellipse'`` (or ``True`` for backward compatibility):
        Linear taper ``max(0, 1 - r)``: 1 at the center, 0 at and beyond the boundary.
    ``'normal'``:
        Gaussian ``exp(-r²/2)``: 1 at the center, smooth infinite tail (≈0.61 at ``r=1``).
    ``'cosine'``:
        ``0.5 * (1 + cos(π min(r, 1)))``: compact support, C¹ at the boundary.
    ``'wendland'``:
        Wendland C² ``(1 - r)⁴ (4r + 1)`` for ``r ≤ 1``, else 0.
    ``'sigmoid'``:
        Logistic taper ``σ(s(1 - r)) / σ(s)``; ``s`` is ``sigmoid_scale`` (default 10).
    ``'inverse_quadratic'``:
        ``1 / (1 + r²)``: 1 at the center, heavy-tailed falloff (0.5 at ``r=1``).
    Callable:
        Custom decay ``f(u, r)`` where ``u`` is ``(N, dim)`` position in the ellipse-aligned
        frame (principal axes, not scaled by semi-axis lengths; world units) and ``r`` is
        ``(N,)`` elliptical distance (canonical radius, 1 on the ellipsoid surface). Must
        return ``(N,)`` values.
    sigmoid_scale : float, optional
        Steepness ``s`` for ``decay='sigmoid'``. Larger values sharpen the transition near ``r=1``.
    """
    _DECAY_MODES = frozenset({
        False, "ellipse", "normal", "cosine", "wendland", "sigmoid", "inverse_quadratic",
    })

    def initField(self,
                  origin: np.ndarray = None,
                  axes: np.ndarray = None,
                  directions: np.ndarray = None,
                  decay=False,
                  sigmoid_scale: float = 10.0):
        
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
        self.origin = _tensor(origin, dt=curlew.dtype, dev=curlew.device)

        # Setup Scaling (Diagonal Matrix D)
        if axes is None:
            axes = np.ones(dim)
        axes_t = _tensor(axes, dt=curlew.dtype, dev=curlew.device)
        D = torch.diag(axes_t)

        # Setup Rotation/Orientation (Matrix R)
        if directions is None:
            R = torch.eye(dim, dtype=curlew.dtype, device=curlew.device)
        else:
            # directions should be (dim, dim)
            R = _tensor(directions, dt=curlew.dtype, dev=curlew.device)
            # Ensure the basis vectors are unit length (normalization)
            R = R / torch.linalg.norm(R, dim=-1, keepdim=True)

        self.R = R
        self.axes = axes_t
        self._axis_aligned = directions is None

        # The Shape Matrix M = R @ D @ R.T; canonical coords are M^{-1}(p - origin)
        M = R @ D @ R.T
        M_inv = R @ torch.diag(1.0 / axes_t) @ R.T
        self.register_buffer("_M_inv", M_inv.contiguous())
        self._affine_is_identity = bool(
            torch.allclose(M, torch.eye(dim, dtype=curlew.dtype, device=curlew.device))
            and torch.allclose(self.origin, torch.zeros(dim, dtype=curlew.dtype, device=curlew.device))
        )
        self.T = _FastAffineTransform(self)
        if callable(decay):
            if not isinstance(decay, Callable):
                raise TypeError("custom decay must be a callable f(u, r)")
            self.decay = decay
        elif decay is True:
            self.decay = "ellipse"
        elif decay is False:
            self.decay = False
        elif decay not in self._DECAY_MODES:
            raise ValueError(
                f"decay must be False, True, a callable f(u, r), or one of "
                f"{sorted(m for m in self._DECAY_MODES if m)}; got {decay!r}"
            )
        else:
            self.decay = decay
        self.sigmoid_scale = float(sigmoid_scale)

    def _ellipse_coords(self, x: torch.Tensor):
        """
        Ellipse-aligned unscaled position ``u`` and elliptical distance ``r``.

        ``x`` is canonical coordinates (unit ellipsoid at ``r=1``).
        """
        axes = self.axes.to(dtype=x.dtype, device=x.device)
        if self._axis_aligned:
            w = x
        else:
            w = x @ self.R.to(dtype=x.dtype, device=x.device)
        u = w * axes.unsqueeze(0)
        r = torch.linalg.norm(w, dim=-1)
        return u, r

    def _decay(self, r: torch.Tensor) -> torch.Tensor:
        """Decay kernels in canonical radius r (1 at center for decay modes)."""
        mode = self.decay
        if mode == "ellipse":
            return torch.clamp(1 - r, min=0)
        if mode == "normal":
            return torch.exp(-0.5 * r**2)
        if mode == "cosine":
            return 0.5 * (1 + torch.cos(torch.pi * torch.clamp(r, max=1)))
        if mode == "wendland":
            t = torch.clamp(1 - r, min=0)
            return t**4 * (4 * r + 1)
        if mode == "sigmoid":
            s = r.new_tensor(self.sigmoid_scale)
            return torch.sigmoid(s * (1 - r)) / torch.sigmoid(s)
        if mode == "inverse_quadratic":
            return 1 / (1 + r**2)
        raise ValueError(f"unknown decay mode {mode!r}")

    def evaluate(self, x: torch.Tensor):
        """
        Evaluate in canonical coordinates (`BaseSF.forward` applies ``self.T`` first).
        """
        if callable(self.decay):
            u, r = self._ellipse_coords(x)
            out = self.decay(u, r)
            if not isinstance(out, torch.Tensor):
                out = torch.as_tensor(out, dtype=x.dtype, device=x.device)
            return out.reshape(r.shape)

        r = torch.linalg.norm(x, dim=-1)
        if self.decay is False:
            return r
        return self._decay(r)

class RectangularPrismField(BaseAF):
    """
    Supports N rectangular prisms. Pass origins/axes/directions as:
      - Single prism:  origin=(dim,), axes=(dim,), directions=(dim,dim)
      - Multi prism:   origin=(N,dim), axes=(N,dim), directions=(N,dim,dim)
    Missing params broadcast to match whichever sets N.
    """
    def initField(self,
                  origin: np.ndarray = None,
                  axes: np.ndarray = None,
                  directions: np.ndarray = None,
                  values: np.ndarray = None,
                  mask: bool = False,
                  reduction: str = "max"  # "max" = union, "min" = intersection, "sum" = blend
                  ):

        # --- Determine dim and n_prisms from whichever arg is provided ---
        def _infer(arr, ndim_single):
            """Return (np.array with batch dim, n, dim)."""
            a = np.asarray(arr, dtype=float)
            if a.ndim == ndim_single:          # single prism
                a = a[np.newaxis]
            return a

        n_prisms, dim = None, None

        if origin is not None:
            origin = _infer(origin, 1)          # (N, dim)
            n_prisms, dim = origin.shape
        if axes is not None:
            axes = _infer(axes, 1)              # (N, dim)
            n_prisms = n_prisms or axes.shape[0]
            dim = dim or axes.shape[1]
        if directions is not None:
            directions = _infer(directions, 2)  # (N, dim, dim)
            n_prisms = n_prisms or directions.shape[0]
            dim = dim or directions.shape[-1]

        dim = dim or getattr(self, "input_dim", 3)
        n_prisms = n_prisms or 1

        self.dim = dim
        self.n_prisms = n_prisms
        
        # Assign per-prism values (default: 1, 2, 3, ...)
        if values is None:
            values = np.arange(1, n_prisms + 1, dtype=float)
        else:
            values = np.asarray(values, dtype=float)
            if values.shape[0] == 1 and n_prisms > 1:
                values = np.broadcast_to(values, (n_prisms,)).copy()
        self.values = _tensor(values, dt=curlew.dtype, dev=curlew.device)  # (N,)
        
        self.mask = mask
        self.reduction = reduction

        # --- Broadcast defaults / singletons to (N, ...) ---
        if origin is None:
            origin = np.zeros((n_prisms, dim))
        elif origin.shape[0] == 1 and n_prisms > 1:
            origin = np.broadcast_to(origin, (n_prisms, dim)).copy()

        if axes is None:
            axes = np.ones((n_prisms, dim))
        elif axes.shape[0] == 1 and n_prisms > 1:
            axes = np.broadcast_to(axes, (n_prisms, dim)).copy()

        if directions is not None and directions.shape[0] == 1 and n_prisms > 1:
            directions = np.broadcast_to(directions, (n_prisms, dim, dim)).copy()

        # --- Build one (dim+1, dim+1) inverse transform per prism ---
        T_invs = []
        for i in range(n_prisms):
            o_i = _tensor(origin[i], dt=curlew.dtype, dev=curlew.device)
            a_i = _tensor(axes[i],   dt=curlew.dtype, dev=curlew.device)
            D_i = torch.diag(a_i)

            if directions is None:
                R_i = torch.eye(dim, dtype=curlew.dtype, device=curlew.device)
            else:
                d_i = directions[i]
                if d_i.shape == (dim - 1, dim):
                    # Only valid for dim == 3
                    d_i = np.vstack([d_i, np.cross(d_i[0], d_i[1])])
                elif d_i.shape != (dim, dim):
                    raise ValueError(
                        f"directions[{i}] shape {d_i.shape}, expected ({dim},{dim}) or ({dim-1},{dim})")
                R_i = _tensor(d_i, dt=curlew.dtype, dev=curlew.device)
                R_i = R_i / torch.linalg.norm(R_i, dim=-1, keepdim=True)

            M_i = R_i @ D_i  # (dim, dim)

            # Assemble (dim+1, dim+1) affine matrix  [M | t ; 0 | 1]
            T_i = torch.eye(dim + 1, dtype=curlew.dtype, device=curlew.device)
            T_i[:dim, :dim] = M_i
            T_i[:dim, dim]  = o_i

            T_invs.append(torch.linalg.inv(T_i))

        # (N, dim+1, dim+1)  — single batched tensor for fast evaluation
        self.T_inv = torch.stack(T_invs)

    def evaluate(self, x: torch.Tensor):
        """
        x : (B, dim)  query points in world space.
        Returns : (B,)  aggregated field value across all N prisms.
        """
        B = x.shape[0]

        x_homo = torch.cat([x, torch.ones(B, 1, device=x.device, dtype=x.dtype)], dim=1)
        x_canon = torch.einsum('nij,bj->nbi', self.T_inv, x_homo)
        x_canon = x_canon[..., :self.dim]

        chebyshev = torch.max(torch.abs(x_canon), dim=-1).values  # (N, B)
        field = 1.0 - chebyshev                                    # (N, B)

        if self.mask:
            field = torch.clamp(field, min=0.0)

        if self.reduction == "label":
            # For each point, pick the prism it's deepest inside,
            # return that prism's assigned value. 0 if outside all.
            inside = field > 0                                      # (N, B)
            masked = torch.where(inside, field, torch.full_like(field, -float('inf')))
            best = torch.argmax(masked, dim=0)                      # (B,)
            any_inside = inside.any(dim=0)                          # (B,)
            return torch.where(any_inside, self.values[best], torch.zeros(B, device=x.device, dtype=x.dtype))

        elif self.reduction == "max":
            return torch.max(field, dim=0).values
        elif self.reduction == "min":
            return torch.min(field, dim=0).values
        elif self.reduction == "sum":
            return torch.sum(field, dim=0)
        else:
            raise ValueError(f"Unknown reduction '{self.reduction}'")