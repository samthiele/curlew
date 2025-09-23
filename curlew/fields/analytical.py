"""
Classes for defining analytical (rather than interpolated) implicit fields, as 
these can also be used to build geological models. These are
implemented to duck-type `curlew.core.NF`, so they can be used as part of
models that also contain interpolated fields.
"""

import numpy as np
import curlew
import torch
from torch import nn
from curlew.geometry import blended_wave

class AF( nn.Module ):
    """
    A parent class to be inherited by analytical fields implementing specific
    geometrical functions.

    Parameters
    ----------
    name : str
        A name for this neural field. Default is "f0" (i.e., bedding).
    input_dim : int, optional
        The dimensionality of the input space (e.g., 3 for (x, y, z)).
    transform : callable
        A function that transforms input coordinates prior to predictions. Must take exactly one argument as input (a tensor of positions) and return the transformed positions. 
    """

    def __init__( self,
                 name: str = 'f0',
                 input_dim: int = 3,
                 transform = None ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.transform = transform
        self.mnorm = 1.0

    def fit(self, epochs, C=None, learning_rate=None, early_stop=(100,1e-4), best=True, vb=True, prefix='Training'):
        """
        Does nothing, but implimented for compatability with fittable (neural) fields. 
        """
        return self.loss()

    def predict(self, X, to_numpy=True, transform=True ):
        """
        Evaluate this analytical field at the specified points.

        Parameters
        ----------
        X : np.ndarray
            An array of shape (N, input_dim) containing the coordinates at which to evaluate
            this neural field.
        to_numpy : bool
            True if the results should be cast to a numpy array rather than a `torch.Tensor`.
        transform : bool
            If True, any defined transform function is applied before encoding and evaluating the field for `x`.

        Returns
        --------
        S : An array of shape (N,1) containig the predicted scalar values
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor( X, device=curlew.device, dtype=curlew.dtype)
        S = self(X, transform=transform)
        if to_numpy:
            return S.cpu().detach().numpy()
        return S

    def forward(self, x: torch.Tensor, transform=True):
        """
        Evaluate this analytical field (using whatever function has
        been specified by the child (implementing) class).
        """
        # apply transform if needed
        if transform and self.transform is not None:
            x = self.transform(x)

        # Pass through specified analytical function and return
        out = self.evaluate( x )
        if len(out.shape) == 1:
            out = out[:, None] # this dimension can be important for consistency with neural fields
        return out

    def evaluate( self, x: torch.Tensor ):
        """
        Evaluate the analytical function defining this scalar field. Must
        be implemented by child classes. 
        """
        assert False, "Please use a child class (e.g., `ALF`) implementing a specific analytical function"

    def bind( self, C ):
        """
        Should not be used, but implemented to warn as such.
        """
        assert False, "Constraints cannot be bound to an analytical field."

    def set_rate(self, lr=1e-2 ):
        """
        Does nothing, but implemented for compatibility.
        """
        return

    def init_optim(self, lr=1e-2):
        """
        Does nothing, but implemented for compatibility.
        """
        return

    def zero(self):
        """
        Does nothing, but implemented for compatibility.
        """
        return

    def step(self):
        """
        Does nothing, but implemented for compatibility.
        """
        return

    def loss(self) -> torch.Tensor:
        """
        Returns zero loss (for inclusion in models with learnable components).
        """
        # 0 loss as field is analytically defined
        loss = torch.tensor(0,
                            dtype=curlew.dtype,
                            device=curlew.device,
                            requires_grad=True)
        out = { self.name : [loss.item(),{}] }
        return loss, out

    def compute_gradient(self, coords: torch.Tensor, normalize: bool = True, transform=True, return_value=False) -> torch.Tensor:
        """
        Compute the gradient of the scalar potential with respect to the input coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            A tensor of shape (N, input_dim) representing the input coordinates.
        normalize : bool, optional
            If True, the gradient is normalized to unit length per sample.
        transform : bool
            If True, any defined transform function is applied before encoding and evaluating the field for `coords`.
        return_value : bool, optional
            If True, both the gradient and the scalar value at the evaluated points are returned.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, input_dim) representing the gradient of the scalar potential at each coordinate.
        torch.Tensor, optional
            A tensor of shape (N, 1) giving the scalar value at the evaluated points, if `return_value` is True.
        """
        coords.requires_grad_(True)

        # Forward pass to get the scalar potential
        potential = self.forward(coords, transform=transform).sum(dim=-1)  # sum in case output_dim > 1
        grad_out = torch.autograd.grad(
            outputs=potential,
            inputs=coords,
            grad_outputs=torch.ones_like(potential),
            create_graph=True,
            retain_graph=True
        )[0]

        if normalize:
            norm = torch.norm(grad_out, dim=-1, keepdim=True) + 1e-8
            grad_out = grad_out / norm

        if return_value:
            return grad_out, potential
        else:
            return grad_out

class ALF( AF ):
    """
    Analytical linear field used to represent planar geometries.
    """

    """
    A parent class to be inherited by analytical fields implementing specific
    geometrical functions.

    Parameters
    ----------
    name : str
        A name for this neural field. Default is "f0" (i.e., bedding).
    input_dim : int, optional
        The dimensionality of the input space (e.g., 3 for (x, y, z)).
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
    transform : callable, optional
        A function that transforms input coordinates prior to predictions. 
        Must take exactly one argument as input (a tensor of positions) and return the transformed positions. 
    """

    def __init__( self,
                 name: str = 'f0',
                 input_dim: int = 3,
                 origin: np.ndarray = None,
                 gradient: np.ndarray = None,
                 normalise: bool = False,
                 transform = None ):
        super().__init__( name=name,
                         input_dim=input_dim,
                         transform=transform )

        # store origin and gradient as torch tensors
        if origin is None:
            origin = np.zeros( input_dim )
        if gradient is None:
            gradient = np.zeros( input_dim )
            gradient[-1] = 1
        self.origin = torch.tensor( origin, dtype=curlew.dtype, device=curlew.device )
        self.normalise = normalise
        self.mnorm = np.linalg.norm(gradient)
        if normalise:
            gradient = gradient / self.mnorm
        self.gradient = torch.tensor( gradient, dtype=curlew.dtype, device=curlew.device )

    def evaluate( self, x: torch.Tensor ):
        """
        Evaluate the linear function determining the scalar field values.
        """
        return torch.sum( (x - self.origin[None,:]) * self.gradient[None,:], axis=-1 )

class ACF( AF ):
    """
    Analytical quadratic field used to represent curved geometries.
    """

    """
    A parent class to be inherited by analytical fields implementing specific
    geometrical functions.

    Parameters
    ----------
    name : str
        A name for this neural field. Default is "f0" (i.e., bedding).
    input_dim : int, optional
        The dimensionality of the input space (e.g., 3 for (x, y, z)).
    origin : np.ndarray, optional
        The point at which this function equals 0. Should be a torch tensor or 
        numpy array of length `input_dim`. If None (default), this will be initialised
        as all zeros.
    gradient : np.ndarray, optional
        The gradient vector of this linear function. Should be a torch tensor or 
        numpy array of length `input_dim`. If None (default), this will be initialised
        as all zeros and one in the last dimensions (vertical).
    curve : np.ndarray, optional
        The curve vector of this linear function. Should be a torch tensor or 
        numpy array of length `input_dim`. If None (default), this will be initialised
        as all zeros.
    normalise : bool, optional
        Normalise the gradient vector to have a length of one (such that the resulting
        field is a distance field). Default is False.
    transform : callable, optional
        A function that transforms input coordinates prior to predictions. 
        Must take exactly one argument as input (a tensor of positions) and return the transformed positions. 
    """

    def __init__( self,
                 name: str = 'f0',
                 input_dim: int = 3,
                 origin: np.ndarray = None,
                 gradient: np.ndarray = None,
                 curve: np.ndarray = None,
                 normalise: bool = False,
                 transform = None ):
        super().__init__( name=name,
                         input_dim=input_dim,
                         transform=transform )

        # store origin and gradient as torch tensors
        if origin is None:
            origin = np.zeros( input_dim )
        if gradient is None:
            gradient = np.zeros( input_dim )
            gradient[-1] = 1
        self.curve = torch.tensor( np.zeros ( input_dim ), dtype=curlew.dtype, device=curlew.device )
        if curve is not None:
            self.curve = torch.tensor( curve, dtype=curlew.dtype, device=curlew.device )
        self.origin = torch.tensor( origin, dtype=curlew.dtype, device=curlew.device )
        self.normalise = normalise
        self.mnorm = np.linalg.norm(gradient)
        if normalise:
            gradient = gradient / self.mnorm
        self.gradient = torch.tensor( gradient, dtype=curlew.dtype, device=curlew.device )

    def evaluate( self, x: torch.Tensor ):
        """
        Evaluate the linear function determining the scalar field values.
        """
        return torch.sum( (x - self.origin[None,:]) * self.gradient[None,:] + self.curve[None,:] * ((x - self.origin[None,:]))**2, axis=-1 )

class APF( AF ):

    """
    Analytical periodic field, used to represent e.g., folded geometries.

    Parameters
    ----------
    name : str
        A name for this neural field. Default is "fold".
    input_dim : int, optional
        The dimensionality of the input space (e.g., 3 for (x, y, z)).
    origin : np.ndarray, optional
        The point at which this function equals 0. Should be a torch tensor or 
        numpy array of length `input_dim`. If None (default), this will be initialised
        as all zeros.
    gradient : np.ndarray, optional
        The gradient vector of a linear function to which the periodic function is added. Should be a torch tensor or 
        numpy array of length `input_dim`. If None (default), this will be initialised as all zeros and 1 in the 
        last dimensions (vertical). It will also be normalised as otherwise the gradient magnitude interferes with
        the `amplitude` argument.
    axialPlane : float, optional
        A vector indicating the normal vector to the axial foliation of the sinusoids / folds. Note that this
        will be normalised to length one (use the wavelength parameter to adjust the wavelength).
    wavelength : float, optional
        Wavelength of the evaluated periodic function. Default is 400.
    amplitude : float, optional
        Amplitude of the evaluated periodic function. Default is 50.
    sharpness : float, optional
        A value between 0 and 1 determining the shape of the periodic function, where 0 gives a sinusoid and 1 gives a triangle-wave.
    transform : callable, optional
        A function that transforms input coordinates prior to predictions. 
        Must take exactly one argument as input (a tensor of positions) and return the transformed positions. 
    """

    def __init__( self,
                 name: str = 'fold',
                 input_dim: int = 3,
                 origin: np.ndarray = None,
                 gradient : np.ndarray = None,
                 axialPlane : np.ndarray = None,
                 wavelength : float = 800,
                 amplitude : float = 150,
                 sharpness: float = 0,
                 transform = None ):
        super().__init__( name=name,
                         input_dim=input_dim,
                         transform=transform )

        # store origin and gradient as torch tensors
        if origin is None:
            origin = np.zeros( input_dim )
        if axialPlane is None:
            axialPlane = np.zeros( input_dim )
            axialPlane[0] = 4
            axialPlane[1] = 1.5
        if gradient is None:
            gradient = np.zeros( input_dim )
            gradient[1] = 1
            
        self.origin = origin
        self.gradient = gradient
        self.gradient = -gradient / np.linalg.norm(self.gradient) * np.pi / 4
        self.axialPlane = np.array(axialPlane)
        self.axialPlane /= np.linalg.norm(self.axialPlane) # normalise to length 1
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.sharpness = sharpness
        
    def evaluate( self, x: torch.Tensor ):
        """
        Evaluate the sinusoidal function determining the scalar field values.
        """
        # evaluate linear component
        linear = torch.sum( (x - self.origin[None,:]) * self.gradient[None,:], axis=-1 )

        # project x onto line perpendicular to axial foliation
        proj = (x-self.origin[None,:]) @ self.axialPlane
        
        return -(linear + blended_wave( proj, f=self.sharpness, A=self.amplitude, T=self.wavelength))
    
# Softramp listric faults
class AEF(AF):
    """
    Analytical softramp field to represent listric fault geometries.
    """

    """
    This class defines a scalar field representing listric (curved).
    The geometry is defined using a modified hyperbolic tangent function to control curvature and asymptotic behavior.
    It inherits from analytical fields implementing specificgeometrical functions.

    Parameters
    ----------
    name : str
        A name for this neural field. Default is "listric fault".
    input_dim : int, optional
        The dimensionality of the input space (e.g., 3 for (x, y, z)).
    origin : np.ndarray
        Origin point where the fault starts (e.g., [x0, y0, z0]).
    fault_depth_scale : float, optional
        Scaling factor controlling the overall vertical extent (depth) of the fault. Default is 1500.0.
    curvature_rate : float, optional
        Controls the horizontal curvature of the fault. A higher value means tighter curvature. Default is 0.001.
    asymptote_factor : float, optional
        Controls how the fault curve approaches its asymptote (flattens out). Values between 0 and 1 recommended. Default is 1.0.
    transform : callable, optional
        Optional spatial transform to apply to inputs.
    """

    def __init__(self,
                 name: str = 'listric fault',
                 input_dim: int = 3,
                 origin: np.ndarray = None,
                 fault_floor: float = 0.,
                 curvature_rate: float = 0.001,
                 fault_ceil: float = 700.,
                 normalise: bool = False,
                 transform=None):
        super().__init__(name=name,
                         input_dim=input_dim,
                         transform=transform)

        if origin is None:
            origin = np.zeros(input_dim)
        self.origin = torch.tensor(origin, dtype=torch.float32)

        # store origin and gradient as torch tensors
        self.field = self.listric()
        self.gradient = self.listric_grad()
        self.k = curvature_rate
        self.y_f = fault_floor
        self.y_s = fault_ceil

    def evaluate(self, x: torch.Tensor):
        """
        Evaluate the scalar field representing a listric fault.
        """
        return self.field(x) / torch.linalg.norm(self.gradient(x), dim=-1)
    
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