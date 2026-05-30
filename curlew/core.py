"""
Define several core curlew types for storing data and hyperparameters. 
"""
import numpy as np
import torch
from dataclasses import dataclass, field
import copy
import curlew
from curlew import _numpy, _tensor
from curlew.geometry import Grid, Transform
from typing import Union

import torch.optim as optim
import torch.nn as nn

# used in the loss functions
ninf = float("-inf")
inf = float("inf")

class LearnableBase(nn.Module):
    """
    Base class for all learnable curlew objects.
    """
    def __init__(self):
        """
        Initialise a new learnable torch module.
        """
        super().__init__()
        self.optim = None # needs to be initialised at some point
        self.frozen = False # if True, optimizer step is ignored
        self.C = None # no constraints by default
        self.model = None  # parent GeoModel; scalar fields can use model.T for global/model coords (e.g. when binding constraints)
        
    def init_optim(self, method=optim.Adam, lr=1e-2, **kwargs):
        """
        Initialise optimiser used for this MLP. This should only be called
        (or re-called) once all relevant learnable parameters have been created.

        Parameters
        ------------
        method : torch.optim.Optimizer
            The optimiser class to use (e.g., `torch.optim.Adam`).
        lr : float
            The learning rate to use for the underlying ADAM optimiser.

        Keywords
        ------------
        Any additional keyword arguments to pass to the optimiser initialisation.
        """
        self.optim = method(self.parameters(), lr=lr, **kwargs)
    
    def set_rate(self, lr=1e-2 ):
        """
        Update the learning rate for this learnable object's optimiser.
        """
        assert self.optim is not None, "Optimiser not initialised. Call `init_optim()` first."
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def loss(self):
        """Compute loss term(s) where relevant and return associated `Pebble` object encapsulating
           these."""
        return Pebble()
    
    def _coord_transform(self):
        """
        Affine transform from global to model coordinates.

        Scalar fields use the parent GeoModel's ``T``. A GeoModel uses its own ``T``.
        Field-local ``Transform`` objects (``self.T`` on ``BaseSF``) are not used here.
        """
        m = getattr(self, "model", None)
        if m is not None:
            return m.T
        if hasattr(self, "events"):  # GeoModel (avoid circular import)
            return getattr(self, "T", None)
        return None

    def bind( self, C ):
        """
        Bind a CSet to this learnable object ready for loss computation (neural fields) or interpolation (interpolators).

        Constraint positions with ``crs='global'`` are mapped into model coordinates using
        the parent GeoModel transform (``model.T`` or ``self.T`` on a GeoModel) before binding.
        This applies to all position arrays, including inequality endpoints and equality traces (``eq``).
        """
        if getattr(C, "crs", "global") == "global":
            T = self._coord_transform()
            if T is not None:
                C = C.to_model(T)
        self.C = C.torch() # make a copy

        # setup deltas for numerical differentiation if not yet defined
        C=self.C # shortand for our copy
        if C.grid is not None:
            if C.delta is None:
                # initialise differentiation step if needed
                C.delta = np.linalg.norm( C.grid.coords()[0,:] - C.grid.coords()[1,:] ) # / 2

            if C._offset is None:
                C._offset = []
                for i in range(self.input_dim):
                    o = [0]*self.input_dim
                    o[i] = C.delta
                    C._offset.append( _tensor( o, dev=curlew.device, dt=curlew.dtype) )

        # pre-allocate inequality clamp tensors
        # (layout matches loss: one block of ns per inequality)
        if C.iq is not None:
            ns = C.iq[0]
            n_iq = len(C.iq[1])
            total_ns = ns * n_iq
            C._iq_low_clamp = torch.empty(total_ns, dtype=curlew.dtype, device=curlew.device)
            C._iq_high_clamp = torch.empty(total_ns, dtype=curlew.dtype, device=curlew.device)
            offset = 0
            for _start, _end, iq in C.iq[1]:
                rel = iq if isinstance(iq, str) else str(iq)
                if rel.strip() == '=' or '=' in rel:
                    raise ValueError(
                        "Equality constraints must be stored in CSet.eq, not iq with relation '='."
                    )
                if '<' in rel:
                    C._iq_low_clamp[offset : offset + ns] = 0.0
                    C._iq_high_clamp[offset : offset + ns] = inf
                elif '>' in rel:
                    C._iq_low_clamp[offset : offset + ns] = ninf
                    C._iq_high_clamp[offset : offset + ns] = 0.0
                else:
                    raise ValueError(f"Unsupported inequality relation {rel!r}; expected '<' or '>'")
                offset += ns

@dataclass
class CSet:
    """
    Set of local constraints used when fitting a specific NF. Note that in the below descriptions *i* refers to the 
    relevant NF's input dimensions, and *o* refers to its output dimensions. N refers to an arbitrary number of 
    constraints, which must be equal for each "position" and "value" pair. Constraints left as None (default) will 
    not be included during training. For most applications it is assumed that many types of constraints will not be defined.

    Attributes:
        vp (torch.tensor or np.ndarray): (N,o) array of value constraint positions (in modern-day coordinates).
        vv (torch.tensor or np.ndarray): (N,o) array of value constraint values.
        gp (torch.tensor or np.ndarray): (N,i) array of gradient constraint position vectors (in modern-day coordinates).
        gv (torch.tensor or np.ndarray): (N,i) array of gradient value vectors.
        gop (torch.tensor or np.ndarray): (N,i) array of gradient orientation constraint positions (in modern-day coordinates).
        gov (torch.tensor or np.ndarray): (N,i) array of gradient orientation value vectors. These differ from `gv` in that the
                                           gradient (younging) direction is not enforced, only the orientiation is considered.
        pp (torch.tensor or np.ndarray): (N,i) array of property position value vectors.
        pv (torch.tensor or np.ndarray): (N,q) array of property value vectors.
        iq (tuple): Inequality constraints. Should be a tuple containing `(N,[(P1, P2, iq),...]`), where each P1 and P2 are (N,d) arrays or tensors
                    defining positions at which to evaluate inequality constraints such as `P1 > P2`. `iq` defines the inequality to evaluate (`<` or `>` only).
                    Note that this inequality is computed for a random set of `N` pairs sampled from `P1` and `P2`. 
        eq (tuple): Equality constraints (traces). Should be a tuple or list containing (T1, T2, T3, ..., T_N), where the scalar value of all T_N should be equal (i.e.
                    minimise the mean-normalised variance of scalar field predictions at these locations).
        grid (tuple, torch.tensor or np.ndarray): A `curlew.geometry.Grid` instance defining the grid used to enforce global constraints (and associated sampling strategy).
        delta (float): The step size used when computing numerical derivatives at the grid points. Default (None) is to initialise
                       as half the distance between the first and second points listed in `grid`. Larger values of delta result
                       in gradients representing larger scale gradients.
        trend (torch.tensor or np.ndarray): an (i,) vector defining a globally preferential gradient direction. 
        crs (str) : The coordinate system positions in this CSet are encoded in. Can be "global" (default), "model" or "event".
    """

    # TODO - add a constraint for marker horizons (loss involves deviation from best-fit plane through reconstructed trace
    #                                               positions )
    
    # local constraints
    vp : torch.tensor = None
    vv : torch.tensor = None
    gp : torch.tensor = None
    gv : torch.tensor = None
    gop : torch.tensor = None
    gov : torch.tensor = None
    pp : torch.tensor = None
    pv : torch.tensor = None
    iq : tuple = None # inequality constraints
    eq : tuple = None # equality constraints
    
    # global constraints
    grid : Grid = None # predefined grid, or params for sampling random ones
    delta : float = None # step to use when computing numerical derivatives 
    trend : torch.tensor = None # global preferential gradient direction vector
    # axis: an (i,) vector defining a globally preferential axis direction.
    
    # also store coordinate system that positions are specified in
    crs : str = 'global'
    
    # place to store offset vectors based on delta used for numerical gradient computation.
    _offset : torch.tensor = field(init=False, default=None)  
    
    def torch(self):
        """
        Return a copy of these constraints cast to pytorch tensors. 
        """
        args = {}
        for k in dir(self):
            if '_' not in k and not callable(getattr(self, k)):
                attr = getattr(self, k)
                if attr is None: continue # easy
                if isinstance(attr, str): args[k] = attr # also easy - e.g., CRS for attribute
                elif k == 'iq': # inequalities are special
                    o = (attr[0], [] )
                    for i in range(len(attr[1])):
                        # convert P1 and P2 to tensor
                        if not isinstance( attr[1][i][0], torch.Tensor): # possibly already a tensor
                            o[1].append( (_tensor( attr[1][i][0], dev=curlew.device, dt=curlew.dtype ),
                                            _tensor( attr[1][i][1], dev=curlew.device, dt=curlew.dtype ),
                                            attr[1][i][2] ) )
                        else:
                            o[1].append( (attr[1][i][0], attr[1][i][1], attr[1][i][2] )) # already tensors
                    attr = o
                elif k == 'eq': # equalities are also special (keep as a list of tensors as shape will differ)
                    attr = [ _tensor( t, dev=curlew.device, dt=curlew.dtype ) for t in attr ]            
                else:
                    if attr is not None:
                        if isinstance( attr, (np.ndarray, list, tuple) ): # convert array-like types to tensor
                            attr = _tensor( attr, dev=curlew.device, dt=curlew.dtype )
                args[k] = attr
        return CSet(**args)
    
    def numpy(self):
        """
        Return a copy of these constraints cast to numpy arrays if necessary.
        """
        args = {}
        for k in dir(self):
            if '_' not in k and not callable(getattr(self, k)):
                attr = getattr(self, k)
                if attr is None: continue # easy
                if isinstance(attr, str): args[k] = attr # also easy - e.g., CRS for attribute
                elif k == 'iq': # inequalities are special
                    o = (attr[0], [] )
                    for i in range(len(attr[1])):
                        p1, p2, rel = attr[1][i]
                        if isinstance(p1, torch.Tensor):
                            p1 = _numpy(p1)
                        if isinstance(p2, torch.Tensor):
                            p2 = _numpy(p2)
                        o[1].append((np.asarray(p1), np.asarray(p2), rel))
                    attr = o   
                elif k == 'eq': # equalities are also special (list of arrays as shape differs)
                    attr = [ _numpy( t ) for t in attr ]    
                else:
                    if attr is not None:
                        if isinstance(attr, torch.Tensor ):
                            attr = _numpy(attr)
                args[k] = attr
        return CSet(**args)

    # TODO - remove these IO functions; not really needed? (or move to IO)
    def toPLY( self, path ):
        """
        Quickly save this CSet to a PLY file for visualisation in a 3D viewer (e.g., CloudCompare).
        """

        from curlew.io import savePLY
        from pathlib import Path
        path = Path(path)
        C = self.numpy()
        if self.vp is not None: savePLY( path / 'value.ply', xyz=C.vp, attr=C.vv[:,None])
        if self.gp is not None: savePLY( path / 'gradient.ply', xyz=C.gp, attr=C.gv)
        if self.gop is not None: savePLY( path / 'orientation.ply', xyz=C.gop, attr=C.gov)
        if self.iq is not None: 
            lkup = {'=':'eq','<':'lt','>':'gt'}
            for i,iq in enumerate(C.iq[1]):
                savePLY( path / str(f'iq_{i}_{lkup[iq[2]]}/lhs.ply'), xyz=iq[0], rgb=[(255,0,0) for i in range(len(iq[0]))])
                savePLY( path / str(f'iq_{i}_{lkup[iq[2]]}/rhs.ply'), xyz=iq[1], rgb=[(0,0,255) for i in range(len(iq[1]))])

    def toCSV( self, path ):
        from pathlib import Path
        import pandas as pd
        
        path = Path(path)
        C = self.numpy()
        def saveCSV( path, xyz, attr=None, names=[], rgb=None ):
            
            cols = ['x','y','z']+names
            if rgb is not None:
                cols += ['r','g','b']
            vals = xyz
            if attr is not None:
                vals = np.hstack([vals, attr])
            if rgb is not None:
                vals = np.hstack([vals, rgb])
            df = pd.DataFrame( vals, columns=cols )
            df.to_csv( path )

        if self.vp is not None: saveCSV( path / 'value.csv', xyz=C.vp, attr=C.vv[:,None], names=['value'])
        if self.gp is not None: saveCSV( path / 'gradient.csv', xyz=C.gp, attr=C.gv, names=['gx','gy', 'gz'])
        if self.gop is not None: saveCSV( path / 'orientation.csv', xyz=C.gop, attr=C.gov, names=['gox','goy', 'goz'])
        if self.iq is not None: 
            lkup = {'=':'eq','<':'lt','>':'gt'}
            for i,iq in enumerate(C.iq[1]):
                saveCSV( path / str(f'iq_{i}_{lkup[iq[2]]}/lhs.csv'), xyz=iq[0], rgb=[(255,0,0) for i in range(len(iq[0]))])
                saveCSV( path / str(f'iq_{i}_{lkup[iq[2]]}/rhs.csv'), xyz=iq[1], rgb=[(0,0,255) for i in range(len(iq[1]))])

    def copy(self):
        """Creates a copy of this CSet instance."""
        out = copy.deepcopy(self)
        if self.grid is not None:
            out.grid = self.grid.copy() # ensure grid is a copy
        return out
    
    def transform(self, f ):
        """
        Apply the specified function to each position stored in this constraint set. Note that this does NOT transform
        values, including values like gradients or orientations that may no longer be valid after (some) spatial transformations.

        Parameters
        ----------
        f : callable
            A function taking a set of points as input, such that `f(x)` returns the transformed positions.

        Returns
        -------
            A copy of this CSet instance with all positions transformed.
        """
        out = self.copy()
        if out.vp is not None: out.vp = f(out.vp)
        if out.gp is not None: out.gp = f(out.gp)
        if out.gop is not None: out.gop = f(out.gop)
        if out.pp is not None: out.pp = f(out.pp)
        if out.iq is not None:
            for i in range(len(out.iq[1])):
                out.iq[1][i] = ( f(out.iq[1][i][0]), # LHS
                                 f(out.iq[1][i][1]), # RHS
                                 out.iq[1][i][2] ) # relation
        if out.eq is not None:
            out.eq = [f(t) for t in out.eq]
        
        if self.grid is not None: 
            from curlew.utils import batchEval
            out.grid._setCache( batchEval( self.grid.coords(), f ) )

        return out

    def _rotate_vectors(self, R, arr):
        """Apply the linear (rotation/scale) part of an affine transform to vectors."""
        if arr is None:
            return None
        R = np.asarray(R)
        if isinstance(arr, torch.Tensor):
            Rt = _tensor(R, dev=arr.device, dt=arr.dtype)
            if arr.ndim == 1:
                return Rt @ arr
            return arr @ Rt.T
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return R @ arr
        return arr @ R.T

    def to_model(self, T: Transform, inverse=False):
        """
        Map constraint positions and direction vectors with the GeoModel homogeneous transform.

        Parameters
        ----------
        T : Transform
            GeoModel transform (global → model when ``inverse=False``).
        inverse : bool
            If False (default), map ``crs='global'`` to ``crs='model'``.
            If True, map ``crs='model'`` back to ``crs='global'`` using ``T.inverse()``.

        Returns
        -------
        CSet
            A copy with the target coordinate reference set.
        """
        if inverse:
            if self.crs != "model":
                return self.copy()
            dst_crs = "global"
        else:
            if self.crs != "global":
                return self.copy()
            dst_crs = "model"

        if T is None or T.isIdentity():
            out = self.copy()
            out.crs = dst_crs
            return out

        Ta = T.inverse() if inverse else T
        out = self.transform(Ta.apply)
        d = Ta.matrix.shape[0] - 1
        R = Ta.matrix[:d, :d]
        out.gv = self._rotate_vectors(R, out.gv)
        out.gov = self._rotate_vectors(R, out.gov)
        if out.trend is not None:
            out.trend = self._rotate_vectors(R, out.trend)
        out.crs = dst_crs
        return out

    def filter(self, f):
        """
        Apply the specified filter to each position stored in this constraint set.

        Parameters
        ----------
        f : callable
            A function taking a set of positions as input, such that `f(x)` returns True if the point should be retained, and False otherwise.
        
        Returns
        -------
            A copy of this CSet instance with the filter applied to potentially remove points.
        """
        out = self.copy()
        def e( arr ):
            mask = f( arr )
            if isinstance(arr, torch.Tensor): mask = _tensor(mask, dev=curlew.device, dt=torch.bool)
            if isinstance(arr, np.ndarray): mask = np.array(mask, dtype=bool)
            return mask
        if out.vp is not None: 
            mask = e(out.vp)
            out.vp = out.vp[mask,:]
            out.vv = out.vv[mask]
        if out.gp is not None: 
            mask = e(out.gp)
            out.gp = out.gp[mask,:]
            out.gv = out.gv[mask, :]
        if out.gop is not None: 
            mask = e(out.gop)
            out.gop = out.gop[mask,:]
            out.gov = out.gov[mask, :]
        if out.pp is not None: 
            mask = e(out.pp)
            out.pp = out.pp[mask,:]
            out.pv = out.pv[mask, ...]
        if out.iq is not None:
            iq_list = []
            for i in range(len(out.iq[1])):
                lhs = out.iq[1][i][0][ e(out.iq[1][i][0]), : ]
                rhs = out.iq[1][i][1][ e(out.iq[1][i][1]), : ]
                if (len(lhs) > 0) and (len(rhs) > 0):
                    iq_list.append((lhs, rhs, out.iq[1][i][2]))

            out.iq = None if len(iq_list) == 0 else (out.iq[0], iq_list)
        if out.eq is not None:
            eq = [e(t) for t in out.eq] # apply filter
            eq = [t for t in eq if len(t) > 0] # drop zero-length arrays
            if len(eq) > 0:
                out.eq = eq
            else:
                out.eq = None # no eq constraints remain
        return out
    
@dataclass
class HSet:
    """
    Set of hyperparameters used for training one or several NFs. Values can be 0 to completely disable a 
    loss function, or a string "1.0" or "0.1" to initialise the hyperparameter as the specified fraction of the `1 / initial_loss`. Note that 
    simpler loss functions (i.e. with most of the different loss components set to 0) can be much easier to optimise, so try to keep things simple.
        
    Attributes:
        value_loss : float  | str
            Factor applied to value losses. Default is 1.
        grad_loss : float  | str
            Factor applied to gradient losses. Default is 1 (as this loss generally ranges between 0 and 1).
        ori_loss : float  | str
            Factor applied to orientation losses. Default is 1 (as this is fixed to the range 0 to 1).
        thick_loss : float  | str
            Factor applied to thickness loss. Default is 0 (disabled); set to 1 or a string fraction to enable.
        mono_loss : float  | str
            Factor applied to monotonicity (divergence) loss. Default is 0 (disabled); ``"0.01"`` initialises automatically. 
        flat_loss : float  | str
            Factor applied to global trend misfit. Default is 0 (disabled); 0.1 is a typical value when enabled.
        prop_loss : float  | str
            Factor applied to property-field (forward model) loss. Default is 0 (disabled).
        iq_loss : float | str
            Factor applied to scale the loss resulting from any provided inequality constraints.
        eq_loss : float | str
            Factor applied to scale the loss resulting from equality (trace) constraints in ``CSet.eq``.
        use_dynamic_loss_weighting : bool
            Enables dynamic task loss weighting based on real-time loss values. Default is False.
            This approach ensures that each task contributes equally in magnitude (≈1)
            while still allowing non-zero gradients. It effectively adjusts the relative 
            gradient scale of each task based on its current loss.
        one_hot : bool
            Enables one-hot encoding of the scalar field value according to the event-ID. Only works with property field HSet()s.
        reuse_worst_half : float
            Fraction of inequality constraint pairs to retain as the "worst" from the previous epoch and reuse in the
            next; the remainder are drawn randomly. 0 = disabled. 0.5 = keep worst 50%, redraw the other 50%.
            Helps convergence by focusing the optimiser on high-loss inequality pairs.
    """
    
    value_loss : float = 1
    grad_loss : float = 1
    ori_loss : float = 0
    thick_loss : float = 0 # 1
    mono_loss : float = 0 # "0.01"
    flat_loss : float = 0 # 0.1
    prop_loss : float = 0 # "1.0"
    iq_loss : float = 0
    eq_loss : float = 0
    use_dynamic_loss_weighting : bool = False
    one_hot : bool = False
    reuse_worst_half : float = 0.5

    def copy(self, **kwargs):
        """
        Creates a copy of the HSet instance. Pass keywords to then update specific parts of this copy.
        
        Keywords
        --------
        Keywords can be provided to adjust hyperparameters after making the copy.

        """
        out = copy.deepcopy( self )
        for k,v in kwargs.items():
            out.__setattr__(k, v)
        return out
    
    def zero(self, **kwargs):
        """
        Set all hyperparameters in the HSet to zero and return. Useful to disable all losses before setting a few relevant ones.

        Keywords
        --------
        Any non-zero hyperparameters can be passed as keywords along with their desired value.
        """
        for k in dir(self):
            if '__' not in k:
                if not callable(getattr(self, k)):
                    setattr(self, k, kwargs.get(k, 0 ) )
        return self
    
@dataclass
class Geode( object ):
    """
    An "egg-like" class containing all the juicy outputs of a curlew model.

    Attributes:
        x (dict): Maps coordinate-system names to (N,d) position arrays. Keys include
                  ``"global"``, ``"model"``, each ``GeoEvent`` name, and ``"<fieldName>_local"``
                  for fields with a non-identity local transform (i.e. only for fields actively using
                  global anisotropy of some form).
        grid (curlew.geometry.Grid): A `curlew.geometry.Grid` class if points were sampled from a regular grid.
        crs (str) : The default coordinate-system key into ``x`` for this Geode.
        lithoID (torch.tensor or np.ndarray): (N,) array of lithology classes defined by isosurfaces described in the relevant `GeoEvent` instance(s).
        softLithoID (torch.tensor): Differentiable proxy for ``lithoID`` (populated when ``predict(..., to_numpy=False)``).
        lithoLookup (dict): A dictionary where keys are lithoID integers and values are the name of the associated isosurfaces.
        structureID (torch.tensor or np.ndarray): (N,) array of structure IDs denoting the index of the `GeoEvent` responsible for each lithology / value
                                                  in the model result.
        softStructureID (torch.tensor): Differentiable proxy for ``structureID`` (populated when ``predict(..., to_numpy=False)``).
        structureLookup (dict): A dictionary where keys are structureIDs and values give the name of the corresponding `GeoEvent`.
        scalar (torch.tensor or np.ndarray): (N,) array of the scalar values evaluated at each `x`.
        properties (torch.tensor or np.ndarray): (N,d) array of property values derived at each `x`.
        propertyNames (list): List of `d` property names corresponding to each dimension of `self.properties`.
        fields (dict): Dict of scalar values at each `x`, keyed by underlying field name (``BaseSF.name``), not ``GeoEvent`` name.
        offsets (dict): Dict containing the individual displacement fields evaluated at each `x` for each `GeoEvent` instance in the model.
    """

    # local constraints
    x : dict = field(default_factory=dict) # coordinate systems -> (N,d) positions
    grid : Grid = None # grid of points at which model was evaluated
    crs : str = None # default coordinate-system key into ``x``

    lithoID : torch.tensor = None
    softLithoID : torch.tensor = None  # differentiable lithology proxy (training only)
    lithoLookup : dict = field(default_factory=dict)

    structureID : torch.tensor = None
    softStructureID : torch.tensor = None  # differentiable structure proxy (training only)
    structureLookup : dict = field(default_factory=dict)

    scalar : torch.tensor = None # scalar field values
    gradient : torch.tensor = None # gradient field values (often left as None)
    properties : torch.tensor = None # forward (property) predictions
    propertyNames : list = field(default_factory=list)

    fields : dict = field(default_factory=dict) # individual scalar fields (keyed by BaseSF.name)
    offsets : dict = field(default_factory=dict) # individual displacement fields (Keyed by GeoEvent name)

    isosurfaces : dict = field(default_factory=dict) # individual isosurfaces (Keyed by GeoEvent name)
    anchors : dict = field(default_factory=dict) # individual anchor points (Keyed by GeoEvent name)
    
    def __post_init__(self):
        """Accept legacy array ``x`` values and normalise to a coordinate dict."""
        if self.x is None:
            self.x = {}
        elif not isinstance(self.x, dict):
            key = self.crs or "model"
            self.x = {key: self.x}

    def coords(self, crs=None):
        """
        Return positions for the requested coordinate system.

        Parameters
        ----------
        crs : str, optional
            Coordinate-system key into ``x``. Defaults to ``self.crs``.
        """
        key = crs or self.crs
        if key is None:
            if len(self.x) == 1:
                key = next(iter(self.x))
            else:
                raise ValueError("No coordinate system specified and Geode.crs is None")
        if key not in self.x:
            raise KeyError(
                f"Coordinates for {key!r} not found in Geode.x (available: {list(self.x)})"
            )
        return self.x[key]

    def set_coords(self, key, coords, primary=False):
        """Store ``coords`` under ``key`` in ``x``; optionally set as default ``crs``."""
        self.x[key] = coords
        if primary or self.crs is None:
            self.crs = key

    def __len__(self):
        if not self.x:
            return 0
        return len(self.coords())

    def getSurface(self, field, name, normals=False):
        """
        Get a mesh for given isosurface for a given field.
        """
        assert self.grid is not None, "Surfaces can only be computed for evaluations on a grid"
        scalar = self.fields[field]
        iso = self.isosurfaces[field][name]
        return self.grid.contour(scalar, iso=iso, normals=normals)

    def getSurfaces(self, normals=False):
        """
        Get a dictionary of meshes for all isosurfaces for all fields.
        """
        out = {}
        for field in self.fields:
            for name in self.isosurfaces[field]:
                out[f'{field}_{name}'] = self.getSurface(field, name, normals=normals)
        return out
        
    @classmethod
    def concat(cls, geodes):
        """
        Concatenate an ordered list of Geodes. Used when e.g., evaluating large models in chunks.
        """
        args = {}
        for g in geodes:
            for k in dir(g):
                if ('_' not in k) and not callable(getattr(g, k)):
                    attr = getattr(g, k)
                    if attr is not None:
                        if k in args:
                            if k == 'x':
                                for k2, v2 in attr.items():
                                    if k2 in args[k]:
                                        if isinstance(v2, torch.Tensor):
                                            args[k][k2] = torch.concat([args[k][k2], v2])
                                        elif isinstance(v2, np.ndarray):
                                            args[k][k2] = np.concatenate([args[k][k2], v2])
                                    else:
                                        args[k][k2] = v2
                            elif isinstance(attr, torch.Tensor ):
                                args[k] = torch.concat([args[k], attr])
                            elif isinstance(attr, np.ndarray ):
                                args[k] = np.concatenate([args[k], attr])
                            elif isinstance(attr, dict):
                                temp = {**args[k], **attr}
                                for k2,v in args[k].items(): # also concatenate any relevant dict entries
                                    if k2 in attr:
                                        if isinstance(v, torch.Tensor ):
                                            temp[k2] = torch.concat( [v, attr[k2]] )
                                        elif isinstance(v, np.ndarray ):
                                            temp[k2] = np.concatenate( [v, attr[k2]] )
                                args[k] = temp
                        else:
                            if k == 'x':
                                args[k] = dict(attr)
                            else:
                                args[k] = attr
        if geodes and geodes[0].crs is not None:
            args.setdefault('crs', geodes[0].crs)
        return Geode(**args)

    def combine(self, younger, weight, soft_litho_weight=None, soft_structure_weight=None):
        """
        Combine the results from this Geode with results from a (typically younger) one, using the 
        specified weights. Both Geodes must be evaluated at the same coordinates.

        Parameters
        ----------
        younger : Geode
            Results from the younger ``GeoEvent`` (same length and coordinate systems as ``self``).
        weight : torch.Tensor
            Hard overprint weights (typically 0/1) for ``scalar``, ``structureID``, and ``lithoID``.
        soft_litho_weight : torch.Tensor, optional
            Differentiable overprint weights for ``softLithoID``. Defaults to ``weight``.
        soft_structure_weight : torch.Tensor, optional
            Differentiable overprint weights for ``softStructureID``. Defaults to ``weight``.

        Returns
        -------
        Geode
            Combined model outputs.
        """
        assert len(self) == len(younger), "Both Geodes must be evaluated at the same coordinates."
        iweight = 1-weight
        if soft_litho_weight is None:
            soft_litho_weight = weight
        if soft_structure_weight is None:
            soft_structure_weight = weight
        isoft_litho = 1 - soft_litho_weight
        isoft_structure = 1 - soft_structure_weight

        # combine basic attributes
        args = dict(x={**self.x, **younger.x}, grid=younger.grid, crs=younger.crs, # always take these from the younger object
                    lithoLookup={**self.lithoLookup, **younger.lithoLookup},
                    structureLookup={**self.structureLookup, **younger.structureLookup},
                    fields={**self.fields, **younger.fields},
                    offsets={**self.offsets, **younger.offsets} )

        # combine gradients (if not None)
        if self.gradient is not None:
            if isinstance(self.gradient, np.ndarray):
                args['gradient'] = self.gradient.copy() # copy (numpy)
            else:
                args['gradient'] = self.gradient.clone() # clone (pytorch)
            if younger.gradient is not None: # overprint with younger values if defined
                if len(younger.gradient) == 2:
                    pass
                args['gradient'] = younger.gradient*weight[:,None] + args['gradient']*iweight[:,None]

        # combine property predictions (if not None)
        if self.properties is not None:
            args['propertyNames'] = self.propertyNames
            assert np.all(np.array(self.propertyNames) == np.array(younger.propertyNames)),\
                f'Property names for {list(self.fields.keys())[0]} do not match {list(younger.fields.keys())[0]}.'
            assert younger.properties is not None,\
                f"Properties must be defined for all generative fields. {list(younger.fields.keys())[0]} is missing."
            args['properties'] = younger.properties*weight[:,None] + self.properties*iweight[:,None]

        # combine scalar values, structure IDs and lithoIDs (if defined)
        args['scalar'] = younger.scalar*weight + self.scalar*iweight
        args['structureID'] = torch.round((younger.structureID*weight + self.structureID*iweight)).to(dtype=torch.int) # round to integer
        if (self.softStructureID is not None) and (younger.softStructureID is not None):
            args['softStructureID'] = younger.softStructureID*soft_structure_weight + self.softStructureID*isoft_structure
        elif younger.softStructureID is not None:
            args['softStructureID'] = younger.softStructureID*soft_structure_weight
        elif self.softStructureID is not None:
            args['softStructureID'] = self.softStructureID*isoft_structure
        if (self.lithoID is not None) and (younger.lithoID is not None):
            args['lithoID'] = torch.round(younger.lithoID*weight + self.lithoID*iweight).to(dtype=torch.int) # round to integer
        if (self.softLithoID is not None) and (younger.softLithoID is not None):
            args['softLithoID'] = younger.softLithoID*soft_litho_weight + self.softLithoID*isoft_litho
        elif younger.softLithoID is not None:
            args['softLithoID'] = younger.softLithoID*soft_litho_weight
        elif self.softLithoID is not None:
            args['softLithoID'] = self.softLithoID*isoft_litho

        return Geode(**args)

    def stackValues(self, mn=0, mx=1):
        """
        Return a copy of this Geode with scalar values scaled so that they vary between mn and mx for each structural field, and increase
        monotonically with the structure ID so that there are no overlaps between structures. This can be useful for plotting or isosurface 
        extraction.

        Parameters
        ----------
        mn : float, optional
            The minimum value to scale the scalar values to, by default 0. 
        mx : float, optional
            The maximum value to scale the scalar values to, by default 1.

        Returns
        -------
        curlew.core.Geode
            A new Geode where the scalar values are scaled to the range [mn, mx] for each structure and offset
            to remove overlaps.
        """
        out = self.numpy()

        # get the unique structure IDs
        ids = np.sort( np.unique(out.structureID) )[::-1]

        # create a new array to hold the stacked values
        stacked = np.zeros_like(out.scalar)

        # loop over each structure ID
        for i, id in enumerate(ids):
            # get the indices of the current structure ID
            idx = np.where(out.structureID == id)[0]

            # scale the scalar values to the range [mn, mx]
            sf = out.scalar[idx]
            if np.max(sf) - np.min(sf) == 0:
                # if all values are the same, set them to mn
                scaled_values = np.full_like(sf, mn)
            else:
                # scale the values to the range [mn, mx]        
                scaled_values = mn + (mx - mn) * (sf - np.min(sf)) / (np.max(sf) - np.min(sf))

            # add an offset based on the index of the structure ID
            stacked[idx] = scaled_values + i * (mx - mn)

        out.scalar = stacked
        return out

    def torch(self):
        """
        Return a copy of these results cast to pytorch tensors (where relevant). 
        """
        args = {}
        for k in dir(self):
            if '_' not in k and not callable(getattr(self, k)):
                attr = getattr(self, k)
                if attr is not None:
                    if isinstance( attr, (np.ndarray, list, tuple) ): # convert array-like types to tensor
                        attr = _tensor( attr, dev=curlew.device, dt=curlew.dtype )
                    elif isinstance(attr, dict):
                        attr = dict(attr)
                        for key in attr.keys():
                            if isinstance( attr[key], (np.ndarray, list, tuple) ):
                                attr[key] = _tensor( attr[key], dev=curlew.device, dt=curlew.dtype )
                args[k] = attr
        return Geode(**args)

    def numpy(self):
        """
        Return a copy of these constraints cast to numpy arrays if necessary.
        """
        args = {}
        for k in dir(self):
            if '_' not in k and not callable(getattr(self, k)):
                attr = getattr(self, k)
                if attr is not None:
                    if isinstance(attr, torch.Tensor ):
                        attr = _numpy(attr)
                    elif isinstance(attr, dict):
                        attr = dict(attr)
                        for key in attr.keys():
                            if isinstance( attr[key], torch.Tensor ):
                                attr[key] = _numpy(attr[key])
                args[k] = attr
        return Geode(**args)

    # TODO - remove these / move to curlew.IO.save
    def toPLY( self, path ):
        """
        Quickly save this result to a PLY file for visualisation in a 3D viewer (e.g., CloudCompare).
        """
        from curlew.io import savePLY
        from pathlib import Path
        path = Path(path)
        R = self.numpy()
        assert R.x, "Coordinates must be defined to save PLY"
        attrs = []
        attr_names = []
        for n in ['lithoID', 'structureID', 'scalar']:
            a = R.__getattribute__(n)
            if a is not None:
                attrs.append(a)
                attr_names.append(n)
        if R.properties is not None:
            attr_names += R.propertyNames
            attrs += R.properties
        attrs = np.vstack( attrs )
        savePLY( path, xyz=R.coords(), attr=attrs, names=attr_names )

    def toCSV( self, path ):
        """
        Quickly save this result to a CSV file for easy interoperability.
        """
        pass # TODO
    
@dataclass
class Pebble( object ):
    """
    A class for encapsulating loss terms computed by one or more learnable objects (anything that inherits
    `curlew.core.LearnableBase`, including e.g., fields, deformations, and models).
    
    Attributes:
        losses (dict of dicts): A dictionary of dictionaries such that `losses[<group>][<item>]` returns a specific loss item and the name of the
                       LearnableBase object that created it.
        weights (dict of dicts): A dict of dicts defining the weight terms (hyperparameters) used to balance each loss in losses during aggregation. 
                        Indexing is the same as `losses`.
        optim (dict): A dictionary that returns the optimiser object used for each group of losses (i.e. `optim[<group>]` gives the optimiser
                      used to update all learnable parameters associated to the `losses[<group>]` dict.
        detached (bool): True if this pebble has been detached for storage (numpy losses, no optimisers).
    """
    
    losses : dict = field(default_factory=dict)
    weights : dict = field(default_factory=dict)
    optim : dict = field(default_factory=dict)
    detached : bool = False

    def _require_active(self, action: str):
        if self.detached:
            raise RuntimeError(f"Cannot {action} on a detached Pebble")

    def detach(self):
        """
        Return a detached copy with numpy loss values and no optimisers, for storing per-iteration summaries.
        """
        losses = {
            g: {
                n: _numpy(v.detach() if isinstance(v, torch.Tensor) else v)
                for n, v in terms.items()
            }
            for g, terms in self.losses.items()
        }
        return Pebble(
            losses=losses,
            weights={g: dict(v) for g, v in self.weights.items()},
            optim={g: None for g in self.losses},
            detached=True,
        )
    def __str__(self):
        """Single-line summary suitable for progress-bar descriptions."""
        if not self.losses:
            return "L=0"

        parts = []
        total = 0.0
        multi_group = len(self.losses) > 1

        for group, terms in self.losses.items():
            for name, loss in terms.items():
                weight = self.weights.get(group, {}).get(name, 1.0)
                if hasattr(loss, "detach"):
                    val = loss.detach()
                    val = float(val.item()) if val.numel() == 1 else None
                elif isinstance(loss, np.ndarray):
                    val = float(loss.item()) if loss.size == 1 else None
                else:
                    val = float(loss) if isinstance(loss, (int, float)) else None

                if val is None:
                    label = f"{group}/{name}" if multi_group else name
                    parts.append(f"{label}=…")
                    continue

                weighted = weight * val
                total += weighted
                label = f"{group}/{name}" if multi_group else name
                parts.append(f"{label}={weighted:.3f}")

        return f"L={total:.3f} " + " ".join(parts)

    def total(self):
        """Return the weighted sum of all loss terms (tensor if active, float if detached)."""
        if self.detached:
            total = 0.0
            for group, terms in self.losses.items():
                for name, loss in terms.items():
                    weight = self.weights.get(group, {}).get(name, 1.0)
                    total += weight * float(np.asarray(loss).item())
            return total

        out = None
        for group, terms in self.losses.items():
            for name, loss in terms.items():
                weight = self.weights.get(group, {}).get(name, 1.0)
                term = weight * loss
                out = term if out is None else out + term
        if out is None:
            return _tensor(0.0).requires_grad_(True)
        return out

    def group_total(self, group):
        """Return the weighted sum of loss terms in one group."""
        terms = self.losses.get(group, {})
        if not terms:
            return 0.0 if self.detached else _tensor(0.0).requires_grad_(True)

        if self.detached:
            total = 0.0
            for name, loss in terms.items():
                weight = self.weights.get(group, {}).get(name, 1.0)
                total += weight * float(np.asarray(loss).item())
            return total

        out = None
        for name, loss in terms.items():
            weight = self.weights.get(group, {}).get(name, 1.0)
            term = weight * loss
            out = term if out is None else out + term
        return out
    
    def push( self, group : str, name : str, loss : torch.Tensor, weight : float = 1.0, optim : torch.optim.Optimizer = None ):
        """
        Add a new loss term to this Pebble object.
        
        Parameters
        ----------
        group : str
            The group of losses to add the new loss term to (typically the 
            name of the field or object that created the loss).
        name : str
            The name of the loss term to add. Useful for evaluating the different components of complex optimisations.
        loss : torch.Tensor
            The loss term to add.
        weight : float, optional
            The weight (hyperparameter) to apply to the loss term. Default is 1.0.
        optim : torch.optim.Optimizer, optional
            The optimiser to use for this loss term. Can be None if an optimiser for this group has already been defined. Can also 
            be a dictionary of optimisers (keyed by group name) if this loss relates to multiple groups.
        """
        self._require_active("push")
        self.losses.setdefault(group, {})[name] = loss
        self.weights.setdefault(group, {})[name] = weight
        if optim is not None:
            if isinstance(optim, dict):
                for k, v in optim.items(): # add multiple optimisers
                    self.optim[k] = v
            else:
                self.optim[group] = optim
        
    def __add__(self, other):
        """Combine two Pebbles by merging their internal dicts."""
        if not isinstance(other, Pebble):
            return NotImplemented
        if self.detached != other.detached:
            raise ValueError("Cannot combine detached and active Pebbles")

        out = Pebble(
            losses={g: dict(v) for g, v in self.losses.items()},
            weights={g: dict(v) for g, v in self.weights.items()},
            optim=dict(self.optim),
            detached=self.detached,
        )

        for attr in ("losses", "weights"):
            dst = getattr(out, attr)
            for g, items in getattr(other, attr).items():
                if g in dst:
                    overlap = set(dst[g]) & set(items)
                    if overlap:
                        raise ValueError(f"Duplicate keys in {attr}[{g!r}]: {overlap}")
                    dst[g].update(items)
                else:
                    dst[g] = dict(items)

        for g, opt in other.optim.items():
            if g in out.optim and out.optim[g] is not None and opt is not None and out.optim[g] is not opt:
                raise ValueError(f"Conflicting optimisers for group {g!r}")
            if opt is not None:
                out.optim[g] = opt

        return out
    
    def zero(self, exclude=[]):
        """
        Zero all optimisers that have been associated with losses included in this pebble, except those named
        in the `exclude` list. 
        """
        self._require_active("zero")
        for n, o in self.optim.items():
            if (n not in exclude) and (o is not None):
                o.zero_grad()
    
    def step(self, exclude=[]):
        """
        Step all optimisers that have been associated with losses included in this pebble, except those named
        in the `exclude` list. 
        """
        self._require_active("step")
        for n, o in self.optim.items():
            if (n not in exclude) and (o is not None):
                o.step()
                
    def backward( self, exclude=[], retain_graph=False):
        """
        Run backward pass to accumulate gradients on each relevant optimiser, excluding those named in the `exclude` list. Set
        `retain_graph` to True if some loss terms are shared between backward passes (e.g., if one learnable object has 
        constructed a loss term by modifying losses from other objects).
        """
        self._require_active("backward")
        for g, o in self.optim.items(): # loop through all (group, optimiser) 
            if g not in exclude:
                loss = None # aggregate loss terms
                for n, v in self.losses.get(g, {}).items(): # loop through all (name, loss) in group
                    h = self.weights.get(g, {}).get(n, 1.0) # hyperparameter (balancing term)
                    term = h * v
                    loss = term if loss is None else loss + term
                
                # run backward
                if loss is not None:
                    loss.backward(retain_graph=retain_graph)
                    
        # TODO - random thought; is there a way to summarise the magnitude of the
        # gradients that result for the main learnable parameters? Might be a useful metric. 