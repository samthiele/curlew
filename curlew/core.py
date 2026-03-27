"""
Define several core curlew types for storing data and hyperparameters. 
"""
import numpy as np
import torch
from dataclasses import dataclass, field
import copy
import curlew
from curlew import _numpy, _tensor
from curlew.geometry import Grid
from typing import Union

import torch.optim as optim
import torch.nn as nn


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
    
    def zero(self):
        """
        Zero gradients in the optimiser for this NF.
        """
        if self.optim is not None:
            self.optim.zero_grad()

    def step(self):
        """
        Step the optimiser for this NF.
        """
        if (self.optim is not None) and not self.frozen:
            self.optim.step()

    def set_rate(self, lr=1e-2 ):
        """
        Update the learning rate for this learnable object's optimiser.
        """
        assert self.optim is not None, "Optimiser not initialised. Call `init_optim()` first."
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def loss(self):
        """Should be implemented by child classess that incur losses."""
        return _tensor(0).requires_grad_(True), dict()
    
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
                    defining positions at which to evaluate inequality constraints such as `P1 > P2`. `iq` defines the inequality to evaluate, and can be `<`, `=` or `>`.
                    Note that this inequality is computed for a random set of `N` pairs sampled from `P1` and `P2`. 
        grid (tuple, torch.tensor or np.ndarray): A `curlew.geometry.Grid` instance defining the grid used to enforce global constraints (and associated sampling strategy).
        delta (float): The step size used when computing numerical derivatives at the grid points. Default (None) is to initialise
                       as half the distance between the first and second points listed in `grid`. Larger values of delta result
                       in gradients representing larger scale gradients.
        trend (torch.tensor or np.ndarray): an (i,) vector defining a globally preferential gradient direction. 
    """

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

    # global constraints
    grid : Grid = None # predefined grid, or params for sampling random ones
    delta : float = None # step to use when computing numerical derivatives 
    trend : torch.tensor = None # global preferential gradient direction vector
    # axis: an (i,) vector defining a globally preferential axis direction.
    
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
                if k == 'iq': # inequalities are special
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
                if k == 'iq': # inequalities are special
                    o = (attr[0], [] )
                    for i in range(len(attr[1])):
                        p1, p2, rel = attr[1][i]
                        if isinstance(p1, torch.Tensor):
                            p1 = _numpy(p1)
                        if isinstance(p2, torch.Tensor):
                            p2 = _numpy(p2)
                        o[1].append((np.asarray(p1), np.asarray(p2), rel))
                    attr = o     
                else:
                    if attr is not None:
                        if isinstance(attr, torch.Tensor ):
                            attr = _numpy(attr)
                args[k] = attr
        return CSet(**args)

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
    
    def transform(self, f, batch=50000 ):
        """
        Apply the specified function to each position stored in this constraint set.

        Parameters
        ----------
        f : callable
            A function taking a set of points as input, such that `f(x)` returns the transformed positions.
        batch : int
            The batch size to use for reconstructing grids (as these can be quite large).
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
        if self.grid is not None: 
            from curlew.utils import batchEval
            out.grid._setCache( batchEval( self.grid.coords(), f ) )

        # TODO -- use autodiff to rotate gradient constriants??

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
            for i in range(len(out.iq[1])):
                out.iq[1][i] = ( out.iq[1][i][0][ e(out.iq[1][i][0]), : ], # LHS
                                 out.iq[1][i][1][ e(out.iq[1][i][1]), : ], # RHS
                                 out.iq[1][i][2] ) # relation
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
            Factor applied to thickness loss. Default is 1 (as this loss is also generally small).
        mono_loss : float  | str
            Factor applied to monotonicity (divergence) loss. Default is "0.01" (initialise automatically). 
        flat_loss : float  | str
            Factor applied to global trend misfit. Default is 0.1 (as this shouldn't be too strongly applied).
        prop_loss : float  | str
            Factor applied to scale the loss resulting from reconstructed property fields (i.e. forward model misfit).
        iq_loss : float | str
            Factor applied to scale the loss resulting from any provided inequality constraints.
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
    thick_loss : float = 1
    mono_loss : float = "0.01"
    flat_loss : float = 0.1
    prop_loss : float = "1.0"
    iq_loss : float = 0
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
        x (torch.tensor or np.ndarray): (N,o) array of value constraint positions (in modern-day coordinates).
        grid (curlew.geometry.Grid): A `curlew.geometry.Grid` class if points (`x`) were sampled from a regular grid.
        crs (str) : A string denoting the coordinate reference used for `x`. Will be `'modern'` if
                    a final result (in modern coordinates), or the name of a specific `GeoField` if result is in field coordinates.
        lithoID (torch.tensor or np.ndarray): (N,) array of lithology classes defined by isosurfaces described in the relevant `GeoField` instance(s).
        lithoLookup (dict): A dictionary where keys are lithoID integers and values are the name of the associated isosurfaces.
        structureID (torch.tensor or np.ndarray): (N,) array of structure IDs denoting the index of the `GeoField` responsible for each lithology / value
                                                  in the model result.
        structureLookup (dict): A dictionary where keys are structureIDs and values give the name of the corresponding `GeoField`.
        scalar (torch.tensor or np.ndarray): (N,) array of the scalar values evaluated at each `x`.
        properties (torch.tensor or np.ndarray): (N,d) array of property values derived at each `x`.
        propertyNames (list): List of `d` property names corresponding to each dimension of `self.properties`.
        fields (dict): Dict containing the individual scalar fields evaluated at each `x` for each `GeoField` instance in the model.
        offsets (dict): Dict containing the individual displacement fields evaluated at each `x` for each `GeoField` instance in the model.
    """

    # local constraints
    x : torch.tensor = None # array of the positions at which points were evaluated
    grid : Grid = None # grid of points at which model was evaluated
    crs : str = None # temporal coordinate system (modern or paleo) associated to these points

    lithoID : torch.tensor = None
    lithoLookup : dict = field(default_factory=dict)

    structureID : torch.tensor = None
    structureLookup : dict = field(default_factory=dict)

    scalar : torch.tensor = None # scalar field values
    gradient : torch.tensor = None # gradient field values (often left as None)
    properties : torch.tensor = None # forward (property) predictions
    propertyNames : list = field(default_factory=list)

    fields : dict = field(default_factory=dict) # individual scalar fields (Keyed by GeoField name)
    offsets : dict = field(default_factory=dict) # individual displacement fields (Keyed by GeoField name)

    isosurfaces : dict = field(default_factory=dict) # individual isosurfaces (Keyed by GeoField name)
    anchors : dict = field(default_factory=dict) # individual anchor points (Keyed by GeoField name)
    
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
                            if isinstance(attr, torch.Tensor ):
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
                            args[k] = attr
        return Geode(**args)

    def combine(self, younger, weight):
        """
        Combine the results from this Geode with results from a (typically younger) one, using the 
        specified weights. Both Geodes must be evaluated at the same coordinates.
        """
        assert len(self.x) == len(younger.x), "Both Geodes must be evaluated at the same coordinates."
        iweight = 1-weight

        # combine basic attributes
        args = dict(x=younger.x, grid=younger.grid, crs=younger.crs, # always take these from the younger object
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
        if (self.lithoID is not None) and (younger.lithoID is not None):
            args['lithoID'] = torch.round(younger.lithoID*weight + self.lithoID*iweight).to(dtype=torch.int) # round to integer

        return Geode(**args)

    def stackValues(self, mn=0, mx=1):
        """
        Scale scalar values so that they vary between mn and mx for each structural field, and then add offsets 
        so that there are no overlaps between structures. This can be useful for e.g., plotting or isosurface 
        extraction.

        Parameters
        ----------
        mn : float, optional
            The minimum value to scale the scalar values to, by default 0. 
        mx : float, optional
            The maximum value to scale the scalar values to, by default 1.

        Returns
        -------
        curlew.geology.geofield.Geode
            A new Geode where the scalar values are scaled to the range [mn, mx] for each structure and offset
            to remove overlaps.
        """
        out = self.numpy()

        # get the unique structure IDs
        ids = np.unique(out.structureID)

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
                        for key in attr.keys(): # also convert any dict entries
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
                    elif isinstance(attr, dict): # also convert any dict entries
                        for key in attr.keys():
                            if isinstance( attr[key], torch.Tensor ):
                                attr[key] = _numpy(attr[key])
                args[k] = attr
        return Geode(**args)

    def toPLY( self, path ):
        """
        Quickly save this result to a PLY file for visualisation in a 3D viewer (e.g., CloudCompare).
        """
        from curlew.io import savePLY
        from pathlib import Path
        path = Path(path)
        R = self.numpy()
        assert R.x is not None, "Coordinates must be defined to save PLY"
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
        savePLY( path, xyz=R.x, attr=attrs, names=attr_names )

    def toCSV( self, path ):
        """
        Quickly save this result to a CSV file for easy interoperability.
        """
        pass # TODO