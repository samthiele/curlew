"""
Define several core curlew types for storing data and hyperparameters. 
"""
import numpy as np
import torch
from dataclasses import dataclass, field
import copy
import curlew


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
        grid (tuple, torch.tensor or np.ndarray): Either a tuple containing (N,[[xmin,xmax],[ymin,ymax],...] to use random grid points during each epoch, or a
                                            (N,i) array of positions (in modern-day coordinates) defining specific points. These points are used to define
                                            where "global" constraints are enforced.
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
    grid : torch.tensor = None # predefined grid, or params for sampling random ones
    sgrid : torch.tensor = None # tensor or array containing the last-used (random) grid
    delta : float = None # step to use when computing numerical derivatives 
    trend : torch.tensor = None # global preferential gradient direction vector
    # axis: an (i,) vector defining a globally preferential axis direction.
    
    # place to store offset vectors based on delta used for numerical gradient computation.
    _offset : torch.tensor = field(init=False, default=None)  

    def torch(self):
        """
        Return a copy of these constraints cast to pytorch tensors with the specified
        data type and hosted on the specified device. 
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
                            o[1].append( (torch.tensor( attr[1][i][0], device=curlew.device, dtype=curlew.dtype ),
                                            torch.tensor( attr[1][i][1], device=curlew.device, dtype=curlew.dtype ),
                                            attr[1][i][2] ) )
                        else:
                            o[1].append( (attr[1][i][0], attr[1][i][1], attr[1][i][2] )) # already tensors
                    attr = o            
                else:
                    if attr is not None:
                        if isinstance( attr, np.ndarray ) or isinstance( attr, list ): # convert nd array or list types to tensor
                            attr = torch.tensor( attr, device=curlew.device, dtype=curlew.dtype )
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
                        # convert P1 and P2 to tensor
                        if isinstance(attr[1][i][0], torch.Tensor ):
                            o[1].append( (attr[1][i][0].cpu().detach().numpy(),
                                        attr[1][i][1].cpu().detach().numpy(),
                                        attr[1][i][2] ) )
                    attr = o     
                else:
                    if attr is not None:
                        if isinstance(attr, torch.Tensor ):
                            attr = attr.cpu().detach().numpy()
                args[k] = attr
        return CSet(**args)

    def toPLY( self, path ):
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
        from curlew.io import savePLY
        from pathlib import Path
        path = Path(path)
        C = self.numpy()
        def saveCSV( path, xyz, attr=None, names=[], rgb=None ):
            import pandas as pd
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
            if isinstance(arr, torch.Tensor): mask = torch.tensor(mask, device=curlew.device, dtype=torch.bool)
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
    
