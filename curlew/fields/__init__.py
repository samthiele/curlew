"""
Import core neural field types from other python files, and define the "base" NF class that these all inherit from.
"""

import curlew
from curlew.core import CSet, HSet, LearnableBase
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy

class BaseSF(LearnableBase):
    """
    Base class for all implicit (scalar) fields, including interpolated, learned or analytical fields.
    """
    
    level = np.inf
    """
    The level of reconstruction to apply when evaluating this field. If "0" then only the drift/trend is returned. 
    If "inf" then the highest level of detail is returned. 
    
    Values between 0 and np.inf will be treated differently by different field types. Default is np.inf.
    """


    def __init__(self, name : str = None, 
                       input_dim: int = 3,
                       output_dim: int = 1,
                       C: CSet = None,
                       H: HSet = None,
                       drift = 0,
                       transform = None,
                       seed : int = 42, **kwargs ):
        """
        Initialise a new scalar field.

        Parameters
        ----------
        name : str
            A (ideally unique) name for this neural field. Should typically match the name of the GeoField instance that uses this field. Defaults
            to the name of this class.
        input_dim : int, optional
            The dimensionality of the input space (e.g., 3 for (x, y, z)).
        output_dim : int, optional
            Dimensionality of the output (usually 1 for a scalar potential).
        C : CSet
            Constraint sent used for learned or interpolated fields. Default is None.
        H : HSet
            Hyperparameters used to tune the loss function for this NF. Default is None.
        drift : int | float | BaseSF
            A constant integer or float (to use a constant value as the drift), or another BaseSF instance (e.g., an AnalyticalField) that
            defines the trend/drift of this field. This trend/drift will be evaluated at each input coordinate and added to the output of the 
            field during the forward call, meaning learnable fields (interpolators) learn a residual relative to this drift. Default is 0 (no drift).
        transform : callable
            A function that transforms input coordinates prior to evaulation. Must take exactly one argument as input (a tensor of positions) and return the transformed positions. 
        seed : callable, optional
            A random seed to (optinally) use for any random operations, if child classess wish.

        Keywords
        ---------
        All keywords are passed to the initField(...) function of the child class, to build the relevant
        neural architecture.

        """
        super().__init__()
        self.name = name
        if self.name is None:
            self.name = str(type(self).__name__) # default name is the name of the field type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = transform
        self.seed = seed # seed to use for any random operations
        self.C = None # will contain constraints if bound
        self.H = None
        if H is not None:
            self.H = H.copy() # will contain hyperparameters if bound
        if C is not None:
            self.bind(C)
        
        self.drift = drift # can be a constant or another field; evaluated during forward pass
        self.mnorm = 0 # cache the average field gradient (can be useful for quick/rough normalisation)
        self.nnorm = 0 # number of evaluations used to compute average gradient
        self.initField( **kwargs ) # call child class init to build the network
    
    def initField(self, **kwargs):
        """
        Build the internal structure of this implicit field. This should be implemented
        by child classes.
        """
        assert False, "BaseNF does not implement initField()"

    def evaluate(x):
        """
        Evaluate this field at the specified input coordinates. This should be
        implemented by child classes.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, input_dim), where N is the batch size.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, output_dim) containing the field values (or predicted values).
        """
        assert False, "BaseNF does not implement eval()"

    def forward(self, x: torch.Tensor, transform=True) -> torch.Tensor:
        """
        Forward operator to derive field predictions from coordinate tensor. This internally calls whatever
        `eval` function the child class has implemented, after first applying relevant transforms to `x`.
        
        Note that this should generally not be called directly (see `self.predict(...)` instead).

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, input_dim), where N is the batch size.
        transform : bool
            If True (default), any defined transform function is applied before encoding and evaluating the field for `x`.
            
        Returns
        -------
        torch.Tensor
            A tensor of shape (N, output_dim), representing the scalar potential.
        """
        # apply transform if needed
        if transform and self.transform is not None:
            x = self.transform(x)

        # TODO - apply global anisotropy transform matrix?

        # evaluate drift
        out = 0
        if isinstance(self.drift, (int, float)):
            out = out + self.drift
        elif isinstance(self.drift, BaseSF):
            out = out + self.drift(x, transform=False)

        # evaluate field
        # N.B. `self.level` can be set to 0 to evaluate only the drift! Some types of field will also use self.level to controll the detail of reconstruction.
        # N.B.B. Analytical fields will always be evaluated, even at level 0, as these are not interpolations.
        if (self.level > 0) or isinstance(self, BaseAF):
            out = self.evaluate(x) + out
            if len(out.shape) == 1:
                out = out[:, None] # add extra dimension if needed (for consistency)
        return out
    
    def bind( self, C ):
        """
        Bind a CSet to this field ready for loss computation (neural fields) or interpolation (interpolators).
        """
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
                    C._offset.append( torch.tensor( o, device=curlew.device, dtype=curlew.dtype) )

    ## MODEL EVALUATION
    def predict(self, X, to_numpy=True, transform=True ):
        """
        Create model predictions at the specified points. 

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
    
    def reset_mnorm(self):
        """Reset accumulation of average gradient magnitude"""
        self.mnorm = 0
        self.nnorm = 0

    def gradient(self, coords: torch.Tensor, 
                       normalize: bool = True, 
                       transform=True, 
                       return_value=False, 
                       retain_graph=False,
                       create_graph=False,
                       accumulate=True) -> torch.Tensor:
        """
        Compute the gradient of the scalar potential with respect to the input coordinates. Note that this only works
        for scalar (i.e. 1-D) fields.

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
        retain_graph : bool, optional
            True if the gradient graph should be retained (to allow e.g., subsequent backpropagation). Default is False.
        create_graph : bool, optional
            True if the gradient value should have an underlying graph to allow it to influence back-prop operations. Default is False.
        accumulate : bool, optional
            True (optional) if the gradient evaluation should contribute to the average gradient estimate. Note that this averaging
            can be reset using `self.reset_mnorm` and accessed through `self.mnorm`.
        
        Returns
        -------
        torch.Tensor
            A tensor of shape (N, input_dim) representing the gradient of the scalar potential at each coordinate.
        torch.Tensor, optional
            A tensor of shape (N, 1) giving the scalar value at the evaluated points, if `return_value` is True.
        """

        # we need to compute gradients
        coords.requires_grad_(True)

        # Forward pass to get the model output and autodiff graph
        potential = self.forward(coords, transform=transform).sum(dim=-1)  # sum in case output_dim > 1

        # Compute gradient
        grad_out = torch.autograd.grad(
            outputs=potential,
            inputs=coords,
            grad_outputs=torch.ones_like(potential),
            create_graph=create_graph,
            retain_graph=retain_graph
        )[0]

        # Accumulate and/or normalize gradients?
        if accumulate or normalize:
            norm = torch.norm(grad_out, dim=-1, keepdim=True) + 1e-8
            if accumulate:
                self.mnorm = (self.mnorm*self.nnorm) + torch.mean(norm, axis=0).item()*len(norm) # update average gradeint
                self.nnorm += len(norm) # update counter holding number of observations
                self.mnorm = self.mnorm / self.nnorm # convert from total to average
            if normalize:
                grad_out = grad_out / norm
        
        # Return
        if return_value:
            return grad_out, potential
        else:
            return grad_out

    ## FITTING (STUBS)
    def loss(self, transform=True) -> torch.Tensor:
        """
        Optionally implemented by child classes to facilitate optimiation and learning. Defaults to 0.
        """
        return torch.tensor(0, dtype=curlew.dtype, device=curlew.device, requires_grad=True), {self.name:(0,{})}
    
    def fit(self, *args):
        """
        Optionally implemented by learnable child classes. If not, simply returns whatever is returned by "loss".

        Returns
        -------
        loss : float
            The loss of the final result.
        details : dict
            A more detailed breakdown of the final loss. 
        """
        loss = self.loss()
        out = { self.name : [loss.item(),{}] }
        return loss, out

class BaseAF( BaseSF ):
    """
    Base class for all analytical fields (those implementing specific geometric implicit functions).
    """
    pass # this does nothing special! But is included to easily distinguish analytical, neural and interpolated fields

class BaseNF(BaseSF):
    """
    A generic base for neural field implementations that learn to translating input coordinates to implicit value (or values). See the other
    child classes in this module (e.g., fourier, geoinr, etc.) for specific implementations.
    """
    def __init__(
            self,
            name : str,
            H: HSet,
            C : CSet = None,
            input_dim: int = 3,
            output_dim: int = 1,
            transform = None,
            seed = 42,
            vloss = nn.MSELoss(),
            scale = 1e2,
            **kwargs
        ):
            """
            Parameters
            ----------
            name : str
                A (ideally unique) name for this neural field. Should typically match the name of the GeoField instance that uses this field.
            H : HSet
                Hyperparameters used to tune the loss function for this NF.
            C : CSet, optinoal
                Constraint sent used when learning this implicit field. Default is None (can be set using `field.bind(...)`).
            input_dim : int, optional
                The dimensionality of the input space (e.g., 3 for (x, y, z)).
            output_dim : int, optional
                Dimensionality of the output (usually 1 for a scalar potential).
            transform : callable
                A function that transforms input coordinates prior to predictions. Must take exactly one argument as input (a tensor of positions) and return the transformed positions. 
            seed : callable, optional
                The random seed to use for any random operations.
            vloss : callable, optional
                The loss function to use for value fitting. Default is mean squared error (`nn.MSELoss()`).
            scale : float, optional
                A scaling factor to apply to outputs of the neural field, as often these struggle to learn functions with a large (>1) amplitude. Default is 1e2. 
                
                This value should be approximately equal to the expected range (max - min) of the scalar field that is being learned. It can be especially important when using a 
                drift (trend), as it determines the extent to which the model initialisation is determined by the drift. Larger values should allow the model to deviate farther from the trend.
                Also note that this term also tends to control the magnitude of residuals (to value or (in)equality constraints), so will also interact with the learning rate.
                
                N.B. The actual implementation of this scale depends on the neural field method being used.

            Keywords
            ---------
            All keywords are passed to the initField(...) function of the child class, to build the relevant
            neural architecture.
            """
            # initialise everything (including calling the initField class of the relevant child class)
            super().__init__(name=name, input_dim=input_dim, output_dim=output_dim, H=H, C=C, transform=transform, seed=seed, **kwargs)

            # store neural field specific properties
            self.closs = torch.nn.CosineSimilarity() # needed by some loss functions
            self.vloss = vloss # loss function to use for value fitting
            self.scale = scale

    ## LEARNING
    def loss(self, transform=True) -> torch.Tensor:
        """
        Compute the loss associated with this neural field given its current state. The `transform` argument
        specifies if constraints need to be transformed from modern to paleo-coordinates before computing loss.
        """
        if self.C is None:
            assert False, "Scalar field has no constraints"

        # move these into local scope for clarity
        C = self.C 
        H = self.H

        # inititialize different loss parts
        L = {}
        for k in ['value_loss', 'grad_loss', 'ori_loss', 'thick_loss', 'mono_loss', 'flat_loss', 'iq_loss']:
            L[k] = 0
        total_loss = 0

        # LOCAL LOSS FUNCTIONS
        # -----------------------------
        # Value Loss
        if (C.vp is not None) and (C.vv is not None) and (isinstance(H.value_loss, str) or (H.value_loss > 0)):
            v_pred = self(C.vp, transform=transform)
            L['value_loss'] = self.vloss( v_pred, C.vv[:,None] )

        # Gradient loss
        self.reset_mnorm() # reset accumulation
        # [ N.B. positions (and thus gradients) are in un-transformed coordinates ]
        if (C.gp is not None) and (isinstance(H.grad_loss, str) or (H.grad_loss > 0)):
            gv_pred = self.gradient(C.gp, normalize=True, transform=transform, accumulate=True, retain_graph=True, create_graph=True) # compute gradient direction 
            L['grad_loss'] = self.vloss(gv_pred, C.gv) # N.B. constraints orientation and younging direction

        # Orientation loss
        # [ N.B. positions (and thus gradients) are in un-transformed coordinates ]
        if (C.gop is not None) and (isinstance(H.ori_loss, str) or (H.ori_loss > 0)):
            gv_pred = self.gradient(C.gop, normalize=True, transform=transform, accumulate=True, retain_graph=True, create_graph=True) # compute gradient direction 
            L['ori_loss'] = torch.clamp( torch.mean( 1 - torch.abs( self.closs(gv_pred, C.gov ) ) ), min=1e-6 ) # N.B.: Orientation loss on its own fits a bit too well, numerical precision crashes avoided with the clamp - AVK

        # GLOBAL LOSS FUNCTIONS
        # -------------------------------
        if C.grid is not None:
            if transform:
                gridL = C.grid.draw(self.transform) # specify transform
            else:
                gridL = C.grid.draw() # no transform
            
            if  isinstance(H.thick_loss, str) or isinstance(H.mono_loss, str) or isinstance(H.flat_loss, str) or \
                (H.thick_loss > 0) or (H.mono_loss > 0) or (H.flat_loss > 0):

                # numerically compute the hessian of our scalar field from the gradient vectors
                # to compute the divergence of the normalised field and so penalise bubbles (local maxima and minima)
                #hess = torch.zeros((gridL.shape[0], self.input_dim, self.input_dim), device=curlew.device, dtype=curlew.dtype)
                ndiv = torch.zeros((gridL.shape[0]), device=curlew.device, dtype=curlew.dtype)
                for j in range(self.input_dim):
                    # compute hessian
                    grad_pos = self.gradient(gridL + C._offset[j], normalize=False, transform=False, accumulate=True, retain_graph=True, create_graph=True)
                    grad_neg = self.gradient(gridL - C._offset[j], normalize=False, transform=False, accumulate=True, retain_graph=True, create_graph=True)
                    #for i in range(self.input_dim):
                    #    hess[:, i, j] = (grad_pos[:, i] - grad_neg[:, i])/(2*C.delta)

                    # compute and accumulate average gradient
                    pnorm = torch.norm(grad_pos, dim=-1 )[:,None]
                    nnorm = torch.norm(grad_neg, dim=-1 )[:,None]

                    # compute divergence of normalised gradient field
                    if isinstance(H.mono_loss, str) or (H.mono_loss > 0):
                        grad_pos = grad_pos / pnorm
                        grad_neg = grad_neg / nnorm
                        ndiv = ndiv + (grad_pos[:,j] - grad_neg[:,j])/(2*C.delta)

                    # compute the percentage deviation in the gradient (at all the points where we evaluated it)
                    if isinstance(H.thick_loss, str) or (H.thick_loss > 0):
                        L['thick_loss'] = L['thick_loss'] + \
                                          torch.mean( (1-(pnorm/torch.clip(torch.mean(pnorm), 1e-8, torch.inf )) )**2 ) + \
                                          torch.mean( (1-(nnorm/torch.clip(torch.mean(nnorm), 1e-8, torch.inf )) )**2 )

                # compute derived thickness and monotonocity loss
                if isinstance(H.mono_loss, str) or (H.mono_loss > 0):
                    L['mono_loss'] = torch.mean(ndiv**2) # (normalised) divergence should be close to 0
                if isinstance(H.thick_loss, str) or (H.thick_loss > 0):
                    # L['thick_loss'] = torch.mean( torch.linalg.det(hess)**2 ) # determinant should be close to 0 [ breaks in 2D, as the trace and determinant can't both be 0 unless all is 0!]
                    L['thick_loss'] = L['thick_loss'] / (2*self.input_dim) # normalise to get average (doesn't change anything, but makes values easier to interpret)

                # Flatness Loss --  gradients everywhere parallel to trend
                if (isinstance(H.flat_loss, str) or (H.flat_loss > 0)) and (C.trend is not None):
                    if transform:
                        gv_at_grid_p = self.gradient(gridL, normalize=True, transform=self.transform, retain_graph=True, create_graph=True) # this requires gradients relative to modern coordinates! 
                    else:
                        gv_at_grid_p = self.gradient(gridL, normalize=True, transform=False, retain_graph=True, create_graph=True)
                    L['flat_loss'] = torch.mean((gv_at_grid_p - C.trend[None,:])**2) # "younging" direction
                    #flat_loss = (1 - self.closs( gv_at_grid_p, C.trend )).mean() # orientation only

        # inequality losses
        if (C.iq is not None) and (isinstance(H.iq_loss, str) or (H.iq_loss > 0)):
            ns = C.iq[0] # number of samples
            for start,end,iq in C.iq[1]:
                # sample N random pairs to evaluate inequality
                six = torch.randint(0, start.shape[0], (ns,), dtype=int, device=curlew.device)
                eix = torch.randint(0, end.shape[0], (ns,), dtype=int, device=curlew.device)

                # evaluate value at these points
                start = self( start[ six, : ], transform=transform )
                end = self( end[ eix, : ], transform=transform )
                delta = start - end

                # compute loss according to the specific inequality
                if '=' in iq:
                    L['iq_loss'] = L['iq_loss'] + torch.mean(delta**2) # basically MSE
                elif '<' in iq:
                    L['iq_loss'] = L['iq_loss'] + torch.mean(torch.clamp(delta,0,torch.inf)**2)
                elif '>' in iq:
                    L['iq_loss'] = L['iq_loss'] + torch.mean(torch.clamp(delta,-torch.inf, 0)**2)

        # parse loss hyperparameters
        # If a hyperparameter is a string, we use this to derive an initial weight 
        # that assumes all (initial) loss terms are equally bad
        for k,v in L.items():
            h = H.__getattribute__(k)
            if isinstance(h, str):
                H.__setattr__(k, float(h) * (1/v).item() if v > 0 else 0.0 )
        
        # Dynamically adjust task weights based on the inverse of real-time loss values.
        # (this ignores the magnitude of each loss term, but preserves it's gradient direction,
        # and is a hacky but sometimes useful way to balance multi-task losses)
        if H.use_dynamic_loss_weighting:
            for k,v in L.items():
                if v > 0:
                    L[k] = 1 / v.item()
        
        # aggregate losses (and store individual parts for debugging)
        out = { self.name : [0,{}] }
        for k,v in L.items():
            alpha = H.__getattribute__(k)
            if (alpha is not None) and (alpha > 0) and (v > 0):
                total_loss = total_loss + alpha*v
                out[self.name][1][k] = (alpha*v.item(), v.item())
        
        out[self.name][0] = total_loss.item()

        # done! 
        return total_loss, out

    def fit(self, epochs, 
                 C : CSet = None, 
                 learning_rate : float = None, 
                 early_stop : tuple = (100,1e-4), 
                 transform : bool = True, 
                 best : bool = True, 
                 vb : bool = True, 
                 prefix : str = 'Training',
                 opt : list = []):
        """
        Train this neural field to fit the specified constraints.

        Parameters
        ----------
        epochs : int
            The number of epochs to train for.
        C : CSet, optional
            The set of constraints to fit this field to. If None, the previously
            bound constraint set will be used.
        learning_rate : float, optional
            Reset this NF's optimiser to the specified learning rate before training.
        early_stop : tuple,
            Tuple containing early stopping criterion. This should be (n,t) such that optimisation
            stops after n iterations with <= t improvement in the loss. Set to None to disable. Note 
            that early stopping is only applied if `best = True`. 
        transform : bool, optional
            True (default) if constraints (C) is in modern coordinates that need to be transformed during fitting. If False, 
            C is considered to have already been transformed to paleo-coordinates. Note that this can be problematic if rotations
            occur (e.g. of gradient constraints!).
        best : bool, optional
            After training set the neural field weights to the best loss.
        vb : bool, optional
            Display a tqdm progress bar to monitor training.
        prefix : str, optional
            The prefix used for the tqdm progress bar.
        opt : list, optional
            An optional list of additional optimisers to include in the training loop (zero() and step() will be called
            on these at the same time as the optimiser used for this NF's internal learnable parameters). Used to allow
            e.g., learnable fault offset.
        Returns
        -------
        loss : float
            The loss of the final (best if best=True) model state.
        details : dict
            A more detailed breakdown of the final loss. 
        """
        # set learning rate if needed
        if learning_rate is not None:
            self.set_rate(learning_rate)

        # bind the constraints
        if C is not None:
            self.bind(C)

        # store best state
        best_loss = np.inf
        best_loss_ = None
        best_state = None

        # for early stopping
        best_count = 0
        eps = 0
        if early_stop is not None:
            eps = early_stop[1]

        # setup progress bar
        bar = range(epochs)
        if vb:
            bar = tqdm(range(epochs), desc=prefix, bar_format="{desc}: {n_fmt}/{total_fmt}|{postfix}")
        for epoch in bar:
            loss, details = self.loss(transform=transform) # compute loss

            if (loss.item() < (best_loss + eps)): # update best state
                best_loss = loss.item()
                best_loss_ = details
                best_state = copy.deepcopy( self.state_dict() )
                best_count = 0
            else: # not necessarily the best; but keep for return
                if best_state == None:
                    best_loss = loss.item()
                    best_loss_ = details
                best_count += 1

            # early stopping?
            if (early_stop is not None) and (best_count > early_stop[0]):
                break

            if vb: # update progress bar
                bar.set_postfix({ k : v[0] for k,v in details[self.name][1].items() })

            # backward pass and update
            self.zero()
            for o in opt:
                if o is not None: o.zero() # can often be None; ignore in that case.
            loss.backward(retain_graph=False)
            self.step()
            for o in opt:
                if o is not None: o.step()

        if best:
            self.load_state_dict(best_state)

        return best_loss, best_loss_ # return summed and detailed loss
    
# import other child classes for easy access
from curlew.fields.analytical import LinearField, QuadraticField, PeriodicField, ListricField
from curlew.fields.fourier import NFF