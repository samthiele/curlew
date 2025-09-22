"""
A class defining a structural scalar (implicit) field. This is where a lot of the magic happens, including
the chaining of multiple neural fields and the application of deformation functions to implement kinematic
events such as faults and sheet intrusions.
"""

import curlew
from curlew.fields import NF
from curlew.geology.interactions import overprint
import numpy as np
import torch
import functools
import copy

def apply_child_undeform(x, sf):
    """
    Placeholder function to avoid Lambda functions.
    (Making models picklable)
    """
    if sf.child is not None:
        return sf.child.undeform(x)
    else:
        return x
    
class SF( object ):
    """
    A linked list of scalar fields, allowing them to be 
    chained so that they combine into time-aware geological models.
    """
    def __init__( self, name, child=None, bound=None, deformation=None, deformation_args={}, sharpness : float = 1e4, **kwargs ):
        """
        Initialise a new SF. 

        Parameters
        ------------
        name : str
            A name for this event.
        child : SF
            A younger (child) SF instance to link this SF to. Any deformation caused by the younger SF will
            then be removed prior to evaluation of this one.
        bound : float, tuple, None
            A float specifying the minimum value at which this SF overprints parent 
            ones (i.e. unconformities), or a (min, max) range if the overprint occur
            over a specific range only (dykes, sills, veins). If None, no overprinting
            will occur and the scalar value outputs ignored (but displacements still applied,
            i.e. faults).
        deformation : function, None
            A function such that f(X1, self) returns an array X0 of shape (N,ndim) that 
            represents the pre-deformation coordinates of X1.
        deformation_args : dict
            A dictionary of arguments to pass to the deformation function, if needed.
        sharpness : float
            The sharpness of the sigmoid function used to combine this SF with results from parent fields. 

        Keywords
        -------------
        All keywords will be passed to `curlew.fields.NF.__init__(...)`.

        """
        self.name = name
        self.eid = -1 # will be set to the position of this SF in the event sequence
        if 'field' in kwargs:
            self.field = kwargs['field'] # field already constructued :-)
        else:
            self.field = NF(name=name, **kwargs) # initialise a neural field
        self.parent = None # used when chaining multiple SFs
        self.parent2 = None # will be defined if this SF represents a domain boundary.
        self.child = None # used when chaining multiple SFs
        self.bound = bound
        self.deformation = deformation # deformation function, if defined
        self.deformation_args = deformation_args # kwargs that will be passed to the deformation function
        self.sharpness = sharpness
        self.isosurfaces = {} # this will hold any added isosurfaces

        # add child-parent relationship
        if child is not None:
            self.child = child
            self.child.parent = self
        
         # define transform function of underlying neural field
        self.field.transform = functools.partial(apply_child_undeform, sf=self)

    def copy(self):
        """
        Create a copy of this SF object for e.g., incorporation into a different model. Note that this will unlink any
        parent, parent2 and child relations.
        """
        out = copy.deepcopy(self)
        out.parent = None
        out.parent2 = None
        out.child = None
        return out
    
    def fit(self, epochs, cache=True, faultBuffer=0, **kwargs):
        """
        Utility function for training fields in isolation. 

        Parameters
        ----------
        epochs : int
            The number of epochs to train for.
        cache : bool | optional
            If True, displacements applied by younger SF instances (linked to this one through `self.child`) will 
            be pre-computed to speed up training (by avoiding repeated evaluations of younger fields that will
            not change).
        faultBuffer : int | optional
            If greater that zero, constraints within this distance of younger faults will be removed (as these can 
            often be reconstructed to misleading positions and are very sensitive to interpolation errors in younger
            fields).
        Keywords
        ----------
        batch_size : int
            The size of the batches used when retro-deforming the passed grid (to save RAM). Default is 50000.
        
        All other keywords are passed to `curlew.fields.NF.fit(...)`. These include:
        learning_rate : float, optional
            Reset each SF's optimiser to the specified learning rate before training.
        best : bool, optional
            After training set neural field weights to the best loss.
        vb : bool, optional
            Display a tqdm progress bar to monitor training.

        Returns
        -------
        loss : float
            The loss of the final (best if best=True) model state.
        details : dict
            A more detailed breakdown of the final loss. 
        """
        # get constraints set to use
        if 'C' in kwargs:
            self.field.bind( kwargs.pop('C', self.field.C) )
        C = self.field.C 

        # enable caching to avoid lots
        # of evaluations of displacement fields
        # that will not change
        C0 = C # no need to reconstruct
        if cache and self.child is not None:
            if faultBuffer > 0:
                from curlew.geology.interactions import faultOffset
                def buffer( arr ): # quick function for masking points on faults
                    f = self.child
                    mask = np.full( len(arr), True )
                    while f is not None:
                        if ((f.deformation is not None) and (f.deformation == faultOffset)): # faults
                            b = f.buffer( arr, f.deformation_args['contact'], faultBuffer )
                        elif (f.parent2 is not None):
                            b = f.buffer( arr, f.bound, faultBuffer ) # domain boundary
                        mask[ b ] = False # identify and remove points within buffer distance of a fault 
                        f = f.child
                    return mask
                C0.filter(buffer)

            # retro-deform other constraints
            C0 = C.numpy().transform( self.child.undeform )
        
        # fit
        try:
            out = self.field.fit(epochs, C=C0, transform=False, **kwargs)
        finally:
            if C0.grid is not None:
                C0.grid._clearCache()
            self.field.bind(C) # ensure this always runs! (e.g., in case of keyboard interrupts)
            
        return out
    
    def forward(self, x: torch.Tensor, undef=True) -> torch.Tensor:
        """
        Forward pass of the network to create a scalar value or property estimate.
        If random Fourier features are enabled, the input is first encoded accordingly.

        This calls the underlying field's `forward(..)` function, but after
        calling `self.undeform` to underform constraint positions prior to interpolation.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, input_dim), where N is the batch size.
        undef : bool
            True if x is specified in undeformed (modern-day) coordinates. Default is True.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, output_dim), representing the scalar potential.
        """
        # set transform that removes any child deformation
        if undef and (self.child is not None):
            self.field.transform = functools.partial(apply_child_undeform, sf=self) # apply this SF's undeform function prior to  field.forward.
            #x = self.child.undeform( x )
        else:
            self.field.transform = None # don't do any transform in field.forward

        # evaluate the underlying field
        return self.field.forward( x )

    def predict(self, x: np.ndarray, combine=False, to_numpy=True, transform=True) -> np.ndarray:
        """
        Predict scalar values belonging to this and/or previous SFs.

        Parameters
        ----------
        x : np.ndarray
            An array of shape (N, input_dim) containing the modern-day coordinates at which to evaluate
            this (and previous) SF.
        combine: bool
            True if previous scalar fields should be aggregated to give a combined
            scalar field output. Otherwise, evaluate the scalar field related to this
            specific structure alone. Default is False.
        to_numpy : bool
            True if the results should be cast to a numpy array rather than a `torch.Tensor`.
        transform : bool
            True if the coordinates should be undeformed (i.e. represent modern-day coordinates) or not (i.e. represent paleo-coordinates).
        """
        # evaluate field and combine with parent ones
        if combine and (self.parent is not None):
            if self.parent2 is not None: # this field represents a domain boundary -- essentially use as a mask

                # predict value of domain scalar field
                domain = self.predict( x, combine=False, to_numpy=False, transform=transform )

                # evaluate isosurfaces to get threshold values, if needed (allows self.bound to containt str isosurface names)
                assert self.bound is not None, "Isosurface must be defined for domain boundary."
                thresh = self.getIsovalue(self.bound) # this thresh separates the domain into two halves
                mask = torch.sigmoid(self.sharpness * (domain - thresh))[:,0]

                # fill in each side of domain [ above and below thresh ]
                out = torch.zeros((len(domain), 2), dtype=curlew.dtype, device=curlew.device)
                for sf, m in zip([self.parent, self.parent2], [mask, (1-mask)]):
                    if not isinstance( sf, SF ): # this side of domain boundary is a constant value (float or int)
                        out = out + m[:,None] * sf
                    else: # this side of domain boundary is a modelled value
                        parent = sf.predict( x, combine=True, to_numpy=False, transform=transform )
                        out = out + m[:,None] * parent
            else: # this field is one in a series of geological "events" -- evaluate accordingly
                if isinstance(self.parent, int) or isinstance(self.parent, float): # int or float parent
                    out = torch.full((len(x), 2), self.parent, device=curlew.device, dtype=curlew.dtype) # create constant (float or int) value parent
                else: # parent is a scalar field (or some variety thereof)
                    out = self.parent.predict( x, combine=True, to_numpy=False, transform=transform )
                if self.bound is not None: # if bound is None we don't change previous values as this is not a generative event
                    sf1 = self.predict( x, combine=False, to_numpy=False, transform=transform )
                    thresh = self.getIsovalues(values=self.bound) # evaluate isosurfaces to get threshold values, if needed (allows self.bound to containt str isosurface names)
                    out = overprint( out, sf1, eid=self.eid, thresh=thresh, sharpness=self.sharpness )
        else: # simply evaluate field! :-)
            if not isinstance(x, torch.Tensor):
                x = torch.tensor( x, device=curlew.device, dtype=curlew.dtype)
            out = torch.full((len(x), 2), self.eid, device=curlew.device, dtype=curlew.dtype) # create output with event ID 
            out[:,0] = self.forward( x, undef=transform ).squeeze() # populate scalar value part

        # return
        if to_numpy:
            out = out.cpu().detach().numpy()
            out[:,1] =  out[:,1].astype(int) # ensure that the event ID is an integer
        return out

    def gradient(self, x: np.ndarray, return_vals=False, normalize=True, **kwds):
        """
        Return the gradient vector of this SF at the specified location. Note that this
        does  not combine the results from previous scalar fields first (i.e. the prediction
        is done using `combine=False`).

        Parameters
        ----------
        x : np.ndarray
            An array of shape (N, input_dim) containing the modern-day coordinates at which to evaluate
            this (and previous) SF.
        return_vals : bool
            True if evaluated scalar values should be returned as well as classes. Default is False. 
        normalize : bool
            True if gradient vectors should be normalised to length 1 (i.e. to represent poles to planes). Default is True. 

        Keywords
        ---------
        Keywords are all passed to `SF.predict(...)`.

        Returns
        --------
        Gradient vectors at the specified locations (`x`). If `return_vals` is `True`,
        tuple (gradients, values) will be returned.
        """

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=curlew.dtype, device=curlew.device, requires_grad=True)

        # generate predictions
        to_numpy = kwds.pop('to_numpy', True)
        pred = self.predict(x, to_numpy=False, **kwds) # keep this as torch tensor

        # get gradients
        grad_out = torch.autograd.grad(
            outputs=pred[:,0], # we only care about scalar values
            inputs=x,
            grad_outputs=torch.ones_like(pred[:,0]),
            create_graph=True,
            retain_graph=False
        )[0]

        # normalise gradients
        if normalize:
            norm = torch.norm(grad_out, dim=-1, keepdim=True) + 1e-8
            grad_out = grad_out / norm

        if to_numpy:
            grad_out = grad_out.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()

        if return_vals:
            return grad_out, pred
        else:
            return grad_out

    def deform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply deformation (displacements) to the passed set of coordinates.

        Translates from paleo-coordinates (i.e. coordinates at the time of this event) to present-day coordinates, 
        for example, by applying child fault offsets.

        Parameters
        ----------
        x : torch.Tensor, np.ndarray
            A tensor of shape (N, input_dim).

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, input_dim), containing the deformed coordinates.
        """
        tonp = False
        if isinstance( x, np.ndarray ):
            x = torch.tensor( x, device=curlew.device, dtype=curlew.dtype)
            tonp = True

        # apply any child deformation
        if self.child is not None:
            x = self.child.deform( x )

        # handle our own displacement
        out = x
        if self._displace is not None:
            out = self._displace(x, inverse=True) # apply this SF's displacement (if defined).

        # cast to numpy if needed
        if tonp:
            return out.cpu().detach().numpy()
        else:
            return out

    def undeform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove deformation (displacements) from the passed set of coordinates.

        Translates from present-day coordinates to coordinates relevant for this SF by removing, 
        for example, child fault offsets.

        Parameters
        ----------
        x : torch.Tensor, np.ndarray
            A tensor of shape (N, input_dim).

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, input_dim), containing the undeformed coordinates.
        """
        if (self.child is None) and (self._displace is None):
            return x # easy

        # evaluate displacement
        tonp = False
        if isinstance( x, np.ndarray ):
            x = torch.tensor( x, device=curlew.device, dtype=curlew.dtype)
            tonp = True

        # remove any child deformation
        if self.child is not None:
            x = self.child.undeform( x ) # undeform to the time-step relevant for this SF

        # handle our own displacement
        out = x
        if self._displace is not None:
            out =  self._displace(x) # apply this SF's displacement (if defined).

        # cast to numpy if needed
        if tonp:
            return out.cpu().detach().numpy()
        else:
            return out

    def _displace(self, x: torch.Tensor, inverse=False) -> torch.Tensor:
        """
        Remove deformation (displacements) caused by this specific event.

        Displaces from post-event coordinates to pre-event coordinates. If this SF has no 
        associated displacement field, the input `x` is returned directly.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, input_dim).
        inverse : bool
            If True, apply the inverse deformation (i.e. from paleo-coordinates to modern-day coordinates).
            Default is False, which applies the deformation from modern-day coordinates to paleo-coordinates.
        Returns
        -------
        torch.Tensor
            A tensor of shape (N, input_dim), containing the undeformed coordinates.
        """
        if self.deformation is None:
            # this SF causes no deformation
            return x
        else:  # apply deformation function
            tonp = False
            if not isinstance(x, torch.Tensor): # make tensor if needed
                x = torch.tensor(x, dtype=curlew.dtype, device=curlew.device)
                tonp = True

            offset = self.displacement(x, inverse=inverse) # get deformation vectors
            x = x - offset # apply deformation to the input coordinates

            if tonp: # cast to numpy if needed
                return x.detach().cpu().numpy() # return as numpy array
            else:
                return x

    def displacement(self, x: np.ndarray, inverse=False) -> np.ndarray:
        """
        Return the deformation vectors associated with this SF at the specified locations. These are the
        displacements that would be remoed during `_displace(...)`.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            An array of shape (N, input_dim) containing the modern-day coordinates at which to evaluate
            this (and previous) SF.
        inverse : bool
            If True, apply the inverse deformation (i.e. from paleo-coordinates to modern-day coordinates).
            Default is False, which applies the deformation from modern-day coordinates to paleo-coordinates.

        Returns
        -------
        np.ndarray | torch.Tensor
            An array of shape (N, input_dim) containing the deformation vectors at the specified locations.
            If `inverse` is True, these vectors represent the deformation from paleo-coordinates to modern-day coordinates.
            Otherwise, they represent the deformation from modern-day coordinates to paleo-coordinates.
        """
        tonp = False
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=curlew.dtype, device=curlew.device)
            tonp = True

        offset = self.deformation(x, self, **self.deformation_args) # apply deformation function
        if inverse: # flip deformation vectors (B to A instead of A to B)
                offset = -offset

        if tonp: # cast to numpy if needed
            return offset.detach().cpu().numpy() # return as numpy array
        else:
            return offset

    def addIsosurface( self, name :str, *, value = None, seed = None):
        """
        Add a (geologically meaningful) isosurfcae to this scalar field. These
        represent e.g., stratigraphic contacts.

        Note that isosurfaces can be defined in two ways:

        1. by specifying their value directly (`value=x`)
        2. by specifying a location (`seed_point`) at with the scalar field
           should be evaluated to determine the scalar value.

        Parameters
        ----------
        name : str
            A friendly name used to refer to this isosurface.
        value : float, None
            A value to explicitely set the isosurface value.
        seed : np.ndarray, None
            A position (x,y,[z]) or list of positions that implicitly define the 
            isosurface value. Whatever value is returned by the model at this 
            location will be used to define the isosurface value). 

            If several points are provided (e.g., known contact locations),
            the mean of their outputs used to determine the isosurface value.
        """
        assert (seed is None) or (value is None), "Either seed or value should be defined, not both."
        assert not( (seed is None) and (value is None)), "Either seed or value should be defined, not both."
        if seed is not None:
            self.isosurfaces[name] = np.array( seed )
        if value is not None:
            self.isosurfaces[name] = value

    def deleteIsosurface( self, name ):
        """
        Remove the specified isosurface from this SF's `isosurfaces` dict, if it exists.
        """
        if name in self.isosurfaces:
            del self.isosurfaces[name]

    def getIsovalue( self, name, offset=0):
        """
        Return the value of a specific isosurface associated with this scalar field.
        """
        return self.getIsovalues( [name], offset=offset )[0]

    def getIsovalues( self, values=None, offset=0 ):
        """
        Evaluate (if necessary) and return the values of all the isosurfaces associated
        with this scalar field.

        Parameters
        ----------
        values : list, None
            A list of isosurface names to evaluate. If None, all isosurfaces will be evaluated.

        Returns
        --------
        Either a list of isosurface values or a dictionary of isosurface names and corresponding
        values (if `values` is None).
        """
        if values is None:
            keys = list(self.isosurfaces.keys())
            out = {}
        else:
            keys = [values] if isinstance(values, int) or isinstance(values, float) or isinstance( values, str) else values
            out = []
        for k in keys:
            if not isinstance(k, str):
                v = k # someone passed us a value directly; but don't complain
            else:
                assert k in self.isosurfaces, f"Isosurface '{k}' not found."
                v = self.isosurfaces[k]

            if isinstance(v, np.ndarray) or isinstance(v, list):
                v = np.array(v)
                if len(v.shape) == 1:
                    v = v[None, :]

                # apply offset if specified
                if offset != 0:
                    g = self.gradient( v, normalize=True )
                    v = v + g * offset # offset seed points by specified distance in gradient direction

                # evaluate (and average) value at seed points
                i = np.mean( self.predict( v )[:, 0] )
            else:
                i = v # easy! 
                if offset != 0: # not so easy...
                    i = i+offset * self.field.mnorm # WARNING: this is a crude hack that assumes a constant gradient
            if values is None:
                out[k] = i
            else:
                out.append(i)
        return out

    def buffer(self, x : np.ndarray, isovalue, width : float, mask = None):
        """
        This method evaluates whether the predicted values for the input `x` fall 
        within a range defined by the isovalue and the specified buffer width. Optionally, 
        a mask can be applied to exclude certain elements from the output.
        
        Parameters:
        -----------
        x : np.ndarray
            The input data for which predictions will be made.
        isovalue : str | float | tuple
            The isosurface (name or value) around which the buffer is calculated. If a tuple is passed, values between isovalue[0] and isovalue[1] will be returned.
        width : float
            The width of the buffer around the isovalue.
        mask : np.ndarray or None, optional
            A boolean mask array of the same shape as `x` that specifies elements 
            to exclude from the output. If `None`, no masking is applied. 
            Default is `None`.
        
        Returns:
        --------
        np.ndarray
            A boolean array indicating whether each element in `x` falls within 
            the buffer range. Elements excluded by the mask are set to `False`.
        """
        pred = self.predict(x)[:,0] # get values at points
        if isinstance(isovalue, tuple) or isinstance(isovalue, list):
            i0 = min(isovalue) # easy!
            i1 = max(isovalue)
        else:
            i0 = self.getIsovalue( isovalue, offset=-width ) # calculate lower buffer isovalue
            i1 = self.getIsovalue( isovalue, offset=width ) # calculate upper buffer isovalue
        out = (pred >= min(i0, i1)) & (pred <= max(i0, i1))
        if mask is not None:
            out[mask] = False
        return out

    def classify( self, x: np.ndarray, return_vals=False, **kwds ):
        """
        Evaluate the model using `self.predict(x, **kwds)` and then bin
        the resulting scalar values according to the isosurfaces associated
        with this scalar field.

        Parameters
        ----------
        x : np.ndarray
            And (N,d) array containing the locations at which to evaluate the model.
        return_vals : bool
            True if evaluated scalar values should be returned as well as classes. Default is False. 
        Returns
        -------
        class_ids : np.ndarray
            Class ID's after classifying according to the defined isosurfaces.
        class_names : np.ndarray
            Array of unit names corresponding to the above class_ids.
        """
        # get and sort isovalues
        iso = np.array( list(self.getIsovalues().values()) )
        ixx = np.argsort(iso)
        inames = np.array( list(self.getIsovalues().keys()) )[ixx]
        pred = self.predict(x, **kwds)
        cids = np.digitize( pred, bins=iso[ixx] )

        # return
        if return_vals:
            return cids, inames, pred
        else:
            return cids, inames