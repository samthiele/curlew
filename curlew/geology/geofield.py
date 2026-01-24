"""
A class defining a structural scalar (implicit) field. This is where a lot of the magic happens, including
the chaining of multiple neural fields and the application of deformation functions to implement kinematic
events such as faults and sheet intrusions.
"""

import curlew
from curlew.core import Geode
from curlew.fields import BaseNF
from curlew.geology.interactions import Overprint, OffsetBase
from curlew.geometry import Grid
from curlew.utils import batchEval

import numpy as np
import torch
import functools
import copy

from typing import Optional, List, Tuple, Union
ArrayLike = Union[np.ndarray, "torch.Tensor"]

def apply_child_undeform(x, sf):
    """
    Placeholder function to avoid Lambda functions. This needs to be dynamic as we may not know yet
    what `sf.child` is -- just that we should check if it is defined while undeforming coordinates
    during the evaluation of a scalar field.
    """
    if sf.child is not None:
        return sf.child.undeform(x)
    else:
        return x

class GeoField( object ):
    """
    A geological (implicit) field and associated objects that deterimine how it interacts with older and younger fields. Typically
    each geological field represents a specific event (e.g., faulting, intrusion, deposition, etc.) that combine to form a geological
    model. This class can also define forward operators that convert the implicit scalar field into estimates of measured properties.  
    """
    def __init__( self, 
                 name : str, 
                 type : BaseNF, 
                 deformation : OffsetBase = None, 
                 overprint : Overprint = None, 
                 propertyField=None,
                 eid=-1,
                 **kwargs ):
        """
        Initialise a new GeologicalField. 

        Parameters
        ------------
        name : str
            A name for this event.
        type : child class of `curlew.fields.BaseNF` (e.g., `curlew.fields.fourier.NFF`). This determines the type of neural or 
                analytical field used to paramaterise this scalar field.
        deformation : function, None
            A function that translates the values of this scalar field into vector displacements, such that f(X1, self) 
            returns an array X0 of shape (N,ndim) that represents the pre-deformation coordinates of X1.
        overprint : curlew.interactions.Overprint, None
            A function that combines the results of this GeologicalField with previous ones.
        propertyField : curlew.geology.property.PropertyModelBase, None
            A custom forward function that translates the implicit scalar field values into estimates of some measured
            property (e.g., density, mineralogy, etc.). See `curlew.geology.property` for details, including implementations
            of constant and learnable forward models. If None, property predictions will not be computed.
        eid : int
            A unique integer denoting the ID of this GeoField (and the associated geological event). If -1 (default), this will 
            be updated when fields are combined into a GeoModel.
        
        Keywords
        -------------
        All keywords will be passed to the __init__ function of the specified `type` class (e.g., `curlew.fields.fourier.NFF`).
        Optionally, a 'field' keyword can be used to pass an already constructed field object (of the specified `type`).
        """
        self.name = name # name of this geological field (and, typically, the corresponding underlying neural or analytical field).
        self.eid = eid # position of this geological field in the event sequence
        self.parent = None # older geological event; used when chaining multiple geological fields
        self.parent2 = None # a second older geological eveint; defined if this geological field represents a domain boundary.
        self.child = None # link to the next-youngest geological event

        # objects that determine deformation, overprint and property prediction
        # to (in combination) determine how this geological field interacts with others
        self.deformation = deformation # deformation function, if defined
        self.overprint = overprint # overprint function, if defined
        self.propertyField = propertyField # forward (property) prediction, if defined

        self.isosurfaces = {} # this will hold any added isosurfaces
        self.llookup = None # this will be defined if lithology IDs have been defined
                            # (when GeoFields are combined into a GeoModel).

        self._lastScalar = {} # temporary storage for last evaluated scalar field (used to retrieve results from e.g., faults)
        self._lastDisp = {} # temporary storage for last evaluated displacement field (used to retrieve results from e.g., faults)

        # build underlying field
        if 'field' in kwargs:
            self.field = kwargs['field'] # field already constructued :-)
        else:
            self.field = type(name=name, **kwargs) # initialise a neural field
        
        # get dimensionality as it is useful to know
        if hasattr(self.field, 'input_dim'): # most cases; but field could also be a float or an int!
            self.input_dim = self.field.input_dim
        else:
            self.input_dim = curlew.default_dim
        
         # assign transform function for determining paleocoordinates in the underlying scalar field
         # (N.B. functools.partial is needed rather than a lambda function to allow pickling)
         # (N.B. this is done like this so classess implementing ScalarField can train
         #  on CSet data in which the coordinates are passed in modern (model) coordinates)
        if not isinstance(self.field, float) and not isinstance(self.field, int):
            self.field.transform = functools.partial(apply_child_undeform, sf=self)

    def copy(self):
        """
        Create a copy of this GeologicalField object for e.g., incorporation into a different model. Note that this will unlink any
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
            If True, displacements applied by younger GeologicalField instances (linked to this one through `self.child`) will 
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
            Reset eachGeologicalField's optimiser to the specified learning rate before training.
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

        assert not (isinstance(self.field, int) or isinstance(self.field, float)), "Cannot fit constant fields."

        # get constraints set to use
        if 'C' in kwargs:
            self.field.bind( kwargs.pop('C', self.field.C) )
        C = self.field.C 

        # of evaluations of displacement fields
        # that will not change
        C0 = C # no need to reconstruct
        if cache and self.child is not None:
            if faultBuffer > 0:
                from curlew.geology.interactions import FaultOffset
                def buffer( arr ): # quick function for masking points on faults
                    f = self.child
                    mask = np.full( len(arr), True )
                    while f is not None:
                        if ((f.deformation is not None) and isinstance(f.deformation, FaultOffset)): # faults
                            b = f.buffer( arr, f.deformation.contact, faultBuffer )
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
            out = self.field.fit(epochs, C=C0, transform=False, 
                                 opt=[self.deformation, self.overprint, self.propertyField], # include possibly learnable elements in these
                                 **kwargs)
        finally:
            if C0.grid is not None:
                C0.grid._clearCache()
            self.field.bind(C) # ensure this always runs! (e.g., in case of keyboard interrupts)
            
        return out
    
    def forward(self, x: torch.Tensor, undef=True) -> torch.Tensor:
        """
        Call the scalar field wrapped by this GeoField instance and return the
        result in a `curlew.core.Geode` instance.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, input_dim), where N is the batch size.
        undef : bool
            True if x is specified in undeformed (modern-day) coordinates. Default is True.

        Returns
        -------
        curlew.core.Geode
            A Geode object containing `x` (the positions that were evaluated) and `scalar` (the resulting values).
        """
        # field is constant -- values are already known! :-)
        if isinstance(self.field, float) or isinstance(self.field, int): 
            value = torch.full( (len(x),1), float(self.field), device=curlew.device, dtype=curlew.dtype)
        else:
            value = self.field.forward(x, transform=undef) # evaluate the underlying field
        return value

    def predict(self, x: ArrayLike, combine=False, to_numpy=True, transform=True, values=None, 
                      litho : bool = True, gradient : bool = False) -> np.ndarray:
        """
        Predict scalar values belonging to this and/or previousGeologicalFields.

        Parameters
        ----------
        x : np.ndarray | torch.tensor | curlew.geometry.Grid
            An array of shape (N, input_dim) containing the modern-day coordinates at which to evaluate
            this (and previous) GeologicalField.
        combine: bool
            True if previous scalar fields should be aggregated to give a combined
            scalar field output. Otherwise, evaluate the scalar field related to this
            specific structure alone. Default is False.
        to_numpy : bool
            True if the results should be cast to a numpy array rather than a `torch.Tensor`.
        transform : bool
            True if the coordinates should be undeformed (i.e. represent modern-day coordinates) or not (i.e. represent 
            this events paleo-coordinates).
        values : torch.Tensor
            Pre-computed results of this GeoField (used sometimes to save recomputing). Default is None (values will be computed).
        litho : bool
            True (default) if lithology codes should be computed.
        gradient : bool
            True if the gradient at each `x` should also be calculated. Default is False.
        """
        
        # check torch types 
        grid = None
        if isinstance(x, Grid):
            grid = x
            x = grid.coords()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor( x, device=curlew.device, dtype=curlew.dtype)

        # operation needs to be done in batches?
        if (len(x) > curlew.batchSize) and (values is None):
                out = batchEval( x, self.predict, combine=combine, to_numpy=to_numpy, transform=True, gradient=gradient,
                                    values=values, litho=litho, batch_size=curlew.batchSize )
                out.grid = grid
                return out
        
        if combine and (self.parent is not None): # COMBINE RESULTS FROM MULTIPLE FIELDS?
            if self.parent2 is not None: # DOMAIN BOUNDARIES (essentially use this field as a mask to combine older fields)
                # predict value of domain scalar field
                # TODO - how to deal with cases where transform is False? (older transforms should be applied, but not younger ones)
                domain = self.predict( x, combine=False, to_numpy=False, transform=True, gradient=gradient )
                #domain = self( x, undef=transform )

                # evaluate isosurfaces to get threshold values, if needed (allows self.bound to containt str isosurface names)
                assert self.overprint is not None, "Overprint must be defined for domain boundary."
                self.overprint.thresh = self.getIsovalue(self.overprint.threshold) # this thresh separates the domain into two halves
                
                # predict parent fields
                # TODO - how to deal with cases where transform is False? (older transforms should be applied, but not younger ones)
                parent = self.parent.predict( x, combine=True, to_numpy=False, transform=True, gradient=gradient )
                parent2 = self.parent2.predict( x, combine=True, to_numpy=False, transform=True, gradient=gradient )

                # apply overprint given domain boundary mask and return
                out = self.overprint.apply( parent, parent2, domain=domain.scalar )
                out.fields[self.name] = domain # also store the results evaluated from this field
            else: # NORMAL CASE - FAULTS OR OTHER GENERATIVE EVENTS
                # TODO - how to deal with cases where transform is False? (older transforms should be applied, but not younger ones)
                parent = self.parent.predict( x, combine=True, to_numpy=False, transform=True, gradient=gradient )

                # GENERATIVE EVENTS (OVERPRINT OLDER FIELDS)
                if self.overprint is not None: 
                    
                    # evaluate this field
                    child = self.predict( x, combine=False, to_numpy=False, transform=True, gradient=gradient )

                    # evaluate isosurfaces to get threshold values
                    # (allows self.bound to containt str isosurface names)
                    self.overprint.thresh = self.getIsovalues(values=self.overprint.threshold)

                    # apply overprinting operation
                    out = self.overprint.apply( parent, child, domain=None )
                
                # NO OVERPRINT DEFINED (PURELY KINEMATIC EVENTS LIKE FAULTS)
                else:
                    out = parent # easy! :-)
          
        else: # evaluate field results and put into a Geode object
            if values is not None:
                scalar = values.squeeze() # already computed (e.g., by a gradient computation)
            else:
                scalar = self.forward( x, undef=transform ).squeeze() # evaluate scalar value 
            if scalar.ndim==0: # if only evaluating one location, ensure result is a vector
                scalar = torch.tensor([scalar.detach()], dtype=curlew.dtype, device=curlew.device)
            structureID = torch.full( (len(scalar),), self.eid, device=curlew.device, dtype=torch.int)
            structureLookup = {self.eid : self.name}

            # evaluate gradient at the chosen positions
            grad = None
            if gradient:
                grad = self.gradient( x, # needs to be copied to avoid multiple passes through gradient tree
                                     return_vals=False, normalize=True, to_numpy=False, transform=transform, retain_graph=True )
            
            # determine lithology IDs based on isosurfaces (if defined)
            lid = 0
            if self.llookup is not None:
                lid = self.llookup.get(self.name, -1)
            lithoID = torch.full( (len(scalar),), lid, device=curlew.device, dtype=torch.int)
            lithoLookup = {-1 : "Undefined", lid : self.name }
            if litho and (self.overprint is not None) and (self.parent2 is None): # only define lithologies for generative events (obviously)
                isosurfaces = self.getIsovalues()
                if len(isosurfaces) > 0:
                    keys = np.array(list(isosurfaces.keys()))
                    values = np.array(list(isosurfaces.values()))
                    ixx = np.argsort(values) # sort these to ensure isosurfaces are applied from smallest to largest
                    for i,(k,v) in enumerate(zip(keys[ixx], values[ixx])):
                        k = f"{self.name}_{k}" # include field name in k to help ensure it is unique!
                        mask = scalar >= v # isosurface is formation top
                        if self.llookup is not None:
                            assert k in self.llookup, "Lithology lookup must contain all isosurfaces in generative fields"
                            i = self.llookup[k]
                        lithoLookup[i] = k # store link betweein ID and lithology name
                        lithoID[mask] = i # update lithology ID array

            # which (temporal) reference was used for these results
            if transform:
                crs="model" # model coordinates
            else:
                crs=self.name # field coordinates

            out = Geode(x=x, grid=grid, crs=crs, 
                        lithoID=lithoID, lithoLookup=lithoLookup,
                        scalar=scalar, gradient=grad,
                        structureID=structureID, structureLookup=structureLookup,
                        fields={self.name : scalar}, offsets={})

            # evaluate property (prediction) field if defined
            if self.propertyField is not None:
                out = self.propertyField.predict(geode=out) # Takes in a Geode and returns updated Geode

        # check if this field has a defined displacement field (e.g., dyke opening vectors or fault offsets)
        if id(x) in self._lastDisp:
            out.offsets[self.name] = self._lastDisp[id(x)]
            del self._lastDisp[id(x)]
        
        # also store values used to compute displacement field too
        if id(x) in self._lastScalar: 
            out.fields[self.name] = self._lastScalar[id(x)]
            del self._lastScalar[id(x)]

        # return
        if to_numpy:
            out = out.numpy()
        out.grid=grid
        return out

    def gradient(self, x: ArrayLike, return_vals=False, normalize=True, to_numpy=True, transform=True, retain_graph=False, create_graph=False):
        """
        Return the gradient vector of this GeologicalField at the specified location. Note that this
        does  not combine the results from previous scalar fields first (i.e. the prediction
        is done using `combine=False`).

        Parameters
        ----------
        x : np.ndarray
            An array of shape (N, input_dim) containing the modern-day coordinates at which to evaluate
            this (and previous) GeologicalField.
        return_vals : bool
            True if evaluated scalar values should be returned as well as classes. Default is False. 
        normalize : bool
            True if gradient vectors should be normalised to length 1 (i.e. to represent poles to planes). Default is True. 
        to_numpy : bool
            True if the results should be cast to a numpy array rather than a `torch.Tensor`.
        transform : bool
            True if results should be transformed into modern-day coordinates.
        retain_graph : bool, optional
            True if the gradient graph should be retained (to allow e.g., subsequent backpropagation). Default is False.
        create_graph : bool, optional
            True if the gradient value should have an underlying graph to allow it to influence back-prop operations. Default is False.
        
        Returns
        --------
        Gradient vectors at the specified locations (`x`). If `return_vals` is `True`,
        tuple (gradients, values) will be returned. If the underlying field is multi-dimensional, 
        a gradient tensor will be returned.
        """

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=curlew.dtype, device=curlew.device, requires_grad=True)
        if not x.requires_grad:
            #x = torch.tensor(x, dtype=curlew.dtype, device=curlew.device, requires_grad=True)
            x = x.detach().clone().requires_grad_(True)
        
        # evalaute scalar value
        pred = self.forward( x, undef=transform ).squeeze()

        # get gradients
        grad_out = torch.autograd.grad(
            outputs=pred,
            inputs=x,
            grad_outputs=torch.ones_like(pred),
            create_graph=create_graph,
            retain_graph=retain_graph,
        )[0]

        # normalise gradients
        if normalize:
            norm = torch.norm(grad_out, dim=-1, keepdim=True) + 1e-8
            grad_out = grad_out / norm

        if to_numpy:
            grad_out = grad_out.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()

        if return_vals: # return gradient array and predictions Geode
            return grad_out, self.predict( x, combine=False, to_numpy=to_numpy, 
                                          transform=transform, values=pred)
        else:
            # return gradient array only
            return grad_out

    def undeform(self, x: ArrayLike) -> torch.Tensor:
        """
        Remove deformation (displacements) from the passed set of coordinates.

        Translates from present-day coordinates to coordinates relevant for this GeologicalField by removing, 
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
        tonp = False
        if isinstance( x, np.ndarray ): # cast numpy to torch if need be (should not be though)
            x = torch.tensor( x, device=curlew.device, dtype=curlew.dtype)
            tonp = True
        
        # remove any child deformation
        if self.child is not None:
            x = self.child.undeform( x ) # undeform to the time-step relevant for this GeologicalField

        # handle our own displacement
        if self.deformation is not None:
                offset = self.displacement(x) # get deformation vectors
                x = x - offset # apply deformation to the input coordinates
        
        # return (in a matching array format)
        if tonp:
            return x.cpu().detach().numpy()
        return x

    def displacement(self, x: ArrayLike) -> np.ndarray:
        """
        Return the displacement vectors associated with this GeologicalField at the specified locations. These are the
        displacements that would be removed during this fields contribution to `undeform(...)`.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            An array of shape (N, input_dim) containing the modern-day coordinates at which to evaluate
            this (and previous) GeologicalField.

        Returns
        -------
        np.ndarray | torch.Tensor
            An array of shape (N, input_dim) containing the deformation vectors at the specified locations.
        """
        tonp = False
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=curlew.dtype, device=curlew.device)
            tonp = True

        if self.deformation is None:
            offset = torch.zeros_like(x) # no deformation
        else:
            offset = self.deformation.eval(x, self)
        
        if tonp: # cast to numpy if needed
            return offset.detach().cpu().numpy() # return as numpy array
        else:
            return offset

    def addIsosurface( self, name :str, *, value = None, seed = None):
        """
        Add a (geologically meaningful) isosurface to this scalar field. These
        represent e.g., stratigraphic contacts and (when determining lithology IDs)
        are interpreted as formation tops.

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
            self.isosurfaces[name] = np.asarray( seed )
        if value is not None:
            self.isosurfaces[name] = value

    def deleteIsosurface( self, name ):
        """
        Remove the specified isosurface from this GeologicalField's `isosurfaces` dict, if it exists.
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
                i = np.mean( self.predict( v, litho=False ).scalar )
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
        pred = self.predict(x).numpy().scalar # get values at points
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

    def loss(self):
        """
        Compute loss associated with the underlying field and learnable property, deformation or overprint objects.
        """

        # constant fields have no loss
        if (isinstance(self.field, int) or isinstance(self.field, float)): return torch.tensor(0.0, device=curlew.device), {}

        loss, details = self.field.loss() 
        for o in [self.propertyField, self.deformation, self.overprint]:
            if o is not None: 
                ll, dets = o.loss()
                loss = loss + ll
                details.update(dets)
        return loss, details

    def zero(self):
        """
        Zero optimiser associated with the underlying field and learnable property, deformation or overprint objects. 
        """
        if not (isinstance(self.field, int) or isinstance(self.field, float)): self.field.zero()
        if self.propertyField is not None: self.propertyField.zero()
        if self.deformation is not None: self.deformation.zero()
        if self.overprint is not None: self.overprint.zero()

    def step(self):
        """
        Step optimiser associated with the underlying field and learnable property, deformation or overprint objects. 
        """
        if not (isinstance(self.field, int) or isinstance(self.field, float)): self.field.step()
        if self.propertyField is not None: self.propertyField.step()
        if self.deformation is not None: self.deformation.step()
        if self.overprint is not None: self.overprint.step()
    
    def set_rate(self, lr=1e-3):
        """
        Set the learning rate for the underlying field and learnable property, deformation or overprint objects.
        """
        for o in [self.field, self.propertyField, self.deformation, self.overprint]:
            if (o is not None) and (not(isinstance(self.field, int) or isinstance(self.field, float))) and (o.optim is not None):
                o.set_rate(lr=lr)
