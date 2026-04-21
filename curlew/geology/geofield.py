"""
A class defining a structural scalar (implicit) field. This is where a lot of the magic happens, including
the chaining of multiple neural fields and the application of deformation functions to implement kinematic
events such as faults and sheet intrusions.
"""

import curlew
from curlew import _tensor, _numpy
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

def apply_child_undeform(x, end, sf):
    """
    Placeholder function to avoid Lambda functions. This needs to be dynamic as we may not know yet
    what `sf.child` is -- just that we should check if it is defined while undeforming coordinates
    during the evaluation of a scalar field.
    """
    if sf == end:
        return x  # reached the end of the line; no need to transform anymore
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
                 anchors=[], 
                 isosurfaces=[],
                 **kwargs ):
        """
        Initialise a new GeologicalField. 

        Parameters
        ------------
        name : str
            A name for this event.
        type : child class of `curlew.fields.BaseNF` (e.g., `curlew.fields.fourier.NFF`). This determines the type of neural or 
                analytical field used to paramaterise this scalar field.
        deformation : curlew.interactions.Deformation, list, None
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
        anchors : list
            Optional list of dicts defining anchors to add to the added field. Anchors will be added by calling: `self.addAnchor(**argsD)` for each args in this list. 
        isosurfaces : list
            Optional list of dicts defining isosurfaces to add to the added field. Isosurfaces will be added by calling: `self.addIsosurfaces(**args)` for each args in this list.
        
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
        self.anchors = {} # named anchor points in modern-day coordinates (paleo positions via getAnchor)
        self.volumes = {} # named boolean volumes (boolean functional domains; evaluated via getVolume)
        self.llookup = None # this will be defined if lithology IDs have been defined
                            # (when GeoFields are combined into a GeoModel).

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
        # attach transform for (non-constant) fields so they can train/evaluate in modern coords
        if not isinstance(self.field, float) and not isinstance(self.field, int):
            self.field.transform = functools.partial(apply_child_undeform, sf=self)

        # Attach anchors / isosurfaces for this underlying field during construction
        for anchor in anchors:
            anchor['field'] = self.field.name # associate to the constructed field
            self.addAnchor(**anchor)
        for iso in isosurfaces:
            iso['field'] = self.field.name # associate to the constructed field
            self.addIsosurface(**iso)
        
    def addField(self, fieldName: str, type: BaseNF = None, anchors=[], isosurfaces=[], **kwargs):
        """
        Add an additional underlying scalar field to this GeoField.

        Parameters
        ----------
        fieldName : str
            Name used to identify this underlying field (for isosurface associations and outputs).
        type : child class of `curlew.fields.BaseNF`, optional
            Type of field to construct if `field` is not passed in kwargs.
        Parameters
        ----------
        field : BaseNF | float | int
            If provided, uses this pre-constructed field (or constant).
        anchors : list
            Optional list of dicts defining anchors to add to the added field. Anchors will be added by calling: `self.addAnchor(**argsD)` for each args in this list. 
        isosurfaces : list
            Optional list of dicts defining isosurfaces to add to the added field. Isosurfaces will be added by calling: `self.addIsosurfaces(**args)` for each args in this list.
        All other keywords are forwarded to the underlying field constructor when `type` is provided.
        """
        assert isinstance(fieldName, str) and len(fieldName) > 0, "fieldName must be a non-empty string."

        if 'field' in kwargs:
            new_field = kwargs['field']
        else:
            assert type is not None, "type must be provided when constructing a new field."
            new_field = type(name=fieldName, **kwargs)

        # Promote existing field to list if needed
        if not isinstance(self.field, list):
            self.field = [self.field]

        # prevent duplicates by underlying field.name when possible
        if hasattr(new_field, "name"):
            for f in self.field:
                if hasattr(f, "name") and (f.name == new_field.name):
                    raise ValueError(f"Underlying field with name '{new_field.name}' already exists.")

        # store our new field :-)
        self.field.append(new_field)
        if not isinstance(new_field, float) and not isinstance(new_field, int): # not sure why you would do this, but hey... 
            new_field.transform = functools.partial(apply_child_undeform, sf=self)

        # Attach anchors / isosurfaces for this underlying field during construction
        for anchor in anchors:
            anchor['field'] = fieldName # associate to this field
            self.addAnchor(**anchor)
        for iso in isosurfaces:
            iso['field'] = fieldName # associate to this field
            self.addIsosurface(**iso)

    def _field_list(self):
        """
        Return `self.field` as a list without mutating.
        """
        if isinstance(self.field, list): return self.field
        else: return [self.field]

    def getField(self, field=0) -> int:
        """
        Return a field associated to this GeoField object by index or name.

        Parameters
        ----------
        field : int | str
            If int, treated as list index. If str, matched against each underlying field's `name`.
        """
        fields = self._field_list()
        if isinstance(field, int):
            return fields[field] # easy!
        if isinstance(field, str):
            for i, f in enumerate(fields):
                if hasattr(f, "name") and (f.name == field):
                    return f
            raise KeyError(
                f"Unknown field '{field}'. Known: {[f.name for f in fields if hasattr(f,'name')]}"
            )
        raise TypeError("field must be an int index or a str field name.")

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
            
        If this GeoField has multiple fittable underlying fields, a list of these losses and details will be returned.
        """
        fields = self._field_list()
        outList = []
        for i, field in enumerate(fields):
            if isinstance(self.field, int) or isinstance(self.field, float):
                continue # no point fitting integers ;-) 
            
            # get CSet associated to this field
            C = field.C 
            assert C is not None, f"Field '{field.name}' has no constraints."
            
            
            # of evaluations of displacement fields
            # that will not change
            C0 = C # no need to reconstruct in many cases
            if cache and self.child is not None:
                #if faultBuffer > 0:
                    #from curlew.geology.interactions import FaultOffset
                    #def buffer( arr ): # quick function for masking points on faults
                    #    f = self.child
                    #    mask = np.full( len(arr), True )
                    #    while f is not None:
                    #        if ((f.deformation is not None) and isinstance(f.deformation, FaultOffset)): # faults
                    #            b = f.buffer( arr, f.deformation.contact, faultBuffer )
                    #        elif (f.parent2 is not None):
                    #            b = f.buffer( arr, f.bound, faultBuffer ) # domain boundary
                    #        mask[ b ] = False # identify and remove points within buffer distance of a fault 
                    #        f = f.child
                    #    return mask
                    #C0.filter(buffer)

                # retro-deform other constraints
                C0 = C.numpy().transform( self.child.undeform )
            
            # inject paleo-coordinate anchors so the field can use them during training if needed
            for name, anchor in self.anchors.items():
                # Only apply anchors associated to the current field
                if isinstance(anchor, tuple) and (anchor[0] != i) and (anchor[0] != field.name):
                    continue # ignore anchors associated to other fields
                pos, direction = self.getAnchor(name, to_numpy=False)
                if direction is not None: # anchor represents a direction
                    setattr(field, name, direction)
                elif pos is not None: # anchor represents a position
                    setattr(field, name, pos)

            # fit
            try:
                out = field.fit(epochs, 
                                C=C0, 
                                transform=False, 
                                opt=[self.deformation, self.overprint, self.propertyField], # include possibly learnable elements in these
                                **kwargs)
            finally:
                if C0.grid is not None:
                    C0.grid._clearCache()
                field.bind(C) # ensure this always runs! (e.g., in case of keyboard interrupts)
            
            outList.append(out)
        
        # return
        if len(outList) == 1:
            return outList[0]
        else:
            return outList
    
    def forward(self, x: torch.Tensor, undef=True, field=0) -> torch.Tensor:
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
        fobj = self.getField(field)

        # constant field
        if isinstance(fobj, float) or isinstance(fobj, int): 
            if isinstance(x, Geode):
                value = x
                value.scalar = torch.full((len(x.x),), float(fobj), device=curlew.device, dtype=curlew.dtype)
            else:
                value = torch.full((len(x), 1), float(fobj), device=curlew.device, dtype=curlew.dtype)
        else:
            # inject paleo-coordinate anchors as parameters so the field can use them if needed
            for aname, anchor in self.anchors.items():
                if isinstance(anchor, tuple) and (self.getField(anchor[0]) != fobj):
                    continue # ignore anchors associated to other fields
                pos, direction = self.getAnchor(aname, to_numpy=False)
                setattr(fobj, aname, pos)
                if direction is not None:
                    setattr(fobj, aname + "_direction", direction)
            value = fobj.forward(x, transform=undef) # evaluate the chosen underlying field
        
        if isinstance(value, Geode):
            # keep legacy key; scalar corresponds to chosen underlying field
            value.fields[self.name] = value.scalar
        return value

    def predict(self, x: ArrayLike, combine=False, to_numpy=True, transform=True, values=None, 
                      litho : bool = True, props=True, isosurfaces=True, gradient : bool = False) -> np.ndarray:
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
        props : bool
            True (default) if properties should be computed when a `propertyField` is defined.
        isosurfaces : bool
            True (default is False) if isosurface (and anchor) values should be computed and stored. Default is True. 
        gradient : bool
            True if the gradient at each `x` should also be calculated. Default is False.
        """
        
        # check torch types 
        grid = None
        if isinstance(x, Grid):
            grid = x
            x = grid.coords()
        if not isinstance(x, torch.Tensor):
            x = _tensor( x )

        # operation needs to be done in batches?
        if (len(x) > curlew.batchSize) and (values is None):
                out = batchEval( x, self.predict, combine=combine, to_numpy=to_numpy, transform=transform, 
                                    values=values, litho=litho, props=props, isosurfaces=isosurfaces, gradient=gradient,
                                    batch_size=curlew.batchSize )
                out.grid = grid
                return out
        
        if not transform:
            transform = self # specify that transforms should stop on encountering this field in the tree
                             # i.e. evaluate older transforms, but not younger ones.
        
        if combine and (self.parent is not None): # COMBINE RESULTS FROM MULTIPLE FIELDS?
            if self.parent2 is not None: # DOMAIN BOUNDARIES (essentially use this field as a mask to combine older fields)
                # predict value of domain scalar field
                domain = self.predict( x, combine=False, to_numpy=False, transform=transform, gradient=gradient )

                # evaluate isosurfaces to get threshold values, if needed (allows self.bound to containt str isosurface names)
                assert self.overprint is not None, "Overprint must be defined for domain boundary."
                self.overprint.thresh = self.getIsovalue(self.overprint.threshold) # this thresh separates the domain into two halves
                
                # predict parent fields
                parent = self.parent.predict( x, combine=True, to_numpy=False, transform=transform, gradient=gradient )
                parent2 = self.parent2.predict( x, combine=True, to_numpy=False, transform=transform, gradient=gradient )

                # apply overprint given domain boundary mask and return
                out = self.overprint.apply( parent, parent2, domain=domain.scalar )
                out.fields[self.name] = domain.scalar # also store the results evaluated from this field
            else: # NORMAL CASE - FAULTS OR OTHER GENERATIVE EVENTS
                parent = self.parent.predict( x, combine=True, to_numpy=False, transform=transform, gradient=gradient )
                child = self.predict( x, combine=False, to_numpy=False, transform=transform, gradient=gradient )

                # GENERATIVE EVENTS (OVERPRINT OLDER FIELDS)
                if self.overprint is not None: 
                    # apply overprinting operation
                    if isinstance(self.overprint, list):
                        out = parent
                        for o in self.overprint: # apply multiple overprint inequalities
                            o.thresh = self.getIsovalues(values=o.threshold)
                            out = o.apply( out, child )
                    else:
                        # evaluate isosurfaces to get threshold values
                        self.overprint.thresh = self.getIsovalues(values=self.overprint.threshold)
                        out = self.overprint.apply( parent, child, domain=None )
                
                # NO OVERPRINT DEFINED (PURELY KINEMATIC EVENTS LIKE FAULTS)
                else:
                    out = parent # easy! :-)
                    out.fields[self.name] = child.scalar # also add scalar values from this field
          
        else: # evaluate field results and put into a Geode object

            # create an output Geode
            out = Geode(x=x)

            if values is not None:
                # already computed (e.g., by a gradient computation) for the first underlying field
                out.scalar = values.squeeze()

                # still populate Geode.fields for backward compatibility + any extra underlying fields
                out.fields[self.name] = out.scalar
                if isinstance(self.field, list) and (len(self.field) > 1):
                    x_eval = out.x
                    for i in range(1, len(self.field)):
                        fobj = self.field[i]
                        fname = fobj.name if hasattr(fobj, "name") else f"{self.name}_{i}"
                        if isinstance(fobj, (float, int)):
                            s = torch.full((len(x_eval),), float(fobj), device=curlew.device, dtype=curlew.dtype)
                        else:
                            v = self.forward(x_eval, undef=transform, field=i)
                            s = v.scalar if isinstance(v, Geode) else v.squeeze()
                        if s.ndim == 0:
                            s = _tensor([s.detach().item()] )
                        out.fields[fname] = s
            else:
                out = self.forward(out, undef=transform, field=0) # evaluate scalar value for first field
                # If multiple fields exist, also evaluate and store the others (scalar stays as field 0).
                if isinstance(self.field, list) and (len(self.field) > 1):
                    x_eval = out.x
                    for i in range(1, len(self.field)):
                        fobj = self.field[i]
                        fname = fobj.name if hasattr(fobj, "name") else f"{self.name}_{i}"
                        v = self.forward(x_eval, undef=transform, field=i)
                        s = v.scalar if isinstance(v, Geode) else v.squeeze()
                        if s.ndim == 0:
                            s = _tensor([s.detach().item()] )
                        out.fields[fname] = s
            if out.scalar.ndim==0: # if only evaluating one location, ensure result is a vector
                out.scalar = _tensor([out.scalar.detach().item()] )
            
            out.structureID = torch.full( (len(out.scalar),), self.eid, device=curlew.device, dtype=torch.int)
            out.structureLookup = {**out.structureLookup, **{self.eid : self.name}}

            
            # evaluate gradient at the chosen positions
            # TODO - evaluate gradient here rather than recomputing?
            if gradient:
                out.gradient = self.gradient( x,
                                     return_vals=False, normalize=True, to_numpy=False, transform=transform, retain_graph=True )
            
            # determine lithology IDs based on isosurfaces (if defined)
            lid = 0
            if self.llookup is not None:
                lid = self.llookup.get(self.name, -1)
            out.lithoID = torch.full( (len(out.scalar),), lid, device=curlew.device, dtype=torch.int)
            out.lithoLookup = {**out.lithoLookup, **{-1 : "Undefined", lid : self.name }}
            iso_values = None
            if litho and (self.overprint is not None) and (self.parent2 is None): # only define lithologies for generative events (obviously)
                iso_values = self.getIsovalues()
                if len(iso_values) > 0:
                    keys = np.array(list(iso_values.keys()))
                    values = np.array(list(iso_values.values()))
                    ixx = np.argsort(values) # sort these to ensure isosurfaces are applied from smallest to largest
                    for i,(k,v) in enumerate(zip(keys[ixx], values[ixx])):
                        k = f"{self.name}_{k}" # include field name in k to help ensure it is unique!
                        mask = out.scalar >= v # isosurface is formation top
                        if self.llookup is not None:
                            assert k in self.llookup, "Lithology lookup must contain all isosurfaces in generative fields"
                            i = self.llookup[k]
                        out.lithoLookup[i] = k # store link betweein ID and lithology name
                        out.lithoID[mask] = i # update lithology ID array

            if isosurfaces:
                # evaluate isosurfaces to get either threshold values (if 
                # not evaluating on a grid), or contours (if evaluating on a grid)
                if iso_values is None:
                    iso_values = self.getIsovalues()
                out.isosurfaces[self.name] = iso_values
                
                if len(self.anchors) > 0:
                    out.anchors[self.name] = {}
                    for n in self.anchors:
                        anchor = self.getAnchor(n, to_numpy=True)
                        out.anchors[self.name][n] = anchor
                
            # which (temporal) reference was used for these results
            if transform:
                out.crs="model" # model coordinates
                out.x = x # store the coordinates that were used for evaluation (not the possibly transformed ones currently in the Geode object)
            else:
                out.crs=transform.name # field coordinates

            # evaluate property (prediction) field if defined
            if (self.propertyField is not None) and props:
                out = self.propertyField.predict(geode=out) # Takes in a Geode and returns updated Geode
            
        # return
        if to_numpy:
            out = out.numpy()
        out.grid=grid # also add grid if defined
        return out

    def gradient(self, x: ArrayLike, return_vals=False, normalize=True, to_numpy=True, transform=True, retain_graph=False, create_graph=False, field=0):
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
        field : int | str, optional
            Which underlying field to differentiate. If int, treated as index into `self.field` (if a list).
            If str, matched against each underlying field's `name`. Default is 0 (first field).
        
        Returns
        --------
        Gradient vectors at the specified locations (`x`). If `return_vals` is `True`,
        tuple (gradients, values) will be returned. If the underlying field is multi-dimensional, 
        a gradient tensor will be returned.
        """

        if not isinstance(x, torch.Tensor):
            x = _tensor(x ).requires_grad_(True)
        if not x.requires_grad:
            #x = torch.tensor(x, dtype=curlew.dtype, device=curlew.device, requires_grad=True)
            x = x.detach().clone().requires_grad_(True)
        
        # evaluate scalar value for the selected underlying field
        # Keep a torch copy for downstream calls (e.g. predict(values=...))
        pred_t = self.forward(x, undef=transform, field=field).squeeze()

        # get gradients
        grad_out = torch.autograd.grad(
            outputs=pred_t,
            inputs=x,
            grad_outputs=torch.ones_like(pred_t),
            create_graph=create_graph,
            retain_graph=retain_graph,
        )[0]

        # normalise gradients
        if normalize:
            norm = torch.norm(grad_out, dim=-1, keepdim=True) + 1e-8
            grad_out = grad_out / norm

        if to_numpy:
            grad_out = _numpy(grad_out)

        if return_vals: # return gradient array and predictions Geode
            # `predict(values=...)` expects torch tensors; always pass the torch version.
            g = self.predict(x, combine=False, to_numpy=to_numpy, transform=transform, values=pred_t)
            # `predict(..., values=pred)` assumes `values` are for the first field; if requesting a different
            # underlying field, update the returned scalar to match that field for consistency.
            if not (isinstance(field, int) and field == 0):
                try:
                    fobj = self.getField(field)
                    fname = fobj.name if hasattr(fobj, "name") else None
                    if fname is not None and hasattr(g, "fields") and (fname in g.fields):
                        g.scalar = g.fields[fname]
                except Exception:
                    # fall back silently; gradient is still correct even if scalar cannot be swapped
                    pass
            return grad_out, g
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
            x = _tensor( x )
            tonp = True
        
        # remove any child deformation
        if self.child is not None:
            x = self.child.undeform( x ) # undeform to the time-step relevant for this GeologicalField

        # handle our own displacement
        if self.deformation is not None:
                offset = self.displacement(x) # get deformation vectors
                if isinstance(x, Geode):
                    #n.b. offset == x in this case, and x.offsets[self.name] was defined by self.displacement call
                    x.x = x.x + x.offsets[self.name]
                else:
                    x = x + offset # apply deformation to the input coordinates
        
        # return (in a matching array format)
        if tonp:
            return _numpy(x)
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
        if isinstance(x, np.ndarray):
            x = _tensor(x )
            tonp = True

        if self.deformation is None:
            offset = torch.zeros_like(x) # no deformation
        else:
            if isinstance(x, Geode): # We are evaluating a full Geode object
                if not isinstance(self.deformation, list):
                    offset = self.deformation.eval(x.x, self) # just one deformation function
                else:
                    offset = self.deformation[0].eval(x.x, self) # evaluate first offset function
                    if len(self.deformation) > 1: # evaluate remaining offset functions
                        for d in self.deformation[1:]:
                            offset = offset + d.eval(x.x, self)
                x.offsets[self.name] = offset # store offset in Geode
                x.isosurfaces[self.name] = self.getIsovalues() # also store fault isosurfaces
                x.anchors[self.name] = {} # also store anchor points 
                for k in self.anchors:
                    x.anchors[self.name][k] = self.getAnchor(k, to_numpy=True)
                return x
            else: # we are just evaluating a bunch of points
                if not isinstance(self.deformation, list):
                    offset = self.deformation.eval(x, self)
                else:
                    offset = self.deformation[0].eval(x, self) # evaluate first offset function
                    if len(self.deformation) > 1: # evaluate remaining offset functions
                        for d in self.deformation[1:]:
                            offset = offset + d.eval(x, self)            
        
        if tonp: # cast to numpy if needed
            return _numpy(offset) # return as numpy array
        else:
            return offset

    def addIsosurface( self, name :str, *, value = None, seed = None, field=0):
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
        field : specify which sub-field of this GeoField instance this isosurface is associated to. Defaults to 0 (first field).
        """
        assert (seed is None) or (value is None), "Either seed or value should be defined, not both."
        assert not( (seed is None) and (value is None)), "Either seed or value should be defined, not both."
        if seed is not None:
            self.isosurfaces[name] = (field, _numpy( seed ))
        if value is not None:
            self.isosurfaces[name] = (field, value)

    def addAnchor( self, name: str, position: ArrayLike = None, *, direction: ArrayLike = None, start: ArrayLike = None, end: ArrayLike = None, field=0 ):
        """
        Add an anchor point or direction to this GeoField. Coordinates are in modern-day (present) coordinates.
        During evaluation they are transformed to reconstructed (paleo) coordinates via undeform and
        exposed on the underlying field (position as ``name``, direction if applicable as ``name + '_direction'``).

        One of the following must be given:

        - **Position only**: ``position`` → anchor is a single point (same as legacy ``addAnchor(name, point)``).
        - **Position + direction**: ``position`` and ``direction`` → two positions are stored (position and
          position + direction). In reconstructed coordinates the direction vector is (end_recon - start_recon)
          and is **normalised**.
        - **Start + end**: ``start`` and ``end`` → two positions are stored. In reconstructed coordinates
          the direction vector is (end_recon - start_recon) and is **not** normalised.

        Parameters
        ----------
        name : str
            A name used to refer to this anchor (used for the field parameter).
        position : array-like, optional
            Single position (x, y, [z]) in modern-day coordinates. If given alone, defines a position anchor.
            If given with ``direction``, base point for a direction anchor.
        direction : array-like, optional
            Direction vector. Used with ``position``; stored as second point = position + direction.
            Reconstructed direction is normalised.
        start : array-like, optional
            Start position for a direction anchor. Must be used with ``end``.
        end : array-like, optional
            End position for a direction anchor. Must be used with ``start``. Reconstructed direction
            (end - start in paleo coordinates) is not normalised.
        field : specify which sub-field of this GeoField instance this anchor is associated to. Defaults to 0 (first field).
        """
        has_pos = position is not None
        has_dir = direction is not None
        has_start = start is not None
        has_end = end is not None

        if has_pos and not has_dir and not has_start and not has_end:
            # Position-only anchor (legacy behaviour)
            self.anchors[name] = (field, _numpy(position))
        elif has_pos and has_dir and not has_start and not has_end:
            # Position + direction: store two points; direction in reconstructed space will be normalised
            start_pt = _numpy(position)
            end_pt = _numpy(position) + _numpy(direction)
            self.anchors[name] = (field, {"start": start_pt, "end": end_pt, "normalize": True})
        elif has_start and has_end and not has_pos and not has_dir:
            # Start + end: store two points; direction in reconstructed space will not be normalised
            self.anchors[name] = (field, {"start": _numpy(start), "end": _numpy(end), "normalize": False})
        else: # Invalid combination
            raise ValueError( "addAnchor requires one of: position only; position and direction; or start and end." )

    def getAnchor( self, name: str, to_numpy: bool = True):
        """
        Return the anchor position (and optionally direction) in reconstructed coordinates for this GeoField.
        If ``self.child`` is not None, applies ``self.child.undeform`` to transform from
        modern-day to this field's reference frame; otherwise uses the stored point(s) as-is.

        Parameters
        ----------
        name : str
            Name of the anchor (as given to addAnchor).
        to_numpy : bool, optional
            If True (default), return numpy array(s); if False, return torch.Tensor(s).
            Pass False when setting parameters on the underlying field (e.g. in forward).
        
        Returns
        -------
        position: np.ndarray | torch.Tensor | None
            Position of shape (1, input_dim).
        direction: np.ndarray | torch.Tensor | None
            Direction of shape (1, input_dim), if a direction anchor is defined.
        """
        assert name in self.anchors, f"Anchor '{name}' not found."
        stored = self.anchors[name]

        # Anchors are stored as (field_selector, spec) where selector is int index or str field name
        if isinstance(stored, tuple) and len(stored) == 2 and isinstance(stored[0], (int, str)):
            stored = stored[1]

        def to_torch(arr):
            a = _numpy(arr)
            if a.ndim == 1:
                a = a[None, :]
            return _tensor(a)

        def undeform_pt(pt):
            t = to_torch(pt)
            if self.child is not None:
                t = self.child.undeform(t)
            return t

        if isinstance(stored, dict):
            # Direction anchor: start, end [, normalize]
            start_recon = undeform_pt(stored["start"])
            end_recon = undeform_pt(stored["end"])
            diff = end_recon - start_recon
            if stored.get("normalize", False):
                norm = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
                diff = diff / norm
            pos_out = start_recon
            dir_out = diff
            if to_numpy:
                pos_out = _numpy(pos_out)
                dir_out = _numpy(dir_out)
            return pos_out, dir_out
        else:
            # Position-only anchor
            pt = undeform_pt(stored)
            if to_numpy:
                pt = _numpy(pt)
            return pt, None

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
        values (if `values` is None). Note that when multiple underlying fields are present, the calculated
        isovalue only corresponds to the field it is associated to (so should not be used with other fields!).
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

            # isosurface spec can be:
            #  seed array/list (defaults to first underlying field)
            #  value float/int (defaults to first underlying field)
            #  tuple(field_selector, seed/value) where selector is int index or str field name
            fieldName = 0 # default to first field
            if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], (int, str)):
                fieldName, v = v # expand
            fobj = self.getField(fieldName)

            if isinstance(v, np.ndarray) or isinstance(v, list):
                v = np.array(v)
                if len(v.shape) == 1:
                    v = v[None, :]

                # apply offset if specified (offset seed points by specified distance in gradient direction)
                if offset != 0:
                    g = self.gradient(v, normalize=True, field=fieldName)
                    v = v + g * offset

                # evaluate (and average) value at seed points for the chosen underlying field
                # Use `forward(field=...)` to avoid evaluating all underlying fields.
                pts = _tensor(v)
                pred = self.forward(Geode(x=pts), undef=True, field=fieldName)
                i = torch.mean(pred.scalar).detach().item()
            else:
                i = v # explicit value
                if offset != 0:
                    # WARNING: crude fallback assumes constant gradient norm for the underlying field.
                    # Only supported for non-constant first field (legacy behaviour).
                    if hasattr(fobj, "mnorm"):
                        i = i + offset * fobj.mnorm
            if values is None:
                out[k] = i
            else:
                out.append(i)
        return out

    def addVolume(self, name: str, expr: str):
        """
        Add a named boolean volume definition.

        Parameters
        ----------
        name : str
            Friendly name used to refer to this volume.
        expr : str
            A python boolean expression (string) that can be evaluated to return a boolean
            numpy / torch array (True inside the volume, False elsewhere). The expression
            is evaluated with:

            - one variable per underlying field name (e.g. GeoField name for field 0, plus
              any added field names)
            - one variable per isosurface name, containing that isosurface's numeric value
            - `np` and `torch`

            Example: ``(ellipse > ellipse_boundary) & (G < linear_y0)``
        """
        assert isinstance(name, str) and len(name) > 0, "name must be a non-empty string."
        assert isinstance(expr, str) and len(expr) > 0, "expr must be a non-empty string."
        self.volumes[name] = expr

    def getVolume(
        self,
        name: str,
        x: ArrayLike = None,
        *,
        geode: Optional[Geode] = None,
        to_numpy: bool = True,
        transform: bool = True,
    ):
        """
        Return the stored volume expression (if `x` and `geode` are None) or evaluate it.

        Parameters
        ----------
        name : str
            Volume name (as given to addVolume).
        x : np.ndarray | torch.Tensor | curlew.geometry.Grid, optional
            Coordinates to evaluate the volume on. If given, `geode` is ignored.
        geode : curlew.core.Geode, optional
            Pre-computed predictions containing `.fields` used to evaluate the volume.
        to_numpy : bool
            If True, return a numpy boolean array. If False, return a torch boolean tensor.
        transform : bool
            Whether to evaluate in modern-day coordinates (True) or this field's paleo frame (False),
            as per `predict`. Only used if `x` is specified.

        Returns
        -------
        str | np.ndarray | torch.Tensor
            Volume expression string (if `x` and `geode` are None), or evaluated boolean mask.
        """
        assert name in self.volumes, f"Volume '{name}' not found."
        expr = self.volumes[name]

        # If no evaluation context requested, just return expression string
        if (x is None) and (geode is None):
            return expr

        # Get field values (all underlying fields) as a Geode
        if x is not None:
            geode = self.predict(
                x,
                combine=False,
                to_numpy=False,
                transform=transform,
                litho=False,
                props=False,
                isosurfaces=False,
            )

        # Build evaluation environment
        # (exposing numpy and torch functions too)
        env = {"np": np, "torch": torch}

        # Populate one variable per field name
        for k, v in getattr(geode, "fields", {}).items():
            # Values may be torch or numpy depending on predict/to_numpy; enforce output format
            if to_numpy:
                env[k] = _numpy(v) if isinstance(v, torch.Tensor) else _numpy(v)
            else:
                env[k] = _tensor(v) if not isinstance(v, torch.Tensor) else v

        # Add isosurface numeric values as scalars
        iso_vals = self.getIsovalues()
        for k, v in iso_vals.items():
            if k in env: assert False, f"Field/isosurface name '{k}' is duplicated."
            env[k] = v

        # Evaluate expression safely (no builtins)
        mask = eval(expr, {"__builtins__": {}}, env)

        # Enforce boolean dtype
        if to_numpy:
            mask = _numpy(mask).astype(bool)
        else:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, device=curlew.device, dtype=torch.bool)
            else:
                mask = mask.bool()
        return mask

    def projectTo(  self, isosurface: str, pts: ArrayLike, nsteps: int = 3, return_normals: bool = False ):
        """
        Project points onto a named implicit isosurface using repeated linearized corrections
        along the underlying scalar field gradient (same idea as one Newton step per iteration:
        if ``f`` were linear along ``g = ∇f``, then ``p ← p + (f_iso - f(p)) g / (g·g)`` lands on
        ``f = f_iso``). After each step the value and **unnormalized** gradient are re-evaluated
        at the new location.

        Parameters
        ----------
        isosurface : str
            Name registered with :py:meth:`addIsosurface` on this ``GeoField``.
        pts : array-like, shape ``(N, ndim)`` or ``(ndim,)``
            Model coordinates to project.
        nsteps : int
            Number of gradient re-evaluations / correction iterations.
        return_normals : bool
            If True, also return the unit normals estimated from the last gradient evaluation.

        Returns
        -------
        np.ndarray or torch tensor (matching input) tuple
            Projected points, same shape as ``pts`` (``(N, ndim)`` or ``(ndim,)``).
            If ``return_normals`` is True, returns ``(projected_pts, normals)`` where
            ``normals`` has the same shape as ``pts``.
        """
        
        assert isosurface in self.isosurfaces, f"Isosurface '{isosurface}' not found."
        spec = self.isosurfaces[isosurface]
        field_sel = 0
        if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], (int, str)):
            field_sel = spec[0]
        iso = float(self.getIsovalue(isosurface))

        return_numpy = not isinstance(pts, torch.Tensor)
        if return_numpy:
            pts = _tensor(pts)

        single = pts.ndim == 1
        if single:
            pts = pts[None, :]

        # ensure we use torch here
        fobj = self.getField(field_sel)
        fname = fobj.name if hasattr(fobj, "name") else None

        g = None
        for _ in range(int(nsteps)):
            g, geode = self.gradient(
                pts,
                return_vals=True,
                normalize=False,
                to_numpy=False,   # stay in torch
                transform=True,
                field=field_sel,
            )
            if hasattr(geode, "fields") and fname is not None and fname in geode.fields:
                f = geode.fields[fname].reshape(-1)
            else:
                f = geode.scalar.reshape(-1)

            gg = (g * g).sum(dim=1) + 1e-12
            pts = pts - ((f - iso) / gg).unsqueeze(1) * g

        if return_normals:
            n = g / (g.norm(dim=1, keepdim=True) + 1e-18)
            if single:
                pts_out = pts[0]
                n_out = n[0]
            else:
                pts_out = pts
                n_out = n
            if return_numpy:
                return _numpy(pts_out), _numpy(n_out)
            return pts_out, n_out

        if single:
            pts = pts[0]
        if return_numpy:
            return _numpy(pts)
        return pts


    def sampleIsosurface(
        self,
        isosurface: str,
        seed_point: ArrayLike,
        target_distance: float,
        volumeName: Optional[str] = None,
        return_normals: bool = False,
        max_samples: int = 1000,
        n_steps: int = 100,
        relax_substeps: int = 1,
        growth_rate: float = 0.8,
        neighbor_cap: int = 8,
        method: str = 'distance',
        seed: int = 1,
    ):
        
        """
        Place samples on an isosurface starting from ``seed_point`` using a spring/bubble
        growth method. Points are iteratively grown outward from the seed across the
        isosurface surface, with repulsion forces keeping them evenly spaced at roughly
        ``target_distance`` apart.

        Parameters
        ----------
        isosurface : str
            Isosurface name (see :py:meth:`addIsosurface`).
        seed_point : array-like, shape ``(ndim,)``
            Starting location in model coordinates. Projected onto the isosurface before use.
        target_distance : float
            Target spacing between samples on the isosurface (model units).
        volumeName : str, optional
            If given, keep only samples where ``getVolume(volumeName, x=...)`` is True.
        max_samples : int, optional
            Maximum number of sample points to place. Default 200.
        n_steps : int, optional
            Number of growth iterations. Default 40.
        relax_substeps : int, optional
            Number of repulsion-relaxation sub-steps per growth iteration. Default 3.
        growth_rate : float, optional
            Probability [0, 1] of attempting to sprout a new point from any existing
            point that has room. Lower values give slower, more uniform coverage. Default 0.8.
        neighbor_cap : int, optional
            A point with this many or more neighbours within ``1.1 * target_distance`` will
            not sprout new candidates. Default 14.
        method : str, optional
            Neighbour-search backend. ``'distance'`` uses a fully-vectorised
            pairwise ``torch.cdist`` and can work well with few points and a big GPU. 
            ``'kdtree'`` uses ``scipy.spatial.cKDTree`` on CPU and is more
            memory-efficient for large point sets. Defaults is 'distance'.
        seed : int, optional
            Random seed for reproducibility. Default 1.

        Returns
        -------
        np.ndarray, shape ``(M, ndim)``
            Projected sample positions on the isosurface (and inside ``volumeName`` if set).
        """
        
        return_numpy = not isinstance(seed_point, torch.Tensor)
        
        method = method.lower()
        assert method in ('distance', 'kdtree'), "method must be 'distance' or 'kdtree'."
        assert isosurface in self.isosurfaces, f"Isosurface '{isosurface}' not found."
        
        if method == 'kdtree':
            try:
                from scipy.spatial import cKDTree
            except ImportError:
                raise ImportError("scipy is required for method='kdtree'. Install with `pip install scipy`.")

        rng = np.random.default_rng(seed)
        td = float(target_distance)

        def _project(pts, return_normals=False):
            # accepts torch tensor or numpy; always returns torch tensor(s)
            if not isinstance(pts, torch.Tensor):
                pts = _tensor(pts)
            return self.projectTo(isosurface, pts, nsteps=6, return_normals=return_normals)

        def _in_vol(pts):
            if volumeName is None:
                return torch.ones(len(pts), dtype=torch.bool, device=curlew.device)
            m = self.getVolume(volumeName, x=pts, to_numpy=False, transform=True)
            return m.reshape(-1)

        def _pairs(p, radius):
            """Return (i_idx, j_idx) tensors for all pairs within radius."""
            if method == 'distance':
                dists = torch.cdist(p, p)
                mask = (dists < radius) & (
                    torch.arange(len(p), device=curlew.device).unsqueeze(1)
                    < torch.arange(len(p), device=curlew.device).unsqueeze(0)
                )
                return mask.nonzero(as_tuple=True)
            else:
                tree = cKDTree(_numpy(p))
                pairs = list(tree.query_pairs(radius))
                if not pairs:
                    empty = torch.zeros(0, dtype=torch.long, device=p.device)
                    return empty, empty
                pairs_t = torch.as_tensor(pairs, dtype=torch.long, device=curlew.device) # N.B. long as these are indices
                return pairs_t[:, 0], pairs_t[:, 1]
        
        def _neighbour_counts(p, radius):
            """Return (N,) tensor of neighbour counts (excluding self)."""
            if method == 'distance':
                return (torch.cdist(p, p) < radius).sum(dim=1) - 1
            else:
                p = _numpy(p)
                tree = cKDTree(p)
                results = tree.query_ball_tree(tree, td * 1.1)
                counts = torch.tensor([len(r) - 1 for r in results], dtype=torch.long, device=curlew.device)
                return _tensor(counts)
        
        def _nearest_dist(cands, p):
            """Return (E,) tensor of each candidate's distance to its nearest point in p."""
            if method == 'distance':
                return torch.cdist(cands, p).min(dim=1).values
            else:
                p = _numpy(p)
                tree = cKDTree(p)
                dists, _ = tree.query(_numpy(cands), k=1)
                return _tensor(dists)
        
        # Seed
        seed_arr = _tensor(seed_point)
        if seed_arr.ndim == 1:
            seed_arr = seed_arr.reshape(1, -1)
        
        # project and filter seed [points]
        p = _project(seed_arr)
        p = p[_in_vol(seed_arr)]
        assert len(p) > 0, (
            f"No valid seed points inside volume '{volumeName}'."
        )

        # Main growth loop
        no_growth_steps = 0
        no_growth_patience = 5  # stop if no new points added for this many consecutive steps
        for i in range(int(n_steps)):

            # --- Repulsion / relaxation ------------------------------------
            for _ in range(int(relax_substeps)):
                p = _project(p)  # project points
                if len(p) < 2: break
                mask = ~_in_vol(p) # we don't move points that have drifed outside the volume (they will be removed later anyway)
                
                # get pairs and apply repulsion force
                if mask.any():
                    i_idx, j_idx = _pairs(p, td * 1.2)
                    if len(i_idx) > 0:
                        diff = p[i_idx] - p[j_idx]
                        dist = diff.norm(dim=1, keepdim=True).clamp(min=1e-9)
                        push = (td - dist) * (diff / dist) * 0.5
                        push = push * mask.unsqueeze(1) # don't push points that have drifed outside the volume
                        forces = torch.zeros_like(p)
                        forces.index_add_(0, i_idx, push)
                        forces.index_add_(0, j_idx, -push)
                        p = p + forces

            if len(p) >= max_samples: break # too many points
            if len(p) == 0: break # weird, but possible with restrictive volume?
            
            # --- Growth front ----------------------------------------------
            p, n_vecs = _project(p, return_normals=True) # project points back onto isosurface (after movements above)
            
            # seed new particles
            N, ndim = p.shape
            counts = _neighbour_counts(p, td * 1.1)
            eligible = (counts < neighbor_cap)
            eligible &= torch.from_numpy(rng.random(N) < growth_rate).to(curlew.device)
            if eligible.any():
                p_elig = p[eligible]
                n_elig = n_vecs[eligible]

                v = torch.randn(len(p_elig), ndim, device=curlew.device)
                v = v - n_elig * (v * n_elig).sum(dim=1, keepdim=True)
                v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)
                cands = p_elig + v * td

                far_enough = _nearest_dist(cands, p) >= td * 0.8
                cands = cands[far_enough]

                # also reject candidates too close to each other
                if len(cands) >= 2:
                    cand_cand_dists = torch.cdist(cands, cands)
                    cand_cand_dists.fill_diagonal_(float("inf"))
                    no_cand_clash = cand_cand_dists.min(dim=1).values >= td * 0.8
                    cands = cands[no_cand_clash]

                # candidates must also be within our volume
                no_growth_steps += 1 # (N.b. will be reset to 0 if we add new samples)
                if len(cands) > 0:
                    cands = _project( cands[:max_samples - len(p)] )
                    cands = cands[_in_vol(cands)]
                    if len(cands) > 0:
                        no_growth_steps = 0
                        p = torch.vstack([p, cands]) # add new samples
                
                if no_growth_steps >= no_growth_patience: # early stopping
                    print(f"Finishing after {i} steps.")
                    break

        # Final projection + volume filter
        if len(p) == 0: # only space for one sample... (hopefully this doesn't happen often!)
            ndim = seed_point.reshape(-1).shape[0]
            return torch.zeros((0, ndim))
        p, n_vecs = _project(p, return_normals=True)
        ok = _in_vol(p)
        p = p[ok]
        n_vecs = n_vecs[ok]
        
        # return requested output and format
        if return_normals:
            if return_numpy:
                return _numpy(p), _numpy(n_vecs)
            return p, n_vecs
        if return_numpy:
            return _numpy(p)
        return p

    def loss(self):
        """
        Compute loss associated with the underlying field and learnable property, deformation or overprint objects.
        """

        # constant fields have no loss
        if (isinstance(self.field, int) or isinstance(self.field, float)): return _tensor(0.0, dev=curlew.device), {}

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
