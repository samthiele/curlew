"""
A class defining a structural scalar (implicit) field. This is where a lot of the magic happens, including
the chaining of multiple neural fields and the application of deformation functions to implement kinematic
events such as faults and sheet intrusions.
"""

import curlew
from curlew.fields import BaseNF, NFF
from curlew.geology.interactions import Overprint, OffsetBase
from curlew.geometry import Grid
from curlew.utils import batchEval

import numpy as np
import torch
import functools
import copy
from dataclasses import dataclass, field

def apply_child_undeform(x, sf):
    """
    Placeholder function to avoid Lambda functions (Making models picklable).
    """
    if sf.child is not None:
        return sf.child.undeform(x)
    else:
        return x

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
                    if isinstance( attr, np.ndarray ) or isinstance( attr, list ): # convert nd array or list types to tensor
                        attr = torch.tensor( attr, device=curlew.device, dtype=curlew.dtype )
                    elif isinstance(attr, dict):
                        for key in attr.keys(): # also convert any dict entries
                            if isinstance( attr[key], np.ndarray ) or isinstance( attr[key], list ):
                                attr[key] = torch.tensor( attr[key], device=curlew.device, dtype=curlew.dtype )
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
                        attr = attr.cpu().detach().numpy()
                    elif isinstance(attr, dict): # also convert any dict entries
                        for key in attr.keys():
                            if isinstance( attr[key], torch.Tensor ):
                                attr[key] = attr[key].cpu().detach().numpy()
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
        
         # pass transform function for determining paleocoordinates to neural field
         # (functools.partial is needed to allow pickling)
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

        # field is constant -- values are already known! :-)
        if isinstance(self.field, float) or isinstance(self.field, int): 
            return torch.full( (len(x),1), float(self.field), device=curlew.device, dtype=curlew.dtype)

        # set transform that removes any child deformation
        if undef and (self.child is not None):
            self.field.transform = functools.partial(apply_child_undeform, sf=self) # apply this GeologicalField's undeform function prior to  field.forward.
            #x = self.child.undeform( x )
        else:
            self.field.transform = None # don't do any transform in field.forward

        # evaluate the underlying field
        return self.field.forward(x)

    def predict(self, x: np.ndarray, combine=False, to_numpy=True, transform=True, values=None, 
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

    def gradient(self, x: np.ndarray, return_vals=False, normalize=True, to_numpy=True, transform=True, retain_graph=False, create_graph=False):
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

    def undeform(self, x: torch.Tensor) -> torch.Tensor:
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
        if (self.child is None) and (self._displace is None):
            return x # easy

        # evaluate displacement
        tonp = False
        if isinstance( x, np.ndarray ):
            x = torch.tensor( x, device=curlew.device, dtype=curlew.dtype)
            tonp = True

        # remove any child deformation
        if self.child is not None:
            x = self.child.undeform( x ) # undeform to the time-step relevant for this GeologicalField

        # handle our own displacement
        out = x
        if self._displace is not None:
            out =  self._displace(x, inverse=False) # apply this GeologicalField's displacement (if defined).

        # cast to numpy if needed
        if tonp:
            return out.cpu().detach().numpy()
        else:
            return out

    def _displace(self, x: torch.Tensor, inverse=False) -> torch.Tensor:
        """
        Remove deformation (displacements) caused by this specific event.

        Displaces from post-event coordinates to pre-event coordinates. If this GeologicalField has no 
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
            # this GeologicalField causes no deformation
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
        Return the deformation vectors associated with this GeologicalField at the specified locations. These are the
        displacements that would be remoed during `_displace(...)`.

        Parameters
        ----------
        x : np.ndarray | torch.Tensor
            An array of shape (N, input_dim) containing the modern-day coordinates at which to evaluate
            this (and previous) GeologicalField.
        inverse : bool
            If True, apply the inverse deformation (i.e. from paleo-coordinates to modern-day coordinates).
            Default is False, which applies the deformation from modern-day coordinates to paleo-coordinates.
            Note that when inverse=True this only returns an approximation of the true inverse deformation, as
            the field is evaluated at `x` rather than at the true paleo-coordinates.

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

        if self.deformation is None:
            offset = torch.zeros_like(x) # no deformation
        else:
            offset = self.deformation.eval(x, self)
        
        #offset = self.deformation(x, self, **self.deformation_args) # apply deformation function
        if inverse: # flip deformation vectors (B to A instead of A to B)
                offset = -offset

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

    # N.B. THE FOLLOWING FUNCTION IS NOT STRICTLY CORRECT AND ONLY WORKS FOR COMLETELY LINEAR FIELDS
    # def deform(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Apply deformation (displacements) to the passed set of coordinates.

    #     Translates from paleo-coordinates (i.e. coordinates at the time of this event) to present-day coordinates, 
    #     for example, by applying child fault offsets.

    #     Parameters
    #     ----------
    #     x : torch.Tensor, np.ndarray
    #         A tensor of shape (N, input_dim).

    #     Returns
    #     -------
    #     torch.Tensor
    #         A tensor of shape (N, input_dim), containing the deformed coordinates.
    #     """
    #     tonp = False
    #     if isinstance( x, np.ndarray ):
    #         x = torch.tensor( x, device=curlew.device, dtype=curlew.dtype)
    #         tonp = True

    #     # apply any child deformation(s)
    #     if self.child is not None:
    #         x = self.child.deform( x )

    #     # handle our own displacement
    #     out = x
    #     if self._displace is not None:
    #         out = self._displace(x, inverse=True) # apply this GeologicalField's displacement (if defined).

    #     # cast to numpy if needed
    #     if tonp:
    #         return out.cpu().detach().numpy()
    #     else:
    #         return out