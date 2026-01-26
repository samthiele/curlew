"""
A class representing a time-aware geological model and
facilitating interactions with the underlying linked-list of GeoField instances
(that represent each geological structure in the model).
"""
import curlew
from curlew.geology.geofield import GeoField
from curlew.fields import BaseSF
from curlew.utils import batchEval
from curlew.geology.interactions import FaultOffset
from curlew.core import Geode
from curlew.geometry import Grid, Transform

import numpy as np
import torch
from tqdm import tqdm
import copy
import io
from pathlib import Path


def _linkF( fields ):
    """
    Link the list of fields (to the final element in the list).
    """
    for i in range(len(fields)):
        if fields[i].parent2 is None:
            fields[i].parent = None  # start from a clean slate
            fields[i].child = None
        if not isinstance( fields[i], GeoField ):
            continue # skip ints, floats etc.
        if (i > 0) and (fields[i].parent is None):
            fields[i].parent = fields[i-1]
        if (i < (len(fields)-1)) and (fields[i].child is None):
            fields[i].child = fields[i+1]

class GeoModel(object):
    """
    A class representing a time-aware geological model and
    facilitating interactions with the underlying linked-list of GeoField instances
    (that represent each geological structure in the model).
    """

    def __init__( self, fields : list, transform=None, grid=None, name=None):
        """
        Construct a GeoModel from a list of GeoFields.

        Parameters
        ----------
        fields : list
            A list of GeoField instances representing geological events, from oldest to youngest. This list 
            can include domain boundaries if needed, but non-domain fields (e.g., faults, stratigraphy, etc.)
            should not be older than these.
        transform : `curlew.core.Transform`
            A Transform object defining the transform from global coordinates to model coordinates. This will be applied
            to all `x` when `self.predict(x)` is called, and can handle e.g., converting UTM to some model coordinate system.
            Defaults to an identity matrix (no transform).
        grid : curlew.geometry.Grid | optional
            An optional grid to associate with this GeoModel instance. This will set the `M.grid` variable but is not
            necessary (i.e. can be `null`; which is the default).
        name : str | optional
            A string name to associate with this GeoModel. Not really used, but can be useful :-)
        """
        # set parent and child properties of underlying GeoFields
        # (i.e. build our linked list / binary tree of GeoFields)
        _linkF(fields)

        # traverse back down linked list / tree and define IDs
        def traverse( node, i = 1):
            node.eid = i
            if isinstance( node.parent, GeoField ):
                i = traverse( node.parent, i+1)
            if isinstance( node.parent2, GeoField):
                i = traverse( node.parent2, i+1)
            return i
        traverse( fields[-1] )

        # accumulate all fields in this model
        self.fields = []
        def traverse_fields( node ):
            if isinstance(node, GeoField):
                self.fields.append(node)
            if isinstance(node.parent, GeoField):
                traverse_fields( node.parent )
            if isinstance(node.parent2, GeoField):
                traverse_fields( node.parent2 )
        traverse_fields( fields[-1] ) # traverse from last field in the list
        self.fields = self.fields[::-1] # we want the youngest field last, so reverse the list
        self.lastEvent = self.fields[-1] # change to evaluate model in some paleo-space
        self.eidLookup = { f.eid : f for f in self.fields } # create a lookup table for translating event IDs to GeoField instances

        self.input_dim = self.fields[0].input_dim # get dimensionality of model from one of the fields in the model
        for f in self.fields: # check dimensionalities all match
            assert f.input_dim == self.input_dim, f"Field {f.name} has a dimensionality of {f.input_dim} not {self.input_dim}."
        if transform is None:
            self.T = Transform(self.input_dim) # thunk -- leave this as None and skip compute? (slightly faster)
        else:
            self.T = transform

        # store "nice-to-have" extras
        self.grid = grid
        self.name = name

    def freeze( self, name=None, geometry=True, params=False ):
        """
        Freeze the specified field or parameter. Used to e.g., optimise
        fault offset while keeping fault geometry fixed.

        Parameters
        ------------
        name, str | GeoField | list:
            The name of the GeoField to freeze. Can also be a list of names or instances. If None, 
            the specified freeze will be applied to all GeoFields in this model. Use `'forward'` to 
            address any defined forward model.
        geometry : bool
            True if the geometry of the specified GeoField should be frozen. Default is True. 
        params : bool
            True if other parameters (e.g., fault slip) associated with the specified GeoFields should be frozen. Default is False.
        """
        if name is None:
            name = [f for f in self.fields] # apply to all
        if not isinstance(name, list) or isinstance(name, tuple):
            name = [name]
        for f in name:
            if isinstance(f, str) or isinstance(f, int):
                f = self[f] # get field by name or ID
            f.field.frozen = geometry # freeze geometry?
            if f.deformation is not None: # freeze potentially learnable properties?
                f.deformation.frozen = params
            if f.propertyField is not None: # freeze potentially learnable properties?
                f.propertyField.frozen = params
            if f.overprint is not None: # freeze potentially learnable properties?
                f.overprint.frozen = params

    def prefit(self, epochs, **kwargs):
        """
        Train all GeoFields in this model to fit their respective constraints
        in isolation, starting with the youngest field.

        Parameters
        ----------
        epochs : int
            The number of epochs to train for.
        
        Keywords
        ----------
        All keywords are passed to `curlew.fields.NF.fit(...)`. These include:
        learning_rate : float, optional
            Reset each GeoField's optimiser to the specified learning rate before training.
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
        out = {}
        for F in self.fields[::-1]:
            _, loss = F.fit( epochs, prefix=F.name, **kwargs )
            out.update(loss) # add outputs
        return out

    def zero(self):
        """
        Zero all (unfrozen) optimisers associated with the neural fields and 
        other learned parameters (e.g., fault offsets) in this model. 
        """
        for f in self.fields:
            f.zero()

    def step(self):
        """
        Step all (unfrozen) optimisers associated with the fields making
        up this model, and (potentially) other leanrned parameters (e.g., fault offsets). 
        """
        for f in self.fields:
            f.step()

    def fit(self, epochs, learning_rate=None, early_stop=(100, 1e-4), best=True, vb=True, prefix='Training'):
        """
        Train all GeoFields in this model to fit the specified constraints
        simultaneously.

        Parameters
        ----------
        epochs : int
            The number of epochs to train each GeoField for.
        learning_rate : float, optional
            Reset each GeoField's optimiser to the specified learning rate before training.
        early_stop : tuple,
            Tuple containing early stopping criterion. This should be (n,t) such that optimisation
            stops after n iterations with <= t improvement in the loss. Set to None to disable. Note 
            that early stopping is only applied if `best = True`. 
        best : bool, optional
            After training set the neural field weights to the best loss.
        vb : bool, optional
            Display a tqdm progress bar to monitor training.
        prefix : str, optional
            The prefix used for the tqdm progress bar.

        Returns
        -------
        loss : float
            The loss of the final (best if best=True) model state.
        details : dict
            A more detailed breakdown of the final loss. 
        """

        # set learning rate if specified
        if learning_rate is not None:
            for F in self.fields:
                F.set_rate( learning_rate )
        
        # setup progress bar
        bar = range(epochs)
        if vb:
            bar = tqdm(range(epochs), desc=prefix, bar_format="{desc}: {n_fmt}/{total_fmt}|{postfix}")

        # iterate
        out = {}
        #best_state = []
        best_loss = np.inf
        best_count = 0
        eps = 0
        if early_stop is not None:
            eps = early_stop[1]
        for epoch in bar:
            loss = 0
            for F in self.fields[::-1]:
                ll, details = F.loss() # compute loss for this field
                loss = loss + ll # accumulate loss
                out.update(details) # store for output

            # also add forward (property) reconstruction loss
            # if self.forward is not None:
            #     pp = self.forward.C.pp # position of property constraints
            #     pv = self.forward.C.pv # value of property constraints
            #     spred = self.fields[-1].predict(pp, combine=True, to_numpy=False) # automatically recursed back throught the linked list.
            #     # One Hot encoding
            #     if self.forward.H.one_hot:
            #         one_hot_encoder = torch.nn.functional.one_hot((spred[:, 1] - 1).long(), num_classes=len(self.fields))
            #         encoded_spred = one_hot_encoder * spred[:, 0][:, None]
            #         ppred = self.forward( encoded_spred )
            #     else:
            #         ppred = self.forward( spred ) # generate property predictions
            #     prop_loss = self.forward.loss_func( ppred, pv ) # compute loss
            #     if isinstance( self.forward.H.prop_loss, str):
            #         self.forward.H.prop_loss = float(self.forward.H.prop_loss) / prop_loss.item()
            #     loss = loss + self.forward.H.prop_loss * prop_loss
            #     out['forward'] = (prop_loss.item(),{})

            # store best state(s)
            if (loss.item() < (best_loss+eps)):
                #best_state = [ copy.deepcopy( F.field.state_dict()  ) for F in self.fields ]
                best_loss = loss.item()
                best_count = 0
            else:
                best_count += 1

            # early stopping
            if (early_stop is not None) and (best_count > early_stop[0]):
                break

            # update progress
            if vb:
                bar.set_postfix({ k : v[0] for k,v in out.items() })

            self.zero() # zero gradients
            loss.backward() # backprop losses
            self.step()

        # set best state
        #if best_state:
        #    for i,F in enumerate(self.fields):
        #        F.field.load_state_dict(best_state[i])

        # return
        return loss.item(), out

    def predict(self, x : np.ndarray, coords="global", **kwargs):
        """
        Create model predictions at the specified points.

        Parameters
        ----------
        x : np.ndarray | torch.tensor | curlew.geometry.Grid
            An array of shape (N, input_dim) containing the coordinates at which to evaluate
            this GeoModel.
        coords : str
            Specify which coordinate system `x` is in. If `coords == "global"` (default), then any defined
            model transform will be applied (to derive model coordinates). If `coords=="model"` then this
            transform will not be applied.
        
        Keywords
        --------
        All keywords are passed directly to `GeoField.predict()`.

        Returns
        --------
        S : An array of shape (N,1) containig the predicted scalar values and corresponding GeoField
            that "created" them.
        """

        # update isosurface lookup (incase the defined isosurfaces have been changed)
        # build lithology lookup (to ensure lithologies from different fields get unique IDs)
        self.llookup = {}
        self.eidLookup = { f.eid : f for f in self.fields } # create a lookup table for translating event IDs to GeoField instances
        n=1 # start at 1, as -1 is 'undefined' and 0 is default for fields with no lithology defined.
        for F in self.fields:
            self.llookup[F.name] = n # potential lithology created by this field (e.g., constant fields)
            n = n + 1
            if F.overprint is not None:  # only relevant for generative (overprinting) events [ as these "create" new rocks ]
                for k in F.isosurfaces.keys():
                    k = f"{F.name}_{k}" # build key using field name and lithology name
                    assert k not in self.llookup, f"All isosurfaces in model must have unique names, but {k} is not unique!"
                    self.llookup[k] = n # assign ID for this lithology
                    n = n + 1 # increment ID
            F.llookup = self.llookup # link lookup to field so it is used during predict(...).
        
        grid = None
        if isinstance(x, Grid):
            grid = x
            x = grid.coords()

        # apply transform to x
        if "global" in coords.lower():
            x = self.T(x) # transform from world to model coordinates

        # generate predictions
        kwargs['to_numpy'] = kwargs.get('to_numpy', True)
        kwargs['combine'] = True # this is necessary....
        out = self.fields[-1].predict(x, **kwargs) # automatically recursed back throught the linked list.
        
        out.grid = grid
        if "global" in coords.lower():
            out.x = x # replace with global coords
            out.crs = "global"
         
        # return
        return out

    def drill( self, start, end, step ):
        """
        Evaluate the model along a line between start and end with an interval of step.

        Parameters
        -----------
        start : np.ndarray
            The start coordinate of the "drillhole"
        end : np.ndarray
            The end coordinate of the "drillhole"
        step : float
            The distance between points along this line

        Returns
        ---------
        drillholes : Geode
            A Geode instance containing the results given by evaluating the model along the drillhole.
        contacts : Geode
            A Geode instance containing the positions and orientations of contacts intersected along the drillhole.
        """
        dir = np.array(end) - np.array(start)
        length = np.linalg.norm(dir)
        dir = (dir / length)*step
        pos = np.array([start+dir*i for i in range( int(length / step) ) ])

        # evaluate model along drillholes
        g = self.predict( pos )

        # find contacts
        c = None
        g.contactMask = np.abs( np.diff( g.lithoID, prepend=g.lithoID[0] ) ) > 0
        if g.contactMask.any():
            cpos = pos[g.contactMask]
            c = self.predict( cpos, gradient=True ) # predict again, at the contact points only

        # return Geode
        return g, c

    # def evaluate( self, grid, topology=False, buffer=None, surfaces=None, vb=True):
    #     """
    #     Evaluate a *curlew* model on a grid and extract isosurfaces, topology and/or fault buffers.

    #     Parameters
    #     ----------
    #     grid : curlew.geometry.Grid | np.ndarray
    #         A structured Grid to evaluate the model on (if surfaces are to be calculated), or an array
    #         of coordinates (unstructured grid). Isosurfaces cannot be calculate for unstructured grids.
    #     topology : bool, optional
    #         True if model topology (fault hangingwall and footwall relations) should be calculated and returned. Default is False. 
    #     buffer : float, optional
    #         If not None, this distance (in model coordinates) will be used to compute a buffer of this size on either side of each fault surface.
    #     surfaces : str | bool, optional
    #         If not None, isosurfaces will be computed and returned. If a string is passed, these will also be saved to PLY in the specified folder.

    #     Returns
    #     -------
    #     A dict containing some of the following keys: 'topology', 'buffer', 'surfaces'.
    #     """

    #     # TODO - extend this to include e.g., lithological classifications, stratigraphic contacts, etc.
    #     from curlew.geometry import Grid
    #     if isinstance(grid, Grid):
    #         gxy = grid.coords()
    #     else:
    #         surfaces = None # disable surfaces
    #         gxy = grid

    #     # setup output array
    #     out = dict()
    #     if buffer:
    #         out['buffer'] = np.zeros( len(gxy) ) # initialise fault buffer
    #     if topology:
    #         out['topology'] = np.zeros( (len(gxy), len(self.fields)) ) # array to store hanging-wall & footwall information
    #     if surfaces:
    #         out['surfaces'] = {}

    #     # recurse through model extracting required info
    #     def recurse( f, dmask, i=0 ):
    #         # evaluate model
    #         if (f.parent2 is not None) or (f.deformation is not None): # ignore stratigraphic fields
    #             if vb:
    #                 print(f"Evaluating field {i}/{len(self.fields)}", end='\r')
    #             pred = batchEval( gxy, f.predict, vb=False)[:,0]
    #             pred[dmask] = np.nan # remove masked areas

    #         # evaluate topology, buffer & recurse
    #         if f.parent2 is not None: # this is a domain boundary
    #             iso = f.getIsovalue( f.bound )

    #             if buffer:
    #                 i0 = f.getIsovalue( f.bound, offset=-buffer ) # lower buffer
    #                 i1 = f.getIsovalue( f.bound, offset=buffer ) # upper buffer
    #                 out['buffer'][ (pred >= min(i0, i1)) & (pred <= max(i0, i1)) & (out['buffer'] == 0) ] = f.eid

    #             footwall = pred < iso
    #             hangingwall = pred >= iso
    #             if topology:
    #                 out['topology'][ footwall, i ] = -1
    #                 out['topology'][ hangingwall, i ] = 1

    #             if isinstance(f.parent, GeoField):
    #                 recurse( f.parent, dmask=(dmask + footwall ), i=i+1 ) # recurse hangingwall objects
    #             if isinstance(f.parent2, GeoField):
    #                 recurse( f.parent2, dmask=(dmask + hangingwall), i=i+1 ) # recurse footwall objects

    #         elif (f.deformation is not None) and isinstance(f, FaultOffset): # this is a fault surface
    #             if topology:
    #                 iso = f.getIsovalue( f.deformation_args['contact'], offset=0 ) # fault surface
    #                 footwall = pred < iso
    #                 hangingwall = pred > iso

    #                 out['topology'][ footwall, i ] = -1
    #                 out['topology'][ hangingwall, i ] = 1

    #             if buffer:
    #                 i0 = f.getIsovalue( f.deformation_args['contact'], offset=-buffer ) # lower buffer
    #                 i1 = f.getIsovalue( f.deformation_args['contact'], offset=buffer ) # upper buffer
    #                 out['buffer'][ (pred >= min(i0, i1)) & (pred <= max(i0, i1)) & (out['buffer'] == 0) ] = f.eid

    #             if isinstance(f.parent, GeoField):
    #                 recurse( f.parent, dmask=dmask, i=i+1 ) # recurse older objects
    #         else:
    #             iso = None
            
    #         if surfaces: # compute isosurface meshes
    #             out['surfaces'][f.name] = {}
    #             for k in f.isosurfaces.keys():
    #                 if grid.ndim == 3: # 3D
    #                     verts, faces = grid.contour( pred, iso=f.getIsovalue(k))
    #                     out['surfaces'][f.name][k] = (verts, np.array(faces))
    #                     if isinstance(surfaces, str) or isinstance(surfaces, Path):
    #                         from curlew.io import savePLY
    #                         savePLY( Path( surfaces ) / str(f.name) / f'{str(k)}.ply',
    #                                 xyz = verts, faces = faces )
    #                 elif grid.ndim == 2: # 2D
    #                     contours = grid.contour( pred, iso=f.getIsovalue(k))
    #                     out['surfaces'][f.name][k] = contours

    #     # traverse from last event in model
    #     recurse( self.fields[-1], np.full( len(gxy), False ) )

    #     return out

    def __getitem__(self, index ):
        """Get fields by name (str) or SID (int)"""
        return self.getField( index )

    def getField( self, eid ):
        """
        Get the scalar field associated with the specified event ID (int) or name (str).

        Parameters
        ----------
        eid : int | str
            The event ID or field name to retrieve.

        Returns
        -------
        GeoField
            The scalar field instance associated with the specified event ID.
        """
        if isinstance(eid, str):
            for f in self.fields:
                if f.name == eid: return f
            assert False, f"A field with name {eid} does not exist in this model."
        else:
            return self.eidLookup.get( int(eid), None)

    def _getPositions(self, G, node, first_x=0, first_y=0, step_x=10, step_y=10, pos=None):
        """
        Recursively calculate the 2D positions of nodes in a hierarchical structure. Used when plotting
        the model tree as a 2D graph.

        Parameters
        ----------
        G : networkx.Digraph
             The directed graph containing the nodes to be positioned.
        node : str
            The current node for which to calculate the position. 
        first_x : int, optional
            The initial x-coordinate for the current node.
        first_y : int, optional
            The initial y-coordinate for the current node.
        step_x : int, optional
            The horizontal step size for moving to the right.
        step_y : int, optional
            The vertical step size for moving down.
        pos : dict, optional
            A dictionary to store the positions of nodes.

        Returns
        -------
        pos : dict
            A dictionary mapping each node to its (x, y) position.
        """
        if pos is None:
            pos = {}

        # Assign the position to the current node
        pos[node] = (first_x, first_y)

        # Get the children of the current node
        children = list(G.successors(node))

        if not children:
            return pos

        # If the node is a domain boundary, handle its children differently
        node_field = next((field for field in self.fields if field.name == node), None)
        if node_field.parent2 is not None and isinstance(node_field, GeoField):
            # Move to the right
            pos = self._getPositions(G, children[0], first_x + step_x, first_y, step_x, step_y, pos)
            # Move down
            pos = self._getPositions(G, children[1], first_x, first_y - step_y, step_x, step_y, pos)
        else:
            # For non-domain boundary nodes, move to the right
            pos = self._getPositions(G, children[0], first_x + step_x, first_y, step_x, step_y, pos)

        return pos

    def _repr_svg_(self):
        """
        Visualize the model tree of a GeoModel and return it as an SVG string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            An SVG string representation of the visualized model tree.
        """
        # Create an empty graph
        try:
            import networkx as nx
        except:
            assert False, "Please install networkx using `pip install networkx`"
        graph = nx.DiGraph()

        domain_boundary_color = '#E35B0E'
        dilative_event_color = "#F0C419"
        generative_event_color = '#31B4C2'
        kinematic_event_color = "#A6340B"
        fixed_value_color = "#FAE8B6"

        for field in self.fields[::-1]:
            # Determine the color based on the event type
            color = None
            if field.parent2 is not None: # domain boundary
                color = domain_boundary_color
            elif field.overprint is not None and field.deformation is not None: # dilative event
                color = dilative_event_color
            elif field.overprint is not None: # generative event
                color = generative_event_color
            elif field.deformation is not None: # kinematic event
                color = kinematic_event_color
            graph.add_node(field.name, label=field.name, color=color)

            # Add edges
            if isinstance(field.parent, GeoField):
                graph.add_edge(field.name, field.parent.name)
            if isinstance(field.parent2, GeoField):
                graph.add_edge(field.name, field.parent2.name)
            if not isinstance(field.parent, GeoField) and field.parent is not None: # Handle fixed values
                graph.add_edge(field.name, str(field.parent))
            if not isinstance(field.parent2, GeoField) and field.parent2 is not None:
                graph.add_edge(field.name, str(field.parent2))

        # Plot
        try:
            import matplotlib.pyplot as plt
        except:
            assert False, "Please install matplotlib to use plotting tools.`"
        
        fig, ax = plt.subplots(1,1, figsize=(8, 4))
        pos = self._getPositions(graph, list(graph.nodes())[0], step_x=1, step_y=1)
        node_colors = [graph.nodes[node].get('color', fixed_value_color) for node in graph.nodes()]
        nx.draw(graph, pos, with_labels=True, arrows=True, node_size=2000,
                node_color=node_colors, font_size=8, ax=ax)
        
        # Legend
        legend_labels = {
            domain_boundary_color : 'Domain',
            dilative_event_color : 'Dilative',
            generative_event_color : 'Generative',
            kinematic_event_color : 'Kinematic',
            fixed_value_color : 'Fixed'
        }
        #ax_legend.axis('off')
        for color, label in legend_labels.items():
            ax.scatter([], [], color=color, label=label, s=200)
        ax.legend(loc='lower center', ncol=5)

        # Save the figure
        buffer = io.StringIO()
        fig.savefig(buffer, format='svg')
        plt.close(fig)
        svg = buffer.getvalue()
        buffer.close()

        return svg