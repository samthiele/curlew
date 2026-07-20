"""
A class representing a time-aware geological model and
facilitating interactions with the underlying linked-list of GeoEvent instances
(that represent each geological structure in the model).
"""
import curlew
from curlew.geology.geoevent import GeoEvent
from curlew.core import Pebble, LearnableBase
from curlew.geometry import Grid, Transform

import numpy as np
import torch
from tqdm import tqdm
import io


def _linkE( events ):
    """
    Link the list of events (to the final element in the list).
    """
    for i in range(len(events)):
        if events[i].parent2 is None:
            events[i].parent = None  # start from a clean slate
            events[i].child = None
        if not isinstance( events[i], GeoEvent ):
            continue # skip ints, floats etc.
        if (i > 0) and (events[i].parent is None):
            events[i].parent = events[i-1]
        if (i < (len(events)-1)) and (events[i].child is None):
            events[i].child = events[i+1]


def _attach_event_model(node, model):
    """Link a GeoEvent and its scalar fields to the parent GeoModel. Gives individual models access to the whole model structure (and e.g., global - model coordinate transform) if needed."""
    node.model = model
    for f in node._field_list():
        if not isinstance(f, (int, float)):
            f.model = model
    if node.propertyField is not None:
        node.propertyField.model = model


def _register_events(node, out, model, i=1):
    """Assign event IDs, link to the GeoModel, and collect events from the youngest node upward."""
    node.eid = i
    _attach_event_model(node, model)
    out.append(node)
    if isinstance(node.parent, GeoEvent):
        i = _register_events(node.parent, out, model, i + 1)
    if isinstance(node.parent2, GeoEvent):
        i = _register_events(node.parent2, out, model, i + 1)
    return i


def _rebind_global_constraints(obj):
    """
    Re-bind constraints that were attached before a GeoModel existed (``crs='global'``).

    ``LearnableBase.bind`` maps global coordinates to model space when ``model.T`` is
    available; this is called from ``GeoModel.__init__`` after the tree is linked.
    """
    if obj.C is None or getattr(obj.C, "crs", None) != "global":
        return
    obj.bind(obj.C.numpy())


class GeoModel( LearnableBase ):
    """
    A class representing a time-aware geological model and
    facilitating interactions with the underlying linked-list of GeoEvent instances
    (that represent each geological structure in the model).
    """

    def __init__( self, events : list, transform=None, grid=None, name=None):
        """
        Construct a GeoModel from a list of GeoEvents.

        Parameters
        ----------
        events : list
            A list of GeoEvent instances representing geological events, from oldest to youngest. This list 
            can include domain boundaries if needed, but non-domain events (e.g., faults, stratigraphy, etc.)
            should not be older than these.
        transform : `curlew.geometry.Transform`
            A Transform object defining the transform from global coordinates to model coordinates. This will be applied
            to all `x` when `self.predict(x)` is called, and can handle e.g., converting UTM to some model coordinate system.
            Defaults to an identity matrix (no transform).
        grid : curlew.geometry.Grid | optional
            An optional grid to associate with this GeoModel instance. This will set the `M.grid` variable but is not
            necessary (i.e. can be `null`; which is the default).
        name : str | optional
            A string name to associate with this GeoModel. Not really used, but can be useful :-)
        """
        super().__init__() # setup learnable aspects
        
        # set parent and child properties of underlying GeoEvents
        # (i.e. build our linked list / binary tree of GeoEvents)
        _linkE(events)

        # assign IDs and collect events from youngest root (same tree walk as before)
        collected = []
        _register_events(events[-1], collected, self)
        self.events = collected[::-1]  # oldest first, youngest last
        self.lastEvent = self.events[-1]
        self.eidLookup = {f.eid: f for f in self.events}

        self.input_dim = self.events[0].input_dim # get dimensionality of model from one of the fields in the model
        for f in self.events: # check dimensionalities all match
            assert f.input_dim == self.input_dim, f"Field {f.name} has a dimensionality of {f.input_dim} not {self.input_dim}."
        if transform is None:
            self.T = Transform(self.input_dim) # thunk -- leave this as None and skip compute? (slightly faster)
        else:
            self.T = transform

        # constraints may have been bound on fields before this GeoModel was built
        for ev in self.events:
            for f in ev._field_list():
                if not isinstance(f, (int, float)):
                    _rebind_global_constraints(f)
            if ev.propertyField is not None:
                _rebind_global_constraints(ev.propertyField)
        _rebind_global_constraints(self)

        # store "nice-to-have" extras
        # (N.B. mostly used if e.g., geomodels are being pickled)
        self.grid = grid
        self.name = name

    def __str__(self):
        from curlew.text import geomodel_str
        return geomodel_str(self)

    def freeze( self, name=None, geometry=True, params=False ):
        """
        Freeze the specified field or parameter. Used to e.g., optimise
        fault offset while keeping fault geometry fixed.

        Parameters
        ----------
        name : str | GeoEvent | list, optional
            The name of the GeoEvent to freeze, or a list of names/instances. If None, apply to all
            events. Use ``'forward'`` to address any defined forward model.
        geometry : bool, optional
            If True (default), freeze field geometry for the selected event(s).
        params : bool, optional
            If True, also freeze learnable parameters (e.g. fault slip). Default is False.
        """
        if name is None:
            name = [f for f in self.events] # apply to all
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
        Train all GeoEvents in this model to fit their respective constraints
        in isolation, starting with the youngest field.

        Parameters
        ----------
        epochs : int
            The number of epochs to train for.
        
        Keywords
        ----------
        All keywords are passed to the fit function of each GeoEvent.

        Returns
        -------
        dict
            Maps each ``curlew.geology.geoevent.GeoEvent`` name to the final ``curlew.core.Pebble`` from isolated training.
        """
        out = {}
        for F in self.events[::-1]:
            _, pebble = F.fit(epochs, prefix=F.name, **kwargs)
            out[F.name] = pebble
        return out

    def fit(self, epochs, early_stop=(100, 1e-4), custom_loss=None, best=True, vb=True, prefix='Training', seed=42):
        """
        Train all GeoEvents in this model to fit the specified constraints
        simultaneously.

        Parameters
        ----------
        epochs : int
            The number of epochs to train each GeoEvent for.
        early_stop : tuple, optional
            Early stopping as ``(n, t)``: stop after ``n`` epochs with improvement ``<= t``. Set to
            None to disable. Only used when ``best=True``.
        custom_loss : list of callable, optional
            Functions ``f(pebble, model, C) -> Pebble`` called each epoch after event losses are
            accumulated. ``pebble`` holds per-event terms; ``model`` is this ``curlew.geology.geomodel.GeoModel``; ``C`` is
            the bound ``curlew.core.CSet`` (or None). Each function should return a ``curlew.core.Pebble`` to merge.
        best : bool, optional
            After training set the neural field weights to the best loss.
        vb : bool, optional
            Display a tqdm progress bar to monitor training.
        prefix : str, optional
            The prefix used for the tqdm progress bar.
        seed : int, optional
            Seed for the Bayes-by-Backprop weight-sampling generator (see Notes).

        Returns
        -------
        loss : float
            The loss of the final (best if best=True) model state.
        pebble : curlew.core.Pebble
            A detailed breakdown of the final loss.

        Notes
        -----
        ``self.events`` is a plain list, not an ``nn.ModuleList`` (each ``GeoEvent`` is not
        itself an ``nn.Module``), so nested :class:`~curlew.fields.series.FSF` potentials
        — and indeed every learnable parameter in the model — are not reachable via
        ``self.modules()``/``self.state_dict()`` (both come back empty on a bare
        ``GeoModel``). Two things that ``BaseSF.fit`` gets "for free" from being a single
        ``nn.Module`` are therefore done explicitly here, per ``field``/``deformation``/
        ``overprint``/``propertyField`` across every event:

        1. Every nested ``FSF`` gets a fresh held weight sample each epoch, so joint
           ``fit`` trains through the same reparameterised Bayes-by-Backprop draws as
           isolated per-event ``fit``/``prefit`` — without this, ``A_rho`` never sees a
           data-fit gradient and every forward pass silently uses the posterior mean
           instead of a weight sample.
        2. Each owner's ``state_dict()`` is snapshotted on improvement and reloaded at the
           end when ``best=True``, so joint ``fit`` actually returns the best-loss weights
           instead of whatever the last (or early-stopped) epoch happened to leave behind.
        """
        from curlew.fields.series import FSF

        owners = []
        for F in self.events:
            for o in (F.field, F.deformation, F.overprint, F.propertyField):
                if o is not None:
                    owners.append(o)
        fsf_modules = [m for o in owners for m in FSF.iter_in(o)]
        bbb_gen = torch.Generator(device=curlew.device).manual_seed(seed) if fsf_modules else None

        bar = range(epochs)
        if vb:
            bar = tqdm(range(epochs), desc=prefix, bar_format="{desc}: {n_fmt}/{total_fmt}|{postfix}")

        best_loss = np.inf
        best_pebble = None
        best_state = None
        best_count = 0
        eps = early_stop[1] if early_stop is not None else 0

        if custom_loss is None:
            custom_loss = []

        for epoch in bar:
            if fsf_modules:
                for m in fsf_modules:
                    m.resample_weights(generator=bbb_gen)
            pebble = Pebble() # initialise loss
            for F in self.events[::-1]:
                pebble = pebble + F.loss() # add loss incurred by each GeoEvent (and associated learnables like deformations, overprints, etc.)
            for loss_func in custom_loss:
                pebble = pebble + loss_func(pebble, self, self.C) # add custom loss functions if they have been defined
            total = pebble.total() # compute total loss
            if total.item() < (best_loss + eps):
                best_loss = total.item()
                best_pebble = pebble.detach()
                best_state = [
                    {k: v.detach().clone() for k, v in o.state_dict().items()} for o in owners
                ]
                best_count = 0
            else:
                if best_pebble is None:
                    best_loss = total.item()
                    best_pebble = pebble.detach()
                best_count += 1

            if (early_stop is not None) and (best_count > early_stop[0]):
                break

            if vb:
                bar.set_description(str(pebble))
                bar.set_postfix({
                    g: f"{pebble.group_total(g).item():.4g}"
                    for g in pebble.losses
                })

            exclude = []
            for F in self.events:
                if getattr(F.field, "frozen", False):
                    exclude.append(F.field.name)
                for o in [F.deformation, F.overprint, F.propertyField]:
                    if o is not None and getattr(o, "frozen", False):
                        exclude.append(getattr(o, "name", None) or f"{F.name}:{type(o).__name__}")

            pebble.zero()
            pebble.total().backward()
            pebble.step(exclude=exclude)

        if best and best_state is not None:
            for o, sd in zip(owners, best_state):
                o.load_state_dict(sd)

        if fsf_modules:
            for m in fsf_modules:
                m.clear_weight_sample()

        return best_loss, best_pebble

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
        All keywords are passed directly to `GeoEvent.predict()`.

        Returns
        -------
        curlew.core.Geode
            Combined model prediction at ``x`` (scalar, structure/lithology IDs, properties, etc.).
        """

        # update isosurface lookup (incase the defined isosurfaces have been changed)
        # build lithology lookup (to ensure lithologies from different events get unique IDs)
        self.llookup = {}
        self.eidLookup = { f.eid : f for f in self.events } # create a lookup table for translating event IDs to GeoEvent instances
        n=1 # start at 1, as -1 is 'undefined' and 0 is default for events with no lithology defined.
        for F in self.events:
            self.llookup[F.name] = n # potential lithology created by this event (e.g., constant fields)
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

        x_global = None
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=curlew.dtype, device=curlew.device)

        if "global" in coords.lower():
            x_global = x
            x = self.T(x) # transform from world to model coordinates

        # generate predictions
        kwargs['to_numpy'] = kwargs.get('to_numpy', True)
        kwargs['combine'] = True # this is necessary....
        out = self.events[-1].predict(x, **kwargs) # automatically recursed back throught the linked list.
        
        out.grid = grid
        if x_global is not None:
            out.set_coords("global", x_global)
        out.set_coords("model", x, primary=("global" not in coords.lower()))
        out.crs = "global" if "global" in coords.lower() else "model"
         
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
        drillholes : curlew.core.Geode
            A `curlew.core.Geode` instance containing the results given by evaluating the model along the drillhole.
        contacts : curlew.core.Geode
            A `curlew.core.Geode` instance containing the positions and orientations of contacts intersected along the drillhole.
        """
        dir = np.array(end) - np.array(start)
        length = np.linalg.norm(dir)
        dir = (dir / length)*step
        pos = np.array([start+dir*i for i in range( int(length / step) ) ])

        # evaluate model along drillholes
        g = self.predict( pos )

        # find contacts
        c = None
        g._contactMask = np.abs( np.diff( g.lithoID, prepend=g.lithoID[0] ) ) > 0
        if g._contactMask.any():
            cpos = pos[g._contactMask]
            c = self.predict( cpos, gradient=True ) # predict again, at the contact points only

        # return Geode
        return g, c

    def __getitem__(self, eid ):
        """Get GeoEvent by name (str) or SID (int). See `self.getField` for more details"""
        if isinstance(eid, str):
            for f in self.events:
                if f.name == eid: return f
            assert False, f"A field with name {eid} does not exist in this model."
        else:
            return self.eidLookup.get( int(eid), None)
    
    def getEvent( self, eid ):
        """
        Get the scalar field associated with the specified event ID (int) or name (str).

        Parameters
        ----------
        eid : int | str
            The event ID or field name to retrieve.

        Returns
        -------
        GeoEvent
            The scalar field instance associated with the specified event ID.
        """
        return self[eid]

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
        node_field = next((field for field in self.events if field.name == node), None)
        if node_field.parent2 is not None and isinstance(node_field, GeoEvent):
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

        for field in self.events[::-1]:
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
            if isinstance(field.parent, GeoEvent):
                graph.add_edge(field.name, field.parent.name)
            if isinstance(field.parent2, GeoEvent):
                graph.add_edge(field.name, field.parent2.name)
            if not isinstance(field.parent, GeoEvent) and field.parent is not None: # Handle fixed values
                graph.add_edge(field.name, str(field.parent))
            if not isinstance(field.parent2, GeoEvent) and field.parent2 is not None:
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