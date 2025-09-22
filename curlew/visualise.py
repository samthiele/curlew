"""
Functions for performing crude 2D plotting using matplotlib. Useful for demonstrations, but will
need to be extended at some point to be more usable during model development. 
"""

import numpy as np
from curlew.geology.SF import SF

def plot2D( sxy, grid, C=None, ticksize=50, lw=1, cmap='rainbow', levels=None, ax=None, alpha=0.3 ):
    """
    Create a 2D plot of a scalar field and optionally overlay associated constraints.

    Parameters
    ----------
    sxy : np.ndarray
        A (N,) array of scalar values corresponding to the above points, or an array of shape (N, 3) 
        containing RGB values to plot instead.
    grid : curlew.geometry.Grid
        A grid defining the points at which the values of `sxy` are located.
    C : np.ndarray, optional
        A constraint set containing the (2D) points to overlay on the plot.
    ticksize : int, optional
        The size of the orientation ticks to add to the plot.
    lw : float, optional
        The linewidth to use for plotting orientation constraints.
    levels : list or None or bool, optional
        A list of contour levels to plot. Set to `None` for automatic selection, or `False` to disable contours.
    ax : matplotlib.axes.Axes, optional
        A Matplotlib axes object on which to plot.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.Axes
        The generated Matplotlib figure and associated axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca() # get current axes
    
    vmn, vmx = None, None
    pxy = grid.coords()
    shape = grid.shape
    if (pxy is not None) and (sxy is not None):
        xmn,xmx = np.percentile(pxy[:,0], (0,100)) # get bounds
        ymn,ymx = np.percentile(pxy[:,1], (0,100))

        if (sxy.shape[-1] == 3) or (sxy.shape[-1] == 4): # RGB or RGBA colours
            # plot colours directly
            si = sxy.reshape( shape + (sxy.shape[-1],) )
            ax.imshow( np.transpose( si, (1,0,2)), alpha=alpha, extent=(xmn,xmx,ymn,ymx), origin='lower' )
        else:
            si = sxy.reshape(shape) # reshape to image
            vmn,vmx = np.percentile(sxy, (0,100))
            
            # plot scalar field and countours
            ax.imshow(si.T, cmap=cmap, alpha=alpha, extent=(xmn,xmx,ymn,ymx), vmin=vmn, vmax=vmx, origin='lower' )
            if not (isinstance(levels, bool) and (levels == False)):
                contour = ax.contour(si.T, cmap=cmap,levels=levels, extent=(xmn,xmx,ymn,ymx),
                                     vmin=vmn, vmax=vmx)
                ax.clabel(contour, inline=True, fontsize=12)

    # plot data
    if C is not None:
        # plot value constraints
        if (C.vp is not None) and (C.pp is None): # don't plot value constraints if a property constraint is defined
            if vmn is None: vmn,vmx = np.percentile( C.vv.squeeze(), (0,100) )
            ax.scatter( C.vp[:,0], C.vp[:,1], c=C.vv.squeeze(), 
                        cmap=cmap, vmin=vmn, vmax=vmx, edgecolors='k', zorder=10, s=ticksize )
        
        # plot gradient constraints
        if C.gp is not None:
            gp = C.gp
            gv = C.gv
            for i,v in enumerate(gv):
                ax.plot( [ gp[i][0], gp[i][0] + v[0]*ticksize ], [ gp[i][1], gp[i][1] + v[1]*ticksize ], color='orange', lw=lw, zorder=10 )
                dx = ticksize*v[1]
                dy = -ticksize*v[0]
                ax.plot([ gp[i][0]-dx,  gp[i][0]+dx],[ gp[i][1]-dy,  gp[i][1]+dy], color='k', lw=lw, zorder=10 )
        
        # plot orientation constraints
        if C.gop is not None:
            gp = C.gop
            gv = C.gov
            for i,v in enumerate(gv):
                dx = ticksize*v[1]
                dy = -ticksize*v[0]
                ax.plot([ gp[i][0]-dx,  gp[i][0]+dx],[ gp[i][1]-dy,  gp[i][1]+dy], color='k', lw=lw, zorder=10 )
        
        # plot property constraints
        if (C.pp is not None) and (C.pv is not None):
            if (C.pv.shape[-1] == 1):
                ax.scatter( C.pp[:,0], C.pp[:,1], c=C.pv[:,0])
            elif (C.pv.shape[-1] >= 3):
                rgb = C.pv[:,[0,1,2]].astype(float)
                rgb -= np.min(rgb, axis=0)[None,:]
                rgb /= np.max(rgb, axis=0)[None,:]
                ax.scatter( C.pp[:,0], C.pp[:,1], c=rgb, s=ticksize/2)
                

        # plot grid for evaluating global constraints
        if C.sgrid is not None:
            ax.scatter( C.sgrid[:,0], C.sgrid[:,1], color='gray', s=ticksize/3 )
    
    return ax.get_figure(), ax

def format_latex_subscript(name):
    """
    Converts 'f2' to LaTeX format '$f_2$'. 
    Works with multiple letters and digits (e.g., 'sigma12' → '$\\sigma_{12}$').
    """
    import re

    match = re.match(r"^([a-zA-Z]+)([0-9]+)$", name)
    if match:
        return f"${match.group(1)}_{{{match.group(2)}}}$"
    else:
        return f"${name}$"  # fallback if not matching pattern

def plotDrill(hole, ax, ticksize=50, lw=1, vmn=0, vmx=20, cmap="tab20b", noval=False):
    """
    Plot the specified drillhole on the given axis with color normalization.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.patheffects as pe
    
    points = np.array([hole['pos'][:, 0], hole['pos'][:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalization to handle vmin/vmax
    norm = Normalize(vmin=vmn, vmax=vmx)
    
    # Plot the borehole
    lc = LineCollection(
        segments, 
        array=hole['classID'], 
        cmap=cmap, 
        norm=norm,
        linewidths=lw
    )
    t = ax.add_collection(lc)
    t.set_path_effects([pe.Stroke(linewidth=lw+5, foreground='k'), pe.Normal()])
    
def plotConstraints(ax, C=None, H=None, ll=1, lw=4, scale=0.001, ac="k", vmn=0, vmx=20, cmap="tab20b"):
    
    if C is not None:
        if (H is None) or (H.value_loss != 0):
            if (C.vp is not None) and (C.pp is None): # don't plot value constraints if a property constraint is defined        
                if vmn is None: vmn, vmx = np.percentile( C.vv.squeeze(), (0,100))
                ax.scatter(C.vp[:,0], C.vp[:,1], c=C.vv.squeeze(), 
                           cmap=cmap, vmin=vmn, vmax=vmx, zorder=12, edgecolor=ac)

        # plot gradient constraints
        if (H is None) or (H.grad_loss != 0):
            if C.gp is not None:
                # Extract the positions and gradients
                positions = C.gp
                gradients = C.gv
                
                # Normalize perpendicular vectors for consistent length
                norms = np.linalg.norm(gradients, axis=1, keepdims=True)
                gradients_unit = gradients / (norms + 1e-8)

                # Plot gradients
                ax.quiver(
                    positions[:, 0], positions[:, 1],
                    gradients_unit[:, 0], gradients_unit[:, 1],
                    color=ac, angles='xy', scale_units='xy', scale=scale, zorder=10, width=lw,
                )
                
                # Compute perpendicular bedding orientations by rotating gradients 90 degrees
                perp_gradients = np.vstack([-gradients_unit[:, 1], gradients_unit[:, 0]]).T

                # Plot perpendicular bedding orientations
                t = ax.quiver(
                    positions[:, 0], positions[:, 1],
                    ll * perp_gradients[:, 0], ll * perp_gradients[:, 1],
                    color=ac, angles='xy', scale_units='xy', scale=scale, zorder=11, width=lw,
                    headlength=0, headwidth=0, headaxislength=0, pivot="middle"
                )

        # Plot orientations
        if (H is None) or (H.ori_loss != 0):
            if C.gop is not None:
                # Extract the positions and gradients
                positions = C.gp
                gradients = C.gv

                # Normalize perpendicular vectors for consistent length
                norms = np.linalg.norm(gradients, axis=1, keepdims=True)
                gradients_unit = gradients / (norms + 1e-8)

                # Compute perpendicular bedding orientations by rotating gradients 90 degrees
                perp_gradients = np.vstack([-gradients_unit[:, 1], gradients_unit[:, 0]]).T

                # Plot perpendicular bedding orientations
                ax.quiver(
                    positions[:, 0], positions[:, 1],
                    ll * perp_gradients[:, 0], ll * perp_gradients[:, 1], scale=scale,
                    color=ac, angles='xy', scale_units='xy', zorder=10, width=lw,
                    headlength=0, headwidth=0, headaxislength=0, pivot="middle"
                )

            
        # Plot property constraints
        if (H is None) or (H.prop_loss != 0):
            if (C.pp is not None) and (C.pv is not None):
                if (C.pv.shape[-1] == 1):
                    ax.scatter( C.pp[:,0], C.pp[:,1], c=C.pv[:,0])
                elif (C.pv.shape[-1] >= 3):
                    rgb = C.pv[:,[0,1,2]].astype(float)
                    rgb -= np.min(rgb, axis=0)[None,:]
                    rgb /= np.max(rgb, axis=0)[None,:]
                    ax.scatter( C.pp[:,0], C.pp[:,1], c=rgb, s=lw/2)
                    
                    
# MODEL TREE                    
def get_positions(M, G, node, first_x=0, first_y=0, step_x=10, step_y=5, pos=None):
    """
    Recursively calculate the positions of nodes in a hierarchical structure.

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
    node_field = next((field for field in M.fields if field.name == node), None)
    if node_field.parent2 is not None and isinstance(node_field, SF):
        # Move to the right
        pos = get_positions(M, G, children[0], first_x + step_x, first_y, step_x, step_y, pos)
        # Move down
        pos = get_positions(M, G, children[1], first_x, first_y - step_y, step_x, step_y, pos)
    else:
        # For non-domain boundary nodes, move to the right
        pos = get_positions(M, G, children[0], first_x + step_x, first_y, step_x, step_y, pos)

    return pos

def showModel(M, axs=None, leg_loc=None, title="c)", node_size=3000, font_size=18):
    """
    Visualize the model tree of a GeoModel.

    Parameters
    ----------
    None
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    import networkx as nx

    # Create an empty graph
    graph = nx.DiGraph()

    domain_boundary_color = '#E35B0E'
    dilative_event_color = "#F0C419"
    generative_event_color = '#31B4C2'
    kinematic_event_color = "#A6340B"
    fixed_value_color = "#FAE8B6"

    for field in M.fields[::-1]:
        # Determine the color based on the event type
        color = None
        if field.parent2 is not None:
            # domain boundary
            color = domain_boundary_color
        elif field.bound is not None and field.deformation is not None: 
            # dilative event
            color = dilative_event_color
        elif field.bound is not None: 
            # generative event
            color = generative_event_color
        elif field.deformation is not None:
            # kinematic event
            color = kinematic_event_color
        graph.add_node(field.name, label=format_latex_subscript(field.name), color=color)

        # Add edges
        if isinstance(field.parent, SF):
            graph.add_edge(field.name, field.parent.name)
        if isinstance(field.parent2, SF):
            graph.add_edge(field.name, field.parent2.name)
        if not isinstance(field.parent, SF) and field.parent is not None: # Handle fixed values
            graph.add_edge(field.name, str(field.parent))
        if not isinstance(field.parent2, SF) and field.parent2 is not None:
            graph.add_edge(field.name, str(field.parent2))

    # Plotting
    if axs is None:
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.01)
        ax_graph = fig.add_subplot(gs[0])
        ax_legend = fig.add_subplot(gs[1])
    else:
        ax_graph = axs
        ax_legend = ax_graph.inset_axes(leg_loc)

    # Model Tree
    pos = get_positions(M, graph, list(graph.nodes())[0])
        
    node_colors = [graph.nodes[n].get('color', fixed_value_color) for n in graph.nodes()]
    labels = {n: graph.nodes[n].get("label", str(n)) for n in graph.nodes()}
    nx.draw(graph, pos, with_labels=True, labels=labels, arrows=True, node_size=node_size,
            node_color=node_colors, font_size=font_size, font_color="k", ax=ax_graph)
    ax_graph.set_title(title, loc="left")

    # Legend
    legend_labels = {
        domain_boundary_color : 'Domain Boundary',
        dilative_event_color  : 'Dilative Event',
        generative_event_color: 'Generative Event',
        kinematic_event_color : 'Kinematic Event',
        fixed_value_color     : 'Fixed Value'
    }
    
    ax_legend.axis('off')
    handles = [Line2D([0], [0], marker='o', color='k', label=label,
                      markerfacecolor=color, markersize=15)
               for color, label in legend_labels.items()]
    ax_legend.legend(handles=handles, loc='center', fontsize=12)
    
    if axs is None:
        plt.close(fig)
        return fig
    else:
        pass

def colour( sf, cmap='tab20', breaks=19 ):
    """
    Apply a Matplotlib colormap to a scalar field to generate a colorized property set.

    The function maps scalar field values to colors using a specified colormap and returns 
    the colors as uint8 values between 0 and 255.

    Parameters
    ----------
    sf : np.ndarray
        A scalar field array representing the values to be colorized.
    cmap : str, optional
        The name of the Matplotlib colormap to use. Default is 'tab20'.
    breaks : int or array-like, optional
        If an integer, defines the number of breakpoints used to segment the scalar field.
        If an array-like object, specifies the exact breakpoints.

    Returns
    -------
    np.ndarray
        An array of shape matching `sf` with RGB color values mapped to the colormap, 
        returned as uint8 values in the range [0, 255].
    """
    
    from matplotlib.colors import BoundaryNorm # do here so matplotlib is not a mandatory dependency
    import matplotlib.pyplot as plt

    if isinstance(breaks, int):
        breaks = np.hstack( [np.min(sf), np.linspace( np.min(sf), np.max(sf), breaks)] )
    else:
        breaks = np.hstack( [np.min(sf), breaks, np.max(sf) ] )
        
    cm = plt.get_cmap(cmap)
    n = cm.N
    
    assert n >= len(breaks), "Number of breaks (%d) must be less than number of colours in colormap (%d)"%(len(breaks), n) 

    norm = BoundaryNorm( breaks, ncolors=n )
    c = cm( norm( sf ) )[..., :3]
    c = (c*255).astype(np.uint8)
    return c