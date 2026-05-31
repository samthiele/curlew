"""

A toolkit for building 2- and 3- dimensional geological models using neural fields.

<img src="https://github.com/samthiele/curlew/blob/main/banner.png?raw=true" style="width:100%; height:auto; display:block;">

## Getting started

### Installation

To install directly from github try: `pip install git+https://github.com/samthiele/curlew.git`.

This should run on most systems: `numpy`, `pytorch` and `tqdm` are the only required dependencies. Matplotlib is handy too, but not required. 

### Tutorials

To help get up to speed with `curlew`, we maintain a set of CoLab tutorial notebooks [here](https://drive.google.com/drive/folders/14OPpL2-zKuJSd2Hh7jobnIYPnxzl0wCI?usp=sharing). 
Note that these are written for the `main` branch of `curlew` so will likely not work out-of-the-box with the `dev` branch. And, if anyone knows a better (free) platform for these then do let us know!

## Support and feedback

If you are using `curlew`, or even just think it's cool, then please give us a star. It makes a difference!

Any questions, ideas, feedback or issues can be raised via [GitHub issues](https://github.com/samthiele/curlew/issues) or
on the [discussions](https://github.com/samthiele/curlew/discussions) page. Don't be afraid to drop us a line.

## Overview

`curlew` is a toolkit for building 2- and 3-dimensional geological models using neural fields and/or other learnable functions. The key idea is to use neural fields 
to learn the implicit representation of geological structures, which can then be used to generate synthetic models, 
interpolate sparse geological data, and perform joint inversion tasks.

When starting with `curlew` it is important to understand the following key classes, as these are the building blocks of a `curlew` model.

**`curlew.fields.BaseSF` (Scalar Fields)**

All implementations of implicit (scalar) fields inherit from `curlew.fields.BaseSF`. These are the basic building blocks of all implicit models. 
Currently, curlew supports a variety of learnable (complex/fitted) and analytical (simple, closed-form) fields. These are summarised below.

*Neural fields (for interpolation tasks):*

- `curlew.fields.fourier.NFF` for fourier-feature based neural fields
- `curlew.fields.series.FSF` for Fourier-series based neural fields (a simplified/faster version of NFF)

*Analytical fields (generally for building synthetic models, but can be useful in real models too)*:

- `curlew.fields.analytical.LinearField` for planar (linear) geometries
- `curlew.fields.analytical.QuadraticField` for quadratic geometries
- `curlew.fields.analytical.PeriodicField` for sine/cosine waves (e.g., folded geometries)
- `curlew.fields.analytical.ListricField` for listric faults.
- `curlew.fields.analytical.EllipsoidalField` for ellipsoidal geometries (e.g., intrusion bodies, finite faults, etc.)

**`curlew.geology.geoevent.GeoEvent` (Geological events)**

Geological meaning is given to scalar fields through the `curlew.geology.geoevent.GeoEvent` class. This encapsulates one or more `curlew.fields.BaseSF` instances and defines 
how they interact with older and younger fields in the model. Currently, two different relationships are supported:

- `curlew.geology.interactions.Overprint` defines overprinting relations (unconformities, onlaps, intrusions and domain boundaries) and determines how an event overprints (truncates) older geology.
- `curlew.geology.interactions.OffsetBase` defines deformations that transform (retro-deform) the points a model is being evaluated at prior to the evaluation of older events. This can be used to represent offset from folds, faults, and sheet-intrusions (each of which has a specific deformation formulation that inherits from the base `curlew.geology.interactions.OffsetBase` class, e.g. `curlew.geology.interactions.FaultOffset`).

`curlew.geology.geoevent.GeoEvent` instances are typically constructed using the factory functions in `curlew.geology`, including `curlew.geology.strati`, `curlew.geology.sheet`, `curlew.geology.fault`, `curlew.geology.fold`, etc.

Geologically meaningful isosurfaces (lithological contacts, fault surfaces, etc.) and volumes (regions between faults, reservoirs, finite faults, etc.) are also defined by the `curlew.geology.geoevent.GeoEvent` class, 
using the `addIsosurface` and `addVolume` methods. Importantly, and unlike most geological modelling tools, isosurfaces can be defined either as fixed values (e.g., `1.0`) or by whichever value the model evaluates
at a given seed point (or the average of multiple seed points). Crucially, the latter form allows isosurfaces to be defined independent of the underlying field values.  

**`curlew.geology.geomodel.GeoModel` (models)**

The `curlew.geology.geomodel.GeoModel` class is used to construct the graph of `curlew.geology.geoevent.GeoEvent` instances that ultimately define model evaluation flow and topology. 
It also provides a high-level interface for evaluating model predictions at arbitrary points in space and defining global to local 
coordinate system transforms (i.e. local grids). Models can be saved and loaded using the `curlew.io.saveModel` and `curlew.io.loadModel` functions.

**`curlew.core.Geode` (model outputs)**

The `curlew.core.Geode` class is used to store the diverse outputs of a model. These include the scalar field values, lithology and structure IDs, 
predicted properties (e.g., density), and applied deformations (e.g., fault offset). The `curlew.core.Geode` class can also be used to store the model's grid, coordinate system transforms, 
and other metadata.

**`curlew.core.Pebble` (losses and optimisers)**

The `curlew.core.Pebble` class is used to store losses and optimisers. This includes the loss values, the weights (hyperparameters) used to balance each loss, 
and the optimisers used to update the model's parameters. Generally it is only necessary to create a new `curlew.core.Pebble` object if you want to 
create a custom loss function or new type of learnable object.

**`curlew.core.CSet` (model constraints)**

Geological information used to fit learnable fields (or offset/overprint objects) are stored in a `curlew.core.CSet` object. When constructing a new model
it is typical to construct a `curlew.core.CSet` for each `curlew.geology.geoevent.GeoEvent` instance based on the available data. These data include (but are not limited to):

- value constraints (target scalar values at specific points. Generally these should be avoided as neural fields are much better at fitting gradient and (in)equality constraints.)
- gradient constraints (strike and dip measurements of bedding or other structural surfaces)
- property constraints (arbitrary property values at specific points, like density or chemistry)
- inequality constraints (known stratigraphic relationships between two sets of points)
- equality constraints (traces or contact surfaces known to be tangential to a field)
- grid constraints (grid covering the model domain, to sample global constraints from)
- trend constraints (a preferred global gradient orientation)

**`curlew.core.HSet` (model hyperparameters)**

The dark-side of neural field interpolation. This class contains the various hyperparameters that can be used to 
balance/tune complex (multi-objective) loss functions. Most fields should use ~1-3 individual loss terms (with the remaining hyperparameters set to 0),
as otherwise hyperparameter optimisation becomes very difficult. It is typical to create a new `curlew.core.HSet` object for each `curlew.geology.geoevent.GeoEvent` instance based on the available data
and properties of the interpolated field.

## Contributing and appreciation

Please star this repository if you found it useful. If you have fixed bugs or added new features then we welcome pull requests.

## Authors and acknowledgment

`curlew` has been developed by Sam Thiele and Akshay Kamath, with valuable input from 
Mike Hillier, Lachlan Grose, Richard Gloaguen and Florian Wellmann.

If you use `curlew` we would appreciate it if you:

1. Cite the following paper (for academic work)

    ```
    Kamath, A.V., Thiele, S.T., Moulard, M., Grose, L., Tolosana-Delgado, R., Hillier, M.J., Wellmann, R., & Gloaguen, R. Curlew 1.0: Implicit geological modelling with neural fields in python. Geoscientific Model Development (preprint online soon) 
    ```

2. Star this repository so that we get a rough idea of our user base

3. Leave a [GitHub issue](https://github.com/samthiele/curlew/issues) if you have questions or comments (Issues do not strictly need to be related to bug reports).

"""
import torch
import numpy as np

device = 'cpu' # can be changed to set device to e.g., gpu
"""The device used to compute operations with pytorch tensors. Change to allow e.g. GPU parallelisation."""

dtype = torch.float64
"""The precision used during pytorch computations. Lower to float32 to save RAM."""

default_dim = 3
"""The default input dimensionality (2D or 3D) to use when creating new models. Default is 3."""

compile = False 
"""Whether to compile the model using torch.compile. This can significantly speed up larger models when using a GPU."""

ccmap = None
"""A colourful (custom) matplotlib (categorical) colormap taylored for `curlew`. Will only be set if `matplotlib` is installed."""

ccramp = None
"""A colourful (custom) matplotlib (continuous) colormap taylored for `curlew`. Will only be set if `matplotlib` is installed."""

ccstrat = None
"""A shuffled version of ccmap, useful for plotting stratigraphic fields as though they have many layers in them. """

batchSize = 512000
"""Divide arrays larger than this size into chunks (batches) to reduce memory usage and avoid out-of-memory crashes."""

mpl=False
try:
    # Define curlew colormap :-) 
    import matplotlib.colors as mcolors
    mpl = True
except:
    pass

if mpl:
    # Define a custom / pretty colour ramp for curlew models
    colors = [
        "#A6340B",  # rich red (not darkest)
        "#E35B0E",  # vibrant orange-red
        "#F39C12",  # medium orange
        "#F0C419",  # bright orange-yellow
        "#FAE8B6",  # soft pale orange (close to white but not pure white)
        "#8CD9E0",  # light cyan blue
        "#31B4C2",  # medium cyan-blue
        "#1B768F",  # medium blue 
        "#054862",  # deeper blue (not darkest)
    ]
    ccmap = mcolors.ListedColormap(colors=colors, name='curlew_categorical')

    # Also as a continuous colormap
    ccramp = mcolors.LinearSegmentedColormap.from_list(
        name="curlew_continuous",
        colors=colors,
        N=256  # resolution of the ramp
    )

    # and a shuffled version for visualising stratigraphies
    _colors = ccramp(np.linspace(0, 1, 255))[:, :3]
    _step = 25 # block shuffle
    for i in np.arange(0,len(_colors), step=_step):
        if i + _step*2 > len(_colors):
            break
        ixx = np.random.choice(np.arange(i,i+_step*2), _step*2, replace=False)
        _colors[i:(i+_step*2), :] = _colors[ixx, :]

    # Create a new colormap
    ccstrat = mcolors.ListedColormap(_colors, name="curlew_stratigraphic")

## Utility functions for converting between numpy and torch tensors
def _tensor(x, dev=None, dt=None):
    """
    Convert array-like or scalar input to a torch tensor on the given device and dtype.
    """
    if dev is None: dev = device # normally use default device
    if dt is None: dt = dtype # normally use default dtype
    if isinstance(x, torch.Tensor):
        return x.to(device=dev, dtype=dt)
    elif isinstance(x, np.ndarray):
        return torch.tensor(x, device=dev, dtype=dt)
    elif isinstance(x, (list, tuple)):
        return torch.tensor(x, device=dev, dtype=dt)
    elif isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_)):
        return torch.tensor(x, device=dev, dtype=dt)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    
def _numpy(x):
    """
    Convert a torch tensor or list to a numpy array.
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    elif isinstance(x, list):
        return np.array(x)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
    
    
# import things we want to expose under the `curlew` namespace
from curlew.fields import BaseNF
from curlew.geology.geomodel import GeoModel
from curlew.geology.geoevent import GeoEvent
from curlew import core
from curlew import synthetic
from curlew import geology
from curlew import geometry
from curlew import visualise
from curlew import text

from curlew.core import CSet, HSet
from curlew.geology import fault, strati, sheet
