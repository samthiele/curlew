"""

A toolkit for building 2- and 3- dimensional geological models using neural fields.

<img src="https://github.com/samthiele/curlew/blob/main/icon.png?raw=true" width="200">

## Getting started

### Installation

To install directly from github try: `pip install git+https://github.com/samthiele/curlew.git`.

This should run on most systems: `numpy`, `pytorch` and `tqdm` are the only required dependencies. Matplotlib is handy too, but not required. 

### Tutorials

To help get up to speed with `curlew`, we maintain a set of CoLab tutorial notebooks [here](https://drive.google.com/drive/folders/14OPpL2-zKuJSd2Hh7jobnIYPnxzl0wCI?usp=sharing). 
Additional examples (used to make figures in the paper listed below) can be found [here](https://github.com/k4m4th/curlew_tutorials).

### Support

Please use [GitHub issues](https://github.com/samthiele/curlew/issues) to report bugs. 

## Contributing and appreciation

Please star this repository if you found it useful. If you have fixed bugs or added new features then we welcome pull requests.

## Authors and acknowledgment

`curlew` has been developed by Sam Thiele and Akshay Kamath, with valuable input from 
Mike Hillier, Lachlan Grose, Richard Gloaguen and Florian Wellmann.

If you use `curlew` we would appreciate it if you:

1) Cite the following paper (for academic work)

```
Kamath, A.V., Thiele, S.T., Moulard, M., Grose, L., Tolosana-Delgado, R., Hillier, M.J., Wellmann, R., & Gloaguen, R. Curlew 1.0: Implicit geological modelling with neural fields in python. Geoscientific Model Development (preprint online soon) 
```

2) Star this repository so that we get a rough idea of our user base

3) Leave a [GitHub issue](https://github.com/samthiele/curlew/issues) if you have questions or comments (Issues do not strictly need to be related to bug reports).

"""
import torch
import numpy as np
from curlew.fields import BaseNF
from curlew.geology.geomodel import GeoModel
from curlew.geology.geofield import GeoField

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

# import things we want to expose under the `curlew` namespace
from curlew import core
from curlew import synthetic
from curlew import geology
from curlew import geometry
from curlew import visualise

from curlew.core import CSet, HSet
from curlew.geology import fault, strati, sheet
