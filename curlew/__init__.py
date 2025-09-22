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
from curlew.fields import NF
from curlew.geology.model import GeoModel
from curlew.geology.SF import SF

device = 'cpu' # can be changed to set device to e.g., gpu
"""The device used to compute operations with pytorch tensors. Change to allow e.g. GPU parallelisation."""

dtype = torch.float64
"""The precision used during pytorch computations. Lower to float32 to save RAM."""

ccmap = None
"""A colourful (custom) matplotlib colormap taylored for `curlew`. Will only be set if `matplotlib` is installed."""

try:
    # Define curlew colormap :-) 
    import matplotlib.colors as mcolors

    # Define the colors extracted manually from the provided logo image
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
    ccmap = mcolors.ListedColormap(colors)
except:
    pass

# import things we want to expose under the `curlew` namespace
from curlew import core
from curlew import data
from curlew import geology
from curlew import geometry
from curlew import visualise

from curlew.core import CSet, HSet
from curlew.geology import fault, strati, sheet
