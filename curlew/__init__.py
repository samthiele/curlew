"""
# curlew

A toolkit for building 2- and 3- dimensional geological models using neural fields.

<img src="../icon.png" width="200">

## Getting started

### Installation

To install directly from github try: `pip install git+https://github.com/samthiele/curlew.git`.

This should run on most systems: `numpy`, `pytorch` and `tqdm` are the only required dependencies. Matplotlib is handy too, but not required. 

## Tutorials

Jupyter notebooks showing how to use 'curlew' and its various features can be found here:

- [GitHub]()
- [CoLab]()

## Support

Please use GitHub issues to report bugs. 

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

3) Leave a GitHub issue if you have questions or comments (Issues do not strictly need to be related to bug reports).

"""
import torch
from curlew.fields import NF
from curlew.geology.model import GeoModel
from curlew.geology.SF import SF

device = 'cpu' # can be changed to set device to e.g., gpu
"""The device used to compute operations with pytorch tensors. Change to allow e.g. GPU parallelisation."""

dtype = torch.float64
"""The precision used during pytorch computations. Lower to float32 to save RAM."""

# import things we want to expose under the `curlew` namespace
from curlew import core
from curlew import data
from curlew import geology
from curlew import geometry
from curlew import visualise

from curlew.core import CSet, HSet
from curlew.geology import fault, strati, sheet
