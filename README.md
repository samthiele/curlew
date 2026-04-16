# curlew

A toolkit for building 2- and 3- dimensional geological models using neural fields.

<img src="https://github.com/samthiele/curlew/blob/main/icon.png?raw=true" width="200">

## Getting started

### Installation

To install directly from github try: `pip install git+https://github.com/samthiele/curlew.git`.

This should run on most systems: `numpy`, `pytorch` and `tqdm` are the only required dependencies. Other recommended (but optional) dependencies include `matplotlib` (for 2D plotting), `plyfile`(for IO with 3D PLY files), and `napari` (for 3D visualisation).

To install all dependencies, including the optional ones, use `pip install curlew[all]`. 

### Tutorials

Quickly get up to speed with `curlew` using these [CoLab tutorials](https://drive.google.com/drive/folders/14OPpL2-zKuJSd2Hh7jobnIYPnxzl0wCI?usp=sharing). 

### Documentation

Documentation is automatically built and served through [GitHub pages](https://samthiele.github.io/curlew/). 

## Support

Please use [GitHub issues](https://github.com/samthiele/curlew/issues) to report bugs or other problems. The [Discussions](https://github.com/samthiele/curlew/discussions) can be used for feature requests, ideas or more general problems (or praise :smirk:). Also feel free to share any cool models you have built through the [Discussions](https://github.com/samthiele/curlew/discussions)!

## Contributing and appreciation

Please star this repository if you found it useful. If you have fixed bugs or added new features then we welcome pull requests.

## Authors and acknowledgment

`curlew` has been developed by Sam Thiele and Akshay Kamath, with valuable input from Mike Hillier, Lachlan Grose, Richard Gloaguen and Florian Wellmann.

If you use `curlew` we would appreciate it if you:

1) Cite the following paper (for academic work)

```
Kamath, A. V., Thiele, S. T., Moulard, M., Grose, L., Tolosana-Delgado, R., Hillier, M., and Gloaguen, R. (2026). Curlew 1.0: Spatio-temporal implicit geological modelling with neural fields in python. Solid Earth. doi:10.31223/X5KX81
```

2) Star this repository so that we get a rough idea of our user base :star:

## Versions and change log

### v1.1 - major restructure, introduced napari-based 3D viewer

Upgrades to the geology/modeling core, especially an improved structure for the `GeoField` class and associated deformation events. Added new field types, including Fourier series fields, which tend to converge faster and better than Fourier neural fields. Also included a prototype napari-based 3D visualization tool that runs nicely in parallel to jupyter notebook environments.

### v1.0 - initial release

Implemented basic fourier-feature based neural fields and overlying data-structures to turn these into 3D model. Developed an implemented comprehensive loss function for fitting neural fields to diverse geological data. 