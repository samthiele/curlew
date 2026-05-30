# curlew

A toolkit for building 2- and 3- dimensional geological models using neural fields.

<img src="https://github.com/samthiele/curlew/blob/main/banner.png?raw=true" height="300">

This approach allows the inclusion of diverse geological data, while the overarching graph-based modelling framework allows complex structures (unconformities, folds, faults, intrusions, etc.) to be constructed sequentially based on a geological history. 

As well interpolation-based 3D modelling, `curlew` can build complex synthetic geology models in which the neural fields are replaced by simple mathematical functions. These synthetic models can then be sampled to create datasets for testing, benchmarking and algorithm development.

Everything is implemented in a differentiable way using `pytorch` to facilitate inversion frameworks and perform optimisation tasks.

## Getting started

### Installation

To install directly from github try: 

> pip install git+https://github.com/samthiele/curlew.git

This should run on most systems: `numpy`, `pytorch` and `tqdm` are the only required dependencies. Other recommended (but optional) dependencies include `matplotlib` (for 2D plotting), `plyfile`(for IO with 3D PLY files), and `napari` (for 3D visualisation).

To install all dependencies, including the optional ones, use: 

> pip install curlew[all] 

### Tutorials

Quickly get up to speed with `curlew` using these [CoLab tutorials](https://drive.google.com/drive/folders/14OPpL2-zKuJSd2Hh7jobnIYPnxzl0wCI?usp=sharing). 

### Documentation

Documentation is automatically built and served through [GitHub Pages](https://samthiele.github.io/curlew/):
- [Stable (`main`)](https://samthiele.github.io/curlew/) — default API reference
- [Development (`dev`)](https://samthiele.github.io/curlew/dev/) — latest changes on the dev branch

A banner at the top of each page links between the two versions.

## Support and feedback

Please use [GitHub issues](https://github.com/samthiele/curlew/issues) to report bugs. For broader ideas or questions, don't hesitate to use the [discussions](https://github.com/samthiele/curlew/discussions) page.

## Contributing and appreciation

Please star this repository if you found it useful. If you have fixed bugs or added new features then we welcome pull requests.

## Authors and acknowledgment

`curlew` has been developed by Sam Thiele and Akshay Kamath, with valuable input from Mike Hillier, Lachlan Grose, Richard Gloaguen and Florian Wellmann.

If you use `curlew` we would appreciate it if you:

1) Cite the following [paper](https://eartharxiv.org/repository/view/10473/) (for academic work)

```
Kamath, A. V., Thiele, S. T., Moulard, M., Grose, L., Tolosana-Delgado, R., Hillier, M., and Gloaguen, R. (2026). Curlew 1.0: Spatio-temporal implicit geological modelling with neural fields in python. Solid Earth. doi:10.31223/X5KX81
```

2) Star this repository so that we get a rough idea of our user base :star:

## Versions and change log

### v1.2 - another significant refactoring (sorry)

- Renamed `GeoField` to `GeoEvent` to avoid confusion with scalar fields themselves (inheriting from `BaseSF`). Updated `GeoEvent` so that they can contain multiple 
separate underlying fields and (using these) define `volumes` within the model. These
volumes can be used to e.g., define finite faults. 

- Updated the `Overprint` class to allow onlap relationships in which the
unconformity geometry is determined by the older (rather than the younger)
implicit field.

- Simplified synthetic model creation so that extracted constraints are 
added directly to the created `BaseSF` instances, and added 2D and 3D options
so that 3D test data can be more easily generated.

- Added the `Pebble` class (inspired by `Geode`) to help more easily create, track
  and combine losses and their associated optimsers.

- Added option for custom losses to be defined at the `GeoModel` level, to enable 
  workflows that optimise all fields as a stack (e.g., fitting to large datasets that
  cannot easily be broken down to the per-field level [ e.g., gravity inversion ]). 

### v1.1 - major restructure, introduced napari-based 3D viewer

Upgrades to the geology/modeling core, especially an improved structure for the `GeoField` class and associated deformation events. Added new field types, including Fourier series fields, which tend to converge faster and better than Fourier neural fields. Also included a prototype napari-based 3D visualization tool that runs nicely in parallel to jupyter notebook environments.

### v1.0 - initial release

3) Leave a [GitHub issue](https://github.com/samthiele/curlew/issues) if you have questions or comments (Issues do not strictly need to be related to bug reports).
