# Task: LDI (restored) model

Build stratigraphy as the image of a reference layer-cake under a learned
diffeomorphism, so topology is preserved by construction.

Read [restoration.md](../restoration.md) for the method. This file is the
step-by-step.

## When to use this

- Folds are tight, inclined, recumbent, or refolded — the cases where a
  smoothness-based interpolant produces closed isosurfaces.
- The sequence is conformable inside the modelled volume. Faults still need
  [lifting](../restoration.md#faults) or a separate event.
- Finite-strain markers should constrain the geometry.
- You need a model you can argue is topologically admissible.

**Do not** use this when a plain `strati` + `fault` history expresses the
geology, when the map itself must contain a true discontinuity, or when dense
data must be honoured exactly and admissibility is not the limit.

## Steps

```
- [ ] Step 1: Environment check
- [ ] Step 2: Collect contacts, bedding, and strain if you have it
- [ ] Step 3: Decide the reference orientation
- [ ] Step 4: Read the wavelength band from the data
- [ ] Step 5: Build an Euler velocity and a restore() event
- [ ] Step 6: Fit
- [ ] Step 7: Validate topology
```

### Step 1: Environment

Follow `python scripts/check_env.py` as in [SKILL.md](../SKILL.md) § Step 1.
Set `curlew.default_dim` before building anything.

### Step 2: Constraints

Contacts go in `CSet.eq` as traces, or as values in `vp`/`vv`. Bedding goes in
`gp`/`gv`. Ordering goes in `iq`. Strain ellipses go in `sp` with `se`/`sr`.
Ask for the length units and the model domain. Wavelengths are in those units.

### Step 3: Reference orientation

Default is flat (`depth_axis=-1`). If the package youngs sideways, set
`depth_axis` to that horizontal axis. This changes optimisation, not the
topological guarantee.

### Step 4: Wavelengths

```python
from curlew.priors.spectral import spectral_band, n_features_for_band

band = spectral_band(C)   # or spectral_band(C, axis=hinge) when cylindrical
n_features = n_features_for_band(band, modes_per_octave=128, n_directions=32)
```

If the moments are unresolved, set `(L_min, L_max)` from the hinge spacing and
the limb-to-limb wavelength you can see, with at least two octaves between
them. See [restoration.md](../restoration.md#wavelengths).

### Step 5: Build

```python
import curlew, torch
from curlew.core import HSet
from curlew.geology import restore
from curlew.geology.geomodel import GeoModel
from curlew.kinematics.euler import euler_velocity

curlew.default_dim = 2

velocity = euler_velocity(
    2, n_features=n_features, n_directions=32, length_scale_range=band,
    stream="alpha", activation="cubic", energy_normalised=True, seed=7,
)
fold = restore(
    "fold", velocity=velocity, C=C,
    H=HSet.restore(grad_loss="1.0", eq_loss="0.1"),
    n_steps=0, integrator="shear", depth_axis=1, init_std=0.0,
    learning_rate=5e-3,
    optimizer={
        "amplitude": (torch.optim.SGD, dict(lr=1e-1, momentum=0.9, nesterov=True)),
        "shape": (torch.optim.Adam, dict(lr=5e-3)),
    },
)
M = GeoModel([fold])
```

For several wavelengths, pass `n_features` as a list (coarse band first) or
wrap `prefit` in `SpectralWarmup`. For a cylindrical 3-D fold, set
`hinge=` and `cylindrical=True`. Details and a strain-prior example are in
[restoration.md](../restoration.md).

### Step 6: Fit

```python
M.prefit(epochs=2000)
```

### Step 7: Validate — mandatory here

Check topology, not just loss. A valid scalar has no interior extrema in the
domain. Contacts on one trace restore to one depth, in stratigraphic order.
On a shear map, `det F` is 1.

A lower loss with interior extrema is a worse model. A looser fit at dense
contacts than an exact interpolant is the intended trade.

## Reporting to the user

State the reference axis, the wavelength band and whether it came from
`spectral_band` or from measured folds, `n_features`, `integrator` and
`n_steps`, any thickness or strain prior, and the extrema check.
