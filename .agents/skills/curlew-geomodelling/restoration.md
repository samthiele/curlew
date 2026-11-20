# Learnable Diffeomorphic Interpolation

How to build a `restore(...)` event. For *when* to choose it, see
[SKILL.md](SKILL.md). The checklist is
[tasks/restored-model.md](tasks/restored-model.md).

Worked notebooks (same API, fuller data plumbing):
https://github.com/k4m4th/ldi

## Principle

A smoothness-based interpolant fits a scalar to contacts and normals. Nothing
stops that scalar from closing on itself, so tight folds grow interior extrema
("bubbles"). A penalty only discourages them, and only as well as its weight.

LDI recovers the deformation instead of the scalar. Present-day stratigraphy
is a reference state carried by a map `Φ`:

```
ψ(x) = ψ_ref(Φ⁻¹(x))
```

`ψ_ref` is a flat layer-cake (original horizontality) unless you pass another
`reference`. `Φ` is a diffeomorphism: smooth, invertible, smooth inverse. Level
sets of `ψ` are images of the reference level sets, so a stack of planes cannot
become a closed bubble. Topology is a property of the construction.

Three consequences:

- **One map, every observation.** Contacts, bedding normals (transported by
  Nanson's relation) and finite-strain markers all constrain `Φ`. Strain is
  data, not an axial surface you have to draw.
- **Volume preservation is the parameterisation.** The velocity is an Euler
  field, `v = ∇α × ∇β` (in 2-D, `v = ∇⊥α`). Any such `v` is divergence-free.
  Composing the Fourier modes as exact shears keeps `det F = 1`.
- **The wavelength band is geological.** It is the range of fold wavelengths
  you expect, not a smoothing knob. A short band cannot build the envelope; a
  long band cannot build the hinges.

The fit is deliberately inexact where an interpolant would be exact. If the
data require a true discontinuity (fault, unconformity, boudin), LDI is the
wrong object for that break.

## What to call

```text
euler_velocity(...)  →  restore(...)  →  GeoModel  →  prefit
        │                    │
        │                    └─ RestorationField
        │                         integrator: FlowOffset
        └─ EulerVelocity over FSF potentials
```

| Piece | Where |
|---|---|
| `euler_velocity` | `curlew.kinematics.euler` |
| `restore` | `curlew.geology` |
| `RestorationField`, `FlowOffset` | `curlew.kinematics.restoration`, `curlew.kinematics.flow` |
| Wavelengths | `curlew.priors.spectral` |
| Thickness / strain priors | `curlew.kinematics.constraints` |
| Fault sheets | `curlew.kinematics.lift.FaultLift` |
| Coarse-to-fine amplitudes | `curlew.kinematics.training.SpectralWarmup` |

Do not import `curlew.fields.clebsch` or `curlew.fields.restoration`. Those
modules are gone.

## Wavelengths

Set `length_scale_range=(L_min, L_max)` in model units before choosing a mode
count. Three ways, in order:

1. **From the constraints**, when you have both traces and bedding:

```python
from curlew.priors.spectral import spectral_band, n_features_for_band

band = spectral_band(C)                       # (L_min, L_max)
# or hand the CSet straight to the velocity:
velocity = euler_velocity(2, length_scale_range=C, modes_per_octave=128,
                          n_directions=32, ...)
```

`spectral_band` reads three moments of the folded layering. Contacts set the
height variance, bedding the tilt variance, and nearby bedding pairs the
curvature variance. From those:

- `λx = 2π √(m2/m4)` — mean hinge spacing
- `λ0 = 2π √(m0/m2)` — mean limb spacing (dropped when the traces are too
  short to resolve heights)
- `L_min = min(λx, λ0)`
- `L_max` is the longer of those, widened so the band covers at least two
  octaves and the diameter of the data. Strain ellipses, if present, can push
  `L_max` further so a bulk distortion is not forced into the short modes.

`return_info=True` gives the moments. If it warns that `λx` is unresolved, the
band is biased long: add closer orientation pairs, or set the band by hand.

2. **By eye.** Measure one hinge spacing and one limb-to-limb wavelength.
   Use the shorter as `L_min` and the longer as `L_max`, and keep at least
   two octaves (`L_max ≥ 4 L_min`) so the map can form both the train and
   the hinges. A cylindrical fold is measured in the profile plane
   (`spectral_band(C, axis=hinge)`).

3. **Mode count from the band**, not the other way around. Wavelengths are
   drawn log-uniformly, so each octave should hold a similar number of modes:

```python
n_features = n_features_for_band(band, modes_per_octave=128, n_directions=32)
```

About a hundred modes per octave, shared across a few dozen directions, is a
sound start. Fewer directions leave angular holes; fewer modes per octave
cannot fill the band.

A **list** of `n_features` splits one band into that many log-equal pieces,
coarsest first. `FlowOffset` composes those velocities. It does not add them.
Use this when one wavelength range is not enough (a long envelope with short
disharmonic folds). `SpectralWarmup` is the lighter version of the same idea:
long modes train first, short modes open over the first fraction of epochs.

```python
from curlew.kinematics.training import SpectralWarmup

with SpectralWarmup(event) as schedule:      # default ramp = 1/6 of epochs
    M.prefit(epochs=2000, schedule=schedule)
```

## Building the map

Default integrator in code is RK4. For folding, prefer a shear composition:
each Fourier direction is an exact sinusoidal shear, and `n_steps=0` applies
every direction once at full amplitude. The map stays invertible at any
amplitude, so short folds are not stepped away. `integrator="rk4"` (or
`"rk2"`) integrates the stationary velocity instead; raise `n_steps` until the
short hinges stop moving, then leave it.

```python
import curlew, torch
from curlew.core import CSet, HSet
from curlew.geology import restore
from curlew.geology.geomodel import GeoModel
from curlew.kinematics.euler import euler_velocity
from curlew.priors.spectral import n_features_for_band, spectral_band

curlew.default_dim = 2

band = spectral_band(C)
n_features = n_features_for_band(band, 128, 32)

velocity = euler_velocity(
    2,
    n_features=n_features,
    n_directions=32,
    length_scale_range=band,
    stream="alpha",                 # 2-D: v = ∇⊥α
    activation="cubic",
    energy_normalised=True,         # requires activation="cubic"
    seed=42,
)

fold = restore(
    "fold",
    velocity=velocity,
    C=C,                            # gp/gv bedding, eq traces, optional sp strain
    H=HSet.restore(grad_loss="1.0", eq_loss="0.1"),
    n_steps=0,
    integrator="shear",
    depth_axis=1,                   # flat reference; see below
    init_std=0.0,                   # start from the identity
    learning_rate=5e-3,
    optimizer={
        "amplitude": (torch.optim.SGD, dict(lr=1e-1, momentum=0.9, nesterov=True)),
        "shape": (torch.optim.Adam, dict(lr=5e-3)),
    },
    pin=anchor,                     # hold one point, or "mean"
)
M = GeoModel([fold])
M.prefit(epochs=2000)
```

`depth_axis=-1` is flat-lying. If the package youngs sideways, point
`depth_axis` along that horizontal axis so the reference is a vertical stack.
Otherwise the fit spends its long modes on a rigid rotation. Any reference
with layer-stack topology is valid; the axis is an optimisation choice.

`init_std=0` is the identity. A positive value redraws amplitudes. `None`
keeps a velocity you already configured.

In 3-D, `stream` defaults to `"beta"` and both potentials are learned. For a
cylindrical fold, freeze the hinge and draw modes in the profile plane:

```python
velocity = euler_velocity(
    3, n_features=n_features, n_directions=32, length_scale_range=band,
    stream="alpha", activation="cubic", energy_normalised=True,
    hinge=hinge, cylindrical=True, seed=42,
)
fold = restore("fold", velocity=velocity, C=C, n_steps=0, integrator="shear",
               depth_axis=0)
```

`slow_hinge=(L_min, L_max)` adds a long Fourier term on that hinge so the axis
itself can curve gently. It requires `cylindrical=True`.

## What `C` means under `restore`

| Field | Meaning |
|---|---|
| `gp` / `gv` | Bedding normals, transported by Nanson (`H.grad_loss`) |
| `gop` / `gov` | Unsigned normals (`H.ori_loss`) |
| `vp` / `vv` | Restored-depth values (`H.value_loss`) |
| `eq` | Horizon traces that must restore flat (`H.eq_loss`) |
| `iq` | Stratigraphic order of restored depth (`H.iq_loss`) |
| `sp` with `se` and/or `sr` | Strain ellipses (`H.strain_loss`) |

`HSet.restore()` turns value, gradient and equality on, and leaves the
monotonicity and thickness penalties off. Those penalties are for a scalar
interpolant. Topology is already built in.

## Thickness and strain priors

Optional. They live on `constraints=`, not in `C`. A string weight such as
`"0.1"` is scaled to the first finite raw loss, so you do not hand-tune a
magnitude.

```python
from curlew.kinematics.constraints import LayerParallelStretch, OrthogonalThickness

fold = restore(
    "fold", velocity=velocity, C=C,
    constraints=[
        LayerParallelStretch("0.1", horizons=(inner_arc, outer_arc)),  # neutral surface
        OrthogonalThickness("0.1"),                                    # restored thickness ⊥ layers
    ],
)
```

`LayerThickness` matches restored-depth gaps. `HenckyStrain` prefers a
low-strain map among those that already fit; keep its weight below the
contacts. Volume preservation does not fix layer thickness. These priors do.

## Faults

A diffeomorphism cannot jump. Two options:

1. **Lift.** `FaultLift` adds a sheet coordinate from fault traces (2-D
   polylines) or meshes (3-D `(verts, faces)`). Pass `lift_dim` into the
   velocity.

```python
from curlew.kinematics.lift import FaultLift

fault_lift = FaultLift([trace])
velocity = euler_velocity(2, n_features=64, length_scale_range=band,
                          lift_dim=fault_lift.lift_dim)
fold = restore("fold", velocity=velocity, C=C, fault_lift=fault_lift,
               n_steps=0, integrator="shear")
```

2. **Separate events.** Split on a `domainBoundary` and restore each block,
   or leave the fault as a younger kinematic `fault` outside the map.

Unconformities stay ordinary events: an older `restore` (or `strati`) and a
younger `strati` with `base=` and `onlap=True`.

## Check the result

- No interior extrema of `ψ` in the domain. A lower loss that closes a
  horizon is a worse model.
- Each `eq` trace restores to one depth, and those depths are in order.
- For a shear map, `det F` is 1. A drift means the integrator, not the geology.
- State the band and why (moments, or the wavelengths you measured), `n_steps`,
  `integrator`, and `depth_axis`.

## Mistakes

- Setting `length_scale_range` by sweeping losses. Read it from the folds.
- Using RK4 with a handful of steps on a multi-wavelength fold. The short
  hinges disappear. Use `integrator="shear"`, `n_steps=0`, or a band split.
- `energy_normalised=True` without `activation="cubic"`.
- Forgetting `lift_dim=fault_lift.lift_dim`.
- Expecting the map to insert a fault, a boudin, or compaction. It will not.
