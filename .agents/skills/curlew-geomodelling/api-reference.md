# Curlew API reference (agent-oriented)

See also: [context.md](context.md) · [structures.md](structures.md)

## Imports

```python
import curlew
from curlew import GeoModel, GeoEvent, CSet, HSet
from curlew.geology import strati, fault, sheet, fold, domainBoundary, restore
from curlew.fields.fourier import NFF
from curlew.fields.series import FSF
from curlew.fields.analytical import LinearField, QuadraticField, PeriodicField, ListricField, EllipsoidalField
from curlew.kinematics.euler import euler_velocity, EulerVelocity
from curlew.kinematics.restoration import RestorationField
from curlew.kinematics.lift import FaultLift
from curlew.priors.spectral import spectral_band, n_features_for_band
from curlew.geology.interactions import Overprint, SheetOffset, FaultOffset, FoldOffset, FlowOffset
from curlew.geometry import Grid, Transform, compute_topology_adjacency
from curlew.core import Geode, Pebble
from curlew import synthetic  # steno, lehmann, hutton, playfair, walker, michell, anderson, seuss, goguel
from curlew.io import saveModel, loadModel, savePLY, loadPLY
```

**Not exported from `curlew.fields`** — import directly from the submodule:
`EshelbyField` (`curlew.fields.eshelby`), `PointKernelField` and the point
kernels (`curlew.fields.point`), `GeoINR` (`curlew.fields.geoinr`),
`RectangularPrismField` (`curlew.fields.analytical`). `FlowOffset` lives in
`curlew.geology.interactions`, not `curlew.fields`.

**Naming trap:** `FSF` is `FourierSeriesField`, *not* "finite strain field".

## Migration from the pre-`GeoEvent` API

`GeoField` was deleted and replaced by `GeoEvent`. There is **no shim** — old
names raise `ImportError`/`AttributeError`/`TypeError`.

| Old | New |
|---|---|
| `curlew.geology.geofield.GeoField` | `curlew.geology.geoevent.GeoEvent` |
| `GeoModel(fields=[...])` | `GeoModel(events=[...])` |
| `M.fields` | `M.events` |
| `M.getField(eid)` | `M.getEvent(eid)` (or `M[eid]`) |
| `VFieldOffset` | `FlowOffset` |
| `VFieldOffset(field=..., dt=...)` | `FlowOffset(velocity=..., direction=...)` |
| `GeoField.loss()` → `(loss, dict)` | `GeoEvent.loss()` → `Pebble` |
| `GeoField.fit(faultBuffer=...)` | removed |
| `GeoField.copy()`, `.buffer()`, `.zero()`, `.step()`, `.set_rate()` | removed; `zero`/`step` now on `Pebble`, `set_rate` on `LearnableBase` |
| `GeoModel.fit(learning_rate=...)` | removed |
| `GeoModel.zero()`, `.step()` | removed |
| `finiteFault(...)` | removed — use `fault(..., modifier=[...])` + `addField` |
| `FSF.w1` / `FSF.w2` | `FSF.A_mu` / `FSF.phi` (plus `A_rho`, `bias`) |
| `curlew.core.Transform` | `curlew.geometry.Transform` |
| `CSet.iq` with `'='` relations | `CSet.eq` (raises `ValueError`) |

## GeoModel

```python
M = GeoModel(events, transform=None, grid=None, name=None)
M.predict(x, coords="global"|"model", **kwargs)  # → Geode
M.prefit(epochs, **kwargs)   # dict name → Pebble, youngest-first isolated fit
M.fit(epochs, early_stop=(100, 1e-4), custom_loss=[], best=True, vb=True)
M.freeze(name=None, geometry=True, params=False)
M[name]  # str name or int eid
M.drill(start, end, step)  # → (geode, contacts_geode)
```

`events`: list oldest→youngest; linking sets `parent`/`child`; `parent2` for domain boundaries.

## GeoEvent (direct use when factories insufficient)

```python
GeoEvent(name, type=FSF, deformation=..., overprint=..., field=prebuilt, anchors=[], isosurfaces=[], **field_kwargs)
ev.addField("aux", type=LinearField, ...)
ev.addIsosurface("top", seed=pts)  # or value=1.0
ev.addAnchor("pin", position=...)   # or direction via start/end
ev.addVolume("reservoir", "(main > 0) & (main < cap)")
ev.fit(epochs, cache=True)
ev.getIsovalues() / ev.getIsovalue("top")
ev.projectTo("top", pts, nsteps=3)
ev.sampleIsosurface("top", seed_point, target_distance, volumeName=None)
```

## Overprint modes

| mode | Replaces parent where domain scalar... |
|------|--------------------------------------|
| `above` | > threshold (unconformities) |
| `below` | < threshold (intrusions below contact) |
| `in` | between threshold pair (dyke walls) |
| `out` | outside pair |

`threshold`: float, isosurface name, or tuple/list of names/values.
`defaultDomain`: str, 'child' (default) for unconformities, 'parent' for onlaps (unconformity geometry is defined by older implicit field and not younger one)

## CSet constraints (data)

CSets store geological information that interpolators or optimisers fit to.
```python
CSet(
    gp=..., gv=...,           # gradient + direction (gradient constraints)
    gop=..., gov=...,         # orientation only (tangent constraints)
    vp=..., vv=...,           # value (use sparingly on NFF/FSF as fitting is hard)
    pp=..., pv=...,           # property forward targets for e.g., density, chemistry
    iq=(ns, [(P1, P2, '>'), ...]), # inequality constraints for stratigraphic relations (overlies, underlies, older-than, younger-than, etc). ns determines number of random samples during each epoch. P1 and P2 are points on which the inequality is applied
    eq=[trace1, trace2, ...], # each (T, d) array should have the same scalar value (geological traces and contacts)
    grid=Grid(...), # grid used to evaluate global constrants. Typically covers the entire model domain.
    delta=10.0, # step size (in world coordinatse) used when computing numerical gradients to evaluate e.g., monotonicity loss.
    trend=np.array([...]), # global preferred gradient direction (often e.g., [0,0,1] for roughly flat stratigraphy)
    crs='global', # coordinate system used to define constraints. Typically 'global', but can become 'model' once a GeoModel.T transform is applied
)
```

Methods: `.torch()`, `.numpy()`, `.transform(func)` for retro-deformed coords.

Bind: `field.bind(C)` or pass `C=` into factory function.

## HSet (loss weights)
Hyperparameters used to balance loss functions.

Active losses: `value_loss`, `grad_loss`, `ori_loss`, `thick_loss`, `mono_loss`, `flat_loss`, `iq_loss`, `eq_loss`, `prop_loss`, plus `kl_loss` (Bayes-by-Backprop complexity cost on `FSF` posteriors) and `energy_loss`. `use_dynamic_loss_weighting`, `one_hot`, and `reuse_worst_half` (default 0.5) control optimisation behaviour.

Helpers: `HSet(...).copy(mono_loss=1.0)`, `.zero(grad_loss=1.0)`.

## Neural field kwargs (via `strati(..., type=NFF, **kw)`)

kwargs are passed to the constructor of the specified type. For example:

**NFF:** `hidden_layers`, `activation`, `rff_features`, `length_scales`, `stochastic_scales`, `learning_rate`, `scale` (output amplitude ~ field range).

**FSF:** `rff_features`, `length_scale_range`, `activation` (None = linear series), `learning_rate`, `freq_sampling` (e.g. `"quasi"` for Sobol directions), `freq_damp` (spectral falloff exponent, default 2.0), `xavier_amplitudes`, `stochastic_scales`, `domain`/`taper`/`taper_frac`/`envelope_floor` (soft box enveloping), `lift_dim` (extra lifted coordinates), plus Bayes-by-Backprop options `posterior_rho_init`, `prior_pi`, `prior_sigma1`, `prior_sigma2`.

`FSF` computes its `gradient` **analytically** (via `gradient_and_hessian`), not by autodiff. `energy_normalised=True` needs `activation="cubic"`.

## Analytical fields

Best defined using type = {some analytical field}. Kwargs are then passed to 
define the field geometry:

| Class | Geometry |
|-------|----------|
| `LinearField` | Plane; `origin`, `gradient`, `normalise`, `learnable` |
| `QuadraticField` | Curved bedding; `origin`, `gradient`, `curve` |
| `PeriodicField` | Folds; `gradient`, `axialPlane` (3D) |
| `ListricField` | Listric fault surface |
| `EllipsoidalField` | Ellipsoid distance or decay modifier (`decay=False`, `'ellipse'`, `'normal'`, `'cosine'`, `'wendland'`, `'sigmoid'`, `'inverse_quadratic'`, `'quadratic'`, or callable `f(u,r)`); intrusions, **finite-fault** slip tapers via `fault(..., modifier=...)` + `addField` |

**Finite fault:** `fault(..., contact=[...], offset=[...], modifier=[field_names])` where each name is an `EllipsoidalField` from `GeoEvent.addField` (slip × modifier at evaluation). Multi-contact = one implicit plane, several patches. See **`synthetic.goguel`** and [structures.md](structures.md) § finite faults.

## New field classes on `dev`

| Class | Module | Purpose |
|---|---|---|
| `FSF` | `curlew.fields.series` | Fourier series field (learnable amplitudes/phases/biases, frozen wavevectors) |
| `EulerVelocity` | `curlew.kinematics.euler` | Divergence-free velocity; `v = ∇⊥α` (2-D), `v = ∇α × ∇β` (3-D) |
| `RestorationField` | `curlew.kinematics.restoration` | Present stratigraphy as pullback of a reference layer-cake |
| `FlowOffset` | `curlew.kinematics.flow` | Integrator: RK2, RK4, or exact shear composition |
| `FaultLift` | `curlew.kinematics.lift` | Sheet labels for faulted restoration |
| `PointKernelField` | `curlew.fields.point` | Implicit field from seed points with normals/values/kernels |
| `EshelbyField` | `curlew.fields.eshelby` | Eshelby oblate-spheroid displacement (faults, dykes, kink folds) |

## Restoration API

See [restoration.md](restoration.md) for method and [tasks/restored-model.md](tasks/restored-model.md) for the workflow.

```python
restore(name, *, velocity, C=None, H=None, base=-np.inf, mode="above", onlap=False,
        lithoSharpness=1.0, structureSharpness=1.0, constraints=None,
        fault_lift=None, reference=None, pin=None, pin_axes=None, **kwargs)

euler_velocity(dim, *, n_features=128, length_scale_range=(0.5, 3.0),
               modes_per_octave=None, lift_dim=0, stream=None, hinge=None,
               cylindrical=False, seed=42, **fsf_kwargs)

# kwargs forwarded to RestorationField:
#   n_steps=20, integrator="rk4", depth_axis=-1, init_std=0.01,
#   signed_normals, sigma_floor, learning_rate, optimizer

spectral_band(C, *, axis=None, min_octaves=2.0) -> (L_min, L_max)
n_features_for_band(band, modes_per_octave=100.0, n_directions=None) -> int

FaultLift(faults, *, ...)   # 2-D polylines (P, 2) or 3-D (verts, faces)
```

Prefer `integrator="shear"` and `n_steps=0` for folds. `integrator` accepts
`"rk2"`, `"rk4"`, `"shear"`. A list `n_features` is a coarse-to-fine
composition, not a sum. `length_scale_range` may be a `CSet`; it is passed
through `spectral_band`.

`RestorationField` members: `integrator`, `inverse_map` (present→reference),
`forward_map` (reference→present), `restored_depth`, `predict_normals`
(Nanson).

`restore` sets `event.deformation = event.field.integrator`, so younger events
undeform through the same `Φ`.

Restoration constraints (`curlew.kinematics.constraints`):

```python
LayerThickness(horizons, thicknesses, weight=1.0, *, mode="relative")
OrthogonalThickness(weight=1.0, *, points=None, grid=None, horizons=None)
LayerParallelStretch(weight=1.0, *, horizons=None)   # neutral surface / bed length
HenckyStrain(weight=1.0, *, horizons=None)
```

A true discontinuity inside one map is still not representable. Split events
or lift. Method, wavelength rules and examples:
[restoration.md](restoration.md).

## Geode

Contains model outputs. 

Attributes: `scalar`, `structureID`, `lithoID`, `softLithoID`, `structureLookup`, `propertyNames`, `gradient`, `fields`, `offsets`, `isosurfaces`, `anchors`, `llookup`, `crs`.

**`geode.x` is now a dict** of coordinate-system → positions (keys include `crs`, `"global"`, `"model"`, and event names), not a bare array. Access via `geode.coords(crs=None)`; set with `geode.set_coords(key, coords, primary=False)`.

`geode.numpy()`, `geode.combine(younger, weight, soft_litho_weight=None, soft_structure_weight=None)`, `geode.stackValues()`.

### Text summary (`curlew.text` / `Geode.summary` / `str(geode)`)

Agent-oriented validation strings live in `curlew.text`:

| Function | Use |
|----------|-----|
| `geode_summary(geode, ...)` | Full model prediction summary |
| `geode_repr(geode)` | Short `repr(geode)` |
| `pebble_str(pebble)` | One-line loss bar (`str(pebble)`) |
| `geode_topology_contact_lines(dict)` | Format topology dict as contact lines |
| `grid_str(grid)` | Voxel counts, step, center, world-space bounds |
| `cset_str`, `field_str`, `geoevent_str`, `geomodel_str` | Summaries for other core types |
| `to_text(obj)` | Dispatch to the appropriate formatter |

`str(grid)`, `str(cset)`, `str(geoevent)`, `str(geomodel)`, and `str(field)` use the same formatters. `Geode.summary()` and `str(geode)` delegate to `geode_summary`. Includes grid metadata, **volumes** per structure and nested lithology (with `WARNING` for negligible units), **lithology/structure topology** contact lists, **mean displacement** per `offsets`, scalar min/max per structure, **property mean/min/max**, undefined `lithoID` (-1), and a warnings block.

```python
from curlew.text import geode_summary
print(geode.summary())                    # or geode_summary(geode)
print(geode.summary(connection="inclusive", volume_fraction_warn=0.01))
```

Requires `geode.grid` for volumes/topology. Scattered (non-grid) evaluation yields a shorter summary.

**Also useful for agents (not all in summary yet):** compare to expected event order / `GeoModel` names; `Pebble` loss totals after `fit`; constraint misfit per `CSet`; isosurface/anchor presence; property ranges; `gradient` vs `trend`; ensemble spread of topology across seeds.

### Topology adjacency (`Geode.topology`)

Requires evaluation on a regular `Grid` (`geode.grid` or pass `grid=`). Computes an **N×N symmetric contact-area matrix** between lithology or structure classes using differentiable convolution over gridded IDs.

**Useful for:** summarising which lithologies or `GeoEvent`s share boundaries on a voxel grid (QC after `M.predict(G)`); checking interpretive history against expected contact patterns (e.g. onlap/offlap, fault ladders, dyke walls — see [geological-reasoning.md](geological-reasoning.md) § quality checks / Thiele topology); comparing models or uncertainty ensembles by contact graph rather than scalar misfit alone; and, during training with `to_numpy=False`, penalising or rewarding specific contacts via differentiable `softLithoID` / `softStructureID` losses. Use `mode="structure"` for event-level adjacency; `plotAdjacency` for a labelled heatmap.

```python
A = geode.topology()                                    # lithology, matrix (default)
A = geode.topology(mode="structure", connection="inclusive")
D = geode.topology(output="dict")                       # {label_i: {label_j: area}}
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `grid` | `geode.grid` | Grid used to reshape ravelled IDs to 2D/3D volume |
| `mode` | `"lithology"` | `"lithology"` / `"litho"` → `lithoID` / `softLithoID`; `"structure"` → `structureID` / `softStructureID` |
| `output` | `"matrix"` | `"matrix"` → `(N,N)` numpy or torch (type matches hard IDs); `"dict"` → nested dict with lookup **names** as keys |
| `connection` | `"conservative"` | `"conservative"`: face-shared neighbours only (4-/6-connected); `"inclusive"`: face, edge, or vertex (8-/26-connected) |
| `symmetrize` | `True` | `(A + A.T) / 2` for mutual contact counts |
| `soft_sharpness` | `10.0` | Softmax temperature when `softLithoID` / `softStructureID` exist (torch path) |

**Row/column order:** `sorted(lithoLookup.keys())` or `sorted(structureLookup.keys())`; else sorted unique IDs in the data.

**Matrix semantics:** `A[i,j]` ≈ voxel contact area between class i and j (scales with resolution). **Diagonal is always 0** (self-contact removed). Off-diagonal only in `output="dict"`.

**Differentiability:** With `predict(..., to_numpy=False)`, soft IDs yield torch matrices; gradients flow through `compute_topology_adjacency` for loss terms on contact area.

Low-level (custom soft class volumes `(B,C,H,W)` or `(B,C,D,H,W)`):

```python
from curlew.geometry import compute_topology_adjacency
A = compute_topology_adjacency(soft_masks, symmetrize=True, connection="conservative")  # (B,C,C)
```

## Grid

Class for defining grids used to (1) evaluate model voxels before visualisation or extracting isosurfaces, and (2) define sampling locations used to extract points at which global points are evaluated.

Specified in several ways. First, if grid center and dimensions are known:

```python
G = Grid(dims, step=(dx, dy, dz), center=(cx, cy, cz), sampleArgs=dict(N=1024))
G.coords()  # evaluation points
```

Otherwise if we want to build a grid with a specific voxel size based on the 
bounds of some set of points:

```python
G = grid(extent = point_set, size=(rx,ry,rz), sampleArgs=dict(N=1024)) # size as integers specifies number of voxels in each dimension. Size as float specifies each voxel dimension directly
```

3D synthetic: `shape` may be voxel counts `(150,150,100)` with physical `EXTENT_3D`; use `_p2`/`_g2` patterns from `synthetic.py` when promoting 2D sections to 3D.

## Transform

```python
T = Transform(input_dim)  # identity
M = GeoModel(events, transform=T) # custom affine: set T matrix for UTM → model space
```

## IO

| Function | Format |
|----------|--------|
| `saveModel(path, M)` | PyTorch pickle `.pt` |
| `loadModel(path, map_location=curlew.device)` | Restores GeoModel/GeoEvent from pytorch pickle (`.pt`)|
| `savePLY` / `loadPLY` | Point clouds + scalars (needs `plyfile`). Returns dictionary containing "vert", "face", "color", "normals", "attr" and "names" keys |
| `saveOBJ` | Mesh export to OBJ format (prefer PLY due to smaller file size) |

## Visualisation

`curlew.visualise` — 2D plotting; `curlew.visualise.napari_viewer.NapariViewer` — 3D plotting.

### Topology heatmap (`plotAdjacency`)

Matplotlib heatmap of `Geode.topology` with row/column labels from lithology/structure lookup (or stringified class IDs).

```python
from curlew.visualise import plotAdjacency

fig, ax = plotAdjacency(geode)                          # lithology adjacency
fig, ax = plotAdjacency(geode, mode="structure")
fig, ax = plotAdjacency(geode, connection="inclusive", ax=existing_ax)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `geode` | — | `curlew.core.Geode` evaluated on a grid |
| `grid` | `geode.grid` | Passed to `geode.topology` |
| `mode` | `"lithology"` | Same as `Geode.topology` |
| `ax` | `None` | Matplotlib axes; new figure if `None` |
| `cmap` | `curlew.ccramp.reversed()` | Heatmap colormap |
| `colorbar` | `True` | Show contact-area colorbar |
| `title` / `cbar_label` | auto / `"Contact area"` | Axes title and colorbar label |
| `**topology_kwargs` | — | e.g. `connection`, `symmetrize`, `soft_sharpness` (`output` forced to `"matrix"`) |

Returns `(fig, ax)`. Requires matplotlib.

## Validation hooks

| Principle | Curlew mechanism |
|-----------|------------------|
| Superposition / young-on-old | `CSet.iq`, event order in `GeoModel` |
| Same contact | `CSet.eq` traces |
| No spurious closed horizons | `HSet.mono_loss`, `thick_loss` on `grid` — **or** use `restore`, where this holds by construction |
| Cross-cutting | Younger `fault`/`sheet` after cut `strati` |
| Domain split | `domainBoundary` |
| Fit diagnostics | `Pebble`; `M.drill` for section contacts |

Automated knowledge-consistency truth tables (Parquer et al.) are not in curlew — encode rules as `iq`/`eq` or check exported volumes externally.

## Uncertainty hooks

| Source (Wellmann et al.) | Curlew lever |
|--------------------------|--------------|
| Input data | Perturb `CSet`; `crs` + `Transform` |
| Interpolation | `HSet`, `seed`, `length_scales` |
| Structure / history | Alternate `GeoModel` event lists |
| Kinematics | `fault(..., offset=(v, min, max))`, `learn_sigma`, `modifier` (finite fault) |

Ensemble: vary `seed` on `NFF`/`FSF`, `prefit`, `saveModel`; compare `lithoID` across runs (Kamath et al.).

## Tests as spec

- `tests/test_geology.py` — `test_hutton`, `test_playfair`, unconformity/dyke workflows; multi-field / `EllipsoidalField`
- `tests/test_analytic.py` — `EllipsoidalField` decay modes
- `tests/test_synthetic.py` — includes **`goguel`**
- `tests/test_interactions.py` — overprint and offset behaviour
- `tests/restore/` — restoration, shear composition, Euler velocity, lift, training
- `tests/test_fsf.py` — `FSF` parameterisation and analytic gradients
- `tests/test_pointFields.py` — `PointKernelField`
- `tests/test_eshelby.py` — `EshelbyField`
- `tests/test_basic.py` — `Geode.topology` / `summary`, `Geode.coords`

Literature ↔ API: [context.md](context.md).
