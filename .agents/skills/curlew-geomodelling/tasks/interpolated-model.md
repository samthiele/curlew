# Build an interpolated geological model

**Goal:** Fit implicit scalar fields to **real structural data** and site-specific geological knowledge using `NFF`, `FSF`, or learnable analytical fields.

Primary references: `tests/test_geology.py`, [api-reference.md](../api-reference.md), [geological-reasoning.md](../geological-reasoning.md).

## Workflow

```
- [ ] Confirm constraints per event (from extract-data or user data)
- [ ] Decide pathway: smoothness-based interpolation, or `restore` if folding is tight
- [ ] Plan GeoModel event order (oldest → youngest)
- [ ] Attach grid + delta to each CSet for global losses
- [ ] Build factories (strati, fault, sheet, …) with type=NFF or FSF
- [ ] prefit per event → fit → predict → validate
```

## Implicit interpolation (concept)

Each `GeoEvent` owns one or more **scalar fields** φ(x). Geology is encoded as:

- **Contacts** — isosurfaces φ = c (prefer **seed**-based thresholds)
- **Orientations** — ∇φ aligned with bedding/fault poles (`gp`/`gv`)
- **Relations** — equal φ along traces (`eq`); young-on-old inequalities (`iq`)
- **Global shape** — monotonicity, thickness, trend via `HSet` losses on a `grid`

A single global φ cannot represent full history; curlew stacks **events** with overprints and retro-deformation (`undeform`). See [context.md](../context.md).

### When the folds are tight, switch pathway

A smoothness-based interpolant places no constraint on the **topology** of the
surfaces it produces. Tight or recumbent folds are therefore liable to acquire
interior extrema — closed isosurfaces enclosing rock that cannot exist. If the
user's data describe tight folding, or a built model shows closed horizons,
use `restore(...)` instead; there the topology is preserved by construction.
See [restored-model.md](restored-model.md) and [restoration.md](../restoration.md).

Do **not** simply raise `mono_loss`/`thick_loss` and hope: those penalise
extrema only approximately and add hyperparameters to tune.

## Constraint → interpolator mapping

| Data | CSet | HSet (typical) |
|------|------|----------------|
| Bedding dips | `gp`, `gv` | `grad_loss`, `mono_loss`, `thick_loss` |
| Contact traces | `eq` | `eq_loss` |
| Younger above older | `iq` | `iq_loss` |
| Sparse scalar pins | `vp`, `vv` | `value_loss` (use sparingly on NFF) |
| Properties | `pp`, `pv` | `prop_loss` on `GeoModel` |

Always set on each fitted `CSet`:

```python
from curlew.geometry import Grid

G = Grid(dims, step=(10, 10), center=(dims[0]/2, dims[1]/2), sampleArgs=dict(N=1024))
C.grid = G
C.delta = 10.0   # world units for numerical gradients in global losses
```

Prefer `gp`/`gv`, `eq`, `iq` over `vp` on neural fields.

## HSet defaults (starting point)

```python
from curlew import HSet

H = HSet(value_loss="1.0", grad_loss="1.0", mono_loss="0.01", thick_loss="1.0")
# Dyke/sheet: stronger thick_loss — see test_playfair
H_sheet = HSet(value_loss=1, grad_loss=1, mono_loss="0.01", thick_loss=1.0)
```

Tune with `.copy(mono_loss="1.0")` per event.

## Example 1: Unconformity (`test_hutton`)

Folded basement + erosional cover; two `strati` events with `NFF`:

```python
import numpy as np
import curlew
from curlew.synthetic import hutton, extract_constraints  # or user CSet from CSV
from curlew import HSet
from curlew.geology import strati
from curlew.geology.geomodel import GeoModel
from curlew.fields.fourier import NFF
from curlew.geometry import Grid

curlew.default_dim = 2
dims = (2000, 1000)
Ms = hutton(dims, breaks=10)
C = extract_constraints(Ms, ["s0", "s1"])

G = Grid(dims, step=(10, 10), center=(dims[0]/2, dims[1]/2), sampleArgs=dict(N=1024))
for c in C.values():
    c.grid = G
    c.delta = 10

H = HSet(value_loss="1.0", mono_loss="0.01", thick_loss=1.0)
s0 = strati("basement", C=C["s0"], H=H, type=NFF, base=-np.inf,
            rff_features=64, length_scales=[300 * np.pi],
            hidden_layers=[], activation=None)
s1 = strati("unconformity", C=C["s1"], H=H.copy(mono_loss="1.0"), type=NFF, base="base",
            rff_features=64, length_scales=[2000 / 2 * np.pi],
            hidden_layers=[], activation=None)

s1.isosurfaces = Ms["s1"].isosurfaces
s0.isosurfaces = Ms["s0"].isosurfaces
s1.addIsosurface("base", seed=Ms["s1"].field.origin)

M = GeoModel([s0, s1])
loss1 = M.prefit(epochs=1, best=True, vb=False)
loss2 = M.prefit(epochs=200, best=True, vb=False)
# loss should decrease per event

pred = M.predict(Grid(dims, step=(20, 20), center=(dims[0]/2, dims[1]/2)).coords())
```

Event order in `hutton`: `[s0, fold d1, s1]` — fold sits between basement and cover when using full synthetic; neural tests often fit `[s0, s1]` with constraints sampled from the combined truth.

## Example 2: Dyke cross-cutting strat (`test_playfair`)

```python
from curlew.synthetic import playfair, extract_constraints
from curlew.geology import strati, sheet
from curlew.fields.fourier import NFF

Ms = playfair((2000, 1000))
C = extract_constraints(Ms, ["s0", "s1"])

H = HSet(value_loss=1, grad_loss=1, mono_loss="0.01", thick_loss="0.1")
s0 = strati("basement", C=C["s0"], H=H, type=NFF, base=-np.inf,
            rff_features=32, length_scales=[2000/2*np.pi],
            hidden_layers=[], activation=None)

H = HSet(value_loss=1, grad_loss=1, mono_loss="0.01", thick_loss=1.0)
s1 = sheet("dyke", C=C["s1"], H=H, type=NFF,
           contact=("upper", "lower"),
           rff_features=32, length_scales=[2000/2*np.pi],
           hidden_layers=[], activation=None)
s1.addIsosurface("upper", seed=np.array([dims[0]/2 - 25, 0]))
s1.addIsosurface("lower", seed=np.array([dims[0]/2 + 25, 0]))

M = GeoModel([s0, s1])
_, loss1 = M.fit(epochs=1, best=True, vb=False)
_, loss2 = M.fit(epochs=100, best=True, vb=False, early_stop=None)
```

## Example 3: Fault with learnable slip (`test_anderson`)

```python
from curlew.geology import strati, fault
from curlew.fields.analytical import LinearField

s0 = strati("strat", C=..., type=NFF, ...)
fault_ev = fault("fault", C=...,
    type=LinearField,  # or NFF for interpolated fault plane
    shortening=np.array([0.0, -1.0]),
    contact=0,
    offset=(100.0, 0.0, 200.0),  # learnable between 0 and 200
    width=1e-5)
M = GeoModel([s0, fault_ev])
M.freeze("fault", geometry=True, params=False)
M.fit(epochs=500)
```

## Example 4: FSF instead of NFF (`test_hutton_FSF`)

Same as hutton but `type=FSF` from `curlew.fields.series` — useful when a Fourier series parameterisation fits the data better.

## Example 5: 3D onlap (`test_onlap`)

Analytical `LinearField` + `fold` + multiple onlapping `strati` packages — shows 3D `Grid`, named `base` isosurfaces, and `onlap=True`. Read `test_onlap` for full grid extents and event list.

## Fitting strategy

1. **`M.prefit(epochs=…)`** — fit each event in isolation (youngest-first undo); preferred first pass.
2. **`M.fit(epochs=…, early_stop=(100, 1e-4))`** — joint optimisation.
3. **`M.freeze(name, geometry=True, params=False)`** — fix plane geometry, learn slip/offsets.
4. Check **`Pebble`** losses: `str(pebble)` or `pebble.total()` should decrease.

Bind constraints to the **owning event** — fault traces on `M["fault"]["plane_field"]`, not on ellipsoid modifiers (`goguel` pattern).

## Validate (required)

Same as [synthetic-model.md](synthetic-model.md) § 5:

- `print(pred.summary())` — volumes, topology contact lines, warnings
- `plotAdjacency(pred, mode="lithology")` and `mode="structure"`
- Visual cross-sections: [visualise-model.md](visualise-model.md)
- Compare lithology names and unit count to geological brief

## Uncertainty (optional)

Vary `seed` on `NFF`/`FSF`; compare topology adjacency across ensemble ([examples.md](../examples.md) § 10).
