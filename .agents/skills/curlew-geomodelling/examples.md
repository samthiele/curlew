# Curlew modelling examples

Patterns taken from `tests/test_geology.py` and `curlew/synthetic.py`. Adjust dimensions and epochs for production use.

Conceptual background: [context.md](context.md). Structure catalogue: [structures.md](structures.md). Environment: [environment.md](environment.md) (verify `torch` + `curlew` first).

## 1. Layer-cake synthetic (analytical)

```python
import curlew
from curlew.geology import strati
from curlew.geology.geomodel import GeoModel
from curlew.fields.analytical import QuadraticField
from curlew.synthetic import steno, _make_grid, _isosurface_seeds, _physical_extent, _model_ndim, sample

G = _make_grid((1500, 1000))
ndim = _model_ndim()
extent = _physical_extent()

s0 = strati("s0", C=QuadraticField("f0", input_dim=ndim,
    gradient=(0.00001, 1)[:ndim], curve=(-0.00005, 0)[:ndim],
    origin=(1000, 500)[:ndim]))
for seed, name in _isosurface_seeds(extent, ndim, 5):
    s0.addIsosurface(name, seed=seed)

M = GeoModel([s0], grid=G)
sample(M.predict(G), M, pv="rgb")
```

## 2. Unconformity on folded basement (synthetic truth → fit)

```python
from curlew.synthetic import hutton, extract_constraints
from curlew.geology import strati
from curlew.fields.fourier import NFF
from curlew import HSet
from curlew.geometry import Grid
import numpy as np

dims = (2000, 1000)
Ms = hutton(dims, breaks=10)
C = extract_constraints(Ms, ["s0", "s1"])

G = Grid(dims, step=(10, 10), center=(dims[0]/2, dims[1]/2), sampleArgs=dict(N=1024))
for c in C.values():
    c.grid = G
    c.delta = 10

H = HSet(value_loss="1.0", mono_loss="0.01", thick_loss=1.0)
s0 = strati("basement", C=C["s0"], H=H, type=NFF, base=-np.inf,
            rff_features=64, length_scales=[300 * np.pi], hidden_layers=[], activation=None)
s1 = strati("unconformity", C=C["s1"], H=H.copy(mono_loss="1.0"), type=NFF, base="base",
            rff_features=64, length_scales=[2000 / 2 * np.pi], hidden_layers=[], activation=None)

s1.isosurfaces = Ms["s1"].isosurfaces
s0.isosurfaces = Ms["s0"].isosurfaces
s1.addIsosurface("base", seed=Ms["s1"].field.origin)

M = GeoModel([s0, s1])
M.prefit(epochs=200, vb=True)
pred = M.predict(Grid(dims, step=(20, 20), center=(dims[0]/2, dims[1]/2)).coords())
```

Event order in `hutton` synthetic: `[s0, fold d1, s1]` — fold between basement and cover.

## 3. Dyke cross-cutting stratigraphy

```python
from curlew.synthetic import playfair, extract_constraints
from curlew.geology import strati, sheet
from curlew.fields.analytical import QuadraticField, LinearField

Ms = playfair((2000, 1000), width=50, addFault=False)
# Ms.events: s0 strat, s1 sheet — constraints split via sample() in builder

# Interpolation analogue:
s0 = strati("s0", C=..., type=NFF, ...)
s1 = sheet("s1", C=LinearField(...), contact=(-50, 50), aperture=2)
M = GeoModel([s0, s1])  # dyke younger than strat
```

## 4. Fault with learnable offset

```python
from curlew.geology import strati, fault
from curlew.fields.analytical import LinearField
import numpy as np

s0 = strati("strat", C=...)
fault_ev = fault(
    "fault",
    C=LinearField("ff", origin=(1050, 500), gradient=(-0.82, 0.57)),
    shortening=np.array([0.0, -1.0]),
    contact=0,
    offset=(100.0, 0.0, 200.0),  # learnable between 0 and 200
    width=1e-5,
)
M = GeoModel([s0, fault_ev])
M.freeze("fault", geometry=True, params=False)  # geometry fixed, slip free
M.fit(epochs=500)
```

## 5. Building from a text description (template)

**User:** "Flat-lying Tertiary cover on eroded Cretaceous monocline, cut by E-W normal fault."

```python
# Events (oldest → youngest):
# 1. cretaceous — strati, monocline (QuadraticField or NFF + dips)
# 2. erosion — strati cover, base="cretaceous_top" or onlap on parent
# 3. fault — fault(), shortening vertical, contact named "fault_surface"

events = [cretaceous, cover, normal_fault]
M = GeoModel(events)
# CSet per event from mapped data; iq for young-on-old; prefit then predict
```

## 6. Inequality constraint example

```python
from curlew.core import CSet
import numpy as np

n = 256
P_old = np.random.randn(n, 2) * 100 + [500, 200]
P_young = np.random.randn(n, 2) * 100 + [500, 400]
C = CSet(
    gp=orient_positions,
    gv=orient_vectors,
    iq=(n, [(P_young, P_old, ">")]),  # young unit above old
    grid=grid,
    delta=10.0,
)
```

## 7. Save and reload

```python
from curlew.io import saveModel, loadModel

saveModel("models/site_a.pt", M)
M2 = loadModel("models/site_a.pt")
```

## 8. Description → synthetic prototype first

When data are sparse or logic is uncertain:

1. Build closest analytical model: `Ms = curlew.synthetic.hutton(shape)` (or `playfair`, `walker`, …).
2. Visualise / sanity-check: `Ms.predict(Ms.grid)`.
3. `extract_constraints(Ms, event_names)` for training targets.
4. Swap analytical `C=` for `CSet` + `NFF`/`FSF` while preserving `isosurfaces` and event order.

Synthetic builders: `steno`, `lehmann`, `hutton`, `playfair`, `walker`, `michell`, `anderson`, `seuss`, **`goguel`** (finite normal faults).

## 9. Finite normal faults (`goguel`)

Right-dipping normal faults with slip limited by cosine-tapered ellipsoids on a shared fault plane. Constraints: bedding on `s0`, fault traces on `s1['s1']` only (not on `fe*`), global property on `M`.

```python
import curlew
from curlew.synthetic import goguel, extract_constraints

curlew.default_dim = 2
M = goguel(dip=60.0, zstep=300, offset=0.2, n_faults=3)  # samples + binds constraints

g = M.predict(M.grid)
g.stackValues(mn=0, mx=1)

# Re-fit targets (goguel already bound): s0 bedding, s1 plane field only — not fe0–fe2
strat_C = M["s0"].field.C
fault_C = M["s1"]["s1"].C
prop_C = M.C
```

Visualise modifier extent: `sum(g.fields[f"fe{i}"] for i in range(3))` on the grid (`sandbox/plot_goguel_2d.py`). Full pattern: [structures.md](structures.md) § finite faults.

## 10. Simple uncertainty ensemble (RFF seed)

```python
import curlew
from curlew.fields.fourier import NFF

ensemble = []
for seed in range(5):
    curlew  # ensure device/dtype set
    s0 = strati("basement", C=C_basement, H=H, type=NFF, seed=seed,
                rff_features=64, length_scales=[300 * np.pi], ...)
    M = GeoModel([s0, s1])
    M.prefit(epochs=100, vb=False)
    ensemble.append(M.predict(G.coords()))
# Compare lithoID (information entropy) or scalar variance across ensemble (Kamath: RFF stochasticity)
```

## 11. Validation against synthetic truth

```python
Ms = hutton(dims)
pred = M.predict(G.coords())
truth = Ms.predict(G.coords())
# Structure/litho agreement; loss should drop vs random init
assert (pred.structureID == truth.structureID).mean() > 0.9  # tune threshold
```
