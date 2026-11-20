# Build a synthetic geological model

**Goal:** Construct an analytical (or lightly parametric) `GeoModel` from a text description, sketch, or image **without real site data**. Use `curlew.synthetic` as the primary reference implementation.

## Workflow

```
- [ ] Match description to a synthetic prototype (structures.md index)
- [ ] Set curlew.default_dim (2 or 3)
- [ ] Plan GeoModel event order (oldest → youngest)
- [ ] Build with curlew.geology factories + analytical fields
- [ ] predict → summarise → visualise → validate topology
```

## 1. Choose a synthetic prototype

Start from the closest builder in `curlew/synthetic.py`; inspect with `print(M)`:

| Function | Geological story | Event order |
|----------|------------------|-------------|
| `steno` | Layer-cake strat | `[s0]` |
| `lehmann` | Folded strat (`PeriodicField`) | `[s0]` |
| `hutton` | Strat + fold + unconformable cover | `[s0, d1, s1]` |
| `playfair` | Strat + dyke [+ fault] | `[s0, s1]` or `[s0, s1, s2]` |
| `walker` | Strat + dyke swarm [+ fault] | `[s0, s1]` or `[s0, s1, s2]` |
| `michell` | Strat + thrust | `[s0, s1]` |
| `anderson` | Strat + two normal faults | `[s0, s1, s2]` |
| `goguel` | Finite normal faults (ellipsoid modifiers) | `[s0, s1]` |
| `seuss` | Strat, dyke, intrusion domain, listric faults, cover | tree — see source |

```python
import curlew
from curlew.synthetic import hutton, _make_grid, _isosurface_seeds, _physical_extent, _model_ndim

curlew.default_dim = 2
M_ref = hutton(shape=(2000, 1000))
print(M_ref)
pred = M_ref.predict(M_ref.grid)
print(pred.summary())
```

Full structure catalogue: [structures.md](../structures.md).

## 2. Geological planning

Before coding, read [geological-reasoning.md](../geological-reasoning.md):

1. List events **oldest → youngest**.
2. Classify each: generative (`overprint`), kinematic (`deformation`), or both (`sheet`).
3. Pick factories: `strati`, `fault`, `sheet`, `fold`, `domainBoundary`.
4. Prefer **seed-based** `addIsosurface(name, seed=…)` over fixed `value`.

Set dimension:

```python
curlew.default_dim = 2   # section model
# curlew.default_dim = 3  # volume; use EXTENT_3D / VOXELS_3D patterns in synthetic.py
```

## 3. Build pattern (analytical)

Typical layer-cake (`steno` pattern):

```python
import curlew
import numpy as np
from curlew.geology import strati
from curlew.geology.geomodel import GeoModel
from curlew.fields.analytical import QuadraticField, LinearField
from curlew.synthetic import _make_grid, _isosurface_seeds, _physical_extent, _model_ndim

G = _make_grid((1500, 1000))
ndim = _model_ndim()
extent = _physical_extent()

s0 = strati("basement", C=QuadraticField("f0", input_dim=ndim,
    gradient=(0.00001, 1)[:ndim], curve=(-0.00005, 0)[:ndim],
    origin=(1000, 500)[:ndim]), base=-np.inf)
for seed, name in _isosurface_seeds(extent, ndim, 5):
    s0.addIsosurface(name, seed=seed)

M = GeoModel([s0], grid=G)
```

Unconformity stack (`hutton` pattern):

```python
from curlew.geology import strati, fold

s0 = strati("basement", C=..., base=-np.inf)
d1 = fold("d1", origin=..., compression=..., extension=..., wavelength=2000, amplitude=250)
s1 = strati("cover", C=..., base="erosional_base")  # or base=0.0
s1.addIsosurface("erosional_base", seed=erosion_point)
M = GeoModel([s0, d1, s1], grid=G)
```

Dyke cross-cutting strat (`playfair`):

```python
from curlew.geology import strati, sheet

s0 = strati("host", C=...)
s1 = sheet("dyke", C=LinearField(..., gradient=steep_normal),
           contact=(-50, 50), aperture=2)
M = GeoModel([s0, s1])  # dyke younger than host
```

More patterns: [examples.md](../examples.md).

## 4. Helpers from synthetic.py

| Helper | Use |
|--------|-----|
| `_make_grid(shape)` | Default `Grid` for 2D or 3D |
| `_isosurface_seeds(extent, ndim, n)` | Evenly spaced formation-top seeds |
| `_p2(pt, ndim)`, `_g2(vec, ndim)` | Promote 2D section coords to 3D |
| `extract_constraints(M, event_names)` | Sample constraints from truth model (for benchmarking neural fit) |
| `sample(pred, M, ...)` | Colour-coded constraint overlay plots |

## 5. Validate (required)

After `pred = M.predict(G.coords())` or `M.predict(G)`:

1. **Summaries:** `print(M)`, `print(pred.summary())` — volumes, topology contacts, warnings.
2. **Topology:** `pred.topology(mode="lithology")` and `mode="structure"`; or `plotAdjacency(pred)`.
3. **Visual:** [visualise-model.md](visualise-model.md) — `plot2D` on `lithoID`, `structureID`, `scalar`.
4. **Geological checks** ([geological-reasoning.md](../geological-reasoning.md) § quality):
   - Event order respects cross-cutting
   - Unit count and contact graph match the request
   - No “bubble” horizons (use `mono_loss` if fitting neural fields later)

## 6. Optional: neural benchmark

To test interpolation logic against synthetic truth:

```python
from curlew.synthetic import hutton, extract_constraints
from curlew.fields.fourier import NFF
from curlew import HSet

Ms = hutton((2000, 1000))
C = extract_constraints(Ms, ["s0", "s1"])
# → swap analytical C= for NFF + CSet; preserve isosurfaces and event order
# See test_hutton in tests/test_geology.py and [interpolated-model.md](interpolated-model.md)
```

## Anti-patterns

- Skipping prototype match — reinventing `hutton`/`playfair` from scratch.
- Wrong event order; dyke before host strat.
- `PeriodicField` strat **and** kinematic `fold` on the same package.
- Building without `predict` + topology check.
