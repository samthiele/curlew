# Visualise and extract data from a model

**Goal:** Inspect an existing `GeoModel` or saved checkpoint — 2D sections, 3D volumes, drill traces, topology plots, and interactive Napari views.

Requires environment probe ([environment.md](../environment.md)).

## Quick reference: `curlew.visualise`

| Function | Use |
|----------|-----|
| `plot2D(values, grid, C=None)` | Scalar, `lithoID`, `structureID`, or RGB on a **2D** `Grid` |
| `plotDrill2D(hole, ax)` | Lithology-coloured drill trace on a section |
| `plotConstraints2D(ax, C=, H=)` | Overlay constraint types |
| `plotAdjacency(geode, mode="lithology"\|"structure")` | Topology heatmap |
| `colour(sf, breaks, cmap=…)` | Map scalar → RGB for `plot2D` |
| `showModel(M)` | Event-tree graph (needs `networkx`, `matplotlib`) |

3D interactive: `curlew.visualise.napari_viewer.NapariViewer` (requires `pip install napari[all]`).

## Load and predict

```python
import curlew
from curlew.io import loadModel
from curlew.geometry import Grid

M = loadModel("models/site_a.pt")   # or in-memory GeoModel
G = Grid((2000, 1000), step=(20, 20), center=(1000, 500))
pred = M.predict(G)                 # Geode with .grid attached
```

## 2D section plots (matplotlib)

Always save PNG and inspect before reporting success.

```python
import matplotlib.pyplot as plt
from curlew.visualise import plot2D, plotAdjacency

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plot2D(pred.lithoID, pred.grid, ax=axes[0], cmap="tab20", levels=False)
axes[0].set_title("lithoID")

plot2D(pred.structureID, pred.grid, ax=axes[1], cmap="tab10", levels=False)
axes[1].set_title("structureID")

plot2D(pred.scalar, pred.grid, ax=axes[2], C=M["basement"].field.C if hasattr(M["basement"], "field") else None)
axes[2].set_title("scalar")

plt.tight_layout()
plt.savefig("validation_section.png", dpi=150)
```

Per-event scalar from `pred.fields`:

```python
plot2D(pred.fields["f0"], pred.grid, levels=20)
```

Combined / stacked scalars:

```python
pred2 = pred.stackValues(mn=0, mx=1)
plot2D(pred2.scalar, pred.grid)
```

## 3D model → 2D slice for validation

`plot2D` automatically treats a `Grid` with one dimension of size 1 as a **2D section** (drops singleton axes).

### Axis-aligned slice (fixed y, x–z section)

```python
import numpy as np
from curlew.geometry import Grid

# Full model extent (example)
dims_3d = (2000, 600, 1000)
center = (1000, 300, 500)

# Thin slab at y = center[1]: second dimension length 1
G_xz = Grid(
    dims=(dims_3d[0], 1, dims_3d[2]),
    step=(20, 600, 20),           # large y-step → single y layer at center
    center=center,
)
pred_slice = M.predict(G_xz)

from curlew.visualise import plot2D
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plot2D(pred_slice.lithoID, pred_slice.grid, ax=ax, levels=False)
ax.set_title("x–z section at y={:.0f}".format(center[1]))
plt.savefig("slice_xz_litho.png", dpi=150)
```

Swap axes for **map view** (fixed z):

```python
G_xy = Grid(dims=(2000, 600, 1), step=(20, 20, 1000), center=center)
pred_map = M.predict(G_xy)
plot2D(pred_map.structureID, pred_map.grid)
```

### Arbitrary plane (`geometry.section`)

For oblique sections (e.g. perpendicular to strike):

```python
from curlew.geometry import section, Grid
import numpy as np

origin = np.array([1000.0, 300.0, 500.0])
normal = np.array([0.0, 1.0, 0.0])   # plane normal → x–z section

sec_shape, pts = section(
    dims=(100, 80),                   # nx, ny in-plane
    origin=origin,
    normal=normal,
    width=2000,
    height=1000,
    step=20,
)

pred = M.predict(pts)
# Build a 2D Grid for plot2D: use sec_shape as Grid.shape via custom Grid or reshape manually
si = pred.lithoID.reshape(sec_shape)

import matplotlib.pyplot as plt
plt.imshow(si.T, origin="lower", aspect="auto")
plt.savefig("oblique_section.png", dpi=150)
```

For repeated validation, prefer axis-aligned `Grid` slabs — they attach `geode.grid` for `summary()` and topology.

## Drill traces

```python
import numpy as np

start = np.array([500.0, 0.0, 200.0])
end   = np.array([500.0, 1000.0, 800.0])
hole, contacts = M.drill(start, end, step=10.0)

print(hole.summary())
print(contacts)  # Geode at lithology contacts, with gradients if present

# On a 2D section figure:
from curlew.visualise import plotDrill2D
fig, ax = plt.subplots()
plot2D(pred_slice.lithoID, pred_slice.grid, ax=ax, levels=False)
plotDrill2D(hole, ax)
```

## Topology visualisation

```python
from curlew.visualise import plotAdjacency

plotAdjacency(pred, mode="lithology", title="Lithology contacts")
plotAdjacency(pred, mode="structure", connection="conservative")
```

## Napari 3D template (interactive)

Requires `napari[all]` and a display (or save screenshots from scripted views).

```python
import curlew
from curlew.geometry import Grid
from curlew.visualise.napari_viewer import NapariViewer

# Build or load model
# M = ...
G3 = Grid((1500, 600, 1000), step=(30, 30, 30), center=(750, 300, 500))
pred = M.predict(G3)

viewer = NapariViewer(title="Curlew model", ndim=3)
layers = viewer.addGeode(
    pred,
    lithoID=True,
    scalar=True,
    surfaces=True,
    displacement=False,
)

# Optional: overlay constraints
# viewer.addConstraints(M["basement"].field.C, grid=G3)

viewer.show()
try:
    import napari
    napari.run()   # keep Qt event loop alive in scripts
except ImportError:
    print("Install napari: pip install napari[all]")
```

2D Napari (section models):

```python
viewer = NapariViewer(title="Section", ndim=2)
viewer.addGeode(pred, lithoID=True, scalar=True, surfaces=True)
viewer.show()
```

`NapariViewer.addGeode` adds volume layers for 3D grids, contour polylines for 2D isosurfaces, and optional displacement vectors. Coordinate convention: napari **(z, y, x)** — handled internally when `grid=` is passed.

## Export

```python
from curlew.io import saveModel, savePLY

saveModel("models/site_a.pt", M)
# savePLY for external mesh tools — see curlew.io
```

## Validation checklist

```
- [ ] Section orientation matches geological intent (strike, dip direction, map view)
- [ ] lithoID unit count and names match brief
- [ ] structureID reflects expected event stack
- [ ] Faults/dykes visible on sections where they should cross-cut
- [ ] Topology adjacency: no impossible contacts (young-on-old without unconformity)
- [ ] PNG saved and inspected (do not rely on code-only assertions)
```

See also [geological-qa.md](geological-qa.md) for interpreting summaries and topology.
