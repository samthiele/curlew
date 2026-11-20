# Extract geological data for curlew

**Goal:** Turn maps, reports, borehole logs, images, or GIS files into **curlew-ready constraints** and a **relative-age event list**. This file defines what curlew needs; use separate agent skills (vision, PDF, shapefile, CSV, spreadsheet) for the actual extraction.

## Output checklist

Before handing off to [synthetic-model.md](synthetic-model.md) or [interpolated-model.md](interpolated-model.md):

```
- [ ] Coordinate system and units documented (CRS, m vs ft). Prefer metric coordinates, up as +z (or +y in 2D) and east as +x. 
- [ ] Each array of observations mapped to a CSet field where possible (below)
- [ ] Observations assigned to the owning GeoEvent (strat package, fault, dyke, …)
- [ ] Relative age / superposition encoded (event order and/or iq)
- [ ] Named contacts for isosurfaces (formation tops, fault traces, dyke walls)
- [ ] Data saved in a loadable form (CSV, PLY, JSON, Python dict, CSet) where needed.
```

## What curlew needs: constraint types (`CSet`)

Bind each `CSet` to the **GeoEvent that owns** the geology being measured.

| CSet field | Geological observation | Typical source |
|------------|------------------------|----------------|
| `gp`, `gv` | Bedding / foliation **pole** (gradient direction) | Dip/strike tables, structural measurements, map symbols |
| `gop`, `gov` | unsigned gradient orientations; use for strike/dip measurements with unknown younging  | Dip/strike tables with unclear younging direction, foliations |
| `eq` | Points on the **same contact** (equal scalar along a trace) | Mapped contacts, digitised polylines, seismic horizons, traces |
| `iq` | **Stratigraphic inequality** (younger above older: `'>'`) | cross-section interpretation, stratigraphic relations / columns |
| `vp`, `vv` | Fixed scalar at a point | Rare; prefer `eq` or seed-based isosurfaces |
| `pp`, `pv` | Property targets (density, grade, …) | Assay, geophysics, petrophysics |
| `grid`, `delta` | Domain for global losses (`mono_loss`, `thick_loss`, …) | Model extent from data, map or GIS bounds |
| `trend` | Regional depositional “up” direction | Basin-scale dip, general fault orientation vector |

Full API: [api-reference.md](../api-reference.md) § CSet.

**Prefer** `gp`/`gv`, `eq`, and `iq` over scattered `vp` on neural fields (`NFF`/`FSF`).

## Stratigraphic and age relationships

Curlew does **not** infer age automatically. Encode geology explicitly:

| Relationship | How to encode |
|--------------|---------------|
| Conformable stack (A below B below C) | One `strati` event; isosurfaces as formation **tops**; `iq`: points in B `>` points in A |
| Unconformity (younger on eroded older) | Younger `strati` with `base=` erosion isosurface; event **after** basement |
| Onlap | `strati(..., onlap=True)`; cover event after surface being onlapped |
| Fault cuts strata | `fault` event **after** all units it offsets |
| Dyke / vein cross-cuts | `sheet` event **after** host `strati` |
| Cross-cutting pair (fault A cut by fault B) | Older fault first in `GeoModel`, younger second |

See [geological-reasoning.md](../geological-reasoning.md) for principles and [structures.md](../structures.md) for factory mapping.

## Event list (deliverable)

Produce a draft **oldest → youngest** event list before coding:

```text
1. basement_strati — units: …
2. fold_d1 — kinematic (optional)
3. cover_strati — base=erosion_surface, units: …
4. normal_fault_f1 — contact at …
```

## Faults

Faults require an implicit (or analytical) field to represent fault surface geometry, and a shortening direction. Shortening direction should be determined from fault geometry if the type of fault is known: e.g., normal fault, thrust fault, dextral fault, synistral fault, left-lateral fault, right-lateral fault. 

Some text state  regional shortening direction at different times; this informs the slip on faults of this age.

**Fault offset is often difficult to determine. If not mentioned, stop data extraction and prompt the user for input on the expected fault slip magnitude.**

## Common loading patterns

### CSV / spreadsheet (orientations, contacts, collars)

Typical columns → curlew:

```python
import numpy as np
import pandas as pd
from curlew.core import CSet

df = pd.read_csv("bedding.csv")  # x, y, z, dip, dip_dir  OR  gx, gy, gz
gp = df[["x", "y", "z"]].to_numpy()
gv = df[["gx", "gy", "gz"]].to_numpy() # gradient specified directly. If not, convert from strike/dip (RHS) or dip-dip_dir angles. 
gv /= np.linalg.norm(gv, axis=1, keepdims=True)
C_strat = CSet(gp=gp, gv=gv, crs="global")
```

Contact polylines (one row per point, `contact_id` groups traces):

```python
traces = []
for cid, grp in df.groupby("contact_id"):
    traces.append(grp[["x", "y", "z"]].to_numpy())
C_strat = CSet(eq=traces, crs="global")
```

Inequalities from borehole stratigraphy:

```python
# Points in unit B must have higher scalar than unit A at same location
iq = (1024, [(P_young, P_old, ">")])
C_strat = CSet(iq=iq, crs="global")
```

### GIS / shapefile

Use a **shapefile or GeoPackage skill** to read geometries, then convert:

- **Points** → `gp`/`gv` (if gp, gv, strike, dip or dip-dir attributes exist) or isosurface **seeds** if points sit along a single contact.
- **Lines** → `eq` traces (resample to ~uniform spacing along polyline. Downsample to less than 100 points per trace).
- **Polygons** → unit labels for `iq` or domain masks (via `addVolume` later)

Ensure exported coordinates match the model CRS (`crs='global'` unless `GeoModel` transform is applied).

### Reports and PDFs

Use a **PDF skill** to extract tables (dip lists, strat columns, collar surveys). Treat narrative text as interpretive — confirm event order with the user when ambiguous.

### Maps and images

Use a **vision skill** to digitise contacts, faults, and unit boundaries. Record:

- Pixel → world transform (scale bar, grid lines, or georeferenced raster)
- Which side of each contact is younger
- Approximate dip symbols → `gv` vectors

Cross-check digitised geometry against any tabulated measurements in the same document.

## Handoff to modelling

| Next step | When |
|-----------|------|
| [synthetic-model.md](synthetic-model.md) | Sparse or no data; prototype-first to test logic |
| [interpolated-model.md](interpolated-model.md) | Enough constraints to fit `NFF`/`FSF` or analytical fields |

Always attach: `CSet` per event name, proposed `GeoModel` event order, isosurface **names** and seed points, model extent (`Grid` dims/center/step).

## External extraction skills

This skill does **not** perform OCR, vision, or GIS I/O. After reading this file, invoke the appropriate platform skill, for example:

- PDF / document table extraction
- Image / map digitisation (vision)
- CSV / Excel parsing
- Shapefile / GeoPackage / GeoJSON

Then map outputs into `CSet` fields using the tables above.
