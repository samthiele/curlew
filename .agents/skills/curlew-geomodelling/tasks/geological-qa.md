# Geological reasoning and model interrogation

**Goal:** Answer geological questions, interpret model results, and explain relationships — with or without building new code. When a model exists, use curlew’s text summaries, topology, and drill tools.

## Background reading

| Resource | Use |
|----------|-----|
| [geological-reasoning.md](../geological-reasoning.md) | Superposition, cross-cutting, event order, constraint assignment, quality checks |
| [context.md](../context.md) | Implicit modelling theory, isosurfaces, literature links |
| [structures.md](../structures.md) | Structure types → curlew factories |
| Rey (2016) structural geology textbook | [Patrice Rey iBook](https://geoscience-prey.sydney.edu.au/iBooks/Patrice_Textbook.pdf) — planar/linear structures, deformation, workflow foundations |

For **building** or **changing** models, switch to [synthetic-model.md](synthetic-model.md) or [interpolated-model.md](interpolated-model.md). For **plots**, see [visualise-model.md](visualise-model.md).

## When to answer vs when to model

| Question type | Approach |
|---------------|----------|
| “What is an unconformity?” / “Which is younger, fault or dyke?” | Answer from principles ([geological-reasoning.md](../geological-reasoning.md)); cite rules |
| “Does this model respect cross-cutting?” | Load model → `summary()` + topology + sections |
| “Build a model of …” | Not Q&A — route to synthetic or interpolated task |

Environment probe: **optional** for pure conceptual Q&A; **required** if running `predict`, `drill`, or plots.

## Interrogating an existing model

```python
from curlew.io import loadModel
from curlew.geometry import Grid

M = loadModel("models/site_a.pt")
print(M)                    # GeoModel event graph summary

G = Grid((2000, 1000), step=(20, 20), center=(1000, 500))
pred = M.predict(G)

print(pred.summary())       # volumes, topology, warnings
print(pred.lithoLookup)
print(pred.structureLookup)
```

Per-event inspection:

```python
for name in ["basement", "cover", "fault1"]:
    ev = M[name]
    print(ev)
    if ev.field and ev.field.C:
        print(ev.field.C)
```

Loss state after fitting:

```python
from curlew.text import pebble_str
# after M.fit → pebble objects in return dict or event.pebble
```

## Drill: 1D geological log through the model

```python
import numpy as np

start = np.array([x0, y0, z0])
end   = np.array([x1, y1, z1])
hole, contacts = M.drill(start, end, step=5.0)

print(hole.summary())
# contacts: Geode at lithology boundaries with orientations (if gradient=True path used)
```

Use drill to answer: *Which units does this path intersect, in order?* *Where are contacts?* Compare to expected stratigraphy from the brief.

## Topology adjacency — what it tells you

`Geode.topology()` computes **contact area** between lithology or structure classes on a regular grid. It is the primary discrete validation tool beyond scalar misfit.

```python
A_litho = pred.topology(mode="lithology", output="matrix")
A_struct = pred.topology(mode="structure", output="matrix")

topo_dict = pred.topology(mode="lithology", output="dict", connection="conservative")
# dict[a][b] = shared face area between classes a and b
```

**How to read it:**

- **Non-zero off-diagonal entries** — those lithologies/structures share a boundary in the model volume.
- **Missing expected contact** (e.g. cover should touch basement across unconformity) — wrong event order, bad `base`, or fit failure.
- **Forbidden contact** (e.g. young unit below old without erosion event) — geological inconsistency ([geological-reasoning.md](../geological-reasoning.md) § quality checks; Thiele topology papers in [context.md](../context.md)).
- **`connection="conservative"`** (default) — shared **faces** only (4-/6-connected); stricter QC.
- **`connection="inclusive"`** — includes edge/corner touches; more permissive.

Human-readable summary (included in `pred.summary()`):

```text
Lithology topology (conservative contact area):
  basement_i0 — basement_i1: 1.234e+05
  cover_i0 — basement_i2: 4.56e+04
  ...
```

Plot:

```python
from curlew.visualise import plotAdjacency
plotAdjacency(pred, mode="lithology")
```

Compare **structure** topology to verify the event graph matches interpretive history (which `GeoEvent` bounds touch which).

## Typical reasoning questions

| Question | Where to look |
|----------|---------------|
| Relative age of fault vs dyke | Event order in `print(M)`; cross-cutting table in [geological-reasoning.md](../geological-reasoning.md) |
| Is unconformity erosional or onlap? | `Overprint` mode, `onlap` flag on `strati`; cover `base` isosurface |
| How many formations? | `len(np.unique(pred.lithoID))`, `lithoLookup` |
| Does fold affect cover? | `structureID` on section; fold event position in list |
| Model uncertainty | Re-fit with different `seed` on NFF; compare topology across ensemble |

## Reporting format

When answering geological questions about a model, structure the response:

1. **Interpretive summary** — geologist-readable history implied by the model.
2. **Evidence** — event list, key topology contacts, section or drill observations.
3. **Caveats** — assumptions (polarity, onlap, pinching), data gaps, fit quality.
4. **Suggested next step** — if inconsistent, what to change (event order, constraints, isosurface seeds).

## Anti-patterns

- Asserting age relationships without checking event order or topology.
- Ignoring `WARNING` lines in `pred.summary()` (negligible units, undefined lithoID -1).
- Using scalar plots alone when discrete topology is wrong.
