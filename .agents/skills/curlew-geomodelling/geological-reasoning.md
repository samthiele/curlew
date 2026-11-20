# Geological reasoning for curlew models

See also: [context.md](context.md) · [structures.md](structures.md) · [environment.md](environment.md)

## History-first modelling

Curlew is **not** a single global implicit function. It is a **geological history**:

1. Decompose the user's description into **discrete events** (deposition, folding, faulting, intrusion, erosion/unconformity, etc.).
2. Order events from **oldest to youngest** in `GeoModel([e0, e1, ..., eN])`.
3. For each event, decide:
   - **Generative** (`overprint` set): creates new rock volume or replaces older scalar/lithology (strat, unconformity, intrusion envelope).
   - **Kinematic** (`deformation` only): moves coordinates; may not change lithology unless paired with overprint (dyke uses both).
   - **Combined**: e.g. `sheet` — offset host + overprint intrusion interval.

Evaluation walks **youngest → oldest**: younger deformations are removed via `undeform` before fitting or evaluating older fields.

## Core geological principles (Steno, Hutton, Lyell)

These are **universal rules** for inferring **relative time** (what happened before what). They constrain any `GeoModel` event list and justify validation checks (Parquer; Wellmann). They are independent of curlew’s scalar convention but must be respected by event order and constraints.

### Principle of superposition

In an **undeformed** sedimentary sequence, **younger beds lie on top of older beds**. The bottom of the pile is oldest; the top is youngest.

| Implication | Modelling |
|-------------|-----------|
| Vertical order of strata = time order | One `strati` event per depositional **package**; **units** as isosurfaces (tops), sorted younging upward |
| Younger unit has higher stratigraphic position | `iq`: points in upper unit **greater** scalar than points in lower unit (`'>'`) |
| Cannot bury young under old without an event | If model shows old rock above young, need **unconformity** (`strati` + `base`) or wrong event order |

**Curlew:** Superposition is **not** automatic from a single interpolated field—it is enforced by **event order** (older packages first in `GeoModel`), **overprints** at unconformities, and **`iq`** constraints between point sets.

### Principle of original horizontality

Sedimentary layers are deposited approximately **horizontal** (perpendicular to gravity at deposition). They may later be tilted, folded, or faulted.

| Implication | Modelling |
|-------------|-----------|
| Bedding poles ≈ depositional “up” at time of deposition | `gp`/`gv` (gradient) constraints on the owning `strati` |
| Smooth interpolators can create closed “bubbles” | Violates horizontality between data — use `mono_loss` on `grid` (Kamath) |
| Strongly tilted modern dips | Either post-depositional deformation (`fold`, `fault`) or measure in correct **paleo** frame via `undeform` |

At large scale, layers are often not exactly horizontal when deposited. Hence curlew `synthetic` examples often uses a `QuadraticField` to represent flat but non-planar deposition (in a broad basin).

**Curlew:** `HSet.mono_loss`, `thick_loss`, optional `trend` for regional dip; do not rely on value pins alone.

### Principle of lateral continuity (related)

Layers extend laterally until they **thin, pinch out, or meet a barrier** (facies change, unconformity, fault). Used when correlating horizons between boreholes or along strike.

| Implication | Modelling |
|-------------|-----------|
| Same named horizon should be one surface | `eq` traces: equal scalar along a contact line |
| Lateral termination | Often an **unconformity** or **domain boundary**, not a single global φ |

Pinch-out relations are difficult to represent in implicit geological models, but can be constructed by combining several stratigraphic series with onlap and/or unconformity relations between them.

### Cross-cutting relationships

**Definition:** Feature **A** cross-cuts feature **B** if A truncates or displaces B. **A is younger than B.**

Classic cross-cutting types:

| Relationship | Younger | Older | Rule |
|--------------|---------|-------|------|
| Fault cuts bedding | Fault | Strat | Fault event **after** `strati` it offsets |
| Dyke cuts strata | `sheet` | `strati` | `sheet` **after** host package |
| Unconformity truncates strata | Cover `strati` | Basement `strati` | Younger package with `base` on erosion surface |
| Fold affects bedding | `fold` (or post-fold deposit) | Pre-fold strat | `fold` before deposits that are not folded, or fold only pre-fold events |
| Intrusion contact | Intrusion / chamber | Host rock | Intrusion event younger than host |

Constrains **Timing → `GeoModel`:**

```
GeoModel([oldest_event, ..., youngest_event])   # list order = relative age
```

- Each **additional** cross-cutting structure adds a **younger** `GeoEvent` at the end of its branch (or after everything it cuts in a linear history).
- **Two faults:** order by which fault offsets the other’s trace (younger fault later)—see `anderson`.
- **Domain boundaries:** sub-histories on `lt` / `gt` are **not** globally comparable in age; only order **within** each branch and relative to the boundary event.

**Curlew:** Cross-cutting is **not** inferred automatically. You must:

1. Place events in correct order.
2. Use `undeform` so older fields are evaluated in coordinates valid **before** younger faults/intrusions.
3. Validate: younger structures must not appear in `parent` chain **below** (older than) what they cut.

### Other relative-age rules (brief)

Various other rules can be used to interpret age relations from 
reported observation data.

| Rule | Statement | Curlew analogue |
|------|-----------|-----------------|
| **Inclusions** | Clasts in rock A came from rock B → B older than A | Not built-in; model as separate units + younging `iq` |
| **Unconformity** | Gap in record + erosion surface | `strati(..., base=...)`, `mode="above"` |
| **Onlap** | Young beds lap onto older surface | `strati(..., onlap=True)` |
| **Uniformitarianism** (Lyell) | Present processes explain past | Informs choice of `fault`/`sheet` kinematics, not a separate class |

### Principles vs curlew scalar convention

Curlew often assigns **increasing** scalar φ toward **formation tops** within a package. That is a **modelling convention**, not a law of nature. What matters geologically:

- **Relative** order (inequalities, event stack, overprint).
- **Orientation** (gradients).
- **Contacts** (isosurfaces from seeds or equal scalar along traces).

Absolute φ values are arbitrary unless pinned—and pinning is discouraged for neural fields. However, in general, scalar fields that increase in the time direction (stratigraphic height) are preferred.

## Superposition and contacts in curlew

Implementation of the principles above:

- **Stratigraphic series:** one scalar field per package; younger units → higher φ toward tops (`strati` + `addIsosurface` as formation tops).
- **Unconformity:** younger `strati` overprints older only where φ **above** `base` (`mode="above"`); below `base`, parent geology remains.
- **Onlap:** `onlap=True` → basal unconformity geometry from **parent** (`Overprint.defaultDomain='parent'`).
- **Offlap / erosional top:** default `onlap=False` (child field defines erosional surface).

## Faults and shear zones

- Geometry: implicit surface `contact` (0, named isosurface, or list for parallel faults).
- Slip: `shortening` vector projected onto fault plane tangent; `offset` magnitude (fixed or learnable tuple).
- `width`: sigmoid sharpness — small ≈ brittle fault, large ≈ ductile zone; tuple `(core, ductile, proportion)` for mixed behaviour.
- `modifier`: name(s) of fields on the fault `GeoEvent` (e.g. `EllipsoidalField` from `addField`) scaling slip to zero outside a patch — **finite fault**; see **`goguel`**.
- `polarity`: which side moves (hangingwall vs footwall).
- Faults in list order should respect cross-cutting relations (younger fault later in list).

## Intrusions (dykes, sills)

- `sheet`: host field defines orientation; `contact=(lower, upper)` scalar bounds; `aperture` scales opening in gradient direction.
- Multiple parallel intrusions: `contact` and `aperture` as **lists** of matching length.

## Folds

- `fold` builds analytical bedding field + periodic displacement; place in event list where folding occurs relative to subsequent faulting/deposition.
- `estStrain=True` estimates bulk shortening to restore bed length.

## Domain boundaries

Use when **no** stratigraphic continuity across a surface (basin margin, fault with unknown offset correlation):

- `domainBoundary` with `gt` / `lt` as separate sub-model chains ending at the boundary event.
- Do not duplicate those sub-events in the top-level `GeoModel` list — they hang off `parent` / `parent2`.

## Translating natural language → events

| User says | Typical curlew construct |
|-----------|-------------------------|
| Layer-cake stratigraphy | `strati` + `LinearField`/`QuadraticField`/`NFF` + isosurfaces — `steno` |
| Angular unconformity on folded basement | `strati` → `fold` → `strati` (cover) — `hutton` |
| Dyke cross-cutting strata | `strati` then `sheet` — `playfair`, `walker` |
| Dyke swarm cross-cutting strata | `strati` then `sheet` with multiple dyke isosurfaces, `walker` |
| Normal / thrust / conjugate faults | `fault` (order by age) — `anderson`, `michell` |
| Finite normal fault (patch slip) | `fault` + `EllipsoidalField` modifiers — **`goguel`** |
| Listric fault | `fault` + `ListricField` — `seuss` |
| Basin fill onlap onto basement | `strati(..., onlap=True)` or `domainBoundary` |
| Magma chamber / intrusion domain | `domainBoundary` + constant `GeoEvent` — `seuss` |
| Pluton / caldera | Workarounds in [structures.md](structures.md) |

Full table: [structures.md](structures.md).

## Reasoning checklist for agents

1. Apply **superposition** and **cross-cutting** to build a relative timeline before coding.
2. List events **oldest → youngest** in `GeoModel` (branches at `domainBoundary` are ordered only within `lt` / `gt` chains).
3. Classify each event: generative (`overprint`), kinematic (`deformation`), or both (`sheet`).
4. Enforce **original horizontality** on depositional packages: `grad_loss` + `mono_loss`; reject bubble isosurfaces.
5. Bind constraints to the event that **owns** the geology being measured.
6. Use `iq` for young-on-old, `eq` for continuous contacts; minimise `vp` on neural fields.
7. Document assumptions: onlap, fault polarity, fold age vs post-fold cover.
8. Match a **synthetic** prototype when unsure (`structures.md` index).

## Constraint assignment

- Bedding orientation → `gp`/`gv` on the **event** that owns that stratigraphic package.
- Fault surface traces → `eq` / `gp`/`gv` on the **fault-plane sub-field** (`M['fault']['plane_name']`), not on ellipsoid modifier fields — see **`goguel`** sampling pattern.
- Contact picks → prefer `eq` traces or inequalities (`iq`) between point sets, not scattered `vp`.
- Younger-on-older: `iq` with `P_young > P_old` (relation `'>'`).
- Post-fault orientations: measure in **present** coordinates; fitting uses retro-deformation when `cache=True` in isolated `fit`.

## Interpretive and historical reasoning (Frodeman; Wellmann)

- Observations are **interpretation-laden** (you must recognise a bedding plane before measuring dip). Encode interpreter choices as explicit events and named isosurfaces, not implicit defaults.
- Explanation is **retrodictive**: infer past states from present geometry — matches `undeform` / youngest-first `GeoModel` evaluation.
- Prefer **relational** constraints (`iq`, `eq`) because scalar level is arbitrary (Kamath; Hillier et al.).

## Quality checks after build

1. `M.predict(grid)` — `structureID` reflects event IDs; litho names prefixed `{event}_{isosurface}`.
2. `M._repr_svg_()` — event graph (needs `networkx`, `matplotlib`).
3. Compare to synthetic truth via `extract_constraints` + re-fit if benchmarking.
4. Loss decreasing under `prefit` per event before joint `fit`.
5. **Knowledge consistency** (Parquer): no younger-on-older violations unless an erosion/unconformity event exists; dyke/fault order respects cross-cutting.
6. **Topology** (Thiele et al.): offlap/onlap/fault-ladder patterns in litho adjacency — flag impossible contact graphs.

## Uncertainty scenarios

- **Data:** jitter constraint locations, re-fit.
- **Interpolation:** multiple `seed` for `NFF`/`FSF` (Kamath RFF stochasticity).
- Report both scalar misfit (`Pebble`) and discrete topology class across ensemble.
