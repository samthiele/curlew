# Geological structures → curlew GeoEvents

Structure types → factory functions, interactions, fields, and `curlew.synthetic` benchmarks. See also [context.md](context.md) · [api-reference.md](api-reference.md)

## Quick reference

| Structure | GeoEvent factory | Interaction | Typical field | Synthetic example |
|-----------|------------------|-------------|---------------|-------------------|
| Stratigraphic series / units | `strati` | `Overprint` (`base`, `mode="above"`) | `QuadraticField`, `NFF`, `FSF` | `steno`, `hutton` (s0), `seuss` (s0, s2) |
| Unconformity (erosional) | `strati` (younger package) | `Overprint` + finite `base` | same | `hutton` (s1, `base=0`) |
| Onlap | `strati(..., onlap=True)` | `Overprint`, `defaultDomain='parent'` | same | Build from `hutton`; set `onlap=True` on cover |
| Offlap | Younger truncates older tops | `strati` + isosurfaces | same | Topology per Thiele et al.; same as unconformity stack |
| Fold (kinematic) | `fold` | `FoldOffset` | `LinearField` + periodic wave | `hutton` (`d1`); `lehmann` uses `PeriodicField` only |
| Fold (implicit bedding) | `strati` | — | `PeriodicField` | `lehmann` |
| Normal fault | `fault` | `FaultOffset` | `LinearField` | `anderson`, `playfair`/`walker` (`addFault=True`), `goguel` (finite) |
| Thrust / reverse fault | `fault` | `FaultOffset` | `LinearField` | `michell` (`shortening` ≈ horizontal) |
| Conjugate fault set | Multiple `fault` events | Each `FaultOffset` | `LinearField` each | `anderson` (s1 + s2, different dips) |
| Listric fault | `fault` | `FaultOffset` | `ListricField` | `seuss` (`f1`, `f2`) |
| Ductile shear zone | `fault` | `FaultOffset`, large `width` | `LinearField` / interpolated | Tune `width`; optional `(core, ductile, proportion)` tuple |
| Dyke | `sheet` | `SheetOffset` + `Overprint` `mode='in'` | Steep `LinearField` | `playfair`, `walker`, `seuss` (`s1`) |
| Sill / sheet intrusion | `sheet` | same | Subhorizontal `LinearField` | Same factory; gradient ≈ bedding |
| Vein | `sheet` | same, narrow `contact` | `LinearField` | Narrow `contact` in `playfair` |
| Pluton / stock / batholith | `stock` (**stub**) | — | — | **Not implemented** — see workarounds below |
| Magma chamber (subsurface body) | `domainBoundary` + constant `GeoEvent` | Mask `lt`/`gt` | `QuadraticField` boundary | `seuss` (`intrusion` = -2, `d1`) |
| Calderas / volcanic subsidence | — | — | — | **No dedicated factory** — composite workaround |
| Domain boundary (no strat continuity) | `domainBoundary` | `Overprint` on boundary field | `LinearField` / `QuadraticField` | `seuss` (`d1`, `d2`) |
| Finite fault (limited slip) | `fault` + `modifier` or `addVolume` | `FaultOffset` | `EllipsoidalField` per patch (`addField`) | **`goguel`**; `tests/test_geology.py`, `tests/test_analytic.py` |

> `finiteFault` was **removed** on `dev`. Use `fault(..., modifier=['fe0', ...])` with an
> `EllipsoidalField` attached via `addField` instead.

### Folding-dominated, topologically constrained cases

When folds are tight enough that a smoothness-based interpolant would create
closed isosurfaces, use the restoration pathway instead of the kinematic row
above. `restore(...)` models the stratigraphy as a flat reference layer-cake
carried by a volume-preserving diffeomorphism, so topology is preserved by
construction.

| Structure | Factory | Notes | Reference |
|---|---|---|---|
| Folded sequence (any tightness) | `restore` + `euler_velocity` | Read `length_scale_range` with `spectral_band` | [restoration.md](restoration.md) |
| Refolded / isoclinal folds | `restore` (may set `depth_axis` for vertical reference) | Reference orientation is an optimisation choice | [restoration.md](restoration.md) |
| Restored + strain data | `restore` + `constraints=` | Strain markers constrain the map directly | [restoration.md](restoration.md#thickness-and-strain-priors) |
| Faulted + folded | `restore` + `FaultLift`, or `domainBoundary` split | True discontinuities need lifting or separate events | [restoration.md](restoration.md#faults) |

**Faults, dykes, and unconformities still use the kinematic factories above.**
Restoration handles folding only; it cannot represent a topological
discontinuity inside a single map.

**Event order:** always list structures **oldest → youngest** in `GeoModel([...])` (see [geological-reasoning.md](geological-reasoning.md) — superposition, cross-cutting, and timing). Faults and dykes that cross-cut strat must be **younger** than the units they offset.

---

## Stratigraphy

### Stratigraphic series and units

A **series** is one depositional `GeoEvent` with a single implicit scalar field \(\phi\). **Units** (formations, members) are **not** separate events, rather they are **isosurfaces** on that field, interpreted as formation **tops** (\(\phi \ge c\)) when assigning `lithoID`. Lithologies within a single series are **conformable**.

Note that bedding (sedimentary layer) orientation is measured based on its tangent plane relative to horizontal. This is at 90 degrees to the gradient vector used  during implicit field modelling.

Hence, *horizontal bedding* implies a vertical gradient, while *vertical bedding* implies a horizontal gradient.

```python
s0 = strati("basement", C=QuadraticField(...), base=-np.inf)
for seed, name in _isosurface_seeds(extent, ndim, 5):
    s0.addIsosurface(name, seed=seed)  # prefer seed over fixed value
```

| Concept | Curlew |
|---------|--------|
| Layer-cake bedding | One `strati`; many `addIsosurface` |
| Younging direction | `CSet.gp`/`gv` on this event’s field |
| Unit contacts | Named isosurfaces (`i0`, `S1`, …) |

**Synthetic:** `steno` — simple curved layer-cake. `seuss` — `s0` / `s2` with `nlayers` isosurfaces.

**Fitting:** `NFF`/`FSF` + `HSet(grad_loss, mono_loss)`; avoid relying on `value_loss` alone. Bubbles (non-monotonic scalar fields)
are geologically invalid representations of a stratigraphic series,
due to the principle of superposition and principle of original
horizontality.

---

### Unconformities

An **unconformity** is modelled as a **younger** `strati` that **overprints** older geology only where \(\phi_\text{new} > \text{base}\).

```python
s_cover = strati("cover", C=..., base="erosional_base", mode="above")  # or base=0.0
s_cover.addIsosurface("erosional_base", seed=erosion_point)
```

| Parameter | Meaning |
|-----------|---------|
| `base=-np.inf` | No cut-off; entire field is new package (basal strat) |
| `base=0` or named iso | Erosion surface; below = parent geology preserved |
| `mode="above"` | Standard erosional unconformity |

**Synthetic:** `hutton` — folded `s0`, fold event `d1`, cover `s1` with `base=0`.

---

### Onlap and offlap

**Onlap:** younger beds lap onto an **older** erosional surface; the **basal contact geometry** follows the **parent** field.

```python
cover = strati("cover", C=..., base="unconformity", onlap=True)
```

Sets `Overprint.defaultDomain = 'parent'`.

**Offlap:** younger units truncate older ones at a surface—same event stack as unconformity; topology differs (Thiele et al., 2016a). Model with younger `strati` + isosurfaces; verify adjacency in `lithoID`.

**Synthetic:** No dedicated builder; start from `hutton` and set `onlap=True` on the cover event.

---

## Faults

All faults use `fault(name, C=..., shortening=..., contact=..., offset=..., width=...)`. Kinematic only (**no lithology overprint** unless you add one manually). Slip direction: `shortening` projected onto fault-plane tangent (`FaultOffset`).

### Normal faults

Extensional: hangingwall drops relative to footwall. Use **vertical or near-vertical** `shortening` (slip driven by horizontal extension). Normal faults typically dip at ~60 degrees to horizontal as they form under an extensional stress regime (vertical sigma 1).

```python
fault("fn", C=LinearField(..., gradient=fault_plane_normal),
      shortening=np.array([0, -1]),  # 2D example
      contact=0, offset=100, width=1e-5)
```

| Knob | Effect |
|------|--------|
| `width` small | Brittle fault |
| `width` large | Ductile shear zone |
| `offset=(v, min, max)` | Learnable slip |
| `polarity` (on `FaultOffset`) | Which block moves |

**Synthetic:** `anderson` — two normal faults (`s1`, `s2`) different positions/dips. `playfair` / `walker` with `addFault=True` — single normal fault after strat + dyke.

---

### Thrust / reverse faults

Compressional: slip vector roughly in **transport direction** (horizontal component dominant in section models). Thrust faults typically dip at ~30 degrees to horizontal as they form under a compressive stress regime (vertical sigma 3).

```python
fault("thrust", C=LinearField(...),
      shortening=np.array([-1, 0]),  # 2D: horizontal compression
      offset=100, width=1e-5)
```

**Synthetic:** `michell` — strat + thrust `s1`.

---

### Strike-slip faults

Faults formed due to plates sliding past each other. Slip is largely horizontal. Dip is typically steep (>80 degrees).

Offset is dextral if slip field has a clockwise (top to the right transport) vorticity and synistral if anticlockwise (top to the left transport).

### Conjugate faults

Two (or more) faults with intersecting or paired orientations, **same or different** slip. Each fault = **separate** `GeoEvent`, ordered by cross-cutting age (older fault first if it is cut by the younger).

```python
f_old = fault("f1", ..., offset=offset1, ...)
f_young = fault("f2", ..., offset=offset2, ...)
GeoModel([strat, f_old, f_young])
```

**Parallel multi-faults** on one surface: pass `contact`, `offset`, `width` as **lists** to a single `fault()` (shared implicit surface, different slips).

**Synthetic:** `anderson` — conjugate-style **normal** pair; pattern applies to thrust pairs too.

---

### Listric faults

Curved fault that flattens with depth. Use **`ListricField`** as the fault surface, then `fault(..., C=ListricField(...))`.

```python
fault("listric", C=ListricField("ff", input_dim=2,
      origin=[x0, z0], fault_floor=0, fault_ceil=700,
      curvature_rate=0.006),
      offset=100, shortening=[1, -1], n_steps=3, width=1e-9)
```

Increase `n_steps` for strongly curved surfaces (Euler integration).

**Synthetic:** `seuss` — `f1` (older listric) and `f2` (younger), with `domainBoundary` unconformities.

---

### Finite faults and fault zones

**Infinite fault:** default `fault` — slip everywhere on the isosurface (sigmoid across contact).

**Finite fault (limited slip patch):** `FaultOffset` multiplies displacement by a **modifier field** (name passed to `modifier=`, typically from `addField`). Values are in \([0,1]\) at the fault centre and taper to zero outside the patch. Do **not** sample constraints on modifier fields—only on the shared fault-plane field.

**Pattern (reference: `curlew.synthetic.goguel`):**

```python
# Shared plane; list contact / offset / modifier for parallel finite faults on one surface
s1 = fault(
    "s1",
    type=LinearField,
    origin=...,
    gradient=...,              # plane normal; dip toward +x: (sin(dip), cos(dip)) in 2D section
    shortening=(0, -1),        # normal fault: vertical shortening → top-down slip
    offset=[slip0, slip1, ...],
    contact=["ff0", "ff1", ...],
    width=0,                   # brittle
    modifier=["fe0", "fe1", ...],
    n_steps=1,
)
for i, seed in enumerate(fault_seeds):
    s1.addIsosurface(f"ff{i}", seed=seed, field="s1")
    s1.addField(
        f"fe{i}",
        field=EllipsoidalField(
            f"fe{i}",
            input_dim=ndim,
            origin=seed,
            axes=[major, 0.75 * major],
            directions=[strike, normal, ...],  # 2D: [tangent, plane normal]
            decay="cosine",                    # or 'ellipse', 'normal', 'wendland', callable f(u,r), ...
        ),
    )
GeoModel([strati, s1], grid=G)  # strat older than fault
```

| `EllipsoidalField.decay` | Role |
|--------------------------|------|
| `False` | Elliptical distance (not used as modifier) |
| `'ellipse'` / `True` | Linear taper `max(0, 1−r)` |
| `'normal'`, `'cosine'`, `'wendland'`, `'sigmoid'`, `'inverse_quadratic'` | Smooth tapers; `'cosine'` is default in **`goguel`** |
| `'quadratic'` | `r²`: 0 at the centre, 1 on the unit ellipse, grows outside |
| Callable `f(u, r)` | Custom taper; `u` = ellipse-aligned position (world units), `r` = elliptical radius |

**Constraint sampling (`goguel` / `michell` pattern):** `sample(M.predict(G))` for bedding on `s0` + global property; `sample(s1.predict(G), breaks=[getIsovalue('ff0'), ...])` for fault traces on **`s1` only**; bind `M['s0'].field`, `M['s1']['s1']`, `M` (property). Zero fault `vv` if orientations only.

**Alternatives:** `addVolume("fault_patch", expr)` — boolean mask; combine with interpretation.

**Shear zone:** same `fault`, large `width` or `width=(core, ductile, proportion)`.

**Synthetic:** **`goguel`** — horizontal `s0`, three right-dipping normal finite faults on one `s1` event; `tests/test_geology.py` (`test_isosurfaces_and_volumes`, multi-field events); `tests/test_analytic.py` (`EllipsoidalField` decay modes).

---

## Intrusions (sheet geometry)

`sheet` models **tabular** intrusions: implicit wall pair + **Mode I opening** (`aperture`) along host \(\nabla\phi\).

```python
sheet("dyke", C=LinearField(..., gradient=steep), contact=(-w, w), aperture=2)
```

| Type | Geometry hint | Curlew |
|------|---------------|--------|
| **Dyke** | Steep `LinearField` gradient | `sheet`, default `playfair` |
| **Sill** | Gradient subparallel to bedding | `sheet`, shallow `LinearField` |
| **Vein** | Narrow `contact` interval | `sheet`, small half-width |
| **Dyke swarm** | `contact=[(a,b), (c,d), ...]`, `aperture=[...]` | `walker` |

**Interactions:** `SheetOffset` (displace host) + `Overprint(mode='in')` per wall pair.

**Synthetic:** `playfair` (one dyke), `walker` (many), `seuss` (`s1`).

**Not sheet-like:** subvertical **plutons**, **laccoliths**, **magma chambers** — see below.

---

## Igneous bodies (volumetric)

### Plutons, stocks, batholiths

`stock(name, C, H, contact=0)` is a **stub** — not implemented.

**Workarounds:**

| Approach | Pattern |
|----------|---------|
| **Ellipsoidal intrusion** | `strati(..., mode="below")` with `C=EllipsoidalField(...)` or constant body + overprint |
| **Domain + constant lithology** | `GeoEvent("pluton", field=-1, type=int)` + `domainBoundary(..., lt=[pluton], gt=[host])` |
| **Volume mask** | `addVolume("pluton", "(dist_field < 1)")` on ellipsoid distance field |

See `tests/test_geology.py`, `tests/test_analytic.py`, and **`curlew.synthetic.goguel`** for `EllipsoidalField` (including finite-fault modifiers).

---

### Magma chambers

Subsurface **volumetric** melt region, often with internal stratigraphy absent. **`seuss`** pattern:

```python
intrusion = GeoEvent("i0", field=-2, type=int)  # constant scalar package
d1 = domainBoundary("d1", C=QuadraticField(...), bound=0,
                    lt=[intrusion], gt=[s0, s1])  # below boundary = intrusion
```

The boundary field’s isosurface separates chamber from host + earlier intrusions.

---

### Calderas

Ring faults, subsidence block, intracaldera fill — **no single curlew factory**. Build as **ordered** `GeoModel`:

1. Pre-caldera strat (`strati`)
2. Ring faults (`fault` list or domain boundaries)
3. Collapse / fill package (`strati` with `base` = caldera floor iso)
4. Optional `sheet` dykes radial to centre

Use `domainBoundary` where fault displacement breaks stratigraphic correlation across blocks.

---

## Folds

Two complementary approaches — **do not combine** on the same rock package (see SKILL anti-patterns).

| Approach | Factory | Use when |
|----------|---------|----------|
| **Implicit folded bedding** | `strati` + `PeriodicField` | Layered series with wave-like \(\phi\) |
| **Kinematic fold** | `fold(...)` | Undo folding via `FoldOffset`; needs `origin`, `compression`, `extension`, `wavelength` |

```python
# Kinematic (hutton)
d1 = fold("d1", origin=..., compression=..., extension=...,
          wavelength=2000, amplitude=250, sharpness=0.7)
GeoModel([s0, d1, s1])
```

**Synthetic:** `lehmann` — `PeriodicField` strat only. `hutton` — basement + `fold` + unconformable cover.

---

## Domain boundaries

Where **footwall and hangingwall are not one continuous stratigraphic series** (basin edge, intrusion wall, post-tectonic cover above faulted stack):

```python
d2 = domainBoundary("d2", C=boundary_field, bound=0,
                    lt=[older_branch],   # footwall chain (ends with oldest GeoEvent)
                    gt=[younger_cover])  # hangingwall chain
GeoModel([d2, f_young])  # youngest root(s) listed; branches linked via lt/gt
```

`lt` / `gt` may be floats (constants), single `GeoEvent`, or **lists** forming sub-histories (`seuss`: `lt=[d1, f1]`, `gt=[s2]`).

**Synthetic:** `seuss` — intrusion boundary `d1`, unconformity `d2` over faulted + intruded pile.

---

## Structure → constraint checklist

When building from a geological description:

1. Name each **event** (deposition, fold, fault, intrusion, unconformity).
2. Pick **factory** from tables above.
3. Add **isosurfaces** (`seed` from contacts) for units and named `base`/`contact`.
4. Bind **CSet** per event (orientations on the owning package).
5. Order `GeoModel` oldest → youngest.
6. Pick closest **`synthetic.*`** prototype and `extract_constraints` if training neural fields.

## Synthetic model index

| Function | Structures modelled | `GeoModel` event order (oldest → youngest) |
|----------|---------------------|------------------------------------------|
| `steno` | Layer-cake strat | `[s0]` |
| `lehmann` | Folded strat (`PeriodicField`) | `[s0]` |
| `hutton` | Strat + fold + unconformity | `[s0, fold d1, s1]` |
| `playfair` | Strat + dyke [+ normal fault] | `[s0, s1]` or `[s0, s1, s2]` |
| `walker` | Strat + dyke swarm [+ fault] | `[s0, s1]` or `[s0, s1, s2]` |
| `michell` | Strat + thrust | `[s0, s1]` |
| `anderson` | Strat + 2 normal faults | `[s0, s1, s2]` |
| `seuss` | Strat, dyke, intrusion domain, listric faults, cover, unconformity domain | `[d2, f2]` (tree: see `synthetic.seuss`) |

Load and inspect: `M = curlew.synthetic.hutton(shape=(2000, 1000)); print(M)`.
