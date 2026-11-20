---
name: curlew-geomodelling
description: >-
  Builds implicit geological models with curlew (GeoModel, GeoEvent, neural/analytical
  fields, faults, unconformities, dykes, folds). Use when constructing or editing
  geological models from descriptions, constraints, or data via tools in the curlew
  repository; extracting geological data from maps, reports, or images; visualising
  or validating models; answering geological reasoning questions; or extending the
  curlew codebase. Trigger terms: curlew, implicit modelling, synthetic model,
  interpolated model, geological model, drillhole, topology, napari,
  diffeomorphic, restoration, LDI.
---

# Curlew geomodelling

Curlew models geology as a **time-ordered graph of events** (`GeoEvent` + `GeoModel`).

Docs: https://samthiele.github.io/curlew/dev/ · Source: https://github.com/samthiele/curlew/tree/dev

## Workflow

```
Task progress:
- [ ] Step 1: Verify Python environment (if code will run)
- [ ] Step 2: Classify task type
- [ ] Step 3: Follow the task playbook
- [ ] Step 4: Validate (types 2–3: always; type 5: inspect outputs)
```

### Step 1: Python environment

Before curlew/PyTorch code, `pytest`, or training, follow **[environment.md](environment.md)**.

```bash
python scripts/check_env.py
```

On `MISSING_DEPS`, stop and ask the user — do not `pip install` without consent. After `SUCCESS`, reuse the printed `PYTHON:` for the session.

**Skip the probe** for pure geological Q&A with no code (type 6). Defer for data extraction (type 1) until curlew code is needed.

### Step 2: Classify the task

| Type | When | Playbook |
|------|------|----------|
| **Extract data** | Map, PDF, image, spreadsheet, or report → constraints for modelling | [tasks/extract-data.md](tasks/extract-data.md) |
| **Synthetic model** | Text description, sketch, or image; no site data | [tasks/synthetic-model.md](tasks/synthetic-model.md) |
| **Interpolated model** | Real orientations, contacts, boreholes, geologist knowledge | [tasks/interpolated-model.md](tasks/interpolated-model.md) |
| **LDI / restored model** | Folds where spurious extrema ("bubbles") are a risk; strain data; topology required by construction | [tasks/restored-model.md](tasks/restored-model.md) |
| **Visualise / extract** | Model exists; plots, sections, drills, exports | [tasks/visualise-model.md](tasks/visualise-model.md) |
| **Geological Q&A** | Explain concepts, interpret relationships, no build requested | [tasks/geological-qa.md](tasks/geological-qa.md) |
| **Extend codebase** | Change curlew source, tests, or API | [tasks/extend-codebase.md](tasks/extend-codebase.md) |

If ambiguous, ask one clarifying question. Many requests are **sequences** (e.g. extract → interpolate → visualise).

**Validation** after building (types 2–3) is mandatory — see the validate step in each playbook; do not treat it as a separate task type.

## Core model (all build tasks)

```
GeoModel → GeoEvent → field (BaseSF) + deformation? + overprint?
```

- **Event order:** `GeoModel([...])` lists events **oldest → youngest**.
- **Outputs:** `M.predict(x)` → `Geode`.
- **Training:** `M.prefit` (per event), then `M.fit`.
- **IO:** `saveModel` / `loadModel`.
- **Attributes:** `M.events` (was `M.fields`), `M.getEvent(eid)` (was `getField`).

### Naming: `GeoEvent`, not `GeoField`

`GeoField` no longer exists. The class is `GeoEvent` (`curlew/geology/geoevent.py`)
and the class it replaced was deleted. If you produce code using `GeoField`,
`M.fields`, `getField`, `VFieldOffset`, or `finiteFault`, it is for an obsolete
API — see [api-reference.md](api-reference.md) § Migration.

### Two ways to build stratigraphy

1. **Kinematic / structural** (default) — `strati`, `fault`, `fold`, `sheet`,
   `domainBoundary`. A scalar field is interpolated and then *deformed* by
   offsets. Fast and flexible, but cannot guarantee topological admissibility.
2. **Learnable Diffeomorphic Interpolation (LDI)** — `restore(...)`. Fitting
   and restoring are the same map. Present-day stratigraphy is a reference
   layer-cake carried by a learned diffeomorphism `Φ`,
   `ψ(x) = ψ_ref(Φ⁻¹(x))`. Because `Φ` is smooth and invertible, `ψ` cannot
   form closed isosurfaces the reference does not have. Volume preservation
   comes from the velocity (`v = ∇α × ∇β`), not from a penalty. Contacts,
   bedding and finite strain all constrain that one map. Where data are dense
   the fit is a little looser than an exact interpolant; that trade is
   intended.

Choose (1) for most work. Choose (2) when folds are tight enough that a
smoothness-based interpolant would create closed isosurfaces, or when
finite-strain data must constrain the fit. Wavelengths, the shear integrator
and worked examples are in [restoration.md](restoration.md); the checklist is
[tasks/restored-model.md](tasks/restored-model.md).

## Anti-patterns

- Running code before environment probe succeeds.
- Wrong event order; duplicate isosurface names; one φ for whole history.
- Global losses (`mono_loss`, `thick_loss`, `trend_loss`, …) without `grid`.
- Fold implicit field (`PeriodicField`) + kinematic folding on the same package.
- Using the old API names (`GeoField`, `M.fields`, `VFieldOffset`).
- Expecting `restore` to cut a true fault inside one map. Use `FaultLift`
  or a separate event.
- Reaching for `restore` when a plain `strati` + `fault` history suffices.
- Importing `clebsch_velocity` or `curlew.fields.restoration`. The velocity
  is `curlew.kinematics.euler.euler_velocity`.

## Reference files

| File | Contents |
|------|----------|
| [environment.md](environment.md) | Probe, conda/venv/uv paths |
| [structures.md](structures.md) | Structure types, factories, synthetic index |
| [restoration.md](restoration.md) | LDI: principle, wavelengths, shear map, examples |
| [geological-reasoning.md](geological-reasoning.md) | Superposition, cross-cutting, constraints |
| [examples.md](examples.md) | Code patterns |
| [api-reference.md](api-reference.md) | CSet, HSet, factories, API detail |
| [context.md](context.md) | Implicit modelling theory, restoration framing, literature |

Key repo paths: `curlew/synthetic.py`, `tests/test_geology.py`, `curlew/geology/__init__.py`, `curlew/kinematics/`, `curlew/priors/spectral.py`, `tests/restore/`, `curlew/visualise/`, `sandbox/`
