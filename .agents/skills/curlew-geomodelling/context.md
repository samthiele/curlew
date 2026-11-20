# Literature context for curlew geomodelling

Key structural geology and geological modelling literature. See also: [structures.md](structures.md) · [geological-reasoning.md](geological-reasoning.md) · [examples.md](examples.md) · [api-reference.md](api-reference.md)

| # | Paper | Focus | Citation |
|---|------|--------| ----- |
| 0 | https://geoscience-prey.sydney.edu.au/iBooks/Patrice_Textbook.pdf| Structural geology foundations, deformation features, workflow | Rey 2016|
| 1 | https://serc.carleton.edu/files/research_on_learning/synthesis/journalclub/frodeman-1995.pdf | Geology as interpretive + historical reasoning | Frodeman 1995 |
| 2 | https://publications.rwth-aachen.de/record/754773/files/754773.pdf | 3D geomodel concepts, construction workflow, uncertainty | Wellmann & Caumon, 2018 |
| 3 | https://www.sciencedirect.com/science/article/pii/S0191814116301110 | Topology of geological models (characterisation) | Thiele et al., 2016a |
| 4 | https://www.sciencedirect.com/science/article/pii/S0191814116301122 | Topological uncertainty, ensembles, validation | Thiele et al., 2016b |
| 5 | https://gmd.copernicus.org/articles/19/3455/2026/ | Curlew 1.0 theory and implementation | Kamath et al., 2026 |
| 6 | https://gmd.copernicus.org/articles/18/71/2025/ | Automated geological consistency checking | Parquer et al., 2025 |

---

## 1. Geological structures

**Core idea:** Structural geology describes **geometry** (shape of features), **kinematics** (paths and relative motion), and **dynamics** (forces). At model scale, structures are discrete features whose relationships matter for fluid flow, hazards, and resources (Thiele et al., 2016a; Wellmann & Caumon, 2018).

**Feature types** (Rey, 2016; field practice):

- **Planar:** bedding, cleavage/foliation, faults, unconformities — written using strike / dip or dip / dip direction angles. Strike recorded using right-hand rule (RHR) or approximate compass direction of dip (N, NE, S, SW, etc.)
- **Linear:** fold axes, lineations, intersection lineations - written using trend and plunge angles
- **Brittle:** fractures, faults (kinematics: block motion, displacement magnitude), joints.
- **Ductile:** shear zones, folds 
- **Igneous:** dikes / dykes (extensional openings filled with magma), sills, sheet-intrusions, plutons, laccoliths, stocks
- **Hydrothermal:** veins and other hydrothermally altered or filled fractures

**Structural models as surfaces:** Wellmann & Caumon emphasise that full Earth history is not simulated; **structural geomodels** approximate history with **3D surfaces** (horizons, faults, intrusions) that mark discrete events—depositional changes, deformation, erosion—at the scale of the study.

### Implicit geological modelling — core principles

**Scalar field representation.** In implicit (potential-field) modelling, geology is encoded by one or more continuous functions (scalar fields) \(\phi(\mathbf{x})\) over model coordinates. A **geological contact** is not drawn as an explicit mesh first; it is the **isosurface** (level set) where \(\phi\) takes a chosen value:

\[
\{\mathbf{x} : \phi(\mathbf{x}) = c\}
\]

Structural orientations constrain \(\nabla\phi\) (bedding poles, fault-plane normals). Sparse contact picks can constrain **value** at points, but in under-constrained problems it is often better to constrain **relationships** between points (same contact → equal \(\phi\); younger above older → inequality) rather than absolute scalar levels—because \(\phi\) has no unique physical unit until isosurfaces are defined (Kamath et al., 2026; Hillier et al., 2023).

**Role of isosurfaces.** Isosurfaces are the bridge from abstract \(\phi\) to geological meaning:

| Geological concept | Implicit expression |
|------------------|-------------------|
| Stratigraphic horizon / formation top | Isosurface on depositional \(\phi\); younger units often correspond to **higher** \(\phi\) |
| Fault / shear contact | Isosurface on fault \(\phi\) (commonly `contact=0` or **named** isosurface) |
| Unconformity base | Threshold in `Overprint` (`base=`) — may be float or **named** isosurface |
| Dyke walls | Pair of isosurfaces (`contact=(lower, upper)`) bounding intrusion interval |
| Domain boundary | Isosurface separating footwall / hangingwall sub-models |

On **generative** events (`strati`, intrusions with overprint), isosurfaces are also used to assign **lithology IDs**: each named surface is treated as a **formation top** (\(\phi \ge c\)), applied from smallest to largest threshold so units stack consistently.

**Role of volumes.** Where isosurfaces define **boundaries**, **volumes** define **regions** (which side of a surface, which rock body, which fault influence zone). Examples:

- Rock unit between two horizons → band bounded by two isosurface values.
- **Finite fault** → displacement modifier active only inside an ellipsoid or custom region (`fault(..., modifier=...)` with `EllipsoidalField` on `addField`; reference model **`curlew.synthetic.goguel`**).
- Reservoir or alteration halo → boolean combination of \(\phi\) inequalities.

In curlew, explicit volumetric domains are **boolean expressions** over field values and resolved isosurface constants (see `addVolume` below), not separate volumetric meshes.

**Spatio-temporal implicit models (curlew).** A single global \(\phi\) cannot represent full geological history with unconformities and faults. Curlew uses **one or more implicit field per `GeoEvent`**, combined through a **time-ordered graph**: overprints replace older scalar/lithology; deformations retro-transform coordinates before older fields are evaluated (`undeform`). Contacts and units are defined **per-event**, then merged at prediction time via `GeoModel.predict`.

### Topological admissibility and the restoration pathway

**Why smoothness alone is not enough.** Every interpolator in the Lajaunie
family — co-kriging, radial basis functions, discrete smooth interpolation, and
neural fields — is *smoothness-based*. It fits \(\phi\) to contacts and
orientations, and nothing constrains the **topology** of the resulting level
sets. In geologically deformed regions the field therefore tends to develop
**interior extrema** — closed isosurfaces enclosing rock that cannot exist.
These are commonly called *bubbles*.

Within a single conformable sequence, a bubble is a **topological
inadmissibility**: a stack of layers cannot close on itself. Mitigations in
practice are unsatisfying — penalties on spurious extrema only enforce
topology approximately and only as well as their hyperparameter weighting
allows, while corrective practices (adding fictitious control points, manually
disconnecting sections) are labour-intensive and can actively harm the model.

**Original horizontality as a constraint.** Structural complexity in geology is
driven by deformation. Present-day geometry is best understood as a *deformed
version of a simpler reference state* — by the principle of original
horizontality, a sequence of flat, horizontal layers. Excluding
**discontinuity-introducing** deformation (faults, intrusions), deformation does
not change the topology of that layer-cake.

**The restoration reformulation.** This suggests a different framing: instead of
interpolating \(\phi\) directly, recover the **deformation map** \(\Phi\) that
carries the reference to the present state, so that

\[
\psi_p(\mathbf{x}) = \psi_r\left(\Phi^{-1}(\mathbf{x})\right)
\]

If \(\Phi\) is a **diffeomorphism** — smooth and invertible with a smooth
inverse — then \(\psi_p\) inherits the topology of \(\psi_r\). A stack of parallel
planes has no closed level sets, so no field obtained from it this way can
either. **Topology becomes a property of the construction, not of the loss.**

This has several consequences:

| Property | Consequence |
|---|---|
| Topology fixed by construction | No data, prior, or hyperparameter can produce an inadmissible geometry |
| Map is a deformation | **Finite-strain markers constrain it directly** — strain, which no scalar-field interpolator can use |
| Physical rules imposed on the velocity | Volume preservation becomes exact (\(\det F = 1\)), not a penalty term |
| Wavelength spectrum | Describes the **folds** (a geological property), rather than acting as an opaque smoothing parameter |
| Exact interpolation traded for admissibility | Where data are dense, fit is slightly looser than an exact interpolant — intended, not a defect |

**What it cannot do.** A continuous map cannot represent a **true
topological discontinuity**: faults, unconformities, and intrusions still need
the conventional treatment (separate events, `domainBoundary`, `Overprint`),
or lifting into an extra dimension. Boudins and other topological changes from
extreme ductile deformation are likewise out of reach. Volume *change*
(compaction) would require a velocity with prescribed divergence.

In curlew this construction is **Learnable Diffeomorphic Interpolation**:
the `restore(...)` pathway. A Fourier velocity supplies `Φ`, and a shear
composition keeps it volume-preserving. The conventional overprint/offset
pathway remains the right choice when a plain history of `strati` + `fault`
expresses the geology directly. See [restoration.md](restoration.md).

### Isosurfaces and volumes in curlew

**Defining isosurfaces** — `GeoEvent.addIsosurface(name, *, value=None, seed=None, field=0)`:

Provide **either** `value` **or** `seed`, not both.

| Mode | When to use | Behaviour |
|------|-------------|-----------|
| **`seed`** (preferred) | Known contact location(s) in present-day coords | At evaluation, \(\phi\) is evaluated at seed point(s) in the correct paleo frame (`undeform` via child chain); **mean** scalar at seeds becomes threshold \(c\). Decouples contact definition from arbitrary training scales—especially important for neural fields (`NFF`/`FSF`). |
| **`value`** | Fixed level already agreed (e.g. `0` for fault surface, `-np.inf` basement cut-off via `strati(base=...)`) | Stores constant \(c\); no re-evaluation. |

**Seed shapes:** one point `(x, y[, z])`, or an array/list of points `(N, ndim)`; multi-point seeds average the field response (useful for multi-point traces).

**Multi-field events:** `field=0` (default) or field name/index when `GeoEvent` has several underlying `BaseSF` instances (`addField`); isosurface thresholds apply only to that sub-field.

**Agents:** Prefer **seed-based** isosurfaces from mapped contacts or synthetic truth (`synthetic._isosurface_seeds`, `extract_constraints`); copy `isosurfaces` dict from reference models when benchmarking. Use **named** isosurfaces for `base`, `contact`, and `bound` so thresholds stay consistent after neural fitting. 

**Topology of structures** (Thiele et al., 2016a):

- **Cellular (1st order):** which rock volumes share interfaces (contact adjacency).
- **Structural / lithological:** higher-order adjacency of interfaces and edges.
- Classic patterns: layer-cake (linear chain), fault “ladder”, onlap, offlap/unconformity networks.

**Curlew mapping:** see the full catalogue in [structures.md](structures.md) (stratigraphy, faults, intrusions, plutons, synthetic index).

---

## 2. Structural reasoning

**Interpretive science** (Frodeman, 1995; Wellmann & Caumon, 2018):

- Geology is **hermeneutic**: observations (e.g. dip direction) are meaningless without interpretive framework—you must know *what* to measure.
- Reasoning blends **inductive** data gathering and **model-driven** prediction (Darwinian vs Baconian workflows; Rey).
- **Facts → hypothesis → model → theory**: models must be testable, lean in hypotheses, and physically valid (Rey on tectonic models).
- **Multiple working hypotheses**: Uncertainty often requires exploring several possible solutions and comparing to see which match observed data.

**Historical science** (Frodeman; Wellmann):

- Present-day geometry is the end state of a **sequence of processes**; explanation requires reconstructing order and causality.
- **Cross-cutting relationships** and superposition constrain event order. **Key links to model topology** and order of events in a GeoModel.
- “Retro-deformation” is the computational analogue: evaluate older geology in **paleo-coordinates** (curlew: `undeform`, `GeoEvent.child` chain).

**Reasoning checklist for agents:**

1. List **events** in time order (oldest first in `GeoModel`); though events in different branches of a domain boundary are incomparable (undefined temporal relationship).
2. For each event: generative (new rock / overprint) vs kinematic (displacement only) vs both.
3. Assign **constraints** to the event that owns the geology (not a younger unit).
4. Prefer **orientation + inequality** over arbitrary scalar values (Kamath et al.; Hillier et al.).
5. State **assumptions** explicitly (onlap, fault polarity, which side moves).
6. Regional tectonic knowledge (e.g., shortening/transport direction, vergence, stretching lineation, bulk strain) constrains transport direction on faults (and folds).

**Curlew mapping:**

- Event history graph: `GeoModel.events`, `parent`/`child`, `parent` and `parent2` indicate domain boundary (`parent` and `parent2` sub-models are incomparable), `M._repr_svg_()` for plotting.
- Cross-cutting: younger events later in list; e.g., `fault` after `strati` it cuts.
- Interpretation in coords: `CSet.crs='global'` → `bind` updates `CSet.crs` based on model `Transform`.
- Paleo frame for fitting: `GeoEvent.fit(cache=True)` with `child.undeform`.

---

## 3. Geological model construction

**Geologic realism** (Wellmann & Caumon): models capture **essential** geometric aspects of history for purpose and scale; not every joint or fault-core detail. Complexity is a **modelling choice**, not a copy of nature. Understanding of model application or focus is crucial.

**Typical workflow** (Wellmann §1.2; Kamath §3; curlew):

1. **Conceptual model** / **Synthetic model** — which structures, event order, 2D/2.5D/3D, scale. Highly simplified geometry. In `curlew` these can be constructed using analytical fields (see `curlew.synthetic.py`).
2. **Mathematical representation** — implicit fields, parameters (neural field representation in `curlew`).
3. **Structure definition** — number of horizons, faults, isosurfaces, domains.
4. **Conditioning** — fit to sparse data (contacts, orientations, inequalities).
5. **Property / forward models** — fit to density, geochemistry via `propertyField` (Kamath). Better models have an implicit field that explains more variance in arbitrary observed data (as "geological structure" is a shared latent explanatory variable).

**Implicit + spatio-temporal (curlew / Kamath):**

- One  (or more) scalar field per **event**; **temporally directed graph** combines them.
- **Generative:** `Overprint` (unconformity, onlap, plutons, stocks); creates new rocks (lithologies) but does not move older ones (overprints them via e.g., erosion or magma assimilation).
- **Kinematic:** `FaultOffset`, `SheetOffset`, `FoldOffset` via `undeform`; displaces older rocks but does not create new lithologies.
- **Dilative:** creates new rock and deforms old ones; sheet intrusions, dykes, sills.
- Train **per event** (`prefit`) or **jointly** (`fit`, `custom_loss`). It is better to initialise fields separately using `prefit` before optimising the (more difficult) global optimisation using `fit`. 

**Constraints hierarchy** (Kamath §2; curlew `CSet` / `HSet`):

| Tier | Role | Curlew |
|------|------|--------|
| Local gradient constraint | Bedding younging | `gp`, `gv` → `grad_loss` |
| Local orientation (tangent) constraint| Strike/dip only | `gop`, `gov` → `ori_loss` |
| Relational (in)equality | Same contact; young-on-old | `eq`, `iq` → `eq_loss`, `iq_loss` |
| Global | Smooth, monotonic, thickness | `grid`, `delta`, `trend` → `mono_loss`, `thick_loss`, `flat_loss` |
| Value pins | Fixed scalar at point | `vp`, `vv` → `value_loss` (discouraged for NFF due to convergence challenges) |

**Anti-artefact:** Biharmonic-style smooth interpolators create **closed isosurface “bubbles”** violating original horizontality (Kamath; Thiele et al.). Curlew: `mono_loss` (Hessian/divergence on grid samples), `thick_loss`, avoid over-reliance on `value_loss`.

**Practical build sequence:**

1. Match a **synthetic** analogue (`curlew.synthetic.*`) or construct a new *conceptual* (synthetic) model using analytical fields but approximately matching world coordinates and geometry.
2. Use `NFF`/`FSF` for constructing parallel model that is fit to data.
3. `addIsosurface` from seeds; `base`/`contact` as names or floats.
4. `GeoModel([...])` → `prefit` → `predict` → visualise (`visualise`, napari).

**For agents**: Text summaries of model outputs (`curlew.core.Geode` instances) can be created using `curlew.core.Geode.summary`. This contains core information for checking model outputs and matching (validating) this against geological context and user instructions.

---

## 4. Geological model validation

Validation asks whether a model is **fit for purpose**—not merely whether it interpolates data (Parquer et al., 2025; Wellmann).

**Two axes** (Parquer):

- **Data consistency:** geometry honours observations (contacts, orientations).
- **Knowledge consistency:** geometry honours **universal** principles (superposition, permissible contact types, event histories).

Many algorithms optimise data fit but violate principles (e.g. older unit above younger without erosion event; Fig. 1 Parquer; bubble artefacts Kamath).

**Validation activities:**

| Check | Literature | Curlew practice |
|-------|------------|-----------------|
| Depositional order | Parquer truth tables; Steno | `iq` with `>`; event order in `GeoModel` |
| Cross-cutting logic | Rey; Parquer temporal relations | Fault/dyke younger than cut units |
| No spurious closed horizons | Kamath; Hillier “bubbles” | Enable `mono_loss`; inspect `predict` isosurfaces |
| Topology sensibility | Thiele et al. 2016b | Compare litho adjacency from `lithoID` / voxel export |
| Visual / section sanity | Wellmann | `M.drill`, `visualise`, synthetic ground truth |
| Consistency automation | Parquer CC tables | Manual: encode rules as `iq`/`eq`; future: external checker on exported meshes |

**Topology as validation filter** (Thiele et al., 2016b): Impossible adjacency (e.g. layer-cake topology with fault ladder violated) can flag **geologically impossible** realisations even when voxel lithology looks similar.

**Curlew-specific checks after `predict`:**

- `structureID` / `lithoID` change plausibly along sections.
- `getIsovalues()` stable after fit (named bases resolve).
- Loss components in `Pebble` decreasing under `prefit`.
- Compare to `synthetic.extract_constraints` benchmark when available.

---

## 5. Geological uncertainty

**Sources** (Wellmann §2; Mann/Jones classifications):

| Stage | Uncertainty | Example |
|-------|-------------|---------|
| (a) Conceptual model | Lack of knowledge, interpreter bias | Which faults exist? |
| (b) Mathematical choice | Method limits | Single global implicit vs event graph |
| (c) Model structure | # layers, fault topology | Extra unconformity? |
| (d) Data | Position, measurement error | GPS, dip uncertainty |

Wellmann **Type 1** = input data; **Type 2** = interpolation; **Type 3** = existence/structure unknown.

Kamath: uncertainty from **data sparsity** and **erroneous/subjective constraints**; ensembles needed because plausible geological solutions are under-sampled.

**Quantification approaches:**

1. **Perturb input data** → ensemble of models (Monte Carlo; Wellmann & Regenauer-Lieb; Lindsay et al.) — perturb `CSet` points, re-fit.
2. **Lithology variability maps** — voxel disagreement across ensemble.
3. **Topology variability** (Thiele et al., 2016b): small input changes → often **distinct topology networks**; faults/unconformities dominate topological uncertainty.
4. **RFF seed ensembles** (Kamath): different `seed` / RFF draws → multiple implicit geometries **without** perturbing data (`NFF`/`FSF` init).

**Topological uncertainty** (Thiele et al., 2016b):

- Compare adjacency graphs between realisations.
- High sensitivity near faults and unconformities.
- Use as **geodiversity metric** and to cluster “geologically distinct” model families.

**Curlew mapping:**

| Technique | Implementation |
|-----------|----------------|
| Data perturbation | Jitter `gp`/`gv`/seeds in `CSet`; re-run `prefit`/`fit` |
| Interpolation variability | `HSet` weights, multiple `seed` on `BaseNF` |
| Structural existence | Alternate `GeoModel` event lists (compare topologies) |
| Aleatory / measurement | Document `crs`, coordinate precision; narrow `offset` ranges on `fault` |
| Epistemic / conceptual | Scenario models: different `GeoModel([...])` histories |
| Report | `Pebble` loss breakdown; litho/structure maps from `Geode`; save ensembles via `saveModel` |

**Communication:** Separate **data uncertainty** (can pdf) from **interpretive uncertainty** (scenario ensembles, topology classes)—aligned with Frodeman/Wellmann on subjectivity.

---

## Reading order for agents

1. **0 + 1** — vocabulary and reasoning style.
2. **2** — industry/academic geomodel workflow and uncertainty taxonomy.
3. **5** — curlew-specific methods (read alongside `curlew/__init__.py` docstring).
4. **3 + 4** — topology for validation and ensemble analysis.
5. **6** — formal consistency when validating against principles.

When implementing, always tie literature concepts to concrete APIs in [api-reference.md](api-reference.md).
