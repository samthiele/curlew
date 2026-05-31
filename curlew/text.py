"""
Text summaries of curlew objects for logging, progress bars, and agent-side validation.

Most `curlew` objects implement a `__str__` method that then calls the relevant
implementation here. These aim to produce human-readable summaries of the object's state, with
the main aim of enabling LLM-agent comprehension of the model and its outputs (to enable agent-based
model construction and validation).

This is very much a work in progress - so if you have any cool ideas then do let us know!
"""

import numpy as np

# helper functions
def _geode_class_label(lookup, class_id):
    if lookup:
        return str(lookup.get(class_id, class_id))
    return str(class_id)

def _geode_topology_contact_lines(topo_dict, min_contact=0.0):
    """Format symmetric topology dict as unique contact lines (i—j: area)."""
    lines = []
    seen = set()
    for a, row in topo_dict.items():
        for b, val in row.items():
            pair = tuple(sorted((str(a), str(b))))
            if pair in seen:
                continue
            seen.add(pair)
            if hasattr(val, "detach"):
                val = float(val.detach().cpu().item())
            else:
                val = float(val)
            if val > min_contact:
                lines.append(f"  {pair[0]} — {pair[1]}: {val:.4g}")
    return sorted(lines) if lines else ["  (no inter-class contacts)"]

# main summary functions
def geode_summary(
    geode,
    volume_fraction_warn=1e-3,
    min_voxels=1,
    connection="conservative",
    topology_kwargs=None,
):
    """
    Text summary of gridded model results for logging and agent-side validation.

    See :meth:`curlew.core.Geode.summary` for parameter documentation.
    """
    g = geode.numpy()
    lines = ["Model prediction summary:"]
    warnings = []
    topology_kwargs = dict(topology_kwargs or {})
    topology_kwargs.setdefault("connection", connection)

    n_pts = len(g)
    if n_pts == 0:
        if g.lithoID is not None:
            n_pts = len(g.lithoID)
        elif g.structureID is not None:
            n_pts = len(g.structureID)
        elif g.grid is not None:
            n_pts = int(np.prod(g.grid.shape))
    lines.append(f"  Points: {n_pts}")
    if g.crs:
        lines.append(f"  Coordinate system: {g.crs}")

    if g.grid is not None:
        cell_vol = float(np.prod(g.grid.step))
        lines.append(
            f"  Grid: shape {g.grid.shape}, step {g.grid.step}, cell volume {cell_vol:.6g}"
        )
    else:
        lines.append("  Grid: (none — volume/topology unavailable)")
        cell_vol = None

    if cell_vol is not None and g.structureID is not None:
        sids = sorted(g.structureLookup.keys()) if g.structureLookup else sorted(
            np.unique(g.structureID).tolist()
        )
        lines.append("")
        lines.append("Volumes (structure → lithology):")
        for sid in sids:
            smask = g.structureID == sid
            n_s = int(smask.sum())
            frac_s = n_s / n_pts if n_pts else 0.0
            sname = _geode_class_label(g.structureLookup, sid)
            line = (
                f"  {sname} [id={sid}]: {frac_s * 100:.2f}% "
                f"({n_s} voxels, volume≈{n_s * cell_vol:.6g})"
            )
            if frac_s < volume_fraction_warn or n_s < min_voxels:
                warnings.append(
                    f"Structure {sname!r} is negligible ({frac_s * 100:.3f}%, {n_s} voxels)"
                )
                line = "WARNING: " + line.strip()
            lines.append(line)

            if g.lithoID is not None and n_s > 0:
                litho_in_s = g.lithoID[smask]
                lids = sorted(g.lithoLookup.keys()) if g.lithoLookup else sorted(
                    np.unique(litho_in_s).tolist()
                )
                for lid in lids:
                    n_l = int((litho_in_s == lid).sum())
                    if n_l == 0:
                        continue
                    frac_l = n_l / n_pts
                    lname = _geode_class_label(g.lithoLookup, lid)
                    sub = f"    {lname} [id={lid}]: {frac_l * 100:.2f}% of model ({n_l} voxels)"
                    if frac_l < volume_fraction_warn or n_l < min_voxels:
                        warnings.append(
                            f"Lithology {lname!r} in {sname!r} is negligible "
                            f"({frac_l * 100:.3f}%, {n_l} voxels)"
                        )
                        sub = "WARNING: " + sub.strip()
                    lines.append(sub)
    elif g.lithoID is not None and cell_vol is not None:
        lines.append("")
        lines.append("Volumes (lithology only; no structureID):")
        lids = sorted(g.lithoLookup.keys()) if g.lithoLookup else sorted(
            np.unique(g.lithoID).tolist()
        )
        for lid in lids:
            n_l = int((g.lithoID == lid).sum())
            frac_l = n_l / n_pts if n_pts else 0.0
            lname = _geode_class_label(g.lithoLookup, lid)
            line = f"  {lname} [id={lid}]: {frac_l * 100:.2f}% ({n_l} voxels)"
            if frac_l < volume_fraction_warn or n_l < min_voxels:
                warnings.append(
                    f"Lithology {lname!r} is negligible ({frac_l * 100:.3f}%, {n_l} voxels)"
                )
                line = "WARNING: " + line.strip()
            lines.append(line)

    if g.lithoID is not None:
        n_undef = int((g.lithoID == -1).sum())
        if n_undef > 0:
            warnings.append(
                f"{n_undef} voxels ({100 * n_undef / n_pts:.2f}%) have undefined lithoID (-1)"
            )

    if g.grid is not None:
        topo_kw = {**topology_kwargs, "output": "dict"}
        if g.lithoID is not None:
            lines.append("")
            conn = topo_kw.get("connection", "conservative")
            lines.append(f"Lithology topology ({conn} contact area):")
            try:
                litho_topo = g.topology(mode="lithology", **topo_kw)
                lines.extend(_geode_topology_contact_lines(litho_topo))
            except Exception as exc:
                lines.append(f"  (unavailable: {exc})")
        if g.structureID is not None:
            lines.append("")
            conn = topo_kw.get("connection", "conservative")
            lines.append(f"Structure topology ({conn} contact area):")
            try:
                struct_topo = g.topology(mode="structure", **topo_kw)
                lines.extend(_geode_topology_contact_lines(struct_topo))
            except Exception as exc:
                lines.append(f"  (unavailable: {exc})")

    if g.offsets:
        lines.append("")
        lines.append("Displacements (mean over voxels with |u| > 0):")
        for name, disp in g.offsets.items():
            d = np.asarray(disp, dtype=float)
            if d.ndim == 1:
                d = d.reshape(-1, 1)
            mag = np.linalg.norm(d, axis=1)
            active = mag > 0
            if not np.any(active):
                lines.append(f"  {name}: zero everywhere")
                continue
            mean_u = d[active].mean(axis=0)
            mean_mag = float(mag[active].mean())
            direction = mean_u / (np.linalg.norm(mean_u) + 1e-30)
            dir_str = ", ".join(f"{v:.4g}" for v in direction)
            lines.append(
                f"  {name}: mean |u|={mean_mag:.4g}, mean direction [{dir_str}]"
            )

    if g.scalar is not None and g.structureID is not None:
        lines.append("")
        lines.append("Scalar field range by structure:")
        for sid in sorted(
            g.structureLookup.keys() if g.structureLookup else np.unique(g.structureID).tolist()
        ):
            smask = g.structureID == sid
            if not np.any(smask):
                continue
            sf = np.asarray(g.scalar)[smask]
            sname = _geode_class_label(g.structureLookup, sid)
            lines.append(f"  {sname}: min={np.min(sf):.4g}, max={np.max(sf):.4g}")

    if g.properties is not None and g.propertyNames:
        props = np.asarray(g.properties, dtype=float)
        if props.ndim == 1:
            props = props.reshape(-1, 1)
        lines.append("")
        lines.append("Properties (model-wide mean, min, max):")
        n_prop = min(len(g.propertyNames), props.shape[1] if props.ndim > 1 else 1)
        for j in range(n_prop):
            col = props[:, j] if props.ndim > 1 else props.ravel()
            pname = g.propertyNames[j]
            lines.append(
                f"  {pname}: mean={np.mean(col):.4g}, "
                f"min={np.min(col):.4g}, max={np.max(col):.4g}"
            )

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in warnings)

    return "\n".join(lines)

def geode_repr(geode):
    """Shorter summary of a Geode object"""
    parts = [f"{len(geode)} pts"]
    if geode.grid is not None:
        parts.append(f"grid{geode.grid.shape}")
    if geode.structureLookup:
        parts.append(f"{len(geode.structureLookup)} structures")
    return f"Geode({', '.join(parts)})"


def grid_str(grid):
    """Summary of voxel resolution, center, bounds, and voxel counts for a Grid."""
    axis_names = ("x", "y", "z")[: grid.ndim]
    n_vox = int(np.prod(grid.shape))

    lines = [f"Grid ({grid.ndim}D):"]
    lines.append(f"  voxels per axis: {grid.shape} (total {n_vox})")
    lines.append(f"  resolution (step): {grid.step}")
    lines.append(f"  physical dims: {grid.dims}")
    lines.append(f"  center: {tuple(np.asarray(grid.center).tolist())}")
    
    pts = grid.coords()
    vmin = pts.min(axis=0)
    vmax = pts.max(axis=0)
    for i, name in enumerate(axis_names):
        lines.append(f"  bounds {name}: [{vmin[i]:.6g}, {vmax[i]:.6g}]")

    rot = grid.matrix[: grid.ndim, : grid.ndim]
    if not np.allclose(rot, np.eye(grid.ndim)):
        lines.append("  rotation: applied (non-axis-aligned)")
    else:
        lines.append("  rotation: none (axis-aligned)")

    return "\n".join(lines)


def pebble_str(pebble):
    """Single-line loss summary suitable for progress-bar descriptions."""
    if not pebble.losses:
        return "L=0"

    parts = []
    total = 0.0
    multi_group = len(pebble.losses) > 1

    for group, terms in pebble.losses.items():
        for name, loss in terms.items():
            weight = pebble.weights.get(group, {}).get(name, 1.0)
            if hasattr(loss, "detach"):
                val = loss.detach()
                val = float(val.item()) if val.numel() == 1 else None
            elif isinstance(loss, np.ndarray):
                val = float(loss.item()) if loss.size == 1 else None
            else:
                val = float(loss) if isinstance(loss, (int, float)) else None

            if val is None:
                label = f"{group}/{name}" if multi_group else name
                parts.append(f"{label}=…")
                continue

            weighted = weight * val
            total += weighted
            label = f"{group}/{name}" if multi_group else name
            parts.append(f"{label}={weighted:.3f}")

    return f"L={total:.3f} " + " ".join(parts)

def cset_str(cset):
    """Summary of constraint counts and types in a CSet."""
    C = cset.numpy() if hasattr(cset, "numpy") else cset
    lines = ["CSet:"]
    if getattr(C, "crs", None):
        lines.append(f"  crs: {C.crs}")
    if C.vp is not None:
        lines.append(f"  value points: {len(C.vp)}")
    if C.gp is not None:
        lines.append(f"  gradient points: {len(C.gp)}")
    if C.gop is not None:
        lines.append(f"  orientation points: {len(C.gop)}")
    if C.pp is not None:
        lines.append(f"  property points: {len(C.pp)}")
    if C.eq is not None:
        lines.append(f"  equality traces: {len(C.eq)}")
    if C.iq is not None:
        lines.append(f"  inequalities: {len(C.iq[1])} pairs ({C.iq[0]} samples/epoch)")
    if C.grid is not None:
        lines.append("  grid:")
        for line in grid_str(C.grid).splitlines():
            lines.append(f"    {line}")
    if getattr(C, "trend", None) is not None:
        lines.append(f"  trend: {np.asarray(C.trend).tolist()}")
    if len(lines) == 1:
        lines.append("  (no constraints defined)")
    return "\n".join(lines)

def field_str(field):
    """Summary of a scalar field (analytical or neural)."""
    if isinstance(field, (int, float)):
        return f"Scalar field constant: {field}"
    name = getattr(field, "name", None) or type(field).__name__
    lines = [f"Scalar field {name!r} ({type(field).__name__}):"]
    if hasattr(field, "input_dim"):
        lines.append(f"  input_dim: {field.input_dim}")
    if getattr(field, "C", None) is not None:
        lines.append("  constraints: bound")
    else:
        lines.append("  constraints: none")
    if getattr(field, "H", None) is not None:
        active = [
            k
            for k in dir(field.H)
            if not k.startswith("_")
            and not callable(getattr(field.H, k))
            and getattr(field.H, k, 0) not in (0, "0", "0.0", None)
        ]
        if active:
            lines.append(f"  active losses: {', '.join(active[:6])}")
    return "\n".join(lines)

def geoevent_str(event):
    """Summary of a GeoEvent and its interaction objects."""
    lines = [f"GeoEvent {event.name!r} (eid={event.eid}):"]
    lines.append("  field:")
    for line in field_str(event.field).splitlines():
        lines.append(f"    {line}")
    if event.deformation is not None:
        lines.append(f"  deformation: {type(event.deformation).__name__}")
    if event.overprint is not None:
        mode = getattr(event.overprint, "mode", None)
        lines.append(f"  overprint: {type(event.overprint).__name__}" + (f" mode={mode!r}" if mode else ""))
    if event.propertyField is not None:
        lines.append(f"  propertyField: {type(event.propertyField).__name__}")
    if event.isosurfaces:
        lines.append(f"  isosurfaces: {len(event.isosurfaces)} ({', '.join(list(event.isosurfaces)[:5])})")
    if event.volumes:
        lines.append(f"  volumes: {len(event.volumes)}")
    if event.parent is not None and hasattr(event.parent, "name"):
        lines.append(f"  parent: {event.parent.name!r}")
    if event.parent2 is not None and hasattr(event.parent2, "name"):
        lines.append(f"  parent2: {event.parent2.name!r}")
    return "\n".join(lines)

def geomodel_str(model):
    """Summary of a GeoModel event list and grid."""
    title = f"GeoModel {model.name!r}:" if getattr(model, "name", None) else "GeoModel:"
    lines = [title]
    if getattr(model, "events", None):
        names = [e.name for e in model.events]
        lines.append(f"  events (oldest→youngest): {names}")
    if getattr(model, "grid", None) is not None:
        lines.append("  grid:")
        for line in grid_str(model.grid).splitlines():
            lines.append(f"    {line}")
    if getattr(model, "T", None) is not None:
        lines.append("  transform: defined (global↔model)")
    if getattr(model, "C", None) is not None:
        lines.append("  global property constraints: bound")
    return "\n".join(lines)

def to_text(obj):
    """
    Return a human-readable text summary of a curlew object.

    Supports :class:`~curlew.core.Geode`, :class:`~curlew.core.CSet`,
    :class:`~curlew.core.Pebble`, :class:`~curlew.geometry.Grid`,
    :class:`~curlew.geology.geoevent.GeoEvent`, :class:`~curlew.geology.geomodel.GeoModel`,
    and scalar field instances.
    """
    from curlew.core import CSet, Geode, Pebble
    from curlew.geometry import Grid

    if isinstance(obj, Grid):
        return grid_str(obj)
    if isinstance(obj, Geode):
        return geode_summary(obj)
    if isinstance(obj, Pebble):
        return pebble_str(obj)
    if isinstance(obj, CSet):
        return cset_str(obj)

    from curlew.geology.geoevent import GeoEvent
    from curlew.geology.geomodel import GeoModel
    from curlew.fields import BaseSF

    if isinstance(obj, GeoEvent):
        return geoevent_str(obj)
    if isinstance(obj, GeoModel):
        return geomodel_str(obj)
    if isinstance(obj, BaseSF):
        return field_str(obj)

    raise TypeError(
        f"No text summary for type {type(obj)!r}; supported types include "
        "Geode, CSet, Pebble, GeoEvent, GeoModel, and scalar fields."
    )
