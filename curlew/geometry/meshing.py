"""
Octree dual contouring of an implicit callable.

:func:`dual_contour` never materialises a voxel grid. It samples ``value_and_grad``
at octree (or quadtree) corners, Hermite zero-crossings on sign-changing edges,
and one QEF vertex per occupied leaf (Ju et al. 2002). Cost follows surface
area × depth, not volume.
"""
from __future__ import annotations

import itertools
from typing import Callable, Optional, Tuple, Union

import numpy as np

import curlew
from curlew import _numpy, _tensor

Bounds = Union[np.ndarray, "object"]


def dual_contour(
    value_and_grad: Callable,
    bounds: Bounds,
    iso: float = 0.0,
    max_depth: int = 8,
    min_depth: int = 0,
    thin_factor: float = 1.0,
    bisection: int = 8,
    qef_tol: float = 0.1,
    batch_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract an isosurface from ``value_and_grad`` on an adaptive octree (3-D)
    or quadtree (2-D).

    Parameters
    ----------
    value_and_grad : callable
        ``f(x) -> (values, grads)`` with ``x`` a ``(N, dim)`` torch tensor.
        ``values`` is ``(N,)``; ``grads`` is ``(N, dim)``. Either torch or NumPy.
    bounds : (2, dim) array or :class:`~curlew.geometry.Grid`
        Axis-aligned box, or a ``Grid`` (its world AABB; the grid is not sampled).
    iso : float
        Isovalue. A corner is treated as inside when ``φ < iso``.
    max_depth : int
        Finest tree level. A depth of ``L`` matches a uniform grid of ``2^L`` cells.
    min_depth : int
        Force subdivision to at least this level (even in empty space).
    thin_factor : float
        Lipschitz refine: split a same-sign cell when
        ``0 < min|φ − iso| ≤ thin_factor · max||∇φ|| · half_diag``. ``0`` disables.
    bisection : int
        Zero-crossing refinements per intersecting edge.
    qef_tol : float
        Relative SVD cutoff for the QEF (feature detection).
    batch_size : int, optional
        Oracle batch size. Defaults to ``curlew.batchSize``.

    Returns
    -------
    verts : ndarray, shape ``(V, dim)``
    faces : ndarray
        ``(F, 3)`` triangles in 3-D, or ``(E, 2)`` segments in 2-D.
    """
    if max_depth < 0 or min_depth < 0 or min_depth > max_depth:
        raise ValueError("require 0 <= min_depth <= max_depth")
    lo, hi = _aabb(bounds)
    dim = int(lo.shape[0])
    if dim not in (2, 3):
        raise ValueError(f"dual_contour supports 2-D or 3-D, got dim={dim}")
    extent = hi - lo
    if np.any(extent <= 0):
        raise ValueError("bounds must have positive extent on every axis")

    n = 1 << max_depth
    bs = int(batch_size if batch_size is not None else curlew.batchSize)
    sampler = _Sampler(value_and_grad, lo, extent, n, dim, bs)

    corners = list(itertools.product((0, 1), repeat=dim))
    leaves = _collect_leaves(
        sampler, corners, dim, iso, max_depth, min_depth, thin_factor, extent, n,
    )
    if not leaves:
        return np.zeros((0, dim), dtype=np.float64), np.zeros(
            (0, dim if dim == 2 else 3), dtype=np.int64,
        )

    hermite = _hermite_edges(sampler, leaves, corners, dim, iso, bisection)
    if not hermite:
        return np.zeros((0, dim), dtype=np.float64), np.zeros(
            (0, dim if dim == 2 else 3), dtype=np.int64,
        )

    cell_size = extent / n
    verts, index = _qef_vertices(leaves, hermite, lo, cell_size, dim, qef_tol)
    faces = _emit_faces(verts, hermite, index, dim)
    return _compact(verts, faces)


def _compact(verts, faces):
    if len(faces) == 0:
        return verts[:0], faces
    used = np.unique(faces.ravel())
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return verts[used], remap[faces]


def _aabb(bounds: Bounds) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(bounds, "axes") and hasattr(bounds, "matrix"):
        grid = bounds
        lo = np.array([float(a[0]) for a in grid.axes], dtype=np.float64)
        hi = np.array([float(a[-1]) for a in grid.axes], dtype=np.float64)
        corners = np.array(list(itertools.product(*zip(lo, hi))), dtype=np.float64)
        h = np.hstack([corners, np.ones((len(corners), 1))])
        world = (h @ np.asarray(grid.matrix, dtype=np.float64).T)[:, : grid.ndim]
        return world.min(0), world.max(0)
    b = np.asarray(bounds, dtype=np.float64)
    if b.ndim != 2 or b.shape[0] != 2:
        raise ValueError("bounds must be shape (2, dim) or a Grid")
    return np.minimum(b[0], b[1]), np.maximum(b[0], b[1])


class _Sampler:
    """Corner cache + batched oracle. Keys are finest-level integer corners."""

    def __init__(self, fn, origin, extent, n, dim, batch_size):
        self.fn = fn
        self.origin = np.asarray(origin, dtype=np.float64)
        self.extent = np.asarray(extent, dtype=np.float64)
        self.n = int(n)
        self.dim = int(dim)
        self.batch_size = int(batch_size)
        self.val = {}
        self.grad = {}

    def world(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + (np.asarray(ijk, dtype=np.float64) / self.n) * self.extent

    def eval_world(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(pts) == 0:
            return np.empty(0, dtype=np.float64), np.empty((0, self.dim), dtype=np.float64)
        x = _tensor(pts)
        vs, gs = [], []
        for i in range(0, len(x), self.batch_size):
            chunk = x[i : i + self.batch_size]
            out = self.fn(chunk)
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                raise TypeError("value_and_grad must return (values, grads)")
            v, g = out
            v = _numpy(v).reshape(-1).astype(np.float64, copy=False)
            g = _numpy(g).astype(np.float64, copy=False)
            if g.ndim == 1:
                g = g.reshape(-1, self.dim)
            if v.shape[0] != len(chunk) or g.shape != (len(chunk), self.dim):
                raise ValueError(
                    f"value_and_grad returned values {v.shape}, grads {g.shape} "
                    f"for {len(chunk)} points of dim {self.dim}"
                )
            vs.append(v)
            gs.append(g)
        return np.concatenate(vs), np.concatenate(gs)

    def fetch(self, keys):
        """Evaluate any uncached integer corners; return (values, grads) aligned to ``keys``."""
        missing = list(dict.fromkeys(k for k in keys if k not in self.val))
        if missing:
            pts = self.world(np.asarray(missing, dtype=np.float64))
            v, g = self.eval_world(pts)
            for i, k in enumerate(missing):
                self.val[k] = float(v[i])
                self.grad[k] = g[i]
        return (
            np.fromiter((self.val[k] for k in keys), dtype=np.float64, count=len(keys)),
            np.stack([self.grad[k] for k in keys], axis=0),
        )


def _cell_corners(ijk, level, max_depth, corners, dim):
    step = 1 << (max_depth - level)
    base = tuple(ijk[d] * step for d in range(dim))
    return [tuple(base[d] + off[d] * step for d in range(dim)) for off in corners]


def _children(ijk, dim):
    out = []
    for off in itertools.product((0, 1), repeat=dim):
        out.append(tuple(2 * ijk[d] + off[d] for d in range(dim)))
    return out


def _collect_leaves(sampler, corners, dim, iso, max_depth, min_depth, thin_factor, extent, n):
    current = [(0,) * dim]
    leaves = []
    for level in range(max_depth + 1):
        if not current:
            break
        keys = []
        per_cell = []
        for ijk in current:
            ck = _cell_corners(ijk, level, max_depth, corners, dim)
            per_cell.append(ck)
            keys.extend(ck)
        sampler.fetch(keys)
        step = 1 << (max_depth - level)
        half_diag = 0.5 * np.linalg.norm((step / n) * extent)
        nxt = []
        for ijk, ck in zip(current, per_cell):
            vals = np.fromiter((sampler.val[k] for k in ck), dtype=np.float64, count=len(ck))
            inside = vals < iso
            sign_change = bool(inside.any()) and (not bool(inside.all()))
            maybe_thin = False
            if thin_factor > 0 and not sign_change:
                g = np.stack([sampler.grad[k] for k in ck], axis=0)
                gmax = float(np.linalg.norm(g, axis=1).max())
                phi_min = float(np.abs(vals - iso).min())
                # A corner exactly on the iso is already owned by a neighbour
                # with a sign change (or the surface lies on a face).
                maybe_thin = (
                    phi_min > 0.0
                    and phi_min <= thin_factor * gmax * half_diag
                )
            split = level < min_depth or sign_change or maybe_thin
            if not split:
                continue
            if level == max_depth:
                if sign_change:
                    leaves.append(ijk)
            else:
                nxt.extend(_children(ijk, dim))
        current = nxt
    return leaves


def _hermite_edges(sampler, leaves, corners, dim, iso, bisection):
    """Unique finest intersecting edges → (lower, axis) → (point, normal, flip)."""
    edge_ends = []
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            diff = [a[d] - b[d] for d in range(dim)]
            if sum(d != 0 for d in diff) == 1:
                edge_ends.append((a, b))

    seen = {}
    for ijk in leaves:
        for a, b in edge_ends:
            ka = tuple(ijk[d] + a[d] for d in range(dim))
            kb = tuple(ijk[d] + b[d] for d in range(dim))
            axis = next(d for d in range(dim) if ka[d] != kb[d])
            lower, upper = (ka, kb) if ka[axis] < kb[axis] else (kb, ka)
            key = (lower, axis)
            if key in seen:
                continue
            v0, v1 = sampler.val[lower], sampler.val[upper]
            if (v0 < iso) == (v1 < iso):
                continue
            seen[key] = (v0, v1, (v0 < iso))

    if not seen:
        return {}

    keys = list(seen.keys())
    p0 = sampler.world(np.array([k[0] for k in keys], dtype=np.float64))
    p1 = np.empty_like(p0)
    for i, (lower, axis) in enumerate(keys):
        up = list(lower)
        up[axis] += 1
        p1[i] = sampler.world(np.array(up, dtype=np.float64))
    v0 = np.array([seen[k][0] for k in keys], dtype=np.float64)
    v1 = np.array([seen[k][1] for k in keys], dtype=np.float64)

    for _ in range(max(0, int(bisection))):
        mid = 0.5 * (p0 + p1)
        vm, _ = sampler.eval_world(mid)
        pick = (v0 < iso) != (vm < iso)
        p1 = np.where(pick[:, None], mid, p1)
        v1 = np.where(pick, vm, v1)
        p0 = np.where(pick[:, None], p0, mid)
        v0 = np.where(pick, v0, vm)
    # Secant on the final interval — exact for a linear field, and lands on
    # an endpoint that sits on the iso rather than a midpoint one step short.
    denom = v1 - v0
    t = np.divide(iso - v0, denom, out=np.full_like(denom, 0.5), where=np.abs(denom) > 1e-30)
    t = np.clip(t, 0.0, 1.0)
    pts = (1.0 - t)[:, None] * p0 + t[:, None] * p1
    _, grads = sampler.eval_world(pts)

    out = {}
    for i, key in enumerate(keys):
        out[key] = (pts[i], grads[i], seen[key][2])
    return out


def _qef_vertices(leaves, hermite, origin, cell_size, dim, qef_tol):
    # Gather Hermite samples per leaf via the leaf's own edges.
    edge_of = {}
    for (lower, axis), sample in hermite.items():
        for cell in _face_cells(lower, axis, dim):
            edge_of.setdefault(cell, []).append(sample)

    verts = np.empty((len(leaves), dim), dtype=np.float64)
    index = {}
    nkeep = 0
    for ijk in leaves:
        samples = edge_of.get(ijk)
        if not samples:
            continue
        pts = np.stack([s[0] for s in samples], axis=0)
        nrm = np.stack([s[1] for s in samples], axis=0)
        lo = origin + np.asarray(ijk, dtype=np.float64) * cell_size
        hi = lo + cell_size
        verts[nkeep] = _qef_min(pts, nrm, lo, hi, qef_tol)
        index[ijk] = nkeep
        nkeep += 1
    return verts[:nkeep], index


def _qef_min(points, normals, lo, hi, rel):
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    n = normals / np.maximum(nrm, 1e-15)
    mass = points.mean(axis=0)
    b = np.einsum("ij,ij->i", n, points - mass)
    try:
        U, S, Vt = np.linalg.svd(n, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.clip(mass, lo, hi)
    cutoff = (rel * S[0]) if S.size else 0.0
    sinv = np.zeros_like(S)
    np.divide(1.0, S, out=sinv, where=S > cutoff)
    x = mass + Vt.T @ (sinv * (U.T @ b))
    return np.clip(x, lo, hi)


def _face_cells(lower, axis, dim):
    """Finest cells that share the unit edge from ``lower`` along ``+axis``, winding +axis."""
    if dim == 2:
        b = 1 - axis
        c0 = [0, 0]
        c1 = [0, 0]
        c0[axis] = c1[axis] = lower[axis]
        c0[b] = lower[b] - 1
        c1[b] = lower[b]
        return [tuple(c0), tuple(c1)]
    b = (axis + 1) % 3
    c = (axis + 2) % 3

    def cell(db, dc):
        p = [0, 0, 0]
        p[axis] = lower[axis]
        p[b] = lower[b] + db
        p[c] = lower[c] + dc
        return tuple(p)

    return [cell(-1, -1), cell(0, -1), cell(0, 0), cell(-1, 0)]


def _emit_faces(verts, hermite, index, dim):
    faces = []
    for (lower, axis), (_pt, grad, low_inside) in hermite.items():
        cells = _face_cells(lower, axis, dim)
        ids = [index[c] for c in cells if c in index]
        if dim == 2:
            if len(ids) != 2:
                continue
            a, b = ids
            if not low_inside:
                a, b = b, a
            faces.append((a, b))
            continue
        if len(ids) == 4:
            cand = ((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3]))
        elif len(ids) == 3:
            cand = ((ids[0], ids[1], ids[2]),)
        else:
            continue
        for tri in cand:
            i, j, k = tri
            e1 = verts[j] - verts[i]
            e2 = verts[k] - verts[i]
            n = np.cross(e1, e2)
            if np.dot(n, grad) < 0:
                j, k = k, j
            faces.append((i, j, k))
    if not faces:
        return np.zeros((0, 2 if dim == 2 else 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64)
