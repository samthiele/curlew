"""Wavelength band for a Fourier velocity, read from a constraint set.

Moments of the folded layering about one shared normal (Rice 1944;
Longuet-Higgins 1957). Contacts that span a fold set the heights; a shorter
trace is slope, not height. Bedding sets the tilts. Quadratic-fit noise is
removed from ``m2`` and ``m4``. Tilts are angles, so an overturned limb stays finite.

* ``m0`` — height variance, contacts averaged in cells of width ``w``
* ``m2`` — variance of tilt from the shared normal
* ``m4`` — variance of curvature
* ``λx = 2π √(m2/m4)`` — mean hinge spacing
* ``λ0 = 2π √(m0/m2)`` — mean limb spacing; dropped when heights are too sparse
  to satisfy ``λ0 ≥ λx``

A curvature measured over a distance ``s`` is low by ``sinc(π s / λ)``. Pairs
closer than ``λx/7``, and windows of width ``≈ λx/4.6``, keep that bias under
3 %. ``λx`` is the stable solution of ``λ ← estimate(λ)``, scanned down from
the data diameter. An estimate counts only with at least ``MIN_SAMPLES``
curvature samples, a standard error under half an octave, and ``λ`` no longer
than the data. If none does, curvature comes from each reading's two nearest
neighbours at any distance, ``λx`` reads long, and the result says so.

A cylindrical fold (normals near a great circle, or ``axis`` given) is measured
in the profile plane. Otherwise ``m2`` and ``m4`` are divided by the
participation ratio of the tilts (1 in one plane, 2 for a dome-and-basin).

Strain ellipses are removed first, ``F̄ = B̄½`` with ``B̄`` the log-Euclidean
mean (Sherwin and Chapple 1968). A mode of wavelength ``ℓ`` changes its
gradient by about ``(π D / ℓ)² / 2`` across a diameter ``D``; keeping that
below the strain's heterogeneity ``r`` requires ``L_max ≥ π D / √(2 r)``.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace
from typing import Optional

import numpy as np

__all__ = ["spectral_band", "spectral_moments", "bulk_strain", "n_features_for_band"]

_SCIPY = None


def _scipy():
    """Lazy import of scipy.spatial.cKDTree and scipy.optimize.brentq."""
    global _SCIPY
    if _SCIPY is None:
        try:
            from scipy.optimize import brentq
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise ImportError(
                "curlew.priors.spectral requires scipy (spectral_band / spectral_moments). "
                "Install with: pip install scipy  or  pip install 'curlew[all]'"
            ) from exc
        _SCIPY = (cKDTree, brentq)
    return _SCIPY


def _check_cset_frame(C) -> None:
    # Positions are taken as given. With a GeoModel transform, convert first (CSet.to_model(T)).
    crs = getattr(C, "crs", "model")
    if crs not in ("model", "event", "global"):
        raise ValueError(f"unsupported CSet.crs for spectral_band: {crs!r}")


def _warn_moment_quality(pick: dict, *, stacklevel: int = 2) -> None:
    if not pick.get("resolved", True):
        warnings.warn(
            "spectral_moments could not resolve λx from curvature pairs; "
            "using nearest-neighbour fallback (lambda_x may be biased high).",
            UserWarning,
            stacklevel=stacklevel,
        )
    if pick.get("lambda_0") is None and pick.get("m0") is not None:
        warnings.warn(
            "horizon heights are under-sampled relative to λx; λ0 was dropped "
            "(L_min uses λx only).",
            UserWarning,
            stacklevel=stacklevel,
        )

PAIR_FRACTION = 1.0 / 7.0   # pairs closer than λx/7: curvature biased by under 3 %
MIN_SAMPLES = 8             # effective curvature samples needed to resolve λx
# A quadratic fit across a window w reads curvature low by about (2πw/λ)²/56: the
# window with the same bias as a pair at λx/7 (w ≈ λx/4.6).
WINDOW_FRACTION = math.sqrt(56 * (1 - math.sin(math.pi * PAIR_FRACTION) / (math.pi * PAIR_FRACTION))) / (2 * math.pi)
_SCAN = (8, 10)             # λ scan: steps per octave, octaves below the diameter
_CYLINDRICITY = 0.2         # normals count as cylindrical below this eigenvalue ratio
_FALLBACK_NEIGHBOURS = 2    # unresolved curvature: nearest along-layer neighbours per reading


# --------------------------------------------------------------------------- data

def _np(a):
    """An array as float64 numpy, from numpy or torch."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=float)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-12)


def _horizons(C):
    """Contacts grouped by horizon: each ``eq`` trace, and ``vp`` grouped by value."""
    out = []
    eq = getattr(C, "eq", None)
    if eq is not None:
        out += [_np(t) for t in eq]
    vp, vv = getattr(C, "vp", None), getattr(C, "vv", None)
    if vp is not None and vv is not None:
        vp, vv = _np(vp), _np(vv).reshape(len(_np(vp)), -1)[:, 0]
        for v in np.unique(vv):
            out.append(vp[vv == v])
    return [h.reshape(-1, h.shape[-1]) for h in out if h.size]


def _bedding(C):
    """Positions, unit normals and polarity flags (True for ``gv``, False for ``gov``)."""
    P, N, signed = [], [], []
    for pos, vec, s in (("gp", "gv", True), ("gop", "gov", False)):
        p, v = getattr(C, pos, None), getattr(C, vec, None)
        if p is not None and v is not None:
            p, v = _np(p), _np(v)
            P.append(p.reshape(-1, p.shape[-1]))
            N.append(_unit(v.reshape(-1, v.shape[-1])))
            signed.append(np.full(len(P[-1]), s))
    if not P:
        return None, None, None
    return np.vstack(P), np.vstack(N), np.concatenate(signed)


def _orientations_only(C):
    """``C`` reduced to its bedding readings: no contacts, so no heights and no contact windows."""
    out = SimpleNamespace(crs=getattr(C, "crs", "model"))
    for name in ("gp", "gv", "gop", "gov"):
        setattr(out, name, getattr(C, name, None))
    return out


def _first_normal(N, signed, d):
    """The mean signed normal, or the principal direction when they cancel or carry no polarity."""
    nbar = N[signed].mean(0) if signed.any() else np.zeros(d)
    if np.linalg.norm(nbar) < 0.2:
        nbar = np.linalg.eigh(N.T @ N)[1][:, -1]
        if not signed.any() and nbar[-1] < 0:
            nbar = -nbar
    return _unit(nbar)


def _profile(N, axis, start):
    """Rows spanning the space the moments live in (the data's own, or the profile plane), and the axis."""
    d = N.shape[1]
    if d == 3 and axis is not False:
        if axis is None:
            w, v = np.linalg.eigh(N.T @ N)
            axis = v[:, 0] if w[0] < _CYLINDRICITY * w[1] else None
        else:
            axis = _unit(_np(axis).ravel())
        if axis is not None:
            e1 = _unit(start - (start @ axis) * axis)
            return np.stack([np.cross(axis, e1), e1]), axis
    return np.eye(d), None


def _declustered(X, w):
    """Weights 1/(samples within w/2): dense runs count per length, not per datum."""
    if len(X) == 0:
        return np.zeros(0)
    cKDTree = _scipy()[0]
    return 1.0 / np.asarray(cKDTree(X).query_ball_point(X, 0.5 * w, return_length=True), dtype=float)


def _cell_means(H, w):
    """One point per occupied cell of size w: the mean of the contacts in it."""
    _, inv = np.unique(np.floor(H / w).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=H[:, k]) for k in range(H.shape[1])], axis=1) / n[:, None]


# --------------------------------------------------------------------------- pieces

def _pairs(X, N, *, radius=None, neighbours=None):
    """
    Curvature pairs of bedding: indices, distance along the layering, curvature,
    and the pairs left out as lying across the layering.

    With ``radius``, every pair closer than it whose separation lies within 45°
    of both bedding planes. With ``neighbours``, each reading's nearest that many
    such readings (within 45° of its own plane) at any distance: the unresolved
    fallback.
    """
    if len(X) < 2:
        e = np.zeros((0, 2), int)
        return e, np.zeros(0), np.zeros(0), e
    cKDTree = _scipy()[0]
    # Readings closer than this are the same place (e.g. repeated along a projected fold
    # axis): they measure no turning, and would count as zero curvature.
    tol = 1e-9 * max(float(np.ptp(X, axis=0).max()), 1e-30)
    if radius is not None:
        ij = cKDTree(X).query_pairs(radius, output_type="ndarray").reshape(-1, 2)
        sep = X[ij[:, 1]] - X[ij[:, 0]]
        s2 = np.sum(sep ** 2, axis=1)
        ca2 = np.einsum("pd,pd->p", sep, N[ij[:, 0]]) ** 2
        cb2 = np.einsum("pd,pd->p", sep, N[ij[:, 1]]) ** 2
        apart = s2 > tol ** 2
        ok = apart & (np.maximum(ca2, cb2) <= 0.5 * s2)
        keep, skipped = ij[ok], ij[apart & ~ok]
        along = np.sqrt(np.clip(s2 - 0.5 * (ca2 + cb2), 0, None))[ok]
    else:
        kq = min(len(X) - 1, max(4 * neighbours, 8))
        _, idx = cKDTree(X).query(X, kq + 1)
        found, skipped = {}, set()
        for a in range(len(X)):
            taken = 0
            for b in idx[a, 1:]:
                if taken == neighbours:
                    break
                sep = X[b] - X[a]
                s = float(np.linalg.norm(sep))
                if s <= tol:
                    continue
                al = float(np.linalg.norm(sep - (sep @ N[a]) * N[a]))
                if al < s / math.sqrt(2):
                    skipped.add((min(a, b), max(a, b)))
                    continue
                found[(min(a, b), max(a, b))] = al
                taken += 1
        keep = np.array(list(found), int).reshape(-1, 2)
        along = np.array(list(found.values()), float)
        skipped = np.array(sorted(skipped - set(found)), int).reshape(-1, 2)
    turn = np.arccos(np.clip(np.einsum("pd,pd->p", N[keep[:, 0]], N[keep[:, 1]]), -1, 1))
    return keep, along, turn / along.clip(1e-12), skipped


def _fits(H, H3, w, tree=None):
    """
    Local quadratic fits of width ``w`` to one horizon.

    Neighbours are sought in ``H3`` and fitted in ``H``. A window needs one
    point more than the fit has parameters, so it also measures the noise.
    Returns centre, unit normal, tilt and curvature-squared noise variances,
    curvature squared, residual sum of squares, degrees of freedom, and
    ``f2`` (the share of the window in the frame; below 1 when a trace runs
    oblique to a profile). ``None`` when there are too few points.
    """
    de = H.shape[1]
    npar = 3 if de == 2 else 6
    if len(H) <= npar:
        return None
    if tree is None:
        tree = _scipy()[0](H3)
    # About four centres per window. Overlapping windows repeat each other.
    free = np.ones(len(H3), bool)
    centres = []
    for i in range(len(H3)):
        if free[i]:
            centres.append(i)
            free[tree.query_ball_point(H3[i], 0.25 * w)] = False
    rows = []
    for c, c3 in zip(H[centres], H3[centres]):
        idx = tree.query_ball_point(c3, 0.5 * w)
        if len(idx) <= npar:
            continue
        Q = H[idx] - c
        Qc = Q - Q.mean(0)
        f2 = min(float(np.sum(Qc ** 2) / max(np.sum((H3[idx] - H3[idx].mean(0)) ** 2), 1e-30)), 1.0)
        Vt = np.linalg.svd(Qc, full_matrices=False)[2]
        n0, T = Vt[-1], Vt[:-1]
        x, z = Q @ T.T, Q @ n0
        if de == 2:
            A = np.c_[np.ones(len(x)), x[:, 0], x[:, 0] ** 2]
        else:
            A = np.c_[np.ones(len(x)), x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 0] * x[:, 1], x[:, 1] ** 2]
        G = np.linalg.pinv(A.T @ A)
        coef = G @ A.T @ z
        res = z - A @ coef
        if de == 2:
            slope = coef[1]
            n = n0 - slope * T[0]
            k2 = (2 * coef[2]) ** 2 / (1 + slope * slope) ** 3
            k2_var, t_var = 4 * G[2, 2], G[1, 1]
        else:
            n = n0 - coef[1:3] @ T
            hxx, hxy, hyy = 2 * coef[3], coef[4], 2 * coef[5]
            k2 = 0.5 * (hxx ** 2 + 2 * hxy ** 2 + hyy ** 2)
            k2_var = 0.5 * (4 * G[3, 3] + 2 * G[4, 4] + 4 * G[5, 5])
            t_var = 0.5 * (G[1, 1] + G[2, 2])
        rows.append((c, _unit(n), t_var, k2_var, k2, float(res @ res), len(idx) - npar, f2))
    if not rows:
        return None
    keys = ("centre", "normal", "t_var", "k2_var", "k2", "rss", "dof", "f2")
    return {k: np.asarray(v, float) for k, v in zip(keys, zip(*rows))}


def _trust_region(A, b):
    """The unit vector minimising vᵀAv − 2bᵀv (A symmetric, positive semi-definite)."""
    lam, U = np.linalg.eigh(A)
    c = U.T @ b
    nc = float(np.linalg.norm(c))
    if nc < 1e-12:
        return U[:, 0]
    scale = max(float(np.abs(lam).max()), nc)
    f = lambda mu: float(np.sum(c ** 2 / (lam - mu) ** 2)) - 1.0
    hi = lam[0] - 1e-12 * scale
    if f(hi) <= 0:   # b (almost) has no part along the smallest direction: the "hard case"
        gap = lam - lam[0]
        y = np.where(gap > 1e-12 * scale, c / np.where(gap > 0, gap, 1.0), 0.0)
        y[0] = math.sqrt(max(1.0 - float(y @ y), 0.0))
        return _unit(U @ y)
    lo = lam[0] - nc
    brentq = _scipy()[1]
    mu = lo if f(lo) >= 0 else brentq(f, lo, hi, xtol=1e-14 * scale, maxiter=200)
    return _unit(U @ (c / (lam - mu)))


def _layer_cake(heights, Nb, wb, start, m0, m2, iters=30):
    """
    The best flat layer cake: a unit normal shared by the horizons in
    ``heights`` (contact positions, levels free) and the bedding ``Nb``
    (weights ``wb``), maximising the Gaussian likelihood of Σh²/2m0 + Σθ²/2m2
    (θ the tilt, 1 − cos θ ≈ θ²/2), with m0 and m2 re-estimated in turn.

    Returns the normal, m0 (``None`` without heights) and m2 of the bedding.
    """
    Q = [h - h.mean(0) for h in heights if len(h) >= 2]
    A = sum((q.T @ q for q in Q), np.zeros((Nb.shape[1],) * 2))
    b = (Nb * wb[:, None]).sum(0)
    dof = max(sum(len(q) for q in Q) - len(Q) - (Nb.shape[1] - 1), 1)
    g = start
    for _ in range(iters):
        if Q:
            new = _trust_region(A / m0, b / m2)
        else:
            new = _unit(b) if np.linalg.norm(b) > 0 else start
        if Q:
            m0 = max(float(sum(np.sum((q @ new) ** 2) for q in Q)) / dof, 1e-30)
        m2 = max(float(np.sum(wb * np.arccos(np.clip(Nb @ new, -1, 1)) ** 2) / wb.sum()), 1e-30)
        done = np.linalg.norm(new - g) < 1e-9
        g = new
        if done:
            break
    return g, (m0 if Q else None), m2


# --------------------------------------------------------------------------- moments

def _prepare_spectral_context(C, axis):
    """Static geometry for spectral moments (coordinate trees, profile frame)."""
    _check_cset_frame(C)
    P, N, signed = _bedding(C)
    if P is None or len(P) < 3:
        raise ValueError(
            "spectral_moments needs at least three bedding orientations; "
            "set length_scale_range by hand instead"
        )
    d = P.shape[1]
    start = _first_normal(N, signed, d)
    N = np.where((~signed)[:, None] & ((N @ start) < 0)[:, None], -N, N)
    E, ax = _profile(N, axis, start)
    Xb, Nb, g0 = P @ E.T, _unit(N @ E.T), _unit(start @ E.T)
    H3 = _horizons(C)
    H = [h @ E.T for h in H3]
    D = float(np.linalg.norm(np.ptp(np.vstack([P] + H3), axis=0)))
    cKDTree = _scipy()[0]
    btree = cKDTree(Xb)
    horizon_trees = [cKDTree(h3) for h3 in H3]
    m2_0 = float(np.mean(np.arccos(np.clip(Nb @ g0, -1, 1)) ** 2)) or 1.0
    h0 = [(h - h.mean(0)) @ g0 for h in H if len(h) >= 2]
    m0_0 = float(np.mean(np.concatenate(h0) ** 2)) if h0 else 1.0
    return SimpleNamespace(
        P=P, N=N, signed=signed, d=d, E=E, ax=ax,
        Xb=Xb, Nb=Nb, g0=g0, H3=H3, H=H, D=D,
        btree=btree, horizon_trees=horizon_trees,
        m2_0=m2_0, m0_0=m0_0,
    )


def _longer_mode(kappa, weights, *, min_octaves=1.0, min_weight=0.1, min_count=3):
    """
    Mask of the longer-wavelength curvature group, and a summary of the split.

    Sorted curvatures are two modes when some neighbour gap is at least
    ``min_octaves`` (a factor ``2**min_octaves``) and each side holds at least
    ``min_weight`` of the weight and ``min_count`` entries. The widest such gap
    is the split. A continuous spread, as in one fold, does not split: ``keep``
    is all true and the summary is ``None``.
    """
    k = np.asarray(kappa, float)
    w = np.asarray(weights, float)
    keep = np.ones(len(k), bool)
    pos = np.where(k > 0)[0]
    if len(pos) < 2 * min_count or not w[pos].sum() > 0:
        return keep, None
    order = pos[np.argsort(k[pos])]
    lk = np.log2(k[order])
    gap = np.diff(lk)
    below = np.cumsum(w[order])[:-1] / w[order].sum()          # weight below each gap
    count = np.arange(1, len(order))
    ok = (gap >= min_octaves) & (below >= min_weight) & (1 - below >= min_weight) \
        & (count >= min_count) & (len(order) - count >= min_count)
    if not ok.any():
        return keep, None
    j = int(np.argmax(np.where(ok, gap, -np.inf)))
    longer = np.zeros(len(k), bool)
    longer[order[: j + 1]] = True
    longer[k <= 0] = True                                       # zero curvature belongs to the longer group
    return longer, dict(
        gap_octaves=float(gap[j]), weight_longer=float(below[j]), weight_shorter=float(1 - below[j]),
        curvature_longer=float(np.median(k[order[: j + 1]])), curvature_shorter=float(np.median(k[order[j + 1:]])),
        n_longer=int(longer.sum()), n_shorter=int(len(k) - longer.sum()),
    )


def _moment_at_lam(ctx, lam, *, resolved=True, return_samples=False):
    """Spectral moments at trial scale ``lam`` (see :func:`spectral_moments`)."""
    w = WINDOW_FRACTION * lam
    d = ctx.d
    P, N, E, ax = ctx.P, ctx.N, ctx.E, ctx.ax
    Xb, Nb, g0 = ctx.Xb, ctx.Nb, ctx.g0
    H, H3 = ctx.H, ctx.H3
    btree = ctx.btree
    m2_0, m0_0 = ctx.m2_0, ctx.m0_0

    fits = []
    if resolved:
        for h, h3, tree in zip(H, H3, ctx.horizon_trees):
            f = _fits(h, h3, w, tree=tree)
            if f is not None:
                fits.append(f)
    F = {k: np.concatenate([f[k] for f in fits]) for k in fits[0]} if fits else None
    if F is not None:
        _, j = btree.query(F["centre"], min(2 * d, len(Xb)))
        dots = np.einsum("wd,wkd->wk", F["normal"], Nb[j.reshape(len(F["centre"]), -1)])
        pick = dots[np.arange(len(dots)), np.argmax(np.abs(dots), axis=1)]
        F["normal"] = np.where((pick < 0)[:, None], -F["normal"], F["normal"])
    noise = float(F["rss"].sum() / F["dof"].sum()) if F is not None and F["dof"].sum() > 0 else 0.0

    cells = [_cell_means(h, w) for h in H]
    wb = _declustered(Xb, w)
    # Heights only from horizons spanning a fold: extent along the layering at least
    # λ0 (never taken below λ), found together with λ0.
    g, m0, lam0, seen = g0, None, lam, []
    span = np.zeros(len(H), bool)
    for _ in range(12):
        span = np.array(
            [np.linalg.norm(np.ptp(h - np.outer(h @ g, g), axis=0)) >= lam0 for h in H],
            bool,
        )
        if any(np.array_equal(span, s) for s in seen):
            break
        seen.append(span)
        g, m0, m2b = _layer_cake(
            [c for c, s in zip(cells, span) if s], Nb, wb, g0, m0_0, m2_0
        )
        if m0 is None:
            break
        lam0 = max(lam, 2 * math.pi * math.sqrt(m0 / m2b))

    Ns = Nb if F is None else np.vstack([Nb, F["normal"]])
    Xs = Xb if F is None else np.vstack([Xb, F["centre"]])
    shape = np.ones(len(Xb)) if F is None else np.r_[np.ones(len(Xb)), F["f2"]]
    t_noise = np.zeros(len(Xb)) if F is None else np.r_[np.zeros(len(Xb)), noise * F["t_var"]]
    ws = _declustered(Xs, w) * shape
    cos = np.clip(Ns @ g, -1, 1)
    theta = np.arccos(cos)
    tilt = theta[:, None] * _unit(Ns - cos[:, None] * g)
    S = (tilt * ws[:, None]).T @ tilt / ws.sum()
    tr = float(np.trace(S))
    n_eff = float(tr ** 2 / np.trace(S @ S)) if tr > 0 else 1.0
    m2 = float(np.sum(ws * (theta ** 2 - t_noise)) / ws.sum()) / n_eff

    if resolved:
        ij, along, kappa, skipped = _pairs(Xb, Nb, radius=PAIR_FRACTION * lam)
    else:
        ij, along, kappa, skipped = _pairs(Xb, Nb, neighbours=_FALLBACK_NEIGHBOURS)
    Xk = 0.5 * (Xb[ij[:, 0]] + Xb[ij[:, 1]])
    k2, k_noise, k_shape = kappa ** 2, np.zeros(len(kappa)), np.ones(len(kappa))
    used = np.zeros(0, bool)
    if F is not None:
        fk2, fkn = F["k2"], noise * F["k2_var"]
        used = np.ones(len(fk2), bool)
        for _ in range(20):
            a = np.r_[k2 - k_noise, (fk2 - fkn)[used]]
            wa = _declustered(np.vstack([Xk, F["centre"][used]]), w) * np.r_[k_shape, F["f2"][used] ** 2]
            level = float(np.sum(wa * a) / wa.sum()) if len(a) and wa.sum() > 0 else 0.0
            new = fkn < max(level, 0.0)
            if np.array_equal(new, used):
                break
            used = new
        Xk = np.vstack([Xk, F["centre"][used]])
        k2, k_noise = np.r_[k2, fk2[used]], np.r_[k_noise, fkn[used]]
        k_shape = np.r_[k_shape, F["f2"][used] ** 2]
    if len(k2) == 0:
        return None
    wk = _declustered(Xk, w) * k_shape
    if not wk.sum() > 0:
        return None
    modes = None
    if getattr(ctx, "prefer_longer", False):
        keep_k, modes = _longer_mode(np.sqrt(np.clip(k2 - k_noise, 0.0, None)), wk)
        if modes is not None and wk[keep_k].sum() > 0:      # two distinct curvature modes: the longer one
            wk, k2, k_noise = wk[keep_k], k2[keep_k], k_noise[keep_k]
        else:
            modes = None
    m4 = float(np.sum(wk * (k2 - k_noise)) / wk.sum()) / n_eff
    if not (m4 > 0 and m2 > 0):
        return None
    se = float(np.sqrt(np.sum(wk ** 2 * (k2 - k_noise - m4 * n_eff) ** 2)) / wk.sum()) / n_eff
    lam_x = 2 * math.pi * math.sqrt(m2 / m4)
    lam_0 = 2 * math.pi * math.sqrt(m0 / m2) if m0 else None
    if lam_0 is not None and lam_0 < lam_x:
        lam_0 = None
    out = dict(
        m0=m0, m2=m2, m4=m4, lambda_x=lam_x, lambda_0=lam_0,
        normal=(g @ E).tolist(), normal_from_contacts=m0 is not None,
        resolved=resolved,
        lambda_x_error=0.5 * se / m4 / math.log(2) if resolved else None,
        n_curvature=float(wk.sum() ** 2 / np.sum(wk ** 2)),
        window=w if F is not None else None,
        contact_noise=math.sqrt(noise) if F is not None else None,
        n_eff=n_eff, n_bedding=int(len(P)), n_contacts=int(sum(len(h) for h in H)),
        n_pairs=int(len(ij)), n_windows=int(used.sum()), n_horizons_spanning=int(span.sum()),
        profile=ax is not None, diameter=ctx.D, curvature_modes=modes,
    )
    if return_samples:
        used_cells = (
            [c for c, s in zip(cells, span) if s and len(c) >= 2] if m0 is not None else []
        )
        out["samples"] = dict(
            bedding_xyz=P, bedding_profile=Xb, normals=N, normal=g @ E,
            tilt_angle=theta[:len(Xb)],
            pairs=ij, pair_distance=along, pair_curvature=kappa,
            pairs_skipped=skipped, axis=ax,
            windows=None if F is None else dict(
                centre=F["centre"], normal=F["normal"] @ E,
                curvature=np.sqrt(F["k2"]), used=used,
            ),
            horizons=used_cells,
            heights=[(c - c.mean(0)) @ g for c in used_cells],
        )
    return out


def spectral_moments(C, *, axis=None, return_samples: bool = False, orientations_only: bool = False,
                     prefer_longer: bool = False) -> dict:
    """
    The spectral moments of the folded layering sampled by ``C``, the reference
    normal they are measured about, and the wavelengths they imply (see the
    module docstring).

    Parameters
    ----------
    C : CSet
        Uses the bedding (``gp``/``gv``, and ``gop``/``gov`` without polarity)
        and the contacts (``eq`` traces, or ``vp`` grouped by ``vv``).
    axis : array-like | False | None
        3-D only. The fold axis, projected out; ``False`` never projects;
        ``None`` (default) projects the direction most nearly normal to all
        bedding normals when they lie near a great circle.
    orientations_only : bool
        Ignore contacts. Then ``m0`` and ``lambda_0`` are ``None``, the normal
        and ``m2``, ``m4`` come from the bedding, and ``diameter`` is the
        bedding extent.
    prefer_longer : bool
        If curvatures split into two groups at least an octave apart, each with
        at least 10 % of the weight, ``m4`` uses only the longer group. A few
        tight hinges would otherwise dominate ``m4`` and pull ``λx`` short.
        A continuous spread is left as it is. ``curvature_modes`` records the
        split, or ``None``.
    return_samples : bool
        Also return, under ``samples``, what the moments are averaged from, for
        plotting: the bedding (``bedding_xyz``, ``bedding_profile``,
        ``normals``, ``tilt_angle``), the reference ``normal``, the curvature
        pairs (``pairs``, ``pair_distance``, ``pair_curvature``, and
        ``pairs_skipped`` across the layering), the contact ``windows``
        (``centre``, ``normal``, ``curvature``, ``used``; ``None`` without
        any), the heights (``horizons``, cell-averaged contacts in the frame,
        and ``heights``), ``axis``, and the ``scan`` of λ (``lam``,
        ``lambda_x``, ``resolved``).

    Returns
    -------
    dict with ``m0``, ``m2``, ``m4``, ``lambda_x`` (hinge spacing), ``lambda_0``
    (limb spacing, or ``None``), ``normal`` (the reference, as a list),
    ``normal_from_contacts`` (whether horizons spanning a fold helped set it),
    ``resolved`` (whether ``lambda_x`` is resolved), ``lambda_x_error`` (its
    standard error in octaves), ``n_curvature`` (effective curvature samples),
    ``window`` (``w``), ``contact_noise`` (rms residual of the fits), ``n_eff``,
    ``n_bedding``, ``n_contacts``, ``n_pairs``, ``n_windows``,
    ``n_horizons_spanning``, ``profile`` and ``diameter`` (data extent).
    """
    if orientations_only:
        C = _orientations_only(C)
    ctx = _prepare_spectral_context(C, axis)
    ctx.prefer_longer = bool(prefer_longer)
    D = ctx.D
    steps, octaves = _SCAN
    scan = []

    def trial(lam):
        e = _moment_at_lam(ctx, lam, resolved=True, return_samples=False)
        ok = (
            e is not None
            and e["n_curvature"] >= MIN_SAMPLES
            and e["lambda_x_error"] < 0.5
            and e["lambda_x"] <= D   # longer than the data: no fold was measured
        )
        scan.append((lam, e["lambda_x"] if e else np.nan, ok, e))
        return scan[-1]

    def miss(s):
        return abs(math.log2(s[1] / s[0]))

    # Coarse pass, half an octave a step, then the bracket refined to 1/steps of an octave.
    pick, below, prev = None, False, None
    for lam in D * 2.0 ** (-np.arange(2 * octaves + 1) / 2):
        s = trial(lam)
        if not s[2]:
            if below:
                break
            continue
        if s[1] < lam:
            below, prev = True, s
        elif below:
            pick = s
            for lam_f in prev[0] * 2.0 ** (-np.arange(1, steps // 2) / steps):
                f = trial(lam_f)
                if f[2] and f[1] >= lam_f:
                    pick = f
                    break
            break
    if pick is None and below:
        # never reached: the estimate closest to its own λ, over the fine grid of the run
        run = [s[0] for s in scan if s[2]]
        hi, lo = max(run), min(run)
        n = int(round(steps * math.log2(D / hi)))
        while D * 2.0 ** (-n / steps) >= lo * 2.0 ** (-0.5):
            lam_f = D * 2.0 ** (-n / steps)
            if not any(abs(math.log2(s[0] / lam_f)) < 1e-9 for s in scan):
                trial(lam_f)
            n += 1
        pick = min((s for s in scan if s[2]), key=miss)
    scan.sort(key=lambda s: -s[0])
    chosen_lam = pick[0] if pick is not None else None
    pick = pick[3] if pick is not None else None
    if pick is None:
        first = _moment_at_lam(ctx, D, resolved=False, return_samples=False)
        if first is None:
            raise ValueError(
                "the bedding orientations do not change between neighbours; "
                "set length_scale_range by hand instead"
            )
        chosen_lam = first["lambda_x"]
        resolved_flag = False
        pick = _moment_at_lam(
            ctx, chosen_lam, resolved=False, return_samples=return_samples
        )
    elif return_samples:
        pick = _moment_at_lam(
            ctx, chosen_lam, resolved=True, return_samples=True
        )
    if return_samples:
        pick["samples"]["scan"] = dict(
            lam=np.array([s[0] for s in scan]),
            lambda_x=np.array([s[1] for s in scan]),
            resolved=np.array([s[2] for s in scan]),
        )
    _warn_moment_quality(pick, stacklevel=2)
    return pick


# --------------------------------------------------------------------------- strain

def _spd(L, f):
    """``f`` applied to the eigenvalues of the symmetric matrix ``L``."""
    w, v = np.linalg.eigh(L)
    return (v * f(w)) @ v.T


def _eigenframes_from_bedding(sp, gp, gv):
    """Minor/major eigenvector frames at strain markers from the nearest bedding reading."""
    sp = _np(sp).reshape(-1, sp.shape[-1])
    gp = _np(gp).reshape(-1, gp.shape[-1])
    gv = _unit(_np(gv).reshape(-1, gv.shape[-1]))
    cKDTree = _scipy()[0]
    _, idx = cKDTree(gp).query(sp)
    n = gv[idx]
    d = sp.shape[1]
    frames = np.zeros((len(sp), d, d))
    for i in range(len(sp)):
        ni = n[i]
        if d == 2:
            t = _unit(np.array([-ni[1], ni[0]]))
            frames[i] = np.column_stack([ni, t])
        else:
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(ni, ref)) > 0.95:
                ref = np.array([1.0, 0.0, 0.0])
            t = _unit(np.cross(ni, ref))
            m = _unit(np.cross(t, ni))
            frames[i] = np.column_stack([ni, t, m])
    return frames


def bulk_strain(C):
    """
    The bulk strain of the strain ellipses in ``C``, or ``None`` without them.

    ``se`` holds the eigenvectors of the spatial ``B = F Fᵀ`` (columns minor to
    major) and ``sr`` the ratio of its major to minor eigenvalue. Each marker's
    unit-determinant ``log B`` is built from them (in 3-D with the intermediate
    eigenvalue 1, i.e. plane strain, since only the one ratio is observed), and
    the bulk strain is their mean, the log-Euclidean mean of ``B``.

    If only ``se`` is given, ``sr`` defaults to 1 (no in-plane ratio observed).
    If only ``sr`` is given, ``se`` is built at ``sp`` from bedding normals at
    the nearest ``gp``.

    Returns a dict with ``log_B`` (the mean of ``log B``), ``F`` (``B̄^½``),
    ``heterogeneity`` (``r``, the rms of ``‖log B − log B̄‖`` over ``‖log B̄‖``;
    ``inf`` when the mean strain is zero), ``X_over_Z`` (the bulk ratio of the
    principal stretches) and ``n_markers``.
    """
    se, sr = getattr(C, "se", None), getattr(C, "sr", None)
    if se is None and sr is None:
        return None
    if se is not None:
        V = _np(se)
        n_markers = V.shape[0]
    else:
        sp = getattr(C, "sp", None)
        gp, gv = getattr(C, "gp", None), getattr(C, "gv", None)
        if sp is None or gp is None or gv is None:
            return None
        V = _eigenframes_from_bedding(sp, gp, gv)
        n_markers = V.shape[0]
    if sr is not None:
        r = _np(sr).ravel()
        if len(r) != n_markers:
            raise ValueError(f"se has {n_markers} markers but sr has {len(r)}")
    else:
        r = np.ones(n_markers)
    d = V.shape[-1]
    half = 0.5 * np.log(np.clip(r, 1.0, None))
    lam = np.zeros((len(r), d))
    lam[:, 0], lam[:, -1] = -half, half
    L = np.einsum("nij,nj,nkj->nik", V, lam, V)
    Lbar = L.mean(0)
    norm = float(np.linalg.norm(Lbar))
    spread = float(np.sqrt(np.mean(np.sum((L - Lbar) ** 2, axis=(1, 2)))))
    w = np.linalg.eigvalsh(Lbar)
    return dict(log_B=Lbar, F=_spd(Lbar, lambda e: np.exp(0.5 * e)),
                heterogeneity=spread / norm if norm > 0 else math.inf,
                X_over_Z=float(np.exp(0.5 * (w[-1] - w[0]))), n_markers=int(len(r)))


def _unstrained(C, F):
    """The contacts and bedding of ``C`` pulled back through ``F``: ``x → F⁻¹x``, ``n → Fᵀn``."""
    Fi = np.linalg.inv(F)
    out = SimpleNamespace(eq=None, vp=None, vv=None, gp=None, gv=None, gop=None, gov=None)
    hz = _horizons(C)
    if hz:
        out.eq = [h @ Fi.T for h in hz]
    for pos, vec in (("gp", "gv"), ("gop", "gov")):
        p, v = getattr(C, pos, None), getattr(C, vec, None)
        if p is not None and v is not None:
            p, v = _np(p), _np(v)
            setattr(out, pos, p.reshape(-1, p.shape[-1]) @ Fi.T)
            setattr(out, vec, _unit(v.reshape(-1, v.shape[-1]) @ F))
    return out


# --------------------------------------------------------------------------- band

def spectral_band(C, *, axis=None, min_octaves: float = 2.0, support: float = 1.0,
                  return_info: bool = False, orientations_only: bool = False, prefer_longer: bool = False):
    """
    ``(L_min, L_max)`` from :func:`spectral_moments`.

    ``L_min = min(λx, λ0)``, or ``λx`` alone when ``λ0`` is missing
    (including ``orientations_only``, which also takes ``D`` from the bedding).
    ``prefer_longer`` is passed through to the moments.
    ``L_max = max(λx, λ0, 2**min_octaves · L_min, support · D, π D / √(2 r))``.
    The last term is present only when strain ellipses were removed first.

    Returns
    -------
    (float, float), or ((float, float), dict) with the moments (as
    :func:`spectral_moments` returns them), ``diameter`` and ``strain`` (the
    bulk strain summary and its ``L_max`` term, or ``None``) when
    ``return_info`` is True. The dict holds only numbers, booleans, lists and
    ``None``, so it can be saved as JSON.
    """
    _check_cset_frame(C)
    strain = bulk_strain(C)
    data = _unstrained(C, strain["F"]) if strain is not None else C
    mom = spectral_moments(data, axis=axis, orientations_only=orientations_only, prefer_longer=prefer_longer)
    mom["orientations_only"] = bool(orientations_only)
    D = mom["diameter"]
    scales = [mom["lambda_x"]] + ([mom["lambda_0"]] if mom["lambda_0"] else [])
    lo = min(scales)
    terms = [max(scales), lo * 2.0 ** min_octaves, support * D]
    mom["strain"] = None
    if strain is not None:
        r = strain["heterogeneity"]
        l_bulk = math.pi * D / math.sqrt(2.0 * r) if r > 0 else math.inf
        if math.isfinite(l_bulk):
            terms.append(l_bulk)
        mom["strain"] = dict(X_over_Z=strain["X_over_Z"], heterogeneity=r, n_markers=strain["n_markers"],
                             L_bulk=l_bulk, F=strain["F"].tolist())
    band = (float(lo), float(max(terms)))
    return (band, mom) if return_info else band


def n_features_for_band(length_scale_range, modes_per_octave: float = 100.0,
                        n_directions: Optional[int] = None) -> int:
    """
    Mode count for a band at a fixed number of modes per octave.

    Wavelengths are drawn log-uniformly, so every octave of the band holds the
    same number of modes, ``M / log2(L_max / L_min)``. This returns
    ``ceil(modes_per_octave · log2(L_max / L_min))``, at least one octave's
    worth, rounded up to a multiple of ``n_directions`` when given so that
    every ray carries the same number of wavelengths.
    """
    lo, hi = (float(v) for v in length_scale_range)
    if not 0.0 < lo <= hi:
        raise ValueError(f"length_scale_range must satisfy 0 < min <= max, got {length_scale_range}")
    if modes_per_octave <= 0:
        raise ValueError("modes_per_octave must be positive")
    octaves = max(math.log2(hi / lo), 1.0)
    m = math.ceil(modes_per_octave * octaves)
    if n_directions:
        r = int(n_directions)
        m = r * math.ceil(m / r)
    return int(m)
