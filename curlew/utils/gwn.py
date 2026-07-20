"""
Generalized winding numbers and fault-geometry proximity helpers.

2-D: open polylines, sum of signed subtended angles (radians).
3-D: oriented open triangle soups, sum of signed solid angles (steradians;
Jacobson et al. 2013).
"""
from typing import Tuple

import torch
import torch.nn.functional as F


def _polyline_segment_angles(x: torch.Tensor, verts: torch.Tensor):
    """Per-segment signed subtended angles (N, S) and unit segment tangents (S, 2)."""
    v, p = verts.detach(), x.detach()
    pa = v[:-1].unsqueeze(0) - p.unsqueeze(1)
    pb = v[1:].unsqueeze(0) - p.unsqueeze(1)
    cross = pa[..., 0] * pb[..., 1] - pa[..., 1] * pb[..., 0]
    angles = torch.atan2(cross, (pa * pb).sum(-1))
    seg_tang = F.normalize(v[1:] - v[:-1], dim=-1)
    return angles, seg_tang, pa, pb


def gwn_polyline(x: torch.Tensor, verts: torch.Tensor) -> torch.Tensor:
    """
    Generalized winding number (sum of signed subtended angles, radians) of an
    open polyline at query points ``x`` (N, 2). Trace geometry is detached.
    """
    if verts.shape[-1] != 2:
        raise ValueError(
            "gwn_polyline expects 2-D traces; for 3-D faults use gwn_mesh on a "
            "triangle surface."
        )
    angles, _, _, _ = _polyline_segment_angles(x, verts)
    return angles.sum(-1)


def _solid_angle_triangle(
    x: torch.Tensor,
    v0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
) -> torch.Tensor:
    """Signed solid angle (steradians) of triangle (v0, v1, v2) at points x (N, 3)."""
    a = v0 - x
    b = v1 - x
    c = v2 - x
    al = a.norm(dim=-1)
    bl = b.norm(dim=-1)
    cl = c.norm(dim=-1)
    num = (torch.linalg.cross(a, b, dim=-1) * c).sum(dim=-1)
    den = (
        al * bl * cl
        + (a * b).sum(dim=-1) * cl
        + (b * c).sum(dim=-1) * al
        + (c * a).sum(dim=-1) * bl
    )
    return 2.0 * torch.atan2(num, den)


def gwn_mesh(
    x: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """
    Generalized winding number of an oriented open triangle mesh.

    Sum of signed solid angles (steradians) subtended by each face at query
    points ``x`` (N, 3). ``verts`` is (V, 3); ``faces`` is (F, 3) vertex
    indices. Mesh geometry is detached.
    """
    if x.shape[-1] != 3 or verts.shape[-1] != 3:
        raise ValueError("gwn_mesh expects 3-D coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (F, 3)")
    v, p, f = verts.detach(), x.detach(), faces
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    omega = _solid_angle_triangle(
        p[:, None, :],
        v0[None, :, :],
        v1[None, :, :],
        v2[None, :, :],
    )
    return omega.sum(dim=-1)


def gwn_ribbon_mesh(
    polyline: torch.Tensor,
    normals: torch.Tensor,
    *,
    half_width: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One-sided ribbon mesh along an open 3-D polyline.

    ``normals`` (P, 3) defines the fault-plane orientation at each vertex.
    Returns ``(verts, faces)`` suitable for :func:`gwn_mesh`. ``half_width``
    sets the extrusion; GWN magnitude scales with it, so treat as a shape
    parameter rather than a physical thickness.
    """
    if polyline.ndim != 2 or polyline.shape[1] != 3:
        raise ValueError("polyline must be (P, 3)")
    if normals.shape != polyline.shape:
        raise ValueError("normals must match polyline shape (P, 3)")
    if half_width <= 0:
        raise ValueError("half_width must be > 0")
    v = polyline.detach()
    n = F.normalize(normals.detach(), dim=-1, eps=1e-12)
    offset = n * float(half_width)
    top = v + offset
    verts = torch.cat([v, top], dim=0)
    p = len(v)
    faces = []
    for i in range(p - 1):
        faces.append([i, i + 1, p + i + 1])
        faces.append([i, p + i + 1, p + i])
    return verts, torch.tensor(faces, dtype=torch.long, device=v.device)


def _closest_point_triangle(
    p: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
):
    """Closest point on triangle (a,b,c) for points ``p`` (N, 3); vertices are (3,)."""
    ab, ac = b - a, c - a
    n_hat = F.normalize(torch.linalg.cross(ab, ac), dim=-1, eps=1e-12)
    ap = p - a
    dist_n = (ap * n_hat).sum(-1, keepdim=True)
    q = p - dist_n * n_hat

    def _seg(pa, pb, pts):
        d = pb - pa
        t = ((pts - pa) * d).sum(-1) / (d * d).sum().clamp_min(1e-12)
        t = t.clamp(0.0, 1.0)
        cp = pa + t.unsqueeze(-1) * d
        return cp, (pts - cp).norm(dim=-1)

    cp_a, d_a = _seg(a, b, p)
    cp_b, d_b = _seg(b, c, p)
    cp_c, d_c = _seg(c, a, p)
    cp_v0 = a.unsqueeze(0).expand_as(p)
    d_v0 = (p - cp_v0).norm(dim=-1)
    cp_v1 = b.unsqueeze(0).expand_as(p)
    d_v1 = (p - cp_v1).norm(dim=-1)
    cp_v2 = c.unsqueeze(0).expand_as(p)
    d_v2 = (p - cp_v2).norm(dim=-1)

    v0, v1, v2 = c - a, b - a, q - a
    d00 = (v0 * v0).sum()
    d01 = (v0 * v1).sum()
    d11 = (v1 * v1).sum()
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = (d00 * d11 - d01 * d01).clamp_min(1e-12)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    inside = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)
    cp_face = a + u.unsqueeze(-1) * v1 + v.unsqueeze(-1) * v0
    d_face = (p - cp_face).norm(dim=-1)

    cp, d = cp_a, d_a
    for cp_i, d_i in ((cp_b, d_b), (cp_c, d_c), (cp_v0, d_v0), (cp_v1, d_v1), (cp_v2, d_v2)):
        pick = d_i < d
        cp = torch.where(pick.unsqueeze(-1), cp_i, cp)
        d = torch.where(pick, d_i, d)
    pick = inside & (d_face < d)
    cp = torch.where(pick.unsqueeze(-1), cp_face, cp)
    d = torch.where(pick, d_face, d)
    return cp, d


def mesh_boundary_vertices(faces: torch.Tensor) -> torch.Tensor:
    """Vertex indices on the open boundary of a triangle mesh."""
    counts = {}
    for tri in faces.tolist():
        for i, j in ((0, 1), (1, 2), (2, 0)):
            key = (min(tri[i], tri[j]), max(tri[i], tri[j]))
            counts[key] = counts.get(key, 0) + 1
    verts = set()
    for (i, j), c in counts.items():
        if c == 1:
            verts.add(i)
            verts.add(j)
    if not verts:
        return faces.new_zeros(0)
    return torch.tensor(sorted(verts), dtype=faces.dtype, device=faces.device)


def closest_point_mesh(
    x: torch.Tensor,
    verts: torch.Tensor,
    faces: torch.Tensor,
):
    """
    Closest point, distance and unit face normal on an open triangle mesh.

    Returns ``(cp, dist, normal)`` with shapes ``(N, 3)``, ``(N,)``, ``(N, 3)``.
    """
    v, p, f = verts.detach(), x.detach(), faces
    best_cp = None
    best_d = None
    best_n = None
    for fi in f:
        a, b, c = v[fi[0]], v[fi[1]], v[fi[2]]
        cp, d = _closest_point_triangle(p, a, b, c)
        fn = F.normalize(torch.linalg.cross(b - a, c - a), dim=-1, eps=1e-12)
        n = fn.unsqueeze(0).expand(len(p), -1)
        if best_d is None:
            best_cp, best_d, best_n = cp, d, n
        else:
            pick = d < best_d
            best_cp = torch.where(pick.unsqueeze(-1), cp, best_cp)
            best_d = torch.where(pick, d, best_d)
            best_n = torch.where(pick.unsqueeze(-1), n, best_n)
    return best_cp, best_d, best_n


def blended_tangency_polyline(
    x: torch.Tensor,
    verts: torch.Tensor,
    *,
    blend_power: float = 1.0,
):
    """
    Strike frame from a blend of segment tangents weighted by the subtended-
    angle magnitudes — the same angular visibility GWN already uses.

    ``blend_power`` sharpens the blend: ``1`` is the default soft corner
    rotation; larger values (e.g. ``4``) favour the dominant visible segment;
    very large values approximate closest-segment tangents.

    Returns closest point, blended unit tangent, blended distance.
    """
    if blend_power <= 0:
        raise ValueError(f"blend_power must be > 0, got {blend_power}")
    angles, seg_tang, pa, pb = _polyline_segment_angles(x, verts)
    v, p = verts.detach(), x.detach()
    weights = angles.abs().pow(blend_power)
    w_sum = weights.sum(-1)
    tang = (weights.unsqueeze(-1) * seg_tang.unsqueeze(0)).sum(1)
    small = w_sum < 1e-10
    if small.any():
        i = weights.argmax(dim=1)
        tang[small] = seg_tang[i[small]]
    tang = F.normalize(tang, dim=-1, eps=1e-12)
    a, ab = v[:-1], v[1:] - v[:-1]
    t = ((p[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)
    cp_seg = a.unsqueeze(0) + t.clamp(0.0, 1.0).unsqueeze(-1) * ab.unsqueeze(0)
    dist_seg = (p[:, None, :] - cp_seg).norm(dim=-1)
    dist = (weights * dist_seg).sum(-1) / w_sum.clamp_min(1e-12)
    if small.any():
        rows = torch.arange(len(p), device=p.device)
        dist[small] = dist_seg[rows[small], i[small]]
        cp = cp_seg[rows, 0].clone()
        cp[small] = cp_seg[rows[small], i[small]]
    else:
        cp = (weights.unsqueeze(-1) * cp_seg).sum(1) / w_sum.unsqueeze(-1).clamp_min(1e-12)
    return cp, tang, dist


def closest_point_polyline(x: torch.Tensor, verts: torch.Tensor):
    """Closest point, unit tangent and distance to an open polyline."""
    v, p = verts.detach(), x.detach()
    a, ab = v[:-1], v[1:] - v[:-1]
    t = ((p[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)
    cp = a + t.clamp(0.0, 1.0)[..., None] * ab
    d = (p[:, None] - cp).norm(dim=-1)
    i = d.argmin(dim=1)
    rows = torch.arange(len(p), device=p.device)
    tang = F.normalize(ab[i], dim=-1)
    return cp[rows, i], tang, d[rows, i]


def smoothstep(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoothstep_scalar(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)
