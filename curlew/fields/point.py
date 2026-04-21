"""
Point kernel fields (PKF): implicit fields defined by a set of seed points with optional normals and values.

Distances from query points to seeds are combined with per-seed values via a kernel (e.g. closest, mean,
linear, gaussian, ray, or a callable). Values can be a fixed array per point or a callable (e.g. distance,
signedDistance) that may support onlyClosest for efficiency. Positions, normals and values can optionally
be nn.Parameters for learning during training.
"""

import math
import torch
import torch.nn as nn
from curlew.fields import BaseSF
from curlew import _tensor
import curlew
import numpy as np

from typing import Union, Optional, Callable, Dict
ArrayLike = Union[torch.Tensor, np.ndarray]

# Keys for the learnable dict: which of positions, normals, values are nn.Parameter
LEARNABLE_KEYS = frozenset({"positions", "normals", "values"})

# Small constant used in distance-based kernels to avoid division by zero.
DEFAULT_EPS = 1e-6
# Floor for kernel weights before normalisation; improves far-field and numerical stability.
# Set kernel_kwargs["min_weight"] = 0 to disable for individual fields (or globally by setting MINIMUM_WEIGHT).
MINIMUM_WEIGHT = 1e-9

def _normalize_learnable(learnable: Union[bool, Dict[str, bool]]) -> Dict[str, bool]:
    """
    Convert learnable (bool or dict) to a dict with keys positions, normals, values.

    Parameters
    ----------
    learnable : bool or dict
        If True, all keys are True; if False, all False. If a dict, only listed keys
        are updated; missing keys default to False.

    Returns
    -------
    dict
        Dict with keys in LEARNABLE_KEYS and bool values.
    """
    if isinstance(learnable, bool):
        return {k: learnable for k in LEARNABLE_KEYS}
    out = {k: False for k in LEARNABLE_KEYS}
    for k in LEARNABLE_KEYS:
        if k in learnable:
            out[k] = bool(learnable[k])
    return out


def _aggregate(arr: torch.Tensor, method: str, dim: int = 1) -> torch.Tensor:
    """
    Reduce the value matrix along the seed dimension (fast path for min, max, mean, sum).

    Parameters
    ----------
    arr : torch.Tensor
        Value matrix of shape (M, N) or (M, N, j).
    method : str
        One of "min", "max", "mean", "sum".
    dim : int, optional
        Dimension along which to reduce (default 1, i.e. over seeds).

    Returns
    -------
    torch.Tensor
        Shape (M,) or (M, j) after reduction.
    """
    if method == "min":
        return arr.min(dim=dim).values
    if method == "max":
        return arr.max(dim=dim).values
    if method == "mean":
        return arr.mean(dim=dim)
    if method == "sum":
        return arr.sum(dim=dim)
    raise ValueError(f"Unknown method '{method}'.")

# =============================================================================
# Value functions: (x, vectors, euclid, cid, seeds, normals, value_kwargs=None)
# Return (M, N, j) when cid is None, or (M, j) when cid is set; j=1 for scalar.
# euclid may be None when the kernel does not need it (e.g. sum with a custom callable); use vectors if needed.
# =============================================================================

def dummy_value_function( x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    cid: Optional[torch.Tensor],
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_kwargs: Optional[dict] = None ) -> torch.Tensor:
    """
    Dummy value function documenting the value-function signature (for building your own).

    Value functions define the "value" at each (query, seed) pair. Examples: signed-distance
    for SDFs, or a double-couple for fault offsets. They are called with:
    x (M, dim), vectors (M, N, dim), euclid (M, N), cid (M,) or None, seeds (N, dim),
    normals (N, dim), value_kwargs. Return (M, N, j) when cid is None, or (M, j) when cid set.
    """
    return distance(x, vectors, euclid, cid, seeds, normals, value_kwargs)

def distance(
    x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    cid: Optional[torch.Tensor],
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Euclidean distance from each query to each seed.

    Returns
    -------
    torch.Tensor
        Shape (M, N, 1) when cid is None, or (M, 1) when cid is set.
    """
    if cid is not None:
        M = x.shape[0]
        return euclid[torch.arange(M, device=x.device), cid].unsqueeze(-1)
    return euclid.unsqueeze(-1)


def signedDistance(
    x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    cid: Optional[torch.Tensor],
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Signed distance along each seed normal: (query - seed) · normal.

    Uses vector · normal per (query, seed). If normals is None, falls back to Euclidean distance.
    Returns (M, N, 1) when cid is None, or (M, 1) when cid is set.
    """
    n = normals
    if cid is None:
        if n is not None:
            signed = (vectors * n.unsqueeze(0)).sum(dim=-1)
        else:
            signed = euclid
        return signed.unsqueeze(-1)
    M = x.shape[0]
    vec_nearest = vectors[torch.arange(M, device=vectors.device), cid, :]
    if n is not None:
        n_nearest = n[cid, :]
        return (vec_nearest * n_nearest).sum(dim=-1, keepdim=True)
    return euclid[torch.arange(M, device=x.device), cid].unsqueeze(-1)

def _moment_tensor(
    fault_normals: torch.Tensor,
    slip_directions: torch.Tensor,
    dilation_magnitude: float,
    slip_magnitude: float,
    mu: float = 20e9, # shear modulus
    poisson_ratio: float = 0.25,
) -> torch.Tensor:
    """
    Build physically rigorous moment tensors including Lamé parameters.
    
    If inputs are 2D, returns 3x3 tensors under the Plane Strain assumption.
    """
    n = fault_normals
    d = slip_directions
    
    # compute lame_lambda from shear modulus and poissons ratio
    lame_lambda = (2 * mu * poisson_ratio) / (1 - 2 * poisson_ratio)

    # 1. Validation and Normalization
    if n.shape != d.shape:
        raise ValueError("fault_normals and slip_directions must match.")
    
    input_dim = n.shape[-1]
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True).clamp(min=1e-8)
    d = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp(min=1e-8)
    
    # 2. Upgrade 2D to 3D for Plane Strain if necessary
    N = n.shape[0]
    if input_dim == 2:
        # Pad vectors with a zero Z component
        zeros = torch.zeros((N, 1), device=n.device, dtype=n.dtype)
        n_3d = torch.cat([n, zeros], dim=-1)
        d_3d = torch.cat([d, zeros], dim=-1)
    else:
        n_3d = n
        d_3d = d

    # 3. Compute Outer Products (N, 3, 3)
    # n ⊗ d + d ⊗ n (Shear)
    n_d = n_3d.unsqueeze(-1) * d_3d.unsqueeze(-2)
    d_n = d_3d.unsqueeze(-1) * n_3d.unsqueeze(-2)
    double_couple_potency = n_d + d_n
    
    # n ⊗ n (Tensile/Dilation)
    n_n = n_3d.unsqueeze(-1) * n_3d.unsqueeze(-2)
    
    # 4. Construct the Moment Tensor Components
    # M_ij = λ * (Δu · n) * δ_ij + μ * (Δu_i * n_j + Δu_j * n_i)
    
    # Isotropic part (from dilation)
    # Trace of the opening is dilation_magnitude * (n · n) = dilation_magnitude
    eye = torch.eye(3, device=n.device, dtype=n.dtype).unsqueeze(0)
    M_iso = lame_lambda * dilation_magnitude * eye
    
    # Deviatoric / Shear part
    # Note: Opening also contributes to the diagonal via 2 * mu * n_i * n_j
    M_shear = mu * (slip_magnitude * double_couple_potency)
    M_opening = 2 * mu * (dilation_magnitude * n_n)
    
    M = M_iso + M_shear + M_opening
    
    return M


def _displacementPointMomentTensor(
    r: torch.Tensor,
    M: torch.Tensor,
    mu: float = 20e9,
    nu: float = 0.25,
    r_min: float = 1e-1,
) -> torch.Tensor:
    """
    Displacement at observer due to a point moment tensor in an isotropic
    elastic full space (static 3D formula). M is always (..., 3, 3). When r
    has last dimension 2, in-plane displacement is returned (plane strain).
    Used for far-field comparison and for isotropic/Mogi-type tests.
    """
    out_dim = r.shape[-1]
    if out_dim not in (2, 3):
        raise ValueError("r must have last dimension 2 or 3.")
    if M.shape[-2:] != (3, 3):
        raise ValueError("M must have last two dimensions (3, 3).")
    if out_dim == 2:
        r = torch.cat([r, r.new_zeros((*r.shape[:-1], 1))], dim=-1)
    r_norm = torch.linalg.norm(r, dim=-1, keepdim=True).clamp(min=r_min)
    n = r / r_norm
    n_M_n = (n * torch.einsum("...kl,...l->...k", M, n)).sum(dim=-1, keepdim=True)
    trM = torch.einsum("...kk->...", M).unsqueeze(-1)
    M_n = torch.einsum("...kl,...l->...k", M, n)
    r_sq = (r_norm.squeeze(-1) ** 2).clamp(min=r_min ** 2).unsqueeze(-1)
    coeff = 1.0 / (16.0 * math.pi * mu * (1.0 - nu) * r_sq)
    u = coeff * (2.0 * M_n - trM * n + 3.0 * n_M_n * n)
    if out_dim == 2:
        u = u[..., :2]
    return u

def circular_eshelby_displacement(
    r_vec: torch.Tensor,
    M: torch.Tensor,
    near_radius: float,
    far_radius: float,
    mu: float,
    nu: float,
    r_min: float = 1e-8,
    normals: Optional[torch.Tensor] = None,
    slip_direction: Optional[torch.Tensor] = None,
    slip_magnitude: Optional[float] = None,
    thickness: Optional[float] = None,
    transition_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Displacement from a circular patch (Eshelby-style), with optional sharp
    near-field based on signed distance to the fault plane.

    near_radius: patch size for sharp near-field extent and transition (m).
    far_radius: length scale in far-field decay 1/(r^2 + far_radius^2); can be
      much larger than near_radius so displacement extends farther.

    When normals, slip_direction and slip_magnitude are provided:
      - Signed distance s = r_vec · n.
      - Near-field: u_near = sign(s) * (slip_magnitude/2) * d_unit.
      - Far-field: elastic solution with 1/(r^2 + far_radius^2).
      - Transition uses near_radius so blend is complete by ~near_radius.

    M is total moment (already scaled by patch area). r_vec last dim 2 or 3.
    """
    out_dim = r_vec.shape[-1]
    if out_dim not in (2, 3):
        raise ValueError("r_vec must have last dimension 2 or 3.")
    if M.shape[-2:] != (3, 3):
        raise ValueError("M must have last two dimensions (3, 3).")
    if out_dim == 2:
        r_vec_3d = torch.cat([r_vec, r_vec.new_zeros((*r_vec.shape[:-1], 1))], dim=-1)
    else:
        r_vec_3d = r_vec

    r_mag = torch.linalg.norm(r_vec_3d, dim=-1, keepdim=True).clamp(min=r_min)
    n_ray = r_vec_3d / r_mag
    r2 = r_mag ** 2
    R_far2 = far_radius ** 2
    K = 1.0 / (16.0 * math.pi * mu * (1.0 - nu))
    K_disk = K * (1.0 / (r2 + R_far2))
    trM = torch.einsum("...kk->...", M).unsqueeze(-1)
    M_n = torch.einsum("...kl,...l->...k", M, n_ray)
    n_M_n = (n_ray * M_n).sum(dim=-1, keepdim=True)
    u_far = K_disk * (2.0 * M_n - trM * n_ray + 3.0 * n_M_n * n_ray)

    use_sharp = (
        normals is not None
        and slip_direction is not None
        and slip_magnitude is not None
    )
    if use_sharp:
        n_fault = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-8)
        d_unit = slip_direction / torch.linalg.norm(slip_direction, dim=-1, keepdim=True).clamp(min=1e-8)
        if out_dim == 2:
            n_fault = torch.cat([n_fault, n_fault.new_zeros((*n_fault.shape[:-1], 1))], dim=-1)
            d_unit = torch.cat([d_unit, d_unit.new_zeros((*d_unit.shape[:-1], 1))], dim=-1)
        signed_dist = (r_vec_3d * n_fault).sum(dim=-1, keepdim=True)
        half_slip = (slip_magnitude * 0.5) * d_unit
        u_near = torch.sign(signed_dist) * half_slip
        L = transition_scale if transition_scale is not None else (near_radius * 0.5)
        transition = torch.sigmoid((r_mag.squeeze(-1) - near_radius) / L).unsqueeze(-1)
        u = (1.0 - transition) * u_near + transition * u_far
    else:
        u = u_far

    if out_dim == 2:
        u = u[..., :2]
    return u


def eshelbyDisplacement(
    x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    cid: Optional[torch.Tensor],
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Value function: displacement at observers due to ellipsoidal Eshelby inclusions
    (one per seed), with sharp near-field and elastic far-field.

    value_kwargs must contain: thickness, slip_direction, slip_magnitude,
    dilation_magnitude, mu, and nu (or lam). Patch size and decay scale:
      - near_radius: patch radius for (i) scaling moment by area π*near_radius² and
        (ii) extent of sharp near-field / transition. Required if far_radius not given.
      - far_radius: length scale in far-field decay 1/(r²+far_radius²); can be >> near_radius.
      - radius: optional shorthand; if given and near_radius/far_radius absent, used for both.

    Optional coverage_factor (float): if provided in value_kwargs, M is multiplied by
    it so the far-field sees the correct total moment when circles do not tile the fault
    (e.g. coverage_factor = total_fault_area / (N * π * near_radius²)).

    Returns (M, N, dim) when cid is None, or (M, dim) when cid is set.
    """
    kwargs = value_kwargs or {}
    if normals is None:
        raise ValueError("eshelbyDisplacement requires normals.")
    N, dim = normals.shape
    if dim not in (2, 3):
        raise ValueError("normals must have last dimension 2 or 3.")
    near_radius = kwargs.get("near_radius")
    far_radius = kwargs.get("far_radius")
    if near_radius is None and far_radius is None:
        near_radius = kwargs["radius"]
        far_radius = near_radius
    elif near_radius is None:
        near_radius = far_radius
    elif far_radius is None:
        far_radius = near_radius
    near_radius = float(near_radius)
    far_radius = float(far_radius)
    thickness = kwargs["thickness"]
    slip_magnitude = float(kwargs["slip_magnitude"])
    dilation_magnitude = float(kwargs["dilation_magnitude"])
    mu = float(kwargs["mu"])
    if "nu" in kwargs:
        nu = float(kwargs["nu"])
    elif "lam" in kwargs:
        lam = float(kwargs["lam"])
        nu = lam / (2.0 * (lam + mu))
    else:
        raise ValueError("value_kwargs must contain 'nu' or 'lam'.")
    slip_direction = kwargs["slip_direction"]
    if isinstance(slip_direction, (list, np.ndarray)):
        slip_direction = _tensor(slip_direction, dev=normals.device, dt=normals.dtype)
    else:
        slip_direction = slip_direction.to(device=normals.device, dtype=normals.dtype)
    if slip_direction.dim() == 1:
        slip_direction = slip_direction.unsqueeze(0).expand(N, dim)
    elif slip_direction.shape[0] != N or slip_direction.shape[1] != dim:
        raise ValueError("slip_direction must have shape (dim,) or (N, dim).")
    M = _moment_tensor(
        normals, slip_direction, dilation_magnitude, slip_magnitude,
        mu=mu, poisson_ratio=nu,
    )
    # Scale moment tensor by patch area (circle π*near_radius²) so total moment scales with patch size
    patch_area = math.pi * (near_radius ** 2)
    M = M * patch_area

    coverage_factor = kwargs.get("coverage_factor", 1)
    if coverage_factor is not None:
        M = M * float(coverage_factor)

    if cid is not None:
        M_size = x.shape[0]
        r_cid = vectors[torch.arange(M_size, device=x.device), cid, :]
        M_cid = M[cid]
        n_cid = normals[cid]
        d_cid = slip_direction[cid]
        u = circular_eshelby_displacement(
            r_cid, M_cid, near_radius, far_radius, mu, nu,
            normals=n_cid, slip_direction=d_cid, slip_magnitude=slip_magnitude,
            thickness=thickness,
        )
        return u
    M_expanded = M.unsqueeze(0)
    n_expanded = normals.unsqueeze(0)
    d_expanded = slip_direction.unsqueeze(0)
    u = circular_eshelby_displacement(
        vectors, M_expanded, near_radius, far_radius, mu, nu,
        normals=n_expanded, slip_direction=d_expanded, slip_magnitude=slip_magnitude,
        thickness=thickness,
    )
    return u

# =============================================================================
# Kernel (weight) functions: (x, vectors, euclid, seeds, normals, value_matrix, kernel_kwargs) -> (M, N)
# Weights are normalised per query and combined with value_matrix to give the field output.
# =============================================================================

def dummy_kernel_function(
    x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_matrix: torch.Tensor,
    kernel_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Dummy kernel documenting the kernel signature (for building your own).

    Kernel functions return non-negative weights (M, N). Built-in options: closest, mean,
    linear, gaussian, ray. Custom kernels receive the same arguments and kernel_kwargs from
    the constructor (e.g. sigma, eps, min_weight).
    """
    return linear(x, vectors, euclid, seeds, normals, value_matrix, kernel_kwargs)


def closest(
    x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_matrix: torch.Tensor,
    kernel_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Weight 1 at the nearest seed (by Euclidean distance), 0 elsewhere.

    Returns (M, N) with a single 1 per row at the argmin of euclid.
    """
    M, N = value_matrix.shape[0], value_matrix.shape[1]
    idx = torch.argmin(euclid, dim=1)
    w = torch.zeros(M, N, device=value_matrix.device, dtype=value_matrix.dtype)
    w[torch.arange(M, device=value_matrix.device), idx] = 1.0
    return w


def mean(
    x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_matrix: torch.Tensor,
    kernel_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Uniform weights 1/N per seed (all seeds contribute equally).
    """
    M, N = value_matrix.shape[0], value_matrix.shape[1]
    return torch.ones(M, N, device=value_matrix.device, dtype=value_matrix.dtype) / N


def linear(
    x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_matrix: torch.Tensor,
    kernel_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Inverse-distance weights 1 / (d + eps).

    eps is taken from kernel_kwargs.get("eps", DEFAULT_EPS).
    """
    kwargs = kernel_kwargs or {}
    eps = kwargs.get("eps", DEFAULT_EPS)
    return 1.0 / (euclid + eps)

def gaussian(
    x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_matrix: torch.Tensor,
    kernel_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Gaussian weights exp(-d² / (2 σ²)) in Euclidean distance.

    sigma from kernel_kwargs.get("sigma"); if None, uses mean(euclid) + eps.
    """
    kwargs = kernel_kwargs or {}
    sigma = kwargs.get("sigma")
    eps = kwargs.get("eps", DEFAULT_EPS)
    if sigma is None:
        sigma = euclid.mean().item() + eps
    return torch.exp(-(euclid ** 2) / (2.0 * sigma ** 2))


def ray(
    x: torch.Tensor,
    vectors: torch.Tensor,
    euclid: torch.Tensor,
    seeds: torch.Tensor,
    normals: Optional[torch.Tensor],
    value_matrix: torch.Tensor,
    kernel_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """
    Gaussian in the plane perpendicular to each seed normal (ray-like; no falloff along the normal).

    Query points are projected onto the tangent plane at each seed; weight is a 2D Gaussian in that
    plane. sigma_plane = sigma_base + tan(angle_deg) * d_perp + eps, so angle > 0 widens the cone.
    kernel_kwargs: "sigma" (base width), "angle" (cone half-angle in degrees), "eps". If normals
    is None, falls back to standard Gaussian in full Euclidean distance.
    """
    kwargs = kernel_kwargs or {}
    sigma_base = kwargs.get("sigma")
    eps = kwargs.get("eps", DEFAULT_EPS)
    if sigma_base is None:
        sigma_base = euclid.mean().item() + eps
    angle_deg = kwargs.get("angle", 0.0)
    angle_slope = math.tan(math.radians(angle_deg)) if angle_deg != 0 else 0.0
    if normals is None:
        return torch.exp(-(euclid ** 2) / (2.0 * sigma_base ** 2))
    # vectors (M, N, dim), normals (N, dim) -> dot = (v · n) per (i,j) -> (M, N)
    dot = (vectors * normals.unsqueeze(0)).sum(dim=-1)
    # d_perp^2 = |v|^2 - (v·n)^2; d_perp = distance in tangent plane from seed to (projected) query
    d_perp_sq = (euclid ** 2 - dot ** 2).clamp(min=0.0)
    d_perp = d_perp_sq.sqrt()
    # sigma in plane: base + tan(angle_deg) * d_perp (cone half-angle in degrees); avoid 0 at d_perp=0
    sigma_plane = sigma_base + angle_slope * d_perp + eps
    return torch.exp(-d_perp_sq / (2.0 * sigma_plane ** 2))


# Map from string kernel name to callable; None means use built-in fast path (closest/min/max/mean/sum)
KERNEL_STRINGS = {
    "closest": None,
    "mean": None,
    "min": None,
    "max": None,
    "sum": None,
    "linear": linear,
    "gaussian": gaussian,
    "ray": ray,
}


class PointKernelField(BaseSF):
    """
    Point kernel field (PKF): implicit field defined by interpolation over seed points with optional normals and values.

    Parameters (passed as keyword arguments through constructor)
    ---------------------------------------------------------------
    positions : array-like, required
        Seed positions, shape (N, input_dim). Required.
    normals : array-like, optional
        Outward normals at each seed, shape (N, input_dim). Required for signed-distance values and ray kernel.
    values : array-like or callable, optional
        Per-point values. If array: shape (N,) or (N, j) with j >= 1; (N,) is treated as (N, 1). If callable
        (e.g. point.distance, point.signedDistance), signature (x, vectors, euclid, cid, seeds, normals, value_kwargs)
        -> (M, N, j) or (M, j) when cid is set. Default when None: signedDistance if normals given, else distance.
    value_kwargs : dict, optional
        Passed to the values callable when values is a callable.
    kernel : str or callable, optional
        How to combine per-seed values. String: "closest", "mean", "min", "max", "sum", "linear", "gaussian", "ray"
        (or "nearest" for closest). Callable signature: (x, vectors, euclid, seeds, normals, value_matrix, kernel_kwargs)
        -> (M, N) weights. Default is "closest".
    kernel_kwargs : dict, optional
        Passed to the kernel callable; can include "sigma", "eps", "min_weight", "angle", etc. Weights are clamped
        to at least min_weight (default MINIMUM_WEIGHT) before normalisation unless min_weight=0.
    far_field : str, optional
        When total weight is below eps (query far from all seeds): "mean" uses the mean of the value matrix for
        that query; "zero" uses zero. Default is "zero".
    learnable : bool or dict, optional
        If True, positions, normals and values (when an array) are nn.Parameter; if False, buffers. If a dict,
        use keys "positions", "normals", "values" (each bool); missing keys default to False.
    observer_batch_size : int, optional
        When set and the number of query points M exceeds it, evaluation is done in chunks over observers
        to limit memory (M, N, dim). Default 100_000; pass None for no chunking. See also
        curlew.utils.batchEval for external batching.

    Performance (large M): set observer_batch_size (e.g. 50_000–200_000) to avoid OOM and often improve
    throughput; or use curlew.utils.batchEval to batch externally. For PyTorch 2+, wrapping the value
    callable (e.g. eshelbyDisplacement) or field.evaluate with torch.compile can reduce Python overhead.
    """

    def __init__(
        self,
        name: str = None,
        input_dim: int = None,
        output_dim: int = 1,
        drift=0,
        transform=None,
        local=None,
        seed: int = 42,
        positions: Optional[ArrayLike] = None,
        normals: Optional[ArrayLike] = None,
        values: Optional[Union[ArrayLike, Callable]] = None,
        value_kwargs: Optional[dict] = None,
        kernel: Union[str, Callable] = "closest",
        kernel_kwargs: Optional[dict] = None,
        learnable: Union[bool, Dict[str, bool]] = False,
        far_field: str = "zero",
        observer_batch_size: Optional[int] = 100_000,
        **kwargs,
    ):
        kwargs["positions"] = positions
        kwargs["normals"] = normals
        kwargs["values"] = values
        kwargs["learnable"] = learnable
        kwargs["kernel"] = kernel
        kwargs["kernel_kwargs"] = kernel_kwargs
        kwargs["value_kwargs"] = value_kwargs
        kwargs["far_field"] = far_field
        kwargs["observer_batch_size"] = observer_batch_size
        # BaseSF.__init__ calls self.initField(**kwargs)
        super().__init__(
            name=name,
            input_dim=input_dim,
            output_dim=output_dim,
            drift=drift,
            transform=transform,
            local=local,
            seed=seed,
            **kwargs,
        )

    def initField(
        self,
        positions: Optional[ArrayLike] = None,
        normals: Optional[ArrayLike] = None,
        values: Optional[Union[ArrayLike, Callable]] = None,
        learnable: Union[bool, Dict[str, bool]] = False,
        kernel: Union[str, Callable] = "closest",
        kernel_kwargs: Optional[dict] = None,
        value_kwargs: Optional[dict] = None,
        far_field: str = "mean",
        observer_batch_size: Optional[int] = 100_000,
        **kwargs,
    ):
        """
        Build the point kernel field: store positions, normals, values and kernel; register parameters or buffers.
        """
        self._value_kwargs = value_kwargs if value_kwargs is not None else {}
        self._kernel_kwargs = dict(kernel_kwargs) if kernel_kwargs is not None else {}
        if far_field not in ("mean", "zero"):
            raise ValueError(f"far_field must be 'mean' or 'zero', got {far_field!r}.")
        self._far_field = far_field
        self._observer_batch_size = observer_batch_size

        if positions is None:
            raise ValueError("PointKernelField requires positions (N, input_dim).")
        positions = _tensor(positions)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        n_pts, dim = positions.shape
        if dim != self.input_dim:
            raise ValueError(
                f"positions last dimension ({dim}) must match field input_dim ({self.input_dim}).")

        learnable_dict = _normalize_learnable(learnable)

        # store positions as parameter or buffer
        if learnable_dict["positions"]:
            self.positions = nn.Parameter(positions)
        else:
            self.register_buffer("positions", positions)

        # store normals (normalised) as parameter or buffer, or None
        if normals is not None:
            normals = _tensor(normals)
            if normals.dim() == 1:
                normals = normals.unsqueeze(0)
            if normals.shape[0] != n_pts or normals.shape[1] != dim:
                raise ValueError( "normals must have shape (N, input_dim) matching positions." )
            normals = normals / (
                torch.linalg.norm(normals, dim=-1, keepdim=True).clamp(min=1e-8)
            )
            if learnable_dict["normals"]:
                self.normals = nn.Parameter(normals)
            else:
                self.register_buffer("normals", normals)
        else:
            self.normals = None

        # store values: callable, or array (N,) / (N, j) as parameter or buffer
        self._values_callable = None
        self._values_data = None
        if values is not None:
            if callable(values):
                self._values_callable = values
            else:
                values = _tensor(values)
                if values.dim() == 0:
                    values = values.unsqueeze(0).expand(n_pts)
                if values.shape[0] != n_pts:
                    raise ValueError("values length must match number of positions.")
                # Ensure (N,) -> (N, 1) so values have shape (N, j) with j >= 1
                if values.dim() == 1:
                    values = values.unsqueeze(-1)
                if learnable_dict["values"]:
                    self.values = nn.Parameter(values)
                else:
                    self.register_buffer("values", values)
        else:
            self.values = None

        self._learnable = learnable_dict
        self.mnorm = 1.0

        # resolve kernel: callable or string -> KERNEL_STRINGS entry
        k = kernel
        if callable(k):
            self._kernel_callable = k
            self.kernel = "fancy"
        else:
            k = (k or "closest").lower()
            if k == "nearest":
                k = "closest"
            if k not in KERNEL_STRINGS:
                raise ValueError(
                    f"kernel must be one of {sorted(KERNEL_STRINGS)} or a callable, got {kernel!r}.")
            self.kernel = k
            self._kernel_callable = KERNEL_STRINGS[k]

    def _value_matrix(
        self,
        x: torch.Tensor,
        vectors: torch.Tensor,
        euclid: torch.Tensor,
        cid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the value matrix at each (query, seed) (or at closest seed only when cid is set).

        Uses the values callable if set, else the values buffer, else the default (signedDistance
        if normals present, else distance). Handles shape normalisation for callable return values.

        Returns
        -------
        torch.Tensor
            Shape (M, N, j) when cid is None, or (M, j) when cid is set; j >= 1.
        """
        if euclid is not None:
            M, N = euclid.shape
        else:
            M, N = vectors.shape[0], vectors.shape[1]
        seeds, n = self.positions, self.normals
        value_kwargs = dict(getattr(self, "_value_kwargs", {}))
        if getattr(self, "_values_data", None) is not None:
            value_kwargs["values_data"] = self._values_data
        if self._values_callable is not None:
            out = self._values_callable(x, vectors, euclid, cid, seeds, n, value_kwargs)
            # normalise callable return to (M, N, j) or (M, j)
            if cid is None:
                if out.dim() == 2:
                    if out.shape[0] == N:
                        out = out.unsqueeze(0).expand(M, N, out.shape[1])
                    elif out.shape[0] == M and out.shape[1] == N:
                        out = out.unsqueeze(-1)
                    else:
                        raise ValueError(f"values callable must return (M, N), (M, N, j), or (N, j), got {out.shape}.")
                elif out.dim() != 3 or out.shape[0] != M or out.shape[1] != N:
                    raise ValueError(f"values callable must return (M, N), (M, N, j), or (N, j), got {out.shape}.")
            else:
                if out.dim() == 1:
                    out = out.unsqueeze(-1)
            return out
        if self.values is not None:
            if cid is not None:
                return self.values[cid]
            return self.values.unsqueeze(0).expand(M, N, self.values.shape[-1])
        default_fn = signedDistance if n is not None else distance
        return default_fn(x, vectors, euclid, cid, seeds, n, value_kwargs)

    def _evaluate_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate at x (single chunk). x already on correct device/dtype."""
        p = self.positions
        vectors = x.unsqueeze(1) - p.unsqueeze(0)
        M, N = vectors.shape[0], vectors.shape[1]
        # Only compute euclid when needed: closest (argmin), kernel callable (weights), or default value fn (distance/signedDistance)
        need_euclid = (
            (self.kernel == "closest" and self._kernel_callable is None)
            or self._kernel_callable is not None
            or self._values_callable is None
        )
        euclid = torch.linalg.norm(vectors, dim=-1) if need_euclid else None

        if self.kernel == "closest" and self._kernel_callable is None:
            cid = torch.argmin(euclid, dim=1)
            return self._value_matrix(x, vectors, euclid, cid=cid)

        value_matrix = self._value_matrix(x, vectors, euclid, cid=None)
        kernel_kwargs = dict(getattr(self, "_kernel_kwargs", {}))
        eps_combine = kernel_kwargs.get("eps", DEFAULT_EPS)

        if self._kernel_callable is not None:
            w = self._kernel_callable(x, vectors, euclid, p, self.normals, value_matrix, kernel_kwargs)
            if w.shape != (M, N):
                raise ValueError(f"kernel callable must return weights (M, N), got {w.shape}.")
            min_weight = kernel_kwargs.get("min_weight", MINIMUM_WEIGHT)
            if min_weight > 0:
                w = w.clamp(min=min_weight)
            w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps_combine)
            w_norm = w / w_sum
            result = (w_norm.unsqueeze(-1) * value_matrix).sum(dim=1)
            if self._far_field == "mean":
                data_mean = value_matrix.mean(dim=1)
                result = torch.where(
                    (w_sum.squeeze(-1) < eps_combine).unsqueeze(-1),
                    data_mean,
                    result,
                )
            return result
        return _aggregate(value_matrix, self.kernel, dim=1)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the field at query points.

        Returns (M, j) with j the value dimension (j=1 for scalar). For kernel "closest", the value
        function is called with cid for efficiency. For weighted kernels, when total weight is below
        eps (query far from all seeds), prediction is the value matrix mean (far_field="mean") or zero.

        When observer_batch_size was set at construction, evaluation is chunked over observers
        to limit memory use for large M.
        """
        x = x.to(dtype=curlew.dtype, device=curlew.device)
        M = x.shape[0]
        batch_size = getattr(self, "_observer_batch_size", None)
        if batch_size is not None and M > batch_size:
            out = []
            for start in range(0, M, batch_size):
                end = min(start + batch_size, M)
                out.append(self._evaluate_chunk(x[start:end]))
            return torch.cat(out, dim=0)
        return self._evaluate_chunk(x)



# Short alias for PointKernelField
PKF = PointKernelField
