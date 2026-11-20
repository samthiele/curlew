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
