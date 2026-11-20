"""
Fourier-series neural fields (FSF) for scalar potential representation.

A scalar potential is a random Fourier series. Linear mode uses a Cartesian
pair of learnable amplitudes per mode, ``A cos(Ω·x) + B sin(Ω·x)`` — equivalent
to amplitude-and-phase but well-defined at the origin, where a polar ``(R, φ)``
chart is singular. Activated mode keeps a single amplitude, a frozen random
phase, and a per-mode bias. Values, gradients and Hessians are computed in
closed form (no autograd through the series), which improves speed and
stability for interpolation and — via the raw API — for Euler /
diffeomorphic deformation downstream.

Module layout
-------------
1. Trigonometric and activation kernels  (``_SinCos``, ``_activation_*``)
2. Mode sampling                         (``_draw_fourier_omega`` and helpers)
3. Mode-bank builders                    (``_build_*_fourier_modes``)
4. The field itself                      (:class:`FSF`)
"""
import math
import warnings
from typing import Optional, Tuple, Union

import curlew
import torch
import torch.nn as nn
import torch.nn.functional as F

from curlew.fields import BaseNF

# Offset separating the phase draw from the frequency draw, which reseeds the
# global RNG with ``seed`` (and ``seed + 1`` on the paired-Euler path).
_PHASE_SEED_OFFSET = 104729


# ═══════════════════════════════════════════════════════════════════════════
# 1. Trigonometric and activation kernels
# ═══════════════════════════════════════════════════════════════════════════


class _SinCos(torch.autograd.Function):
    """Return (sin(x), cos(x)) with one transcendental pair per autograd pass."""

    @staticmethod
    def forward(ctx, proj: torch.Tensor):
        s = torch.sin(proj)
        c = torch.cos(proj)
        ctx.save_for_backward(s, c)
        return s, c

    @staticmethod
    def backward(ctx, gs: torch.Tensor, gc: torch.Tensor):
        s, c = ctx.saved_tensors
        return gs * c - gc * s


def _sincos(proj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return _SinCos.apply(proj)


def _activation_derivs(
    u: torch.Tensor, name: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return σ(u), σ'(u), σ''(u) for activated FSF modes."""
    if name == "tanh":
        t = torch.tanh(u)
        return t, 1.0 - t * t, -2.0 * t * (1.0 - t * t)
    if name == "cubic":
        return (
            1.5 * u - 0.5 * u**3,
            1.5 * (1.0 - u * u),
            -3.0 * u,
        )
    if name == "softplus":
        s = torch.sigmoid(u)
        return F.softplus(u), s, s * (1.0 - s)
    raise ValueError(f"unknown activation {name!r}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Mode sampling
# ═══════════════════════════════════════════════════════════════════════════


def _golden_order(n: int) -> torch.Tensor:
    """
    The permutation ``k_j = rank of {j rho}``, ``j = 0..n-1``, ``rho`` the golden ratio.

    Visiting an even set of ``n`` points in this order gives every contiguous
    run the spread of a Kronecker run, while the *set* stays exactly even.
    """
    rho = 0.5 * (5.0**0.5 - 1.0)
    v = torch.frac(torch.arange(n, dtype=torch.float64) * rho)
    return torch.argsort(torch.argsort(v))


def _stratified_directions(n: int, dim: int, seed: int) -> torch.Tensor:
    """
    An **even set** of ``n`` unit axes, visited in **golden-ratio order**.

    For a real sine series with random phase, ``+Omega`` and ``-Omega`` are the
    same mode (the sign is absorbed into the amplitude and phase), so a
    direction only matters as an *axis*: the half-turn ``[0, pi)`` in 2-D, the
    hemisphere in 3-D.

    **The set.** A Sobol draw leaves holes whose size and place depend on the
    seed, and whose largest grows with ``n`` like the largest of ``n`` random
    gaps. A Kronecker sequence bounds that growth but still has up to three gap
    sizes. An even set removes it:

    * 2-D: ``n`` axes exactly ``pi / n`` apart, ``theta_k = pi {(k + 1/2)/n + u0}``.
    * 3-D: a hemispherical Fibonacci lattice, one axis per equal-area latitude
      strip, ``cos Theta_k = (k + 1/2)/n``, with azimuth ``2 pi {k rho + u1}``.
      Its largest empty cap stays at ~1.4-1.5x the radius of a cap holding
      ``1/n`` of the hemisphere, for every ``n``.

    **The order.** Axis ``j`` is the set's point ``k = _golden_order(n)[j]``.
    The caller hands axes out by wavelength rank (longest first), so in natural
    order ``k = j`` the coarsest modes would share one sector (2-D) or one
    latitude strip (3-D). The golden permutation spreads any contiguous run
    like a Kronecker run, at no cost to the set.

    **The seed.** A hemisphere has a privileged pole and equator, and a fixed
    set a privileged orientation. 2-D's offset ``u0`` is a random rotation of
    the half-turn; in 3-D a seeded, uniformly random rotation (Shoemake's
    random unit quaternion) turns the whole lattice. A seed therefore only
    rotates the set: hole sizes and the order of the axes do not depend on it.

    This changes only the *discrepancy* of the draw, not its measure -- the
    axes remain uniform -- so the amplitude prior in
    :meth:`FSF._build_amplitude_denom` is untouched.
    """
    g = torch.Generator().manual_seed(int(seed) + 8191)
    k = _golden_order(n).to(torch.float64)
    if dim == 2:
        off = torch.rand((), generator=g, dtype=torch.float64)
        a = math.pi * torch.frac((k + 0.5) / n + off)
        return torch.stack([torch.cos(a), torch.sin(a)], dim=1)
    rho = 0.5 * (5.0**0.5 - 1.0)
    off = torch.rand(2, generator=g, dtype=torch.float64)
    az = 2.0 * math.pi * torch.frac(k * rho + off[0])
    cz = (k + 0.5) / n                                   # upper hemisphere: axes
    sn = (1.0 - cz * cz).clamp_min(0.0).sqrt()
    d = torch.stack([sn * torch.cos(az), sn * torch.sin(az), cz], dim=1)
    # Uniform random rotation from a random unit quaternion (Shoemake 1992).
    u1, u2, u3 = torch.rand(3, generator=g, dtype=torch.float64)
    s1, s2 = (1.0 - u1).sqrt(), u1.sqrt()
    w, x = s1 * torch.sin(2 * math.pi * u2), s1 * torch.cos(2 * math.pi * u2)
    y, z = s2 * torch.sin(2 * math.pi * u3), s2 * torch.cos(2 * math.pi * u3)
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]),
    ])
    return d @ R.T


def _sample_wavelengths(
    u0: torch.Tensor,
    length_scale_range: Tuple[float, float],
    sampling: str,
) -> torch.Tensor:
    """
    Map unit Sobol coordinate ``u0 ∈ (0, 1)`` to wavelengths in ``length_scale_range``.

    ``"log"`` (default) is log-uniform in λ, so a linear histogram piles up at
    the fine end. ``"uniform"`` is uniform in λ.
    """
    lo = float(length_scale_range[0])
    hi = float(length_scale_range[1])
    # hi == lo is legitimate: a single-wavelength bank (all modes on one
    # radius, directions still spread). Both branches below degenerate to the
    # constant `lo` in that case, so only hi < lo is an error.
    if lo <= 0.0 or hi <= 0.0 or hi < lo:
        raise ValueError(
            f"length_scale_range must satisfy 0 < min <= max, got {length_scale_range}"
        )
    lo_t = torch.tensor(lo, dtype=u0.dtype, device=u0.device)
    hi_t = torch.tensor(hi, dtype=u0.dtype, device=u0.device)
    if sampling == "log":
        log_lo = torch.log10(lo_t)
        log_hi = torch.log10(hi_t)
        ten = torch.tensor(10.0, dtype=u0.dtype, device=u0.device)
        return torch.pow(ten, log_lo + u0 * (log_hi - log_lo))
    if sampling == "uniform":
        return lo_t + u0 * (hi_t - lo_t)
    raise ValueError(
        f"wavelength_sampling must be 'log' or 'uniform', got {sampling!r}"
    )


def _draw_fourier_omega(
    *,
    rff_features: int,
    length_scale_range: Tuple[float, float],
    input_dim: int,
    lift_dim: int,
    stochastic_scales: bool,
    seed: int,
    stratify_directions: bool = False,
    n_directions: Optional[int] = None,
    wavelength_sampling: str = "log",
) -> torch.Tensor:
    """Sample fixed Fourier rows ``Ω_k`` with shape ``(F, input_dim + lift_dim)``."""
    d_in = input_dim + lift_dim

    torch.manual_seed(seed)

    sobol = torch.quasirandom.SobolEngine(dimension=d_in + 1, scramble=True)
    u = (
        sobol.draw(rff_features)
        .to(device=curlew.device, dtype=curlew.dtype)
        .clamp(1e-6, 1 - 1e-6)
    )
    lengths = _sample_wavelengths(u[:, 0], length_scale_range, wavelength_sampling)
    k_raw = torch.special.ndtri(u[:, 1:].clamp(1e-6, 1 - 1e-6))

    if not stochastic_scales:
        k_raw = F.normalize(k_raw, dim=1)

    if stratify_directions or n_directions is not None:
        if stochastic_scales or lift_dim != 0 or input_dim not in (2, 3):
            warnings.warn(
                "stratify_directions / n_directions are only applied for "
                "lift_dim=0, input_dim in (2, 3) and stochastic_scales=False; "
                "ignoring.",
                RuntimeWarning,
            )
        else:
            # longest wavelength first, so contiguous runs = radial bands
            order = torch.argsort(lengths, descending=True)
            n_dir = rff_features if n_directions is None else int(n_directions)
            if not 1 <= n_dir <= rff_features:
                raise ValueError(
                    f"n_directions must be in [1, rff_features={rff_features}], "
                    f"got {n_directions}"
                )
            dirs = _stratified_directions(n_dir, input_dim, seed).to(
                device=k_raw.device, dtype=k_raw.dtype
            )
            # Cycling the direction index over wavelength-sorted modes makes the
            # bank a *polar grid*: n_dir rays, each carrying ~F/n_dir radii
            # spread across the whole range. n_dir == F recovers the plain
            # stratified bank exactly (arange % F is the identity), so the
            # existing path is untouched.
            k_raw = k_raw.clone()
            k_raw[order] = dirs[torch.arange(rff_features) % n_dir]

    return 2.0 * torch.pi * k_raw / lengths.unsqueeze(1)


def _paired_spatial_direction(
    u: torch.Tensor,
    axis: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """
    Orthogonal companion ``v`` with ``|v| = |u|`` row-wise and ``u · v = 0``.

    ``fallback`` supplies random directions when ``u`` is parallel to ``axis``.
    """
    mag = u.norm(dim=1, keepdim=True).clamp(min=1e-12)
    u_hat = u / mag
    ax = axis / axis.norm().clamp(min=1e-12)
    cross = torch.linalg.cross(u_hat, ax.expand_as(u_hat), dim=1)
    cn = cross.norm(dim=1)
    parallel = cn < 1e-6
    if parallel.any():
        w = fallback[parallel]
        w = w - (w * u_hat[parallel]).sum(1, keepdim=True) * u_hat[parallel]
        wn = w.norm(dim=1, keepdim=True).clamp(min=1e-12)
        cross[parallel] = w / wn.squeeze(1)
        cn = cross.norm(dim=1)
    v_hat = cross / cn.unsqueeze(1).clamp(min=1e-12)
    return v_hat * mag


# ═══════════════════════════════════════════════════════════════════════════
# 3. Mode-bank builders
# ═══════════════════════════════════════════════════════════════════════════


def _out_scale(scale: float, rff_features: int) -> torch.Tensor:
    """Global output scale ``s/√F``."""
    return torch.tensor(
        scale / math.sqrt(rff_features), dtype=curlew.dtype, device=curlew.device
    )


def _build_fourier_modes(
    *,
    rff_features: int,
    length_scale_range: Tuple[float, float],
    input_dim: int,
    lift_dim: int,
    stochastic_scales: bool,
    scale: float,
    seed: int,
    stratify_directions: bool = False,
    n_directions: Optional[int] = None,
    wavelength_sampling: str = "log",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample fixed Fourier directions (Sobol quasi-Monte-Carlo) and return
    ``(fourier_projection, freq_scaling, out_scale)``.
    """
    omega = _draw_fourier_omega(
        rff_features=rff_features,
        length_scale_range=length_scale_range,
        input_dim=input_dim,
        lift_dim=lift_dim,
        stochastic_scales=stochastic_scales,
        seed=seed,
        stratify_directions=stratify_directions,
        n_directions=n_directions,
        wavelength_sampling=wavelength_sampling,
    )
    freq_scaling = (omega**2).sum(1).clamp(min=1e-8)
    return omega.T, freq_scaling, _out_scale(scale, rff_features)


def _plane_basis(normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal pair ``(e1, e2)`` spanning the plane perpendicular to ``normal``."""
    n = normal.reshape(-1).to(dtype=curlew.dtype, device=curlew.device)
    if int(n.numel()) != 3:
        raise ValueError(f"plane basis expects a 3-vector, got length {int(n.numel())}")
    n = n / n.norm().clamp_min(1e-12)
    ref = n.new_tensor(
        [0.0, 1.0, 0.0] if float(n[:2].norm()) < 1e-6 else [0.0, 0.0, 1.0]
    )
    e1 = torch.linalg.cross(n, ref)
    e1 = e1 / e1.norm().clamp_min(1e-12)
    e2 = torch.linalg.cross(n, e1)
    e2 = e2 / e2.norm().clamp_min(1e-12)
    return e1, e2


def _build_cylindrical_fourier_modes(
    *,
    hinge,
    rff_features: int,
    length_scale_range: Tuple[float, float],
    lift_dim: int,
    stochastic_scales: bool,
    scale: float,
    seed: int,
    stratify_directions: bool = False,
    n_directions: Optional[int] = None,
    wavelength_sampling: str = "log",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    2-D Fourier bank embedded in the plane perpendicular to ``hinge``.

    Directions and wavelengths are drawn as in 2-D (``n_directions`` on the
    circle, Sobol radii), then lifted to R³ on an orthonormal basis of that
    plane. The result is a cylindrical field: ``Ω · hinge = 0`` identically,
    with ``F`` features and ``n_directions`` in-plane rays — not a 3-D sphere
    draw projected after the fact.
    """
    omega2 = _draw_fourier_omega(
        rff_features=rff_features,
        length_scale_range=length_scale_range,
        input_dim=2,
        lift_dim=lift_dim,
        stochastic_scales=stochastic_scales,
        seed=seed,
        stratify_directions=stratify_directions,
        n_directions=n_directions,
        wavelength_sampling=wavelength_sampling,
    )
    hinge_t = torch.as_tensor(hinge, dtype=curlew.dtype, device=curlew.device)
    e1, e2 = _plane_basis(hinge_t)
    spatial = omega2[:, :2] @ torch.stack((e1, e2), dim=0)
    omega = spatial if lift_dim == 0 else torch.cat([spatial, omega2[:, 2:]], dim=1)
    freq_scaling = (omega**2).sum(1).clamp(min=1e-8)
    return omega.T, freq_scaling, _out_scale(scale, rff_features)


def _build_paired_fourier_modes(
    *,
    rff_features: int,
    length_scale_range: Tuple[float, float],
    input_dim: int,
    lift_dim: int,
    stochastic_scales: bool,
    scale: float,
    seed: int,
    pair_axis=None,
    wavelength_sampling: str = "log",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Paired Euler frequency banks for full 3-D :math:`\\nabla\\alpha\\times\\nabla\\beta`.

    Mode ``k`` shares wavelength (and lift frequencies) in both slots; spatial
    directions are orthogonal companions with equal magnitude so the diagonal
    cross-product term is non-degenerate.

    Returns ``(proj_alpha, proj_beta, freq_scaling, out_scale)``.
    """
    if input_dim != 3:
        raise ValueError(f"paired Fourier modes require input_dim=3, got {input_dim}")

    omega_a = _draw_fourier_omega(
        rff_features=rff_features,
        length_scale_range=length_scale_range,
        input_dim=input_dim,
        lift_dim=lift_dim,
        stochastic_scales=stochastic_scales,
        seed=seed,
        wavelength_sampling=wavelength_sampling,
    )
    torch.manual_seed(seed + 1)
    fallback = torch.randn(
        rff_features, input_dim, dtype=curlew.dtype, device=curlew.device
    )

    if pair_axis is None:
        axis = torch.zeros(input_dim, dtype=curlew.dtype, device=curlew.device)
        axis[2] = 1.0
    else:
        axis = torch.as_tensor(
            pair_axis, dtype=curlew.dtype, device=curlew.device
        ).reshape(-1)
        if axis.numel() != input_dim:
            raise ValueError(
                f"pair_axis must have length {input_dim}, got {axis.numel()}"
            )

    u = omega_a[:, :input_dim]
    omega_b = omega_a.clone()
    omega_b[:, :input_dim] = _paired_spatial_direction(u, axis, fallback)

    freq_scaling = (omega_a**2).sum(1).clamp(min=1e-8)
    return omega_a.T, omega_b.T, freq_scaling, _out_scale(scale, rff_features)


# ═══════════════════════════════════════════════════════════════════════════
# 4. The field
# ═══════════════════════════════════════════════════════════════════════════


class FSF(BaseNF):
    """
    FourierSeriesField — learnable scalar potential as a damped random Fourier series.

    Two operating modes, selected by ``activation``:

    ── Linear mode (``activation=None``, default) ─────────────────────────────

        φ(x) = (s/√F) Σ_k  (A_k cos(Ω_k·x) + B_k sin(Ω_k·x)) / |Ω_k|^{freq_damp}

    ``A_k`` and ``B_k`` are both parameters. That is the Cartesian form of
    ``R_k sin(Ω_k·x + φ_k)``: same span, but the ``(R, φ)`` chart is singular
    as ``R → 0`` (Lázaro-Gredilla et al. 2010), which is where a Euler
    potential starts. With i.i.d. Gaussian ``(A, B)`` the prior over φ is a
    stationary GP.

    With the default ``freq_damp=2`` this is the ``/ |Ω_k|²`` damping. Lower
    values (e.g. ``1``) weaken high-frequency suppression so short-wavelength
    modes contribute more strongly to ``∇φ`` for the same amplitudes.
    ``freq_damp=0`` removes the damping (``D_k = 1``).

    ── Activated mode (``activation`` given) ───────────────────────────────────

        φ(x) = (s/√F) Σ_k C_k σ(sin(Ω_k·x + φ_k) + b_k) / |Ω_k|^{freq_damp}

        The phase ``φ_k`` is a raw angle (``phi_mu``), initialised uniformly
        from a seeded draw, and the signed outer amplitude ``C_k`` controls
        mode strength. Because ``C`` carries the amplitude, the phase lives on
        the circle alone and the angle is its natural chart: ``sin(θ + φ)`` is
        smooth in ``φ`` everywhere. (A Cartesian ``atan2(B, A)`` chart would
        add an unseen radius whose drift rescales the phase learning rate by
        ``1/|(A, B)|``.) Separating phase from amplitude keeps the activation
        input in ``[-1, 1]``. ``bias`` (``b_k``) breaks symmetry for vergent /
        asymmetric folds.

    Analytic derivatives (spatial coordinates only when ``lift_dim > 0``), with
    ``D_k = |Ω_k|^{freq_damp}`` and ``θ = Ω·x``:

        Linear:   ∇φ = (s/√F) Σ (B_k cos θ − A_k sin θ)/D_k · Ω_k
                  H_φ = (s/√F) Σ −(A_k cos θ + B_k sin θ)/D_k · Ω_k Ω_kᵀ

        Activated: chain rule through σ(sin(·)+b) — see ``gradient_and_hessian``.

    Prior interpretation
    --------------------
    The ``1/|Ω|^{freq_damp}`` damping is the high-frequency tail of a **Matérn
    prior with ν = freq_damp**, in any dimension: modes are drawn from
    (uniform on ``S^{d-1}``) × (log-uniform radius), a measure whose density
    with respect to Lebesgue ``dω`` is ``∝ r^{-d}``, so with amplitude variance
    ``v(r) = D(r)^{-2}`` the induced spectral density is

        S(r) ∝ v(r) · r^{-d} = r^{-(2·freq_damp + d)}

    ``freq_damp=0`` drops the extra ``r^{-2ν}`` factor (``D_k = 1``). The
    remaining ``r^{-d}`` is the sampling measure, not amplitude damping.

    This diverges as ``r → 0`` and is kept finite only by the
    ``length_scale_range`` cutoff, so that band is a load-bearing prior
    assumption rather than a pure quadrature choice.

    Parameters passed through ``initField`` / constructor keywords
    ----------------------------------------------------------------
    rff_features, length_scale_range, activation, scale, learning_rate,
    lift_dim, stochastic_scales, stratify_directions, n_directions,
    xavier_amplitudes, freq_damp, wavelength_sampling, energy_normalised
    """

    # ── Construction ────────────────────────────────────────────────────────

    def initField(
        self,
        rff_features: int,
        length_scale_range: Optional[Tuple[float, float]] = None,
        activation: Optional[str] = None,
        scale: float = 1.0,
        learning_rate: float = 1e-1,
        lift_dim: int = 0,
        stochastic_scales: bool = False,
        stratify_directions: bool = False,
        n_directions: Optional[int] = None,
        xavier_amplitudes: bool = True,
        freq_damp: float = 2.0,
        wavelength_sampling: str = "log",
        fourier_projection: Optional[torch.Tensor] = None,
        freq_scaling: Optional[torch.Tensor] = None,
        out_scale: Optional[torch.Tensor] = None,
        energy_normalised: bool = False,
    ):
        """
        Build Fourier frequencies and learnable mode parameters.

        Parameters
        ----------
        rff_features : int
            Number of Fourier modes F.
        length_scale_range : (float, float)
            Minimum and maximum wavelength. Default sampling is log-uniform
            in λ (see ``wavelength_sampling``).
        activation : None | ``"tanh"`` | ``"cubic"`` | ``"softplus"``
            Nonlinearity applied to each mode in activated mode.
        scale : float
            Global output scale (combined with 1/√F in ``out_scale``).
        learning_rate : float
            Adam learning rate for ``init_optim``.
        lift_dim : int
            Extra input coordinates (e.g. fault sheet labels). Derivatives
            are returned with respect to spatial ``input_dim`` only.
        stochastic_scales : bool
            If True, direction vectors are not normalised and |Ω| follows a
            chi-like distribution (as in ``curlew.fields.NFF``).
        stratify_directions : bool
            If True, replace the Sobol directions with an even set of axes
            (2-D: equal spacing; 3-D: hemispherical Fibonacci lattice) in
            golden-ratio order over the wavelength-sorted modes, so no seed
            leaves a hole in the angular coverage. Only applied for ``lift_dim=0``, ``input_dim in (2, 3)``
            and ``stochastic_scales=False``; warns and is ignored otherwise.
            See :func:`_stratified_directions`.
        n_directions : int, optional
            Number of *distinct* wave directions to share across the bank — a
            **polar grid** of ``n_directions`` rays. Implies
            ``stratify_directions``. ``None`` (default) or ``rff_features``
            gives one direction per mode, i.e. exactly the stratified bank.

            Each ray carries about ``rff_features / n_directions`` Sobol
            wavelengths interleaved across the whole ``length_scale_range``
            (not a tensor-product grid: radii differ from ray to ray).

            Sharing a direction is also what makes ``integrator="shear"``
            cheap: those shears commute and merge, cutting sequential depth
            from ``F`` to ``n_directions``. Too few rays and the field cannot
            fold in an arbitrary direction — a genuine trade, not a free win.
        xavier_amplitudes : bool
            If True (default), initialise mode amplitudes with Xavier
            normal (as legacy ``w1``). In linear mode ``A`` and ``B`` share the
            old variance (each scaled by ``1/√2``). Required for geology /
            scalar-field fitting where zero amplitudes stall optimisation.
            Set False for potentials that are scaled manually after construction.
        freq_damp : float
            Exponent ν of the ``1/|Ω|^ν`` amplitude damping. Must be
            non-negative. ``0`` turns damping off (``D_k = 1``).
        wavelength_sampling : ``"log"`` | ``"uniform"``
            How the radial coordinate is mapped onto ``length_scale_range``.
            ``"log"`` (default) is log-uniform in λ — equal mass per decade,
            so a linear histogram of wavelengths piles up at ``λ_min``.
            ``"uniform"`` is uniform in λ.
        fourier_projection, freq_scaling, out_scale
            Optional pre-built frequency buffers (see
            :func:`_build_fourier_modes` / :func:`_build_paired_fourier_modes`).
            When ``fourier_projection`` is set, the other two must be supplied
            and ``rff_features`` / ``length_scale_range`` are not used for
            sampling (but ``rff_features`` must still match the mode count).
        energy_normalised : bool
            Cubic activation only. Divide each mode by the RMS of its own
            gradient, ``√(e(b_k)/e(0))`` with ``e(b) = b⁴ − b²/2 + 5/8`` (the
            full-period mean of ``(σ'(sin θ + b) cos θ)²`` up to a constant).
            ``C_k²`` then fixes the mode's gradient energy ``⟨‖∇φ_k‖²⟩`` (the
            speed, when φ is a stream function) and ``bias`` / ``phi``
            only change its shape. Default False (unchanged behaviour).
        """
        if activation not in (None, "tanh", "cubic", "softplus"):
            raise ValueError(
                "activation must be None, 'tanh', 'cubic', or 'softplus', "
                f"got {activation!r}"
            )
        if freq_damp < 0:
            raise ValueError(f"freq_damp must be non-negative, got {freq_damp}")
        if wavelength_sampling not in ("log", "uniform"):
            raise ValueError(
                "wavelength_sampling must be 'log' or 'uniform', got "
                f"{wavelength_sampling!r}"
            )
        if length_scale_range is None and fourier_projection is None:
            raise ValueError("length_scale_range is required")
        if energy_normalised and activation != "cubic":
            raise ValueError(
                "energy_normalised needs activation='cubic' (closed-form energy), "
                f"got {activation!r}"
            )

        self.activation = activation
        self.lift_dim = int(lift_dim)
        self.freq_damp = float(freq_damp)
        self.stochastic_scales = bool(stochastic_scales)
        self.wavelength_sampling = str(wavelength_sampling)
        self.energy_normalised = bool(energy_normalised)

        # ── Frequency bank ──────────────────────────────────────────────────
        if fourier_projection is None:
            projection, freq_scaling, out_scale = _build_fourier_modes(
                rff_features=rff_features,
                length_scale_range=length_scale_range,
                input_dim=self.input_dim,
                lift_dim=self.lift_dim,
                stochastic_scales=stochastic_scales,
                scale=scale,
                seed=self.seed,
                stratify_directions=stratify_directions,
                n_directions=n_directions,
                wavelength_sampling=self.wavelength_sampling,
            )
        else:
            projection = self._validate_prebuilt(
                fourier_projection, freq_scaling, out_scale, rff_features
            )
        self.register_buffer("fourier_projection", projection)
        self.register_buffer("freq_scaling", freq_scaling)
        self.register_buffer("out_scale", out_scale)

        # ── Learnable mode parameters ───────────────────────────────────────
        zeros = torch.zeros(rff_features, dtype=curlew.dtype, device=curlew.device)
        if activation is None:
            self.A_mu = nn.Parameter(zeros.clone())
            self.B_mu = nn.Parameter(zeros.clone())
            # Cartesian pair (A cos θ + B sin θ). Xavier each then /√2 so the
            # initial field variance matches the old one-amplitude series.
            if xavier_amplitudes:
                nn.init.xavier_normal_(self.A_mu.unsqueeze(0))
                nn.init.xavier_normal_(self.B_mu.unsqueeze(0))
                scale = 1.0 / math.sqrt(2.0)
                self.A_mu.data.mul_(scale)
                self.B_mu.data.mul_(scale)
        else:
            # C is the signed outer amplitude; phi_mu is the raw phase angle.
            self.C_mu = nn.Parameter(zeros.clone())
            if xavier_amplitudes:
                nn.init.xavier_normal_(self.C_mu.unsqueeze(0))
            self.bias = nn.Parameter(zeros.clone())
            # Seed the learnable phase from a private generator (not the
            # global RNG, which `_draw_fourier_omega` reseeds).
            phase_gen = torch.Generator().manual_seed(
                int(self.seed) + _PHASE_SEED_OFFSET
            )
            phi = (
                2.0
                * math.pi
                * torch.rand(rff_features, generator=phase_gen, dtype=torch.float64)
            ).to(dtype=curlew.dtype, device=curlew.device)
            self.phi_mu = nn.Parameter(phi)

        self._build_derived_buffers()
        self.to(curlew.device)
        self.init_optim(lr=learning_rate)

    def _validate_prebuilt(
        self,
        projection: torch.Tensor,
        freq_scaling: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        rff_features: int,
    ) -> torch.Tensor:
        """Check an externally supplied frequency bank and return the projection."""
        if freq_scaling is None or out_scale is None:
            raise ValueError(
                "freq_scaling and out_scale are required when "
                "fourier_projection is pre-built"
            )
        n_modes = projection.shape[1]
        if n_modes != rff_features:
            raise ValueError(
                f"fourier_projection has {n_modes} modes, "
                f"expected rff_features={rff_features}"
            )
        expected_rows = self.input_dim + self.lift_dim
        if projection.shape[0] != expected_rows:
            raise ValueError(
                f"fourier_projection rows {projection.shape[0]} != "
                f"input_dim + lift_dim ({expected_rows})"
            )
        return projection

    def _build_derived_buffers(self) -> None:
        """
        Precompute everything that depends only on the (fixed) frequency bank:
        spatial frequency rows, their outer products, and the amplitude divisor.
        """
        ox = self.fourier_projection[: self.input_dim, :].T.contiguous()
        n_modes, d = ox.shape
        self.register_buffer("omega_spatial", ox)
        self.register_buffer(
            "omega_outer",
            (ox.unsqueeze(-1) * ox.unsqueeze(-2)).reshape(n_modes, d * d).contiguous(),
        )
        # D_k = |Ω_k|^{freq_damp}; freq_scaling is |Ω_k|², hence the half power.
        # freq_damp=0 → D_k = 1 (no damping); avoid 0^0 if any |Ω| is zero.
        if self.freq_damp == 0.0:
            denom = torch.ones_like(self.freq_scaling)
        else:
            denom = self.freq_scaling.pow(0.5 * self.freq_damp)
        self.register_buffer("amplitude_denom", denom)

    def project_frequencies_onto_plane(self, normal) -> "FSF":
        """
        Project every spatial wavevector onto the plane perpendicular to ``normal``.

        After this, ``φ(x + s n) = φ(x)``: the field is invariant along
        ``n``. Wavelengths ``|Ω|`` are kept. Modes that were parallel to
        ``n`` are replaced by an in-plane direction at the same ``|Ω|``.
        Lift-frequency rows (if any) are left unchanged.
        """
        n = torch.as_tensor(
            normal, dtype=self.omega_spatial.dtype, device=self.omega_spatial.device
        ).reshape(-1)
        if int(n.numel()) != self.input_dim:
            raise ValueError(
                f"normal must have length {self.input_dim}, got {int(n.numel())}"
            )
        n = n / n.norm().clamp_min(1e-12)
        ox = self.fourier_projection[: self.input_dim, :].T.contiguous()
        mag = ox.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        ox = ox - (ox @ n).unsqueeze(-1) * n
        inplane = ox.norm(dim=-1, keepdim=True)
        collapsed = inplane.squeeze(-1) < (1e-12 * mag.squeeze(-1))
        if bool(collapsed.any()):
            arb = torch.zeros_like(n)
            arb[int(n.abs().argmin())] = 1.0
            e = arb - (arb @ n) * n
            e = e / e.norm().clamp_min(1e-12)
            ox = ox.clone()
            ox[collapsed] = e
            inplane = ox.norm(dim=-1, keepdim=True)
        ox = ox / inplane * mag
        with torch.no_grad():
            self.fourier_projection[: self.input_dim, :].copy_(ox.T)
            self.freq_scaling.copy_(
                (self.fourier_projection.pow(2).sum(0)).clamp(min=1e-8)
            )
        self._build_derived_buffers()
        return self

    # ── Introspection ───────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        """Spatial dimension (alias for ``input_dim``)."""
        return self.input_dim

    def mode_wavelengths(self) -> torch.Tensor:
        """``λ_k = 2π / |Ω_k^spatial|`` (F,), in model units."""
        return 2.0 * math.pi / self.omega_spatial.norm(dim=1).clamp_min(1e-30)

    def _omega_spatial(self) -> torch.Tensor:
        """Deprecated accessor — read the ``omega_spatial`` buffer directly."""
        return self.omega_spatial

    @classmethod
    def iter_in(cls, module: nn.Module):
        """Yield every :class:`FSF` submodule under ``module``."""
        for m in module.modules():
            if isinstance(m, cls):
                yield m

    # ── Energy ──────────────────────────────────────────────────────────────

    @classmethod
    def kinetic_energy_on(cls, module: nn.Module) -> Optional[torch.Tensor]:
        """Sum :meth:`kinetic_energy` over every :class:`FSF` under ``module``."""
        terms = [m.kinetic_energy() for m in cls.iter_in(module)]
        if not terms:
            return None
        return torch.stack(terms).sum()

    def kinetic_energy(self) -> torch.Tensor:
        """
        Analytic domain-averaged kinetic-energy surrogate as a quadratic in
        the amplitudes:

            ⟨‖∇φ‖²⟩ ≈ out_scale² · Σ_k (A_k² + B_k²) |ω_k^spatial|² / (2 D_k²)

        In activated mode the same expression uses ``C_k²``: ``phi_mu`` is
        the phase and ``C_k`` is the outer amplitude. For a
        nonlinear activation this remains a regularising surrogate rather than
        the exact spatial average of the activated field's gradient energy.
        Weight via ``HSet.energy_loss`` when this field is nested under a
        trained module that exposes that term.
        """
        w = (self.omega_spatial**2).sum(1) / self.amplitude_denom.pow(2)
        if self.activation is None:
            amp2 = self.A_mu.pow(2) + self.B_mu.pow(2)
        else:
            amp2 = self.C_mu.pow(2)
        return 0.5 * self.out_scale**2 * (amp2 * w).sum()

    # ── Raw evaluation (no drift, transform, or normalise) ──────────────────

    def evaluate_raw(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Evaluate φ at local spatial coordinates (optionally with lift coords ``w``).

        Does not apply ``Transform``, drift, or gradient normalisation.
        """
        psi, _, _ = self._rff(x, w, value=True, derivs=False)
        return psi

    def gradient_and_hessian(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spatial gradient (N, dim) and Hessian (N, dim, dim) of φ.

        Used where analytic Hessians are needed (e.g. custom adapters). When only
        the gradient is wanted, prefer :meth:`gradient_raw`. The Hessian is an
        ``(N, F, d, d)`` product and costs about twice the gradient.
        """
        _, grad, hess = self._rff(x, w, value=False, derivs=True)
        return grad, hess

    def gradient_raw(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Spatial gradient ∇φ alone (N, dim), skipping the Hessian.

        Identical to ``gradient_and_hessian(...)[0]`` but ~2.5x cheaper: it
        avoids the ``Ω_k Ω_kᵀ`` outer-product einsum, which the gradient does
        not need in linear mode.
        """
        _, grad, _ = self._rff(x, w, value=False, derivs=True, hess=False)
        return grad

    # ── Geomodelling API (BaseNF) ───────────────────────────────────────────

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate φ at batch of local coordinates (N, input_dim).

        Called from ``forward`` after ``Transform``; lift coordinates are not
        used on this path (set ``w`` only via ``evaluate_raw``).
        """
        return self.evaluate_raw(x, w=None)

    def gradient(
        self,
        coords: torch.Tensor,
        transform: bool = True,
        return_value: bool = False,
        normalize: bool = True,
        accumulate: bool = True,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Analytic gradient ∇φ at query points (geomodelling path).

        Optionally normalises to unit length and accumulates mean magnitude in
        ``mnorm`` (for ``CSet`` / ``loss``). ``retain_graph`` / ``create_graph``
        are accepted for API compatibility with ``BaseNF`` but unused.
        """
        if transform and self.transform is not None:
            coords = self.transform(coords, end=transform)

        pot, grad, _ = self._rff(coords, None, value=True, derivs=True, hess=False)

        if accumulate or normalize:
            norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
            if accumulate:
                self.mnorm = (
                    self.mnorm * self.nnorm + torch.mean(norm, axis=0).item() * len(norm)
                )
                self.nnorm += len(norm)
                self.mnorm = self.mnorm / self.nnorm
            if normalize:
                grad = grad / norm

        if return_value:
            return grad, pot
        return grad

    # ── Per-mode basis derivatives (linearised-Laplace / sensitivity API) ────

    def basis_derivatives(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        *,
        order: int = 2,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Per-mode basis functions ``f_k(x) = ∂φ/∂w_k`` and their spatial derivatives.

        Linear mode is linear in the concatenated amplitudes ``(A_mu, B_mu)``
        (``2F`` columns: cosine then sine). Activated mode is linear in
        ``C_mu`` only (the activation acts on the learned-phase sinusoid, never
        on ``C``). Used to assemble the sensitivity forcing ``B``.

        Parameters
        ----------
        x : torch.Tensor
            Local spatial coordinates ``(N, input_dim)``.
        w : torch.Tensor, optional
            Lift coordinates when ``lift_dim > 0``.
        order : int
            Highest spatial derivative to return (0–2).

        Returns
        -------
        (f, g, h, t) : tuple
            ``f`` ``(N, n_w)``; ``g`` ``(N, n_w, d)``; ``h`` ``(N, n_w, d, d)``;
            ``t`` is always ``None``. ``n_w = 2F`` (linear) or ``F`` (activated
            outer amplitudes). Entries above ``order`` are ``None``.

        Notes
        -----
        Contracting against :meth:`_amplitude_vector` reproduces the field::

            f @ w                        == evaluate_raw(x, w)
            einsum('nkd,k->nd', g, w)    == gradient_and_hessian(x, w)[0]
        """
        if order not in (0, 1, 2):
            raise ValueError(f"order must be 0, 1 or 2, got {order}")

        proj = self._projection(x, w)               # (N, F)
        coef = self.out_scale / self.amplitude_denom  # (F,)
        ox = self.omega_spatial                     # (F, d)
        d = ox.shape[1]

        s, c = _sincos(proj)

        if self.activation is None:
            # Columns: ∂/∂A (cos) then ∂/∂B (sin).
            #   ∂φ/∂A = coef c,  d1 = -s, d2 = -c
            #   ∂φ/∂B = coef s,  d1 =  c, d2 = -s
            f0 = torch.cat([c, s], dim=-1)
            d1 = torch.cat([-s, c], dim=-1)
            d2 = torch.cat([-c, -s], dim=-1)
            coef = torch.cat([coef, coef], dim=0)
            ox = torch.cat([ox, ox], dim=0)
            omega_outer = torch.cat([self.omega_outer, self.omega_outer], dim=0)
        else:
            coef = coef * self._energy_gain()
            u = s + self.bias
            sig, sp, spp = _activation_derivs(u, self.activation)
            f0 = sig
            d1 = sp * c
            d2 = spp * c * c - sp * s
            omega_outer = self.omega_outer

        f = coef * f0
        if order == 0:
            return f, None, None, None

        g = (coef * d1).unsqueeze(-1) * ox
        if order == 1:
            return f, g, None, None

        h = ((coef * d2).unsqueeze(-1) * omega_outer.unsqueeze(0)).view(
            x.shape[0], -1, d, d
        )
        return f, g, h, None

    # ── Internal series evaluation ──────────────────────────────────────────

    def _lifted_coords(
        self, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Concatenate lift coordinates onto ``x`` (identity when ``lift_dim=0``)."""
        if self.lift_dim == 0:
            return x
        if w is None or w.numel() == 0:
            w = x.new_zeros(x.shape[0], self.lift_dim)
        return torch.cat([x, w], dim=-1)

    def _projection(
        self, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mode projections ``Ω_k·x̃`` (linear) or ``Ω_k·x̃ + φ_k`` (activated)."""
        proj = self._lifted_coords(x, w) @ self.fourier_projection
        if self.activation is not None:
            proj = proj + self._phase()
        return proj

    def _phase(self) -> torch.Tensor:
        """Learned activated-mode phase angle."""
        if self.activation is None:
            raise AttributeError("linear FSF has no separate phase")
        return self.phi_mu

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Activated checkpoints saved with the old Cartesian phase chart
        # (A_mu, B_mu) load as the equivalent angle atan2(B, A).
        a, b, phi = prefix + "A_mu", prefix + "B_mu", prefix + "phi_mu"
        if self.activation is not None and phi not in state_dict and a in state_dict:
            state_dict[phi] = torch.atan2(state_dict.pop(b), state_dict.pop(a))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _lift_phase(
        self, x: Optional[torch.Tensor], w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Non-spatial part of the phase: lift labels, plus learned phase when
        activated. Shape ``(F,)`` with no lift, ``(N, F)`` otherwise.
        """
        n_modes = self.omega_spatial.shape[0]
        if self.lift_dim == 0 or w is None:
            extra = self.omega_spatial.new_zeros(n_modes)
        else:
            if x is None:
                raise ValueError("lift phase needs `x` when lift `w` is set")
            extra = (
                self._lifted_coords(x, w) @ self.fourier_projection
                - x[:, : self.input_dim] @ self.omega_spatial.T
            )
        if self.activation is not None:
            return extra + self._phase()
        return extra

    def _energy_gain(self):
        """Per-mode ``√(e(0)/e(b))`` when :attr:`energy_normalised`, else ``1``."""
        if not getattr(self, "energy_normalised", False):
            return 1.0
        b2 = self.bias * self.bias
        return torch.rsqrt((b2 * b2 - 0.5 * b2 + 0.625) / 0.625)

    def _effective_amplitudes(self) -> torch.Tensor:
        """Damped, scaled cosine amplitudes or activated outer amplitudes."""
        if self.activation is None:
            return self.out_scale * self.A_mu / self.amplitude_denom
        return self.out_scale * self.C_mu * self._energy_gain() / self.amplitude_denom

    def _effective_amplitudes_sin(self) -> torch.Tensor:
        """Damped, scaled sine amplitudes ``s·B_k / (√F · D_k)``. Linear mode only."""
        return self.out_scale * self.B_mu / self.amplitude_denom

    def _amplitude_vector(self) -> torch.Tensor:
        """Learnable weights matching :meth:`basis_derivatives` columns."""
        if self.activation is None:
            return torch.cat([self.A_mu, self.B_mu])
        return self.C_mu

    def _rff(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor],
        *,
        value: bool,
        derivs: bool,
        hess: bool = True,
    ):
        """
        Core Fourier series: value, spatial gradient, and Hessian.

        Linear mode is ``A cos(Ω·x) + B sin(Ω·x)``; activated mode is
        ``C σ(sin(Ω·x+φ)+bias)``. ``hess=False`` skips the Hessian
        einsum.
        """
        proj = self._projection(x, w)
        ox = self.omega_spatial
        d = ox.shape[1]
        n = proj.shape[0]
        s, c = _sincos(proj)

        if self.activation is None:
            a = self._effective_amplitudes()
            b = self._effective_amplitudes_sin()
            # value ~ A c + B s;  d/dθ ~ B c - A s;  d²/dθ² ~ -(A c + B s)
            combo = a * c + b * s
            psi = combo.sum(-1) if value else None
            if not derivs:
                return psi, None, None
            grad = (b * c - a * s) @ ox
            if not hess:
                return psi, grad, None
            hessian = (-combo @ self.omega_outer).view(n, d, d)
            return psi, grad, hessian

        amps = self._effective_amplitudes()
        sig, sp, spp = _activation_derivs(s + self.bias, self.activation)

        psi = (amps * sig).sum(-1) if value else None
        if not derivs:
            return psi, None, None
        grad = (amps * sp * c) @ ox
        if not hess:
            return psi, grad, None
        hessian = ((amps * (spp * c * c - sp * s)) @ self.omega_outer).view(n, d, d)
        return psi, grad, hessian
