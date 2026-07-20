"""
Fourier-series neural fields (FSF) for scalar potential representation.

Gradients and Hessians are computed in closed form (no autograd through the
series), which improves speed and stability for interpolation and — via the
raw API — for Clebsch / diffeomorphic deformation downstream.

Mode amplitudes ``A`` carry a Bayes-by-Backprop variational posterior
(Blundell et al. 2015, "Weight Uncertainty in Neural Networks",
arXiv:1505.05424) instead of a point estimate: each amplitude is
``A_mu + softplus(A_rho) * eps`` for held Gaussian noise ``eps``, giving
analytic, reparameterised weight uncertainty (and cheap MC uncertainty via
repeated forward passes) without the tuning finickiness of feature dropout.
"""
import contextlib
import math
from typing import List, Optional, Sequence, Tuple, Union

import curlew
import torch
import torch.nn as nn
import torch.nn.functional as F

from curlew.fields import BaseNF


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


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus, numerically safe for small and large ``y``."""
    return y + torch.log(-torch.expm1(-y))


def _build_fourier_modes(
    *,
    rff_features: int,
    length_scale_range: Tuple[float, float],
    input_dim: int,
    lift_dim: int,
    freq_sampling: str,
    stochastic_scales: bool,
    scale: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample fixed Fourier directions and return ``(fourier_projection, freq_scaling, out_scale)``.
    """
    d_in = input_dim + lift_dim
    torch.manual_seed(seed)
    lo = torch.log10(
        torch.tensor(length_scale_range[0], dtype=curlew.dtype, device=curlew.device)
    )
    hi = torch.log10(
        torch.tensor(length_scale_range[1], dtype=curlew.dtype, device=curlew.device)
    )

    if freq_sampling == "quasi":
        sobol = torch.quasirandom.SobolEngine(dimension=d_in + 1, scramble=True)
        u = sobol.draw(rff_features).to(
            device=curlew.device, dtype=curlew.dtype
        ).clamp(1e-6, 1 - 1e-6)
        lengths = torch.pow(
            torch.tensor(10.0, dtype=curlew.dtype, device=curlew.device),
            lo + u[:, 0] * (hi - lo),
        )
        k_raw = torch.special.ndtri(u[:, 1:].clamp(1e-6, 1 - 1e-6))
    else:
        lengths = torch.pow(
            10.0,
            torch.linspace(lo, hi, rff_features, dtype=curlew.dtype, device=curlew.device),
        )
        k_raw = torch.randn(
            rff_features, d_in, device=curlew.device, dtype=curlew.dtype
        )

    if not stochastic_scales:
        k_raw = F.normalize(k_raw, dim=1)

    omega = 2.0 * torch.pi * k_raw / lengths.unsqueeze(1)
    fourier_projection = omega.T
    freq_scaling = (omega**2).sum(1).clamp(min=1e-8)
    out_scale = torch.tensor(
        scale / math.sqrt(rff_features), dtype=curlew.dtype, device=curlew.device
    )
    return fourier_projection, freq_scaling, out_scale


class FSF(BaseNF):
    """
    FourierSeriesField — learnable scalar potential as a damped random Fourier series.

    Each mode uses amplitude ``A`` and phase ``phi`` (phase shifts the sinusoid;
    amplitude can be regularised or dropped out independently of phase).

    Two operating modes, selected by ``activation``:

    ── Linear mode (``activation=None``, default) ─────────────────────────────

        φ(x) = (s/√F) Σ_k  A_k sin(Ω_k·x + φ_k) / |Ω_k|^{freq_damp}

    With the default ``freq_damp=2`` this matches the legacy ``/ |Ω_k|²``
    damping. Lower values (e.g. ``1``) weaken high-frequency suppression so
    short-wavelength modes contribute more strongly to ``∇φ`` for the same ``A_k``.

    ── Activated mode (``activation`` given) ───────────────────────────────────

        φ(x) = (s/√F) Σ_k  A_k σ( sin(Ω_k·x + φ_k) + b_k ) / |Ω_k|²

        ``bias`` (``b_k``) breaks symmetry for vergent / asymmetric folds.
        The activation input is a sinusoid in [-1, 1], which keeps σ' active
        at initialisation and gives natural hinge loci at sin(·)=0.

    Analytic derivatives (spatial coordinates only when ``lift_dim > 0``):

        Linear:   ∇φ = (s/√F) Σ A_k cos(Ω·x+φ)/|Ω|² · Ω_k
                  H_φ = (s/√F) Σ −A_k sin(Ω·x+φ)/|Ω|² · Ω_k Ω_kᵀ

        Activated: chain rule through σ(sin(·)+b) — see ``gradient_and_hessian``.

    Optional features
    -----------------
    ``lift_dim``        : append fault-sheet coordinates (Clebsch / fault lift).
    ``domain``          : smooth box envelope on φ (and exact envelope derivatives).
    Weight uncertainty (Bayes by Backprop) : ``A`` is a diagonal Gaussian
                          posterior (``A_mu``, ``A_rho``) rather than a point
                          estimate, with a scale-mixture-of-Gaussians prior
                          (``prior_pi``, ``prior_sigma1``, ``prior_sigma2``).
                          :meth:`kl_loss` gives the complexity cost (add it to
                          training via ``HSet.kl_loss``); :meth:`mc_forward` /
                          :meth:`weight_sample` give cheap MC uncertainty.
                          During :meth:`~curlew.fields.BaseNF.fit`, a fresh
                          weight sample is drawn each epoch on all nested
                          :class:`FSF` instances (Clebsch β/α, time-blended
                          layers, or a scalar field fit directly); the field
                          reverts to its posterior mean (``A_mu``) once
                          training finishes, for deterministic downstream use.

    Parameters passed through ``initField`` / constructor keywords
    ----------------------------------------------------------------
    rff_features, length_scale_range, freq_sampling, activation, scale,
    learning_rate, lift_dim, posterior_rho_init, prior_pi, prior_sigma1,
    prior_sigma2, domain, taper, taper_frac, envelope_floor,
    stochastic_scales, xavier_amplitudes, freq_damp
    """

    def initField(
        self,
        rff_features: int,
        length_scale_range: Tuple[float, float],
        freq_sampling: str = "random",
        activation: Optional[str] = None,
        scale: float = 1.0,
        learning_rate: float = 1e-1,
        lift_dim: int = 0,
        posterior_rho_init: float = 0.0,
        prior_pi: float = 0.5,
        prior_sigma1: float = 1.0,
        prior_sigma2: float = 0.002,
        domain: Optional[Sequence[Tuple[float, float]]] = None,
        taper: Union[None, float, Sequence[float]] = None,
        taper_frac: Union[None, float, Sequence[float]] = None,
        envelope_floor: float = 0.0,
        stochastic_scales: bool = True,
        xavier_amplitudes: bool = True,
        freq_damp: float = 2.0,
    ):
        """
        Build Fourier frequencies and learnable mode parameters.

        Parameters
        ----------
        rff_features : int
            Number of Fourier modes F.
        length_scale_range : (float, float)
            Minimum and maximum wavelength for log-uniform frequency sampling.
        freq_sampling : ``"random"`` | ``"quasi"``
            ``"quasi"`` uses a Sobol sequence; ``"random"`` uses evenly spaced
            log wavelengths plus Gaussian directions (legacy FSF behaviour).
        activation : None | ``"tanh"`` | ``"cubic"`` | ``"softplus"``
            Nonlinearity applied to each mode in activated mode.
        scale : float
            Global output scale (combined with 1/√F in ``out_scale``).
        learning_rate : float
            Adam learning rate for ``init_optim``.
        lift_dim : int
            Extra input coordinates (e.g. fault sheet labels). Derivatives
            are returned with respect to spatial ``input_dim`` only.
        posterior_rho_init : float
            Initial value of the (unconstrained) posterior std parameter
            ``A_rho`` for every mode; the initial std is
            ``softplus(posterior_rho_init)``. Default ``-5.0`` gives a small
            initial std (≈0.0067) so training starts close to a point
            estimate and grows uncertainty only where the loss favours it.
        prior_pi : float
            Mixture weight of the wide prior component (Blundell et al. 2015,
            eq. 7). Must lie in (0, 1).
        prior_sigma1, prior_sigma2 : float
            Std of the wide and narrow scale-mixture prior components
            respectively (``prior_sigma1 > prior_sigma2``, e.g. a "spike and
            slab"-like prior). Both must be positive.
        domain, taper, taper_frac, envelope_floor
            Optional separable soft-box envelope on φ.
        stochastic_scales : bool
            If True (default), direction vectors are not normalised and |Ω|
            follows a chi-like distribution (as in ``curlew.fields.NFF``).
        xavier_amplitudes : bool
            If True (default), initialise mode amplitudes ``A`` with Xavier
            normal (as legacy ``w1``). Required for geology / scalar-field
            fitting where zero amplitudes stall optimisation. Set False for
            Clebsch potentials that are scaled manually after construction.
        """
        assert freq_sampling in ("random", "quasi"), (
            f"freq_sampling must be 'random' or 'quasi', got {freq_sampling!r}"
        )
        assert activation in (None, "tanh", "cubic", "softplus"), (
            f"activation must be None, 'tanh', 'cubic', or 'softplus', "
            f"got {activation!r}"
        )
        if not 0.0 < prior_pi < 1.0:
            raise ValueError("prior_pi must be in (0, 1)")
        if not (prior_sigma1 > 0 and prior_sigma2 > 0):
            raise ValueError("prior_sigma1 and prior_sigma2 must be positive")

        self.activation = activation
        self.lift_dim = int(lift_dim)
        self.prior_pi = float(prior_pi)
        self.prior_sigma1 = float(prior_sigma1)
        self.prior_sigma2 = float(prior_sigma2)
        self._posterior_rho_init = float(posterior_rho_init)
        self._eps: Optional[torch.Tensor] = None  # held reparameterisation noise; None => posterior mean
        if freq_damp <= 0:
            raise ValueError(f"freq_damp must be positive, got {freq_damp}")
        self.freq_damp = float(freq_damp)

        projection, freq_scaling, out_scale = _build_fourier_modes(
            rff_features=rff_features,
            length_scale_range=length_scale_range,
            input_dim=self.input_dim,
            lift_dim=self.lift_dim,
            freq_sampling=freq_sampling,
            stochastic_scales=stochastic_scales,
            scale=scale,
            seed=self.seed,
        )
        self.register_buffer("fourier_projection", projection)
        self.register_buffer("freq_scaling", freq_scaling)
        self.register_buffer("out_scale", out_scale)

        # ── Learnable mode parameters ───────────────────────────────────────
        # Amplitudes are a diagonal Gaussian posterior q(A|A_mu, A_rho), sampled via
        # the reparameterisation trick: A = A_mu + softplus(A_rho) * eps (see sampled_A).
        self.A_mu = nn.Parameter(torch.zeros(rff_features, dtype=curlew.dtype, device=curlew.device))
        if xavier_amplitudes:
            nn.init.xavier_normal_(self.A_mu.unsqueeze(0))  # (1, F) for fan_in scaling
        self.A_rho = nn.Parameter(
            torch.full((rff_features,), self._posterior_rho_init, dtype=curlew.dtype, device=curlew.device)
        )
        self.phi = nn.Parameter(torch.zeros(rff_features, dtype=curlew.dtype, device=curlew.device))
        if activation is not None:
            self.bias = nn.Parameter(
                torch.zeros(rff_features, dtype=curlew.dtype, device=curlew.device)
            )

        # ── Optional domain envelope ──────────────────────────────────────────
        self.windowed = domain is not None
        if self.windowed:
            if len(domain) != self.input_dim:
                raise ValueError(f"domain needs {self.input_dim} (lo, hi) pairs")
            bounds = torch.tensor(domain, dtype=curlew.dtype, device=curlew.device)
            if taper is None:
                if taper_frac is None:
                    raise ValueError("supply taper or taper_frac with domain")
                taper = (torch.tensor(taper_frac, dtype=curlew.dtype, device=curlew.device)
                         * (bounds[:, 1] - bounds[:, 0])).tolist()
            width = torch.tensor(taper, dtype=curlew.dtype, device=curlew.device)
            if width.ndim == 0:
                width = width.expand(self.input_dim).clone()
            if not torch.all(width > 0) or not torch.all(bounds[:, 1] > bounds[:, 0]):
                raise ValueError("taper widths and domain extents must be positive")
            self.register_buffer("win_lo", bounds[:, 0])
            self.register_buffer("win_hi", bounds[:, 1])
            self.register_buffer("win_w", width)
            if not 0.0 <= envelope_floor < 1.0:
                raise ValueError("envelope_floor must be in [0, 1)")
            self.envelope_floor = float(envelope_floor)

        self.to(curlew.device)
        self.init_optim(lr=learning_rate)

    @property
    def dim(self) -> int:
        """Spatial dimension (alias for ``input_dim``; used by Clebsch adapters)."""
        return self.input_dim

    # ── Weight uncertainty (Bayes by Backprop) ──────────────────────────────
    # Blundell et al. 2015, "Weight Uncertainty in Neural Networks" (arXiv:1505.05424).

    @classmethod
    def iter_in(cls, module: nn.Module):
        """Yield every :class:`FSF` submodule (e.g. Clebsch β/α potentials)."""
        for m in module.modules():
            if isinstance(m, cls):
                yield m

    @classmethod
    def resample_weights_on(
        cls, module: nn.Module, generator: Optional[torch.Generator] = None
    ) -> None:
        """Draw a fresh held reparameterisation sample on each :class:`FSF` under ``module``."""
        for m in cls.iter_in(module):
            m.resample_weights(generator=generator)

    @classmethod
    def clear_weight_samples_on(cls, module: nn.Module) -> None:
        """Revert every :class:`FSF` under ``module`` to its posterior mean (deterministic)."""
        for m in cls.iter_in(module):
            m.clear_weight_sample()

    @classmethod
    @contextlib.contextmanager
    def weight_sample_on(
        cls, module: nn.Module, generator: Optional[torch.Generator] = None
    ):
        """Hold one coherent weight sample on all nested :class:`FSF` instances."""
        fields = list(cls.iter_in(module))
        prev = [m._eps for m in fields]
        try:
            cls.resample_weights_on(module, generator=generator)
            yield
        finally:
            for m, pm in zip(fields, prev):
                m._eps = pm

    @classmethod
    def kl_loss_on(cls, module: nn.Module) -> torch.Tensor:
        """Sum the Bayes-by-Backprop complexity cost over every :class:`FSF` under ``module``."""
        terms = [m.kl_loss() for m in cls.iter_in(module)]
        if not terms:
            return None
        return torch.stack(terms).sum()

    @classmethod
    def kinetic_energy_on(cls, module: nn.Module) -> Optional[torch.Tensor]:
        """
        Sum :meth:`kinetic_energy` over every :class:`FSF` **and**
        :class:`TemporalFSF` under ``module`` (temporal potentials contribute
        their path-averaged energy with spectral gates folded in).
        """
        terms = [
            m.kinetic_energy()
            for m in module.modules()
            if isinstance(m, (FSF, TemporalFSF))
        ]
        if not terms:
            return None
        return torch.stack(terms).sum()

    def resample_weights(self, generator: Optional[torch.Generator] = None) -> None:
        """
        Draw and hold fresh reparameterisation noise ``eps ~ N(0, I)`` for ``A``.

        The held sample stays fixed (``A = A_mu + softplus(A_rho) * eps``) until
        cleared or resampled — one coherent draw per integration or MC sample.
        """
        self._eps = torch.randn(
            self.A_mu.shape[0], generator=generator, dtype=self.A_mu.dtype, device=self.A_mu.device
        )

    def clear_weight_sample(self) -> None:
        """Revert to the posterior mean ``A_mu`` (deterministic)."""
        self._eps = None

    @contextlib.contextmanager
    def weight_sample(self, generator: Optional[torch.Generator] = None):
        """Hold one random weight sample, then restore the previous state."""
        prev = self._eps
        try:
            self.resample_weights(generator=generator)
            yield
        finally:
            self._eps = prev

    def _sigma(self) -> torch.Tensor:
        """Posterior std ``softplus(A_rho)`` (always positive)."""
        return F.softplus(self.A_rho).clamp_min(1e-8)

    def sampled_A(self) -> torch.Tensor:
        """Current amplitudes: ``A_mu`` if no sample is held, else a reparameterised draw."""
        if self._eps is None:
            return self.A_mu
        return self.A_mu + self._sigma() * self._eps

    @staticmethod
    def _log_gaussian(w: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Elementwise ``log N(w | mu, sigma^2)``."""
        return (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(sigma)
            - (w - mu) ** 2 / (2.0 * sigma ** 2)
        )

    def kl_loss(self) -> torch.Tensor:
        """
        Bayes-by-Backprop complexity cost ``log q(A|theta) - log P(A)`` for the
        currently held weight sample (Blundell et al. 2015, eq. 2), using a
        scale-mixture-of-Gaussians prior (eq. 7). Draws a fresh sample on the
        fly if none is currently held.
        """
        eps = self._eps if self._eps is not None else torch.randn_like(self.A_mu)
        sigma = self._sigma()
        w = self.A_mu + sigma * eps

        log_q = self._log_gaussian(w, self.A_mu, sigma).sum()

        zero = torch.zeros_like(w)
        s1 = w.new_full((), self.prior_sigma1)
        s2 = w.new_full((), self.prior_sigma2)
        log_p1 = self._log_gaussian(w, zero, s1) + math.log(self.prior_pi)
        log_p2 = self._log_gaussian(w, zero, s2) + math.log(1.0 - self.prior_pi)
        log_p = torch.logaddexp(log_p1, log_p2).sum()

        return log_q - log_p

    def kinetic_energy(self) -> torch.Tensor:
        """
        Analytic domain-averaged kinetic energy ``⟨‖∇φ‖²⟩`` as a quadratic in
        the amplitudes (see :meth:`sampled_A` for BBB draws):

            ⟨‖∇φ‖²⟩ ≈ out_scale² · Σ_k A_k² |ω_k^spatial|² / (2 |Ω_k|⁴)

        For a Clebsch streamfunction β, ``v = ∇⊥β`` has ``|v| = |∇β|`` at
        fixed time. Weight via ``HSet.energy_loss`` in restoration flows.
        """
        a = self.sampled_A()
        ox = self._omega_spatial()
        w = (ox**2).sum(1) / self.freq_scaling.pow(self.freq_damp)
        return 0.5 * self.out_scale**2 * (a**2 * w).sum()

    @torch.no_grad()
    def mc_forward(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        n_samples: int = 20,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mean and standard deviation of φ over posterior weight samples.

        Each draw reparameterises ``A ~ q(A|theta)``; useful for a cheap
        uncertainty spread without retraining an ensemble.
        """
        samples = []
        for _ in range(n_samples):
            with self.weight_sample(generator=generator):
                samples.append(self.evaluate_raw(x, w=w))
        stacked = torch.stack(samples)
        return stacked.mean(0), stacked.std(0)

    # ── Raw evaluation (Clebsch / flow — no drift, transform, or normalise) ─

    def evaluate_raw(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Evaluate φ at local spatial coordinates (optionally with lift coords ``w``).

        Does not apply ``Transform``, drift, or gradient normalisation.
        """
        psi, _, _ = self._rff(x, w, value=True, derivs=False)
        if not self.windowed:
            return psi
        envelope, _, _ = self._envelope(x, derivs=False)
        return envelope * psi

    def gradient_and_hessian(
        self, x: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Spatial gradient (N, dim) and Hessian (N, dim, dim) of φ.

        Used by Clebsch velocity construction and Magnus integration. When
        ``windowed``, envelope derivatives are included exactly.
        """
        if not self.windowed:
            _, grad, hess = self._rff(x, w, value=False, derivs=True)
            return grad, hess
        psi, grad, hess = self._rff(x, w, value=True, derivs=True)
        envelope, grad_e, hess_e = self._envelope(x, derivs=True)
        grad = envelope.unsqueeze(-1) * grad + psi.unsqueeze(-1) * grad_e
        cross = torch.einsum("ni,nj->nij", grad_e, grad)
        hess = (
            envelope[:, None, None] * hess
            + psi[:, None, None] * hess_e
            + cross
            + cross.mT
        )
        return grad, hess

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

        grad, pot = self._geomodelling_grad_value(coords)

        if accumulate or normalize:
            norm = torch.norm(grad, dim=-1, keepdim=True) + 1e-8
            if accumulate:
                self.mnorm = (
                    self.mnorm * self.nnorm
                    + torch.mean(norm, axis=0).item() * len(norm)
                )
                self.nnorm += len(norm)
                self.mnorm = self.mnorm / self.nnorm
            if normalize:
                grad = grad / norm

        if return_value:
            return grad, pot
        return grad

    # ── Internal series evaluation ──────────────────────────────────────────

    def _omega_spatial(self) -> torch.Tensor:
        """Spatial frequency rows (F, input_dim)."""
        return self.fourier_projection[: self.input_dim, :].T

    def _amplitude_denom(self) -> torch.Tensor:
        """Per-mode amplitude divisor |Ω|^{freq_damp} (``freq_scaling = |Ω|²``)."""
        return self.freq_scaling.pow(0.5 * self.freq_damp)

    def _effective_amplitudes(self) -> torch.Tensor:
        """Scaled amplitudes (F,) from the current weight sample (or posterior mean)."""
        return self.out_scale * self.sampled_A() / self._amplitude_denom()

    def _lifted_coords(
        self, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.lift_dim == 0:
            return x
        if w is None or w.numel() == 0:
            w = x.new_zeros(x.shape[0], self.lift_dim)
        return torch.cat([x, w], dim=-1)

    def _projection(
        self, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mode projections Ω_k·x̃ + φ_k  (N, F)."""
        lifted = self._lifted_coords(x, w)
        return lifted @ self.fourier_projection + self.phi

    def _rff(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor],
        *,
        value: bool,
        derivs: bool,
    ):
        """
        Core Fourier series: value, spatial gradient, and Hessian.

        Linear mode uses sin(Ω·x+φ); activated mode uses σ(sin(·)+bias).
        """
        proj = self._projection(x, w)
        amps = self._effective_amplitudes()
        ox = self._omega_spatial()

        s = torch.sin(proj)
        c = torch.cos(proj)

        if self.activation is None:
            psi = (amps * s).sum(-1) if value else None
            if not derivs:
                return psi, None, None
            grad = (amps * c) @ ox
            hess = torch.einsum("nk,ki,kj->nij", -(amps * s), ox, ox)
            return psi, grad, hess

        sig, sp, spp = _activation_derivs(s + self.bias, self.activation)
        psi = (amps * sig).sum(-1) if value else None
        if not derivs:
            return psi, None, None
        grad = (amps * sp * c) @ ox
        hess = torch.einsum(
            "nk,ki,kj->nij", amps * (spp * c * c - sp * s), ox, ox
        )
        return psi, grad, hess

    def _envelope(self, x: torch.Tensor, *, derivs: bool):
        """Separable soft-box envelope E(x) and optional derivatives."""
        a = (x - self.win_lo) / self.win_w
        b = (self.win_hi - x) / self.win_w
        s_lo, s_hi = torch.sigmoid(a), torch.sigmoid(b)
        e = s_lo * s_hi
        envelope = e.prod(dim=-1)
        floor = 1.0 - self.envelope_floor
        if not derivs:
            return self.envelope_floor + floor * envelope, None, None

        inv_w = 1.0 / self.win_w
        d_lo = s_lo * (1 - s_lo) * inv_w
        d_hi = -s_hi * (1 - s_hi) * inv_w
        e_p = d_lo * s_hi + s_lo * d_hi
        e_pp = (
            (s_lo * (1 - s_lo) * (1 - 2 * s_lo) * inv_w**2) * s_hi
            + 2 * d_lo * d_hi
            + s_lo * (s_hi * (1 - s_hi) * (1 - 2 * s_hi) * inv_w**2)
        )
        e_safe = e.clamp_min(1e-12)
        r1, r2 = e_p / e_safe, e_pp / e_safe
        grad_e = envelope.unsqueeze(-1) * r1
        hess_e = envelope[:, None, None] * torch.einsum("ni,nj->nij", r1, r1)
        idx = torch.arange(self.input_dim, device=x.device)
        hess_e[:, idx, idx] = envelope.unsqueeze(-1) * r2
        return self.envelope_floor + floor * envelope, floor * grad_e, floor * hess_e

    def _geomodelling_grad_value(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gradient and scalar value at coords (local field frame)."""
        if not self.windowed:
            pot, grad, _ = self._rff(coords, None, value=True, derivs=True)
            return grad, pot
        psi, grad, hess = self._rff(coords, None, value=True, derivs=True)
        envelope, grad_e, hess_e = self._envelope(coords, derivs=True)
        pot = envelope * psi
        grad = envelope.unsqueeze(-1) * grad + psi.unsqueeze(-1) * grad_e
        return grad, pot


class TemporalFSF(BaseNF):
    """
    Fourier-series potential with a prescribed spectral schedule over
    integration time ``t ∈ [0, 1]``.

    Modes are drawn once from ``length_scale_range = (λ_fine, λ_coarse)``.
    By default each mode's **spatial direction** is learnable while its
    spatial frequency magnitude ``|ω_k|`` stays fixed at the drawn value, so
    the temporal gate schedule (which depends only on ``|ω_k|``) is unchanged
    under direction updates.

    Mode strengths are **positive magnitudes** ``A_k = softplus(A_rho_k)``;
    polarity lives in the phase ``φ_k`` (a sign flip is ``φ → φ + π``). Use
    :meth:`set_amplitudes` to set magnitudes (optionally folding a signed
    input into ``φ``).

    The default **lowpass** gate

        g_k(t) = exp(−½ (|ω_k| / ω_c(t))²),   ω_c(t) = 2π / λ_c(t)

    only **opens** finer modes as ``t → 1``; long wavelengths stay active once
    admitted. Set ``spectral_gate="bandpass"`` for a Gaussian window in log‑ω
    that sweeps from coarse to fine (closing long‑λ modes behind the front), or
    ``spectral_gate="reverse"`` to run the cutoff schedule backwards
    (fine at ``t = 0``, coarse at ``t = 1``).

    Lowpass / bandpass center wavelength:

        λ_c(t) = λ_fine · (λ_coarse / λ_fine)^(1 − t).

    Reverse:

        λ_c(t) = λ_fine · (λ_coarse / λ_fine)^t.

    Restoration integrates ``t : 1 → 0``, so lowpass starts at full bandwidth
    (present) and ends coarse (reference).

    Parameters passed through ``initField`` / constructor keywords
    ----------------------------------------------------------------
    rff_features, length_scale_range, freq_sampling, activation, scale,
    learning_rate, lift_dim, domain, taper, taper_frac, envelope_floor,
    stochastic_scales, learnable_directions, init_std, freq_damp,
    spectral_gate, spectral_band_width

    Learnable mode parameters: ``A_rho`` (→ ``A = softplus(A_rho)``), ``phi``,
    optional ``bias``, and (if enabled) ``theta`` / ``dir_raw``.
    """

    def initField(
        self,
        rff_features: int,
        length_scale_range: Tuple[float, float],
        freq_sampling: str = "random",
        activation: Optional[str] = None,
        scale: float = 1.0,
        learning_rate: float = 1e-1,
        lift_dim: int = 0,
        domain: Optional[Sequence[Tuple[float, float]]] = None,
        taper: Union[None, float, Sequence[float]] = None,
        taper_frac: Union[None, float, Sequence[float]] = None,
        envelope_floor: float = 0.0,
        stochastic_scales: bool = True,
        learnable_directions: bool = True,
        init_std: float = 0.0,
        freq_damp: float = 2.0,
        spectral_gate: str = "lowpass",
        spectral_band_width: Optional[float] = None,
    ):
        assert freq_sampling in ("random", "quasi"), (
            f"freq_sampling must be 'random' or 'quasi', got {freq_sampling!r}"
        )
        assert activation in (None, "tanh", "cubic", "softplus"), (
            f"activation must be None, 'tanh', 'cubic', or 'softplus', "
            f"got {activation!r}"
        )
        if spectral_gate not in ("lowpass", "bandpass", "reverse"):
            raise ValueError(
                f"spectral_gate must be 'lowpass', 'bandpass', or 'reverse', "
                f"got {spectral_gate!r}"
            )

        self.activation = activation
        self.lift_dim = int(lift_dim)
        self.learnable_directions = bool(learnable_directions)
        if freq_damp <= 0:
            raise ValueError(f"freq_damp must be positive, got {freq_damp}")
        self.freq_damp = float(freq_damp)
        self.spectral_gate = spectral_gate
        lam_fine, lam_coarse = float(length_scale_range[0]), float(length_scale_range[1])
        if lam_fine <= 0 or lam_coarse <= 0 or lam_coarse < lam_fine:
            raise ValueError(
                f"length_scale_range must be (λ_fine, λ_coarse) with "
                f"0 < λ_fine ≤ λ_coarse, got {length_scale_range}"
            )
        self.register_buffer(
            "lam_fine",
            torch.tensor(lam_fine, dtype=curlew.dtype, device=curlew.device),
        )
        self.register_buffer(
            "lam_coarse",
            torch.tensor(lam_coarse, dtype=curlew.dtype, device=curlew.device),
        )
        omega_fine = 2.0 * math.pi / lam_fine
        omega_coarse = 2.0 * math.pi / lam_coarse
        if spectral_band_width is None:
            spectral_band_width = 0.35 * abs(math.log(omega_fine) - math.log(omega_coarse))
        if spectral_band_width <= 0:
            raise ValueError(
                f"spectral_band_width must be positive, got {spectral_band_width}"
            )
        self.spectral_band_width = float(spectral_band_width)

        projection, freq_scaling, out_scale = _build_fourier_modes(
            rff_features=rff_features,
            length_scale_range=length_scale_range,
            input_dim=self.input_dim,
            lift_dim=self.lift_dim,
            freq_sampling=freq_sampling,
            stochastic_scales=stochastic_scales,
            scale=scale,
            seed=self.seed,
        )
        self.register_buffer("freq_scaling", freq_scaling)
        self.register_buffer("out_scale", out_scale)

        # Spatial rows (F, dim). |ω_spatial| is frozen so spectral gates stay
        # tied to the drawn bank; only the unit direction may be learned.
        ox = projection[: self.input_dim, :].T
        spatial_mag = ox.norm(dim=1).clamp_min(1e-8)
        self.register_buffer("spatial_omega_mag", spatial_mag)

        # Per-mode weight in <|grad phi|^2> for amplitude A_k (depends on freq_damp).
        # Pure direction updates leave |ω_spatial| (and |Ω|) unchanged, so this
        # quadratic weight can stay a fixed buffer.
        energy_w = (ox**2).sum(1) / freq_scaling.pow(self.freq_damp)
        self.register_buffer("energy_w", energy_w)

        if self.learnable_directions:
            if self.input_dim == 2:
                self.theta = nn.Parameter(torch.atan2(ox[:, 1], ox[:, 0]))
            else:
                self.dir_raw = nn.Parameter(ox.detach().clone())
            if self.lift_dim > 0:
                self.register_buffer(
                    "lift_projection", projection[self.input_dim :, :].contiguous()
                )
            # Keep the init bank for inspection / warm-start comparisons.
            self.register_buffer("fourier_projection", projection)
        else:
            self.register_buffer("fourier_projection", projection)

        # Unconstrained magnitude logits; A = softplus(A_rho) ≥ 0. Polarity is
        # carried by φ (sign(A)·sin(θ) ≡ |A|·sin(θ+π)), so signed amplitudes
        # are redundant with phase for linear / odd activated modes (b≡0).
        if init_std > 0:
            a0 = torch.abs(
                torch.randn(rff_features, dtype=curlew.dtype, device=curlew.device)
            ) * float(init_std)
            a0 = a0.clamp_min(1e-8)
            self.A_rho = nn.Parameter(_inv_softplus(a0))
        else:
            # softplus(A_rho) ≈ 0 → inactive bank until training moves ρ
            self.A_rho = nn.Parameter(
                torch.full(
                    (rff_features,),
                    -20.0,
                    dtype=curlew.dtype,
                    device=curlew.device,
                )
            )
        self.phi = nn.Parameter(
            torch.zeros(rff_features, dtype=curlew.dtype, device=curlew.device)
        )
        if activation is not None:
            self.bias = nn.Parameter(
                torch.zeros(rff_features, dtype=curlew.dtype, device=curlew.device)
            )

        self.windowed = domain is not None
        if self.windowed:
            if len(domain) != self.input_dim:
                raise ValueError(f"domain needs {self.input_dim} (lo, hi) pairs")
            bounds = torch.tensor(domain, dtype=curlew.dtype, device=curlew.device)
            if taper is None:
                if taper_frac is None:
                    raise ValueError("supply taper or taper_frac with domain")
                taper = (
                    torch.tensor(taper_frac, dtype=curlew.dtype, device=curlew.device)
                    * (bounds[:, 1] - bounds[:, 0])
                ).tolist()
            width = torch.tensor(taper, dtype=curlew.dtype, device=curlew.device)
            if width.ndim == 0:
                width = width.expand(self.input_dim).clone()
            if not torch.all(width > 0) or not torch.all(bounds[:, 1] > bounds[:, 0]):
                raise ValueError("taper widths and domain extents must be positive")
            self.register_buffer("win_lo", bounds[:, 0])
            self.register_buffer("win_hi", bounds[:, 1])
            self.register_buffer("win_w", width)
            if not 0.0 <= envelope_floor < 1.0:
                raise ValueError("envelope_floor must be in [0, 1)")
            self.envelope_floor = float(envelope_floor)

        self.to(curlew.device)
        self.init_optim(lr=learning_rate)

    @property
    def dim(self) -> int:
        return self.input_dim

    @property
    def A(self) -> torch.Tensor:
        """Positive mode magnitudes ``softplus(A_rho)`` with shape ``(F,)``."""
        return F.softplus(self.A_rho)

    def set_amplitudes(self, values, *, signed: bool = False) -> None:
        """
        Set mode magnitudes from ``values`` (broadcastable to ``(F,)``).

        Parameters
        ----------
        values
            Target amplitudes. Taken in absolute value unless ``signed=False``
            and negatives are rejected.
        signed : bool
            If True, negative entries flip the corresponding phase by ``π``
            and the magnitude is set to ``|values|``. If False (default),
            ``values`` must be non-negative.
        """
        v = torch.as_tensor(values, dtype=self.A_rho.dtype, device=self.A_rho.device)
        if v.ndim == 0:
            v = v.expand_as(self.A_rho)
        else:
            v = v.reshape(-1)
            if v.numel() == 1:
                v = v.expand_as(self.A_rho)
            if v.shape != self.A_rho.shape:
                raise ValueError(
                    f"amplitudes must broadcast to {tuple(self.A_rho.shape)}, "
                    f"got {tuple(v.shape)}"
                )
        if signed:
            neg = v < 0
            if bool(neg.any()):
                self.phi.data[neg] = self.phi.data[neg] + math.pi
            v = v.abs()
        elif bool((v < 0).any()):
            raise ValueError(
                "amplitudes must be non-negative; pass signed=True to fold "
                "signs into phase"
            )
        self.A_rho.data.copy_(_inv_softplus(v.clamp_min(1e-12)))

    def _clamp_t(self, t: Optional[float]) -> float:
        if t is None:
            return 1.0
        return min(max(float(t), 0.0), 1.0)

    def cutoff_wavelength(self, t: Optional[float] = None) -> torch.Tensor:
        """Center cutoff wavelength λ_c(t) for the spectral window."""
        t = self._clamp_t(t)
        ratio = self.lam_coarse / self.lam_fine
        if self.spectral_gate == "reverse":
            return self.lam_fine * ratio ** t
        return self.lam_fine * ratio ** (1.0 - t)

    def _center_omega(self, t: Optional[float]) -> torch.Tensor:
        lam_c = self.cutoff_wavelength(t)
        return (2.0 * math.pi / lam_c).clamp(min=1e-8)

    def _gates_for(self, t: Optional[float], ox: torch.Tensor) -> torch.Tensor:
        """Per-mode gates g_k(t) from spatial frequency rows."""
        omega_mag = ox.norm(dim=1).clamp(min=1e-8)
        if self.spectral_gate == "bandpass":
            log_om = omega_mag.log()
            log_om_c = self._center_omega(t).log()
            sigma = self.spectral_band_width
            return torch.exp(-0.5 * ((log_om - log_om_c) / sigma) ** 2)
        omega_c = self._center_omega(t)
        return torch.exp(-0.5 * (omega_mag / omega_c) ** 2)

    def spectral_gates(self, t: Optional[float] = None) -> torch.Tensor:
        """Per-mode Gaussian gates g_k(t) with shape (F,)."""
        return self._gates_for(t, self._omega_spatial())

    def kinetic_energy(
        self, t: Optional[float] = None, n_quad: int = 8
    ) -> torch.Tensor:
        """
        Analytic domain-averaged kinetic energy ``⟨‖∇φ(·, t)‖²⟩`` with the
        spectral gates folded into the amplitudes:

            E(t) = ½ out_scale² Σ_k (A_k g_k(t))² w_k

        (same quadratic as :meth:`FSF.kinetic_energy`, with the effective
        gated amplitudes in place of ``A``). If ``t`` is None, returns the
        **path-averaged** energy ``∫₀¹ E(t) dt`` via midpoint quadrature with
        ``n_quad`` nodes — the natural regulariser for a time-varying flow.
        """
        def _energy_at(tt: float) -> torch.Tensor:
            g_eff = self.spectral_gates(tt)
            return ((self.A * g_eff) ** 2 * self.energy_w).sum()

        if t is not None:
            e = _energy_at(float(t))
        else:
            nodes = (torch.arange(n_quad, dtype=curlew.dtype) + 0.5) / n_quad
            e = torch.stack([_energy_at(float(tt)) for tt in nodes]).mean()
        return 0.5 * self.out_scale**2 * e

    def evaluate_raw(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        t: Optional[float] = None,
    ) -> torch.Tensor:
        """Evaluate φ(x, t) without geomodelling transform or drift."""
        psi, _, _ = self._rff(x, w, t=t, value=True, derivs=False)
        if not self.windowed:
            return psi
        envelope, _, _ = self._envelope(x, derivs=False)
        return envelope * psi

    def gradient_and_hessian(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        t: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Spatial ∇φ and H_φ at fixed integration time ``t``."""
        if not self.windowed:
            _, grad, hess = self._rff(x, w, t=t, value=False, derivs=True)
            return grad, hess
        psi, grad, hess = self._rff(x, w, t=t, value=True, derivs=True)
        envelope, grad_e, hess_e = self._envelope(x, derivs=True)
        grad = envelope.unsqueeze(-1) * grad + psi.unsqueeze(-1) * grad_e
        cross = torch.einsum("ni,nj->nij", grad_e, grad)
        hess = (
            envelope[:, None, None] * hess
            + psi[:, None, None] * hess_e
            + cross
            + cross.mT
        )
        return grad, hess

    def evaluate(self, x: torch.Tensor, t: Optional[float] = None) -> torch.Tensor:
        """Evaluate at present-day spectrum (``t = 1``) unless overridden."""
        return self.evaluate_raw(x, w=None, t=t)

    def _unit_dir(self) -> torch.Tensor:
        """Learnable spatial unit directions ``û_k`` with shape ``(F, dim)``."""
        if self.input_dim == 2:
            return torch.stack([self.theta.cos(), self.theta.sin()], dim=-1)
        return F.normalize(self.dir_raw, dim=1)

    def _omega_spatial(self) -> torch.Tensor:
        """Spatial frequency rows ``(F, dim)``. Magnitudes are frozen at init."""
        if self.learnable_directions:
            return self.spatial_omega_mag.unsqueeze(-1) * self._unit_dir()
        return self.fourier_projection[: self.input_dim, :].T

    def _current_fourier_projection(self) -> torch.Tensor:
        """
        Full frequency matrix ``(dim + lift_dim, F)`` used in ``Ω·x̃``.

        With ``learnable_directions``, spatial columns are rebuilt from the
        frozen magnitudes and current unit directions; lift rows stay fixed.
        """
        if not self.learnable_directions:
            return self.fourier_projection
        proj = self._omega_spatial().T
        if self.lift_dim > 0:
            proj = torch.cat([proj, self.lift_projection], dim=0)
        return proj

    def _amplitude_denom(self) -> torch.Tensor:
        """Per-mode amplitude divisor |Ω|^{freq_damp} (``freq_scaling = |Ω|²``)."""
        return self.freq_scaling.pow(0.5 * self.freq_damp)

    def _effective_amplitudes(self, t: Optional[float]) -> torch.Tensor:
        return (
            self.out_scale
            * self.A
            * self.spectral_gates(t)
            / self._amplitude_denom()
        )

    def _lifted_coords(
        self, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.lift_dim == 0:
            return x
        if w is None or w.numel() == 0:
            w = x.new_zeros(x.shape[0], self.lift_dim)
        return torch.cat([x, w], dim=-1)

    def _projection(
        self, x: torch.Tensor, w: Optional[torch.Tensor]
    ) -> torch.Tensor:
        lifted = self._lifted_coords(x, w)
        return lifted @ self._current_fourier_projection() + self.phi

    def _rff(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor],
        *,
        t: Optional[float],
        value: bool,
        derivs: bool,
    ):
        proj = self._projection(x, w)
        amps = self._effective_amplitudes(t)
        ox = self._omega_spatial()

        s = torch.sin(proj)
        c = torch.cos(proj)

        if self.activation is None:
            psi = (amps * s).sum(-1) if value else None
            if not derivs:
                return psi, None, None
            grad = (amps * c) @ ox
            hess = torch.einsum("nk,ki,kj->nij", -(amps * s), ox, ox)
            return psi, grad, hess

        sig, sp, spp = _activation_derivs(s + self.bias, self.activation)
        psi = (amps * sig).sum(-1) if value else None
        if not derivs:
            return psi, None, None
        grad = (amps * sp * c) @ ox
        hess = torch.einsum(
            "nk,ki,kj->nij", amps * (spp * c * c - sp * s), ox, ox
        )
        return psi, grad, hess

    def _envelope(self, x: torch.Tensor, *, derivs: bool):
        a = (x - self.win_lo) / self.win_w
        b = (self.win_hi - x) / self.win_w
        s_lo, s_hi = torch.sigmoid(a), torch.sigmoid(b)
        e = s_lo * s_hi
        envelope = e.prod(dim=-1)
        floor = 1.0 - self.envelope_floor
        if not derivs:
            return self.envelope_floor + floor * envelope, None, None

        inv_w = 1.0 / self.win_w
        d_lo = s_lo * (1 - s_lo) * inv_w
        d_hi = -s_hi * (1 - s_hi) * inv_w
        e_p = d_lo * s_hi + s_lo * d_hi
        e_pp = (
            (s_lo * (1 - s_lo) * (1 - 2 * s_lo) * inv_w**2) * s_hi
            + 2 * d_lo * d_hi
            + s_lo * (s_hi * (1 - s_hi) * (1 - 2 * s_hi) * inv_w**2)
        )
        e_safe = e.clamp_min(1e-12)
        r1, r2 = e_p / e_safe, e_pp / e_safe
        grad_e = envelope.unsqueeze(-1) * r1
        hess_e = envelope[:, None, None] * torch.einsum("ni,nj->nij", r1, r1)
        idx = torch.arange(self.input_dim, device=x.device)
        hess_e[:, idx, idx] = envelope.unsqueeze(-1) * r2
        return self.envelope_floor + floor * envelope, floor * grad_e, floor * hess_e
