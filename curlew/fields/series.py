"""
Implement series based neural fields for scalar potential representation.
The strength of these fields is that they are an analytical representation of the scalar potential,
so gradients and hessians can be computed in closed form without backpropagation.  This can improve
speed and stability for some applications.
"""
import curlew
from curlew.core import HSet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple
from curlew.fields import BaseNF

class FSF(BaseNF):
    """
    FourierSeriesField
    
    A Fourier series based neural field that represents scalar potentials as a sum of sinusoidal basis functions.
    There is no multi-layer perceptron in this field, as the Fourier coefficients are directly learned as parameters
    of the model. Optionally, the basis can be passed through non-linear activations to increase expressivity, though
    this is not common in traditional Fourier series representations. Since the entire model is analytical, gradients
    and hessians can be computed in closed form without backpropagation, improving speed and stability for some applications.

    Two operating modes selected by the ``activation`` parameter:

    ── Linear mode (activation=None, default) ──────────────────────────────────

        φ(x) = Σ_k [ w1_k cos(Ω_k·x) + w2_k sin(Ω_k·x) ] / |Ω_k|²

        w1 : cosine weights  (F,)
        w2 : sine   weights  (F,)

    ── Activated mode (activation given) ───────────────────────────────────────

        φ(x) = Σ_k  w1_k · σ( sin(Ω_k·x + w2_k) + bias_k ) / |Ω_k|²

        w1   : amplitude  (F,)  — how much each mode contributes
        w2   : phase      (F,)  — where in space each mode's hinge sits
        bias : asymmetry  (F,)  — shifts σ input, controls fold vergence

        The activation σ is always evaluated on a sinusoid ∈ [−1, 1] shifted
        by a learnable bias.  This keeps σ' away from zero at initialisation
        regardless of the weights, and allows asymmetric fold shapes (vergent
        folds, wider anticlines than synclines, etc.) that the linear mode
        cannot represent without access to higher harmonics.

        Gradient and Hessian are analytic:
            f_k = sin(Ω_k·x + w2_k) + bias_k
            g_k = cos(Ω_k·x + w2_k)          [∂f_k/∂(Ω_k·x)]

            ∇φ_i   = Σ_k  w1_k/|Ω_k|²  ·  σ'(f_k) · g_k · Ω_{k,i}
            H_φ_{ij} = Σ_k  w1_k/|Ω_k|²  · [σ''(f_k)g_k² − σ'(f_k)f_k*]
                       · Ω_{k,i}Ω_{k,j}

            where f_k* = sin(Ω_k·x + w2_k)  [without bias, for Hessian term]

    Available activations
    ---------------------
    ``"tanh"``     smooth sigmoid-like, reaches ±tanh(1+|bias|) ≈ ±0.76–1.0
    ``"cubic"``    3u/2 − u³/2, polynomial, steeper than tanh at origin,
                   exact ±1 at u=±1 before bias shift
    ``"softplus"`` log(1+eᵘ), non-negative output, asymmetric by design

    Parameters
    ----------
    dim                : spatial dimension (2 or 3)
    n_features         : number of RFF modes
    length_scale_range : (L_min, L_max) for log-uniform frequency sampling
    freq_sampling      : ``"random"`` or ``"quasi"``
    activation         : None | ``"tanh"`` | ``"cubic"`` | ``"softplus"``
    """
    def initField(self,
                  rff_features:         int,
                  length_scale_range:   [float, float],
                  freq_sampling:        str = "random",
                  activation:           str = None,
                  scale:                float = 1.0,
                  learning_rate:        float = 1e-1
                  ):
        """
            Initialise and build this neural field.
            
            rff_features : int
                Number of Fourier features (modes) to use in the series.
            length_scale_range : (float, float)
                Tuple specifying the minimum and maximum length scales (wavenumbers) for the Fourier features.
            freq_sampling : str, either "random" or "quasi"
                Method for sampling the Fourier feature frequencies. "random" samples from a log-uniform distribution,
                while "quasi" uses a low-discrepancy Sobol sequence for more uniform coverage of the frequency space.
            activation : str or None
                Activation function to apply to the Fourier basis functions. Can be None for a standard Fourier series,
                or one of "tanh", "cubic", or "softplus" for non-linear variants that can capture more complex patterns.
            learning_rate : float
                The learning rate of the optimizer used to train this NF.
        """
        # ── Validate inputs ───────────────────────────────────────────────────
        assert freq_sampling in ("random", "quasi"), \
            f"freq_sampling must be 'random' or 'quasi', got {freq_sampling!r}"
        assert activation in (None, "tanh", "cubic", "softplus"), \
            f"activation must be None, 'tanh', 'cubic', or 'softplus', got {activation!r}"
        # Store activation for use in forward pass
        self.activation = activation
        self.scale = scale

        # ── Frequency sampling ────────────────────────────────────────────────
        torch.manual_seed(self.seed)
        lo = torch.log10(torch.tensor(length_scale_range[0], dtype=curlew.dtype, device=curlew.device))
        hi = torch.log10(torch.tensor(length_scale_range[1], dtype=curlew.dtype, device=curlew.device))

        if freq_sampling == "quasi":
            sobol   = torch.quasirandom.SobolEngine(dimension=self.input_dim + 1, scramble=False)
            u       = sobol.draw(rff_features).clamp(1e-6, 1-1e-6)
            lengths = torch.pow(torch.tensor(10.0, dtype=curlew.dtype, device=curlew.device), lo + u[:,0]*(hi-lo))
            k_raw   = torch.erfinv(2.0*u[:,1:]-1.0) * (2.0**0.5)
        else:
            lengths = torch.logspace(lo, hi, rff_features, dtype=curlew.dtype, device=curlew.device)
            k_raw   = torch.randn(rff_features, self.input_dim, device=curlew.device, dtype=curlew.dtype)
            
        Omega = 2.0 * torch.pi * k_raw / lengths.unsqueeze(1)     # (F, dim)
        self.register_buffer("fourier_projection", Omega.T)
        self.register_buffer("freq_scaling", (Omega**2).sum(1).clamp(min=1e-8))
        
        # ── Parameters ────────────────────────────────────────────────────────
        # w1, w2 have mode-dependent meaning depending on activation:
        #   activation=None : w1=cosine weights, w2=sine weights
        #   activation given: w1=amplitudes,     w2=phase offsets
        self.w1 = nn.Parameter(torch.zeros(rff_features))
        nn.init.xavier_normal_(self.w1.unsqueeze(0)) # (1, F) for correct fan_in scaling
        self.w2 = nn.Parameter(torch.zeros(rff_features))
        nn.init.xavier_normal_(self.w2.unsqueeze(0)) # (1, F) for correct fan_in scaling

        # Bias: only meaningful when activation is given.
        # Zero init → symmetric fold (upright).  Nonzero → vergent fold.
        # Not created in linear mode to keep the class lightweight.
        if activation is not None:
            self.bias = nn.Parameter(torch.zeros(rff_features))

        # ── Bind activations ───────────────────────────────
        if activation == "tanh":
            self._sigma    = torch.tanh
            self._sigma_p  = self._tanh_p
            self._sigma_pp = self._tanh_pp
        elif activation == "cubic":
            self._sigma    = self._cubic
            self._sigma_p  = self._cubic_p
            self._sigma_pp = self._cubic_pp
        elif activation == "softplus":
            self._sigma    = F.softplus
            self._sigma_p  = torch.sigmoid
            self._sigma_pp = self._softplus_pp
                
        # Push onto device
        self.to(curlew.device)

        # Initialise optimiser used for this field.
        self.init_optim(lr=learning_rate)
        
    # ── Activation functions and analytic derivatives ─────────────────────────
    # All static — no closure over instance state, doesn't meddle with autograd.
    @staticmethod
    def _tanh_p(u: torch.Tensor) -> torch.Tensor:
        t = torch.tanh(u); return 1.0 - t * t

    @staticmethod
    def _tanh_pp(u: torch.Tensor) -> torch.Tensor:
        t = torch.tanh(u); return -2.0 * t * (1.0 - t * t)

    @staticmethod
    def _cubic(u: torch.Tensor) -> torch.Tensor:
        return 1.5 * u - 0.5 * u * u * u

    @staticmethod
    def _cubic_p(u: torch.Tensor) -> torch.Tensor:
        return 1.5 * (1.0 - u * u)

    @staticmethod
    def _cubic_pp(u: torch.Tensor) -> torch.Tensor:
        return -3.0 * u

    @staticmethod
    def _softplus_pp(u: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(u); return s * (1.0 - s)
        
    # Overriden method to compute analytic gradients without backpropagation
    def gradient(self,
                 coords:        torch.Tensor,
                 transform:     bool = True, 
                 return_value:  bool = False,
                 normalize:     bool = True,
                 accumulate:    bool = True,
                 retain_graph:  bool = False, # Does nothing since gradients are analytic, but included for API consistency with BaseNF.gradient
                 create_graph:  bool = False, # Does nothing since gradients are analytic, but included for API consistency with BaseNF.gradient
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic gradient ∇φ at query points.

        Parameters
        ----------
        x : (N, dim)

        Returns
        -------
        grad : (N, dim)
        """
        # apply transform if needed
        if transform and self.transform is not None:
            coords = self.transform(coords, end=transform)
            
        proj = coords @ self.fourier_projection                   # (N, F)

        if self.activation is None:
            # ── Linear mode ───────────────────────────────────────────────────
            c, s  = torch.cos(proj), torch.sin(proj)              # (N, F)
            w1_n  = self.w1                                       # (F,)
            w2_n  = self.w2                                       # (F,)

            # ∇φ_i = Σ_k (−w1_k sin + w2_k cos)(Ω_k·x) Ω_{k,i} / |Ω_k|²
            pot  = self.scale * (self.w1 * c + self.w2 * s).sum(-1)
            grad_out = (-w1_n * s + w2_n * c) @ self.fourier_projection.T    # (N, dim)
            
            # Accumulate and/or normalize gradients?
            if accumulate or normalize:
                norm = torch.norm(grad_out, dim=-1, keepdim=True) + 1e-8
                if accumulate:
                    self.mnorm = (self.mnorm*self.nnorm) + torch.mean(norm, axis=0).item()*len(norm) # update average gradeint
                    self.nnorm += len(norm) # update counter holding number of observations
                    self.mnorm = self.mnorm / self.nnorm # convert from total to average
                if normalize:
                    grad_out = grad_out / norm

            if return_value:
                return grad_out, pot
            else:
                return grad_out

        # ── Activated mode ────────────────────────────────────────────────────
        # f_k = sin(Ω_k·x + w2_k) + bias_k     ← activation input
        # g_k = cos(Ω_k·x + w2_k)               ← ∂sin/∂(Ω_k·x)
        # s_k = sin(Ω_k·x + w2_k)               ← ∂cos/∂(Ω_k·x) · (−1)
        #
        # Chain rule for ∂σ(f_k)/∂x_i:
        #     = σ'(f_k) · g_k · Ω_{k,i}
        #

        s_raw = torch.sin(proj + self.w2)                         # (N, F) sin(Ω·x+w2)
        g     = torch.cos(proj + self.w2)                         # (N, F) cos(Ω·x+w2)
        f     = s_raw + self.bias                                 # (N, F) shifted input
        pot   = self.scale * (self.w1 * self._sigma(f)).sum(-1)

        w1_n  = self.w1                                            # (F,)
        spf   = self._sigma_p(f)                                   # (N, F)  σ'(f)

        # ∇φ_i = Σ_k w1_k/|Ω|² · σ'(f_k) · g_k · Ω_{k,i}
        grad_coeffs = w1_n * spf * g                              # (N, F)
        grad_out = grad_coeffs @ self.fourier_projection.T            # (N, dim)
        
        # Accumulate and/or normalize gradients?
        if accumulate or normalize:
            norm = torch.norm(grad_out, dim=-1, keepdim=True) + 1e-8
            if accumulate:
                self.mnorm = (self.mnorm*self.nnorm) + torch.mean(norm, axis=0).item()*len(norm) # update average gradeint
                self.nnorm += len(norm) # update counter holding number of observations
                self.mnorm = self.mnorm / self.nnorm # convert from total to average
            if normalize:
                grad_out = grad_out / norm

        if return_value:
            return grad_out, pot
        else:
            return grad_out

        return grad
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network to create a scalar value or property estimate.
        The input passes through the encoding first, then the Fourier series is evaluated
        using the learned coefficients and frequencies.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, input_dim), where N is the batch size.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, output_dim), representing the scalar potential.
        """
        proj = x @ self.fourier_projection                                    # (N, F)

        if self.activation is None:
            c, s = torch.cos(proj), torch.sin(proj)
            return self.scale * (self.w1 * c
                               + self.w2 * s).sum(-1)

        f = torch.sin(proj + self.w2) + self.bias
        return self.scale * (self.w1 * self._sigma(f)).sum(-1)