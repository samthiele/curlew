"""
Implementation of various types of "geological" forward models that convert scalar field values to 
measurable rock properties (density, magnetic susceptibility, mineralogy, chemistry, etc.).
"""
import curlew
from curlew import _tensor
from curlew.core import LearnableBase
import torch
import torch.nn as nn
from typing import Optional
import math
_LOG_2PI = math.log(2 * math.pi)

class PropertyModelBase(LearnableBase):
    """
    Base class for geological property models that convert scalar field values to measurable rock properties.
    """
    def predict(self, scalar_field_values: torch.Tensor) -> torch.Tensor:
        """
        Predict measurable rock properties from scalar field values.
        
        Parameters
        ----------
        scalar_field_values : torch.Tensor
            A tensor containing scalar field values.
        
        Returns
        -------
        torch.Tensor
            A tensor containing predicted rock property values.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class ConstantProperty(PropertyModelBase):
    """
    A simple property model that returns a constant property value regardless of the scalar field value.
    """

    def __init__(self, propDict: dict = {}, propertyNames: list = [], learnable: bool = True, noData: Optional[list] = None, learning_rate: float = 1e-2):
        """
        Initialise the constant forward model.
        
        Parameters
        ----------
        propDict : dict
            A dictionary with keys corresponding to lithology names, and values
            corresponding to property values. Property values can be lists of
            floats, but all lists must be of the same length.
        propertyNames : list
            A list of names for the properties. Length equal to the length of the
            property value lists.
        learnable : bool (Optional, defaults to True)
            If True, wraps the values in nn.Parameter and optimises them during training.
        """
        super().__init__()

        # Save the names
        self.propertyNames = propertyNames
        if noData is not None:
            self.noData = _tensor(noData, dev=curlew.device, dt=curlew.dtype)*torch.ones(len(propertyNames), device=curlew.device, dtype=curlew.dtype)
        else:
            self.noData = torch.nan * torch.ones(len(propertyNames), device=curlew.device, dtype=curlew.dtype)

        # Remake the propDict to store parameters if learnable
        if learnable:
            self.propDict = nn.ParameterDict()
            for k, v in propDict.items():
                self.propDict[k] = nn.Parameter(
                    _tensor(v, dt=curlew.dtype, dev=curlew.device)
                )
            self.init_optim(lr=learning_rate)
        else:
            self.propDict = dict()
            for k, v in propDict.items():
                self.propDict[k] = _tensor(v, dt=curlew.dtype, dev=curlew.device)
    
    def predict(self, geode: curlew.geology.geofield.Geode) -> torch.Tensor:
        """
        Using the Geode, predict the property values based on the lithoID. Return an updated
        Geode with property values and names.
        
        Parameters
        ----------
        geode : curlew.geology.geofield.Geode
            A Geode object containing scalar field values and lithoIDs.
        
        Returns
        -------
        torch.Tensor
            A tensor containing predicted rock property values, all set to the constant fill value.
        """
        # Get the litho-lookup table from the geode
        lithoID = geode.lithoID
        lithoLookup = geode.lithoLookup

        outputTensor = torch.ones((geode.scalar.shape[0], len(self.propertyNames)),
                                  dtype=curlew.dtype, device=curlew.device) # Dimensions of (N, P)
        # Loop through the unique lithoIDs and populate the output tensor
        for i in torch.unique(lithoID):
            currentProp = self.propDict.get(lithoLookup[i.item()], self.noData)
            currentMask = lithoID == i
            outputTensor[currentMask] *= currentProp.unsqueeze(0)

        geode.properties = outputTensor
        geode.propertyNames = self.propertyNames

        return geode


class MLPProperty(PropertyModelBase, nn.Module):
    """
    A learnable forward model that uses a single MLP to map scalar field
    values to measurable rock properties across all geological units.

    A single shared network is used deliberately: the curlew scalar field
    encodes stratigraphic position continuously across unit boundaries, so
    a global function ``s → properties`` naturally captures cross-unit
    trends (e.g. compaction gradients that span multiple formations, or
    metamorphic grade that increases monotonically with depth regardless of
    lithology).  Unit identity is not an input — the scalar value alone
    determines the property.

    The raw scalar is fed directly into the MLP as a 1-D input, allowing
    the network to act as a free function approximator over stratigraphic
    position.  Sharp property contrasts at unit boundaries emerge naturally
    as the network learns step-like transitions in the scalar domain.

    Parameters
    ----------
    output_dim : int
        Number of output properties (e.g. 1 for density alone,
        2 for density + susceptibility).
    propertyNames : list of str
        Human-readable names for each output property.  Length must
        equal ``output_dim``.
    hidden_dims : list of int
        Width of each hidden layer.  Depth = ``len(hidden_dims)``.
    activation : nn.Module
        Nonlinearity applied between hidden layers.  Activations with
        sharp gradients (e.g. ``nn.ReLU``, ``nn.Hardswish``) are
        preferred over smooth ones (e.g. ``nn.Tanh``) when the target
        property field contains abrupt contrasts.
    learning_rate : float
        Learning rate for the built-in Adam optimiser.

    Notes
    -----
    A ``Softplus`` activation is applied at the output to enforce physically
    non-negative property values (appropriate for density and susceptibility).
    Replace with ``nn.Identity`` for properties that may be negative.
    """

    def __init__(
        self,
        output_dim:    int       = 1,
        propertyNames: list      = [],
        hidden_dims:   list      = [32, 32, 32],
        activation:    nn.Module = nn.ReLU(),
        scale:         float     = 1.0,
        learning_rate: float     = 1e-3,
    ):
        super().__init__()

        self.output_dim    = output_dim
        self.propertyNames = propertyNames if propertyNames else [f"prop_{i}" for i in range(output_dim)]
        self.scale         = scale

        # ── Single shared MLP: scalar (1-D) → properties (output_dim) ─────
        layer_dims   = [1] + hidden_dims + [output_dim]
        self.network = self._build_mlp(layer_dims, activation).to(
            device=curlew.device, dtype=curlew.dtype
        )

        # ── Optimiser ─────────────────────────────────────────────────────
        self.init_optim(lr=learning_rate)

    # ──────────────────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, geode: "curlew.geology.geofield.Geode") -> "curlew.geology.geofield.Geode":
        """
        Predict property values from the scalar field stored in a Geode.

        All points are passed through the network in a single batched
        forward pass — no unit-level routing or masking.

        Parameters
        ----------
        geode : Geode
            Must carry ``geode.scalar`` of shape (N,).

        Returns
        -------
        Geode
            The same object with ``geode.properties`` (N, output_dim) and
            ``geode.propertyNames`` populated.
        """
        scalar = geode.scalar.to(device=curlew.device, dtype=curlew.dtype)
        geode.properties    = self.network(scalar.unsqueeze(-1)) * self.scale
        geode.propertyNames = self.propertyNames
        return geode

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_mlp(layer_dims, activation):
        layers = []
        for i in range(len(layer_dims) - 2):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        layers.append(nn.Softplus())
        return nn.Sequential(*layers)
    
class GMMPrior(nn.Module):
    """
    Gaussian Mixture Model prior on predicted physical properties.

    Implements the PGI regularisation of Astic et al. (2019):
    penalises predicted properties that are unlikely under a GMM
    fitted to expected rock-type clusters.

    Parameters
    ----------
    means : list of float
        Per-component means (one per rock type).
    stds : list of float
        Per-component standard deviations.
    weights : list of float, optional
        Mixing weights. Defaults to uniform.
    learnable : bool
        If True, means and stds are nn.Parameters and update
        during inversion. Useful for joint rock-physics + geometry
        inversion. If False, the GMM is a fixed prior.
    """
    def __init__(
        self,
        means:     list,
        stds:      list,
        weights:   list   = None,
        learnable: bool   = False,
    ):
        super().__init__()
        n = len(means)
        means_t   = torch.tensor(means,   dtype=curlew.dtype, device=curlew.device)
        stds_t    = torch.tensor(stds,    dtype=curlew.dtype, device=curlew.device)
        weights_t = torch.ones(n,         dtype=curlew.dtype, device=curlew.device) / n if weights is None \
                    else torch.tensor(weights, dtype=curlew.dtype, device=curlew.device)

        if learnable:
            self.means   = nn.Parameter(means_t)
            self.log_std = nn.Parameter(torch.log(stds_t))   # log for positivity
            self.logits  = nn.Parameter(torch.log(weights_t))
        else:
            self.register_buffer("means",   means_t)
            self.register_buffer("log_std", torch.log(stds_t))
            self.register_buffer("logits",  torch.log(weights_t))
            
    def hard_cluster_loss(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Assign each prediction to its nearest GMM component and
        penalise distance to that component's mean. Much harder
        than NLL — actively collapses predictions onto cluster means.
        """
        rho     = rho.squeeze(-1)                          # (N,)
        means   = self.means                               # (K,)
        
        # Distance from each point to each cluster mean
        dist    = (rho.unsqueeze(-1) - means.unsqueeze(0)).pow(2)  # (N, K)
        
        # Hard assignment to nearest cluster
        assign  = dist.argmin(dim=-1)                      # (N,)
        
        # Loss: distance to the assigned cluster mean
        nearest = means[assign]                            # (N,)
        return (rho - nearest).pow(2).mean()
            
    def repulsion_loss(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Penalise predictions that fall between cluster means.
        Computed as the negative of the GMM log-prob, exponentiated
        to sharpen the penalty in low-density regions.
        """
        log_p = self.log_prob(rho).clamp(min=-3.0)   # prevent exp overflow
        return torch.exp(-log_p).mean()

    def log_prob(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Compute log p_GMM(rho) for each point.

        Parameters
        ----------
        rho : Tensor (N, 1) or (N,)
            Predicted density values.

        Returns
        -------
        Tensor (N,) — per-point log probability under the GMM.
        """
        rho = rho.squeeze(-1)                      # (N,)
        stds    = torch.exp(self.log_std)          # positivity enforced
        weights = torch.softmax(self.logits, dim=0)  # sum to 1

        # (N, 1) - (1, K) → (N, K) per-component log Gaussians
        log_gauss = (
            -0.5 * ((rho.unsqueeze(-1) - self.means) / stds) ** 2
            - torch.log(stds)
            - 0.5 * _LOG_2PI
        )
        # log-sum-exp for numerical stability
        log_weighted = log_gauss + torch.log(weights)
        return torch.logsumexp(log_weighted, dim=-1)  # (N,)

    def loss(self, rho: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood — minimise this to pull rho toward GMM clusters."""
        return -self.log_prob(rho).mean()