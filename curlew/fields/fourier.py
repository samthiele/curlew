"""
Implement fourier-feature based neural fields for scalar potential representation.
"""

import curlew
from curlew.core import HSet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy

from curlew.fields import BaseNF

class NFF(BaseNF):
    """
    A generic neural field that maps input coordinates to some value or values. See the other
    child classes in this module (e.g., fourier, geoinr, etc.) for specific implementations.

    Parameters
    ----------
    H : HSet
        Hyperparameters used to tune the loss function for this NF.
    input_dim : int, optional
        The dimensionality of the input space (e.g., 3 for (x, y, z)).
    output_dim : int, optional
        Dimensionality of the scalar output (usually 1 for a scalar potential).
    transform : callable
        A function that transforms input coordinates prior to predictions. Must take exactly one argument as input (a tensor of positions) and return the transformed positions. 

    Attributes
    ----------
    use_rff : bool
        Indicates whether random Fourier feature encoding is applied.
    weight_matrix : torch.Tensor or None
        RFF weight matrix of shape (input_dim, rff_features) if RFF is used, else None.
    bias_vector : torch.Tensor or None
        RFF bias vector of shape (rff_features,) if RFF is used, else None.
    length_scales : torch.Tensor or None
        Length scales for RFF if used, else None.
    mlp : nn.ModuleList
        A sequence of linear layers (and optional activation) forming the MLP.
    """

    def initField(self, hidden_layers: list = [512,],
             activation: nn.Module = nn.SiLU(),
             rff_features: int = 8,
             length_scales: list = [1e2, 2e2, 3e2],
             stochastic_scales : bool = True,
             learning_rate: float = 1e-1,):
        """
            Initialise and build this neural field.
            
            hidden_layers : list of int, optional
                A list of integer sizes for the hidden layers of the MLP.
            activation : nn.Module, optional
                The activation function to use for each hidden layer.
            rff_features : int, optional
                Number of Fourier features for each input dimension (when RFF is used). 
                Set as 0 to disable RFF.
            length_scales : list of float, optional
                A list of length scales (wavenumbers) for scaling the random Fourier features.
            stochastic_scales : bool, default=True
                Whether to normalize Fourier feature direction vectors to exactly preserve 
                `length_scales`. If True (default) no normalisation is performed, such that
                the fourier feature length scales follow a Chi distribution.
            learning_rate : float
                The learning rate of the optimizer used to train this NF.
        """
        # -------------------- Random Fourier Features -------------------- #
        self.use_rff = rff_features > 0
        self.activation = activation
        self.weight_matrix = None
        self.bias_vector = None
        self.length_scales = None

        # wrap in list if only one length scale provided
        if isinstance(length_scales, float) or isinstance(length_scales, int):
            length_scales = [length_scales]
        
        if self.use_rff:
            # Seed for reproducibility
            torch.manual_seed(self.seed)

            # Weight and bias for RFF
            self.weight_matrix = [torch.randn(self.input_dim, rff_features, device=curlew.device, dtype=curlew.dtype ) for i in range(len(length_scales))]
            self.bias_vector = 2 * torch.pi * torch.rand(rff_features, device=curlew.device, dtype=curlew.dtype )
            if not stochastic_scales:
                for i in range(len(length_scales)): # make direction vectors (weights) have a length of 1
                    self.weight_matrix[i] /= torch.norm(self.weight_matrix[i], dim=0)[None,:] # normalise so projection vectors are unit length

            # store length scales as a tensor (these will be multiplied by our weight matrix later)
            self.length_scales = torch.tensor(length_scales, device=curlew.device, dtype=curlew.dtype)
            
        # -------------------- MLP Construction -------------------- #
        # Determine input dimension for the MLP
        if self.use_rff:
            # Each length_scale effectively creates a separate RFF transform
            # For each transform, we get [cos(...), sin(...)] => 2*rff_features
            mlp_input_dim = 2 * rff_features * len(length_scales)
        else:
            # If not using RFF, the input to the MLP is just (input_dim)
            mlp_input_dim = self.input_dim

        # Define layer shapes
        self.dims = [mlp_input_dim] + hidden_layers + [self.output_dim]

        # Build layers in nn.Sequential
        layers = []
        for i in range(len(self.dims) - 2):
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1],
                                    device=curlew.device, dtype=curlew.dtype))
            layers.append(activation)

        # Final layer
        layers.append(nn.Linear(self.dims[-2], self.dims[-1],
                                device=curlew.device, dtype=curlew.dtype))
        self.mlp = nn.Sequential(*layers) # Combine layers into nn.Sequential

        # Xavier initialization
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        # push onto device
        self.to(curlew.device)

        # Initialise optimiser used for this MLP.
        self.init_optim(lr=learning_rate)
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network to create a scalar value or property estimate.

        If random Fourier features are enabled, the input is first encoded accordingly.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, input_dim), where N is the batch size.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, output_dim), representing the scalar potential.
        """
        # encode position as Fourier features if needed
        if self.use_rff:
            x = self._encode_rff(x)

        # Pass through all layers and return
        out = self.scale * self.mlp( x )

        return out

    def _encode_rff(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input coordinates using random Fourier features with the specified
        length scales. For each length scale, we compute cos(W/scale * x + b) and
        sin(W/scale * x + b) and concatenate them.

        Parameters
        ----------
        coords : torch.Tensor
            Tensor of shape (N, input_dim).

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (N, 2 * rff_features * num_scales).
        """
        outputs = []
        for i in range(len(self.length_scales)):
            proj = coords @ (self.weight_matrix[i] / self.length_scales[i]) + self.bias_vector
            cos_part = torch.cos(proj)
            sin_part = torch.sin(proj)
            outputs.append(torch.cat([cos_part, sin_part], dim=-1))
        return torch.cat(outputs, dim=-1)

# import other child classes for easy access
from curlew.fields.analytical import BaseAF, LinearField