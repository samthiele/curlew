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
from curlew.fields import BaseNF

class NFF(BaseNF):
    """
    NeuralFourierField
    
    A neural field that uses random Fourier feature encoding for representing scalar potentials.
    This class inherits from BaseNF and implements the necessary methods for training and evaluation.

    Attributes
    ----------
    use_rff : bool
        Indicates whether random Fourier feature encoding is applied.
    fourier_projection : torch.Tensor or None
        RFF projection matrix of shape (input_dim, rff_features * num_scales) if RFF is used, else None.
    length_scales : torch.Tensor or None
        Length scales for RFF if used, else None.
    mlp : nn.Sequential
        A sequence of linear layers (and optional activation) forming the MLP.
    """

    def initField(self, 
                  hidden_layers: list = [],
                  activation: nn.Module = None,
                  rff_features: int = 8,
                  length_scales: list = [1e2, 2e2, 3e2],
                  stochastic_scales : bool = True,
                  learning_rate: float = 1e-1):
        """
            Initialise and build this neural field.
            
            hidden_layers : list of int, optional
                A list of integer sizes for the hidden layers of the MLP. Default is [,], which indicates the input encoding is directly translated to the output (i.e. no hidden layers).
            activation : nn.Module, optional
                The activation function to use for each hidden layer. Default is None, though `nn.SiLU()` can be useful for some fields.
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

            # Single combined projection matrix for all length scales (vectorised RFF)
            combined_weights = []
            for i in range(len(length_scales)):
                w = 2*torch.pi*torch.randn(self.input_dim,
                                           rff_features, 
                                           device=curlew.device, 
                                           dtype=curlew.dtype)
                if not stochastic_scales:
                    w = w / torch.norm(w, dim=0, keepdim=True)
                combined_weights.append(w / length_scales[i])
            projection = torch.cat(combined_weights, dim=1)
            self.register_buffer("fourier_projection", projection)
            self.length_scales = length_scales # store, just in case we need it later
            
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
            if self.activation is not None:
                layers.append(self.activation)

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
        out = self.scale * self.mlp(x)

        return out

    def _encode_rff(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input coordinates using random Fourier features with the specified
        length scales. Single matrix multiply for all scales, then cos/sin applied once.

        Parameters
        ----------
        coords : torch.Tensor
            Tensor of shape (N, input_dim).

        Returns
        -------
        torch.Tensor
            Encoded tensor of shape (N, 2 * rff_features * num_scales).
        """
        proj = coords @ self.fourier_projection
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

# import other child classes for easy access
from curlew.fields.analytical import BaseAF, LinearField