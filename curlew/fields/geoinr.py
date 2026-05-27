"""
TODO - implement neural field based on the GeoINR approach.
"""

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

class GeoINR(BaseNF):
    """
    GeoINR-inspired neural field for interpolation of geological structures. 
    
    See Hillier et al., 2023 for further details: 
    
    `Hillier, Michael, et al. "GeoINR 1.0: an implicit neural network approach to three-dimensional geological modelling." Geoscientific Model Development 16.23 (2023): 6987-7012.`
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
            learning_rate : float
                The learning rate of the optimizer used to train this NF.
        """
        # -------------------- Random Fourier Features -------------------- #
        self.activation = activation
            
        # -------------------- MLP Construction -------------------- #
        # Determine input dimension for the MLP
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
        # Pass through all layers and return
        out = self.scale * self.mlp(x)
        return out
    
    def loss(self, transform=True) -> torch.Tensor:
        """
        Compute the loss associated with this neural field given its current state.
        """
        C = self.C # curlew-style constraints
        return super().loss(transform) # todo some funky loss 
    
    def fit(self, epochs, C=None, **kwargs):
        """
        Train this neural field using the specified constraints.
        """
        return super().fit(epochs, C=C, **kwargs)
    