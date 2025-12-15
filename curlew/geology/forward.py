"""
Implementation of various types of "geological" forward models that convert scalar field values to 
measurable rock properties (density, magnetic susceptibility, mineralogy, chemistry, etc.).
"""
import curlew
from curlew.core import LearnableBase
import torch
import torch.nn as nn

class ForwardModelBase(LearnableBase):
    """
    Base class for geological forward models that convert scalar field values to measurable rock properties.
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

class ConstantForward(ForwardModelBase):
    """
    A simple forward model that returns a constant property value regardless of the scalar field value.
    """

    def __init__(self, fill_value: float):
        """
        Initialise the constant forward model.
        
        Parameters
        ----------
        fill_value : float
            The constant property value to return.
        """
        self.fill_value = fill_value
    
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
            A tensor containing predicted rock property values, all set to the constant fill value.
        """
        return torch.full_like(scalar_field_values, fill_value=self.fill_value, dtype=curlew.dtype, device=curlew.device)

class MLPForward(ForwardModelBase, nn.Module):
    """
    A learnable forward model that uses a multi-layer perceptron (MLP) to map scalar field values
    to measurable rock properties.
    """

    def __init__(self,
                 output_dim: int = 1,
                 hidden_dims: list = [32, 32, 32],
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 use_rff: bool = False,
                 rff_features: int = 64,
                 rff_length_scales: list = [1.0],
                 learning_rate: float = 1e-3):
        """
        Initialise the MLP forward model.

        Parameters
        ----------
        output_dim : int
            Dimensionality of the output property values.
        hidden_dims : list
            List of integers specifying the number of units in each hidden layer.
        activation : torch.nn.Module
            Activation function to use between layers.
        use_rff : bool
            Whether to use random Fourier features for input encoding.
        rff_features : int
            Number of random Fourier features per length scale.
        rff_length_scales : list
            List of length scales for random Fourier feature encoding.
        learning_rate : float
            Learning rate for the optimiser.
        """
        super().__init__()

        # Define network architecture
        self.dims = [1] + hidden_dims + [output_dim]
        layers = []

        # TODO 

        # Hidden layers
        #for i in range(len(self.dims) - 2):
        #    layers.append(nn.Linear(self.dims[i], self.dims[i+1],