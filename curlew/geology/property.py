"""
Implementation of various types of "geological" forward models that convert scalar field values to 
measurable rock properties (density, magnetic susceptibility, mineralogy, chemistry, etc.).
"""
import curlew
from curlew.core import LearnableBase
import torch
import torch.nn as nn
from typing import Optional

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
            self.noData = torch.tensor(noData, device=curlew.device, dtype=curlew.dtype)*torch.ones(len(propertyNames), device=curlew.device, dtype=curlew.dtype)
        else:
            self.noData = torch.nan * torch.ones(len(propertyNames), device=curlew.device, dtype=curlew.dtype)

        # Remake the propDict to store parameters if learnable
        if learnable:
            self.propDict = nn.ParameterDict()
            for k, v in propDict.items():
                self.propDict[k] = nn.Parameter(
                    torch.tensor(v, dtype=curlew.dtype, device=curlew.device)
                )
            self.init_optim(lr=learning_rate)
        else:
            self.propDict = dict()
            for k, v in propDict.items():
                self.propDict[k] = torch.tensor(v, dtype=curlew.dtype, device=curlew.device)
    
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

        outputTensor = torch.ones((geode.scalar.shape[0], len(self.propertyNames)), dtype=curlew.dtype, device=curlew.device) # Dimensions of (N, P)
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