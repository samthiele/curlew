"""
Import core neural field types from other python files, and define the "base" NF class that these all inherit from.
"""

import curlew
from curlew.core import HSet
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy


class NF(nn.Module):
    """
    A generic neural field that maps input coordinates to some value or values. 

    Parameters
    ----------
    H : HSet
        Hyperparameters used to tune the loss function for this NF.
    name : str
        A name for this neural field. Default is "f0" (i.e., bedding).
    input_dim : int, optional
        The dimensionality of the input space (e.g., 3 for (x, y, z)).
    output_dim : int, optional
        Dimensionality of the scalar output (usually 1 for a scalar potential).
    hidden_layers : list of int, optional
        A list of integer sizes for the hidden layers of the MLP.
    activation : nn.Module, optional
        The activation function to use for each hidden layer.
    transform : callable
        A function that transforms input coordinates prior to predictions. Must take exactly one argument as input (a tensor of positions) and return the transformed positions. 
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
    seed : int, optional
        Random seed for reproducible RFF initialization.

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

    def __init__(
            self,
            H: HSet,
            name: str = 'f0',
            input_dim: int = 3,
            output_dim: int = 1,
            hidden_layers: list = [512,],
            activation: nn.Module = nn.SiLU(),
            loss = nn.MSELoss(),
            transform = None,
            rff_features: int = 8,
            length_scales: list = [1e2, 2e2, 3e2],
            stochastic_scales : bool = True,
            learning_rate: float = 1e-1,
            seed: int = 404,
        ):
            super().__init__()
            self.C = None # will contain constraints once bound
            self.name = name
            self.H = H.copy() # make a copy
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.mnorm = None # will be set to the average gradient magnitude
            self.use_rff = rff_features > 0
            self.activation = activation
            self.loss_func = loss
            self.transform = transform
            self.closs = torch.nn.CosineSimilarity() # needed by some loss functions

            # -------------------- Random Fourier Features -------------------- #
            self.weight_matrix = None
            self.bias_vector = None
            self.length_scales = None

            if self.use_rff:
                # Seed for reproducibility
                torch.manual_seed(seed)

                # Weight and bias for RFF
                self.weight_matrix = torch.randn(input_dim, rff_features, device=curlew.device, dtype=curlew.dtype )
                self.bias_vector = 2 * torch.pi * torch.rand(rff_features, device=curlew.device, dtype=curlew.dtype )
                if not stochastic_scales:
                    self.weight_matrix /= torch.norm(self.weight_matrix, dim=0)[None,:] # normalise so projection vectors are unit length

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
                mlp_input_dim = input_dim

            # Define layer shapes
            self.dims = [mlp_input_dim] + hidden_layers + [output_dim]

            # Build layers in nn.Sequential
            layers = []
            for i in range(len(self.dims) - 2):
                layers.append(nn.Linear(self.dims[i], self.dims[i + 1],
                                        device=curlew.device, dtype=curlew.dtype))
                # layers.append(nn.BatchNorm1d(self.dims[i + 1]))  # Uncomment if BatchNorm is needed
                layers.append(activation)

            # Final layer
            layers.append(nn.Linear(self.dims[-2], self.dims[-1],
                                    device=curlew.device, dtype=curlew.dtype))
            self.mlp = nn.Sequential(*layers) # Combine layers into nn.Sequential

            # Xavier initialization
            for layer in self.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)

            # Initialise optimiser used for this MLP.
            self.init_optim(lr=learning_rate)

            # push onto device
            self.to(curlew.device)

    def fit(self, epochs, C=None, learning_rate=None, early_stop=(100,1e-4), transform=True, best=True, vb=True, prefix='Training'):
        """
        Train this neural field to fit the specified constraints.

        Parameters
        ----------
        epochs : int
            The number of epochs to train for.
        C : CSet, optional
            The set of constraints to fit this field to. If None, the previously
            bound constraint set will be used.
        learning_rate : float, optional
            Reset this NF's optimiser to the specified learning rate before training.
        early_stop : tuple,
            Tuple containing early stopping criterion. This should be (n,t) such that optimisation
            stops after n iterations with <= t improvement in the loss. Set to None to disable. Note 
            that early stopping is only applied if `best = True`. 
        transform : bool, optional
            True (default) if constraints (C) is in modern coordinates that need to be transformed during fitting. If False, 
            C is considered to have already been transformed to paleo-coordinates. Note that this can be problematic if rotations
            occur (e.g. of gradient constraints!).
        best : bool, optional
            After training set the neural field weights to the best loss.
        vb : bool, optional
            Display a tqdm progress bar to monitor training.
        prefix : str, optional
            The prefix used for the tqdm progress bar.
        
        Returns
        -------
        loss : float
            The loss of the final (best if best=True) model state.
        details : dict
            A more detailed breakdown of the final loss. 
        """
        # set learning rate if needed
        if learning_rate is not None:
            self.set_rate(learning_rate)

        # bind the constraints
        if C is not None:
            self.bind(C)

        # store best state
        best_loss = np.inf
        best_loss_ = None
        best_state = None

        # for early stopping
        best_count = 0
        eps = 0
        if early_stop is not None:
            eps = early_stop[1]

        # setup progress bar
        bar = range(epochs)
        if vb:
            bar = tqdm(range(epochs), desc=prefix, bar_format="{desc}: {n_fmt}/{total_fmt}|{postfix}")
        for epoch in bar:
            loss, details = self.loss(transform=transform) # compute loss

            if (loss.item() < (best_loss + eps)): # update best state
                best_loss = loss.item()
                best_loss_ = details
                best_state = copy.deepcopy( self.state_dict() )
                best_count = 0
            else: # not necessarily the best; but keep for return
                if best_state == None:
                    best_loss = loss.item()
                    best_loss_ = details
                best_count += 1

            # early stopping?
            if (early_stop is not None) and (best_count > early_stop[0]):
                break

            if vb: # update progress bar
                bar.set_postfix({ k : v[0] for k,v in details[self.name][1].items() })

            # backward pass and update
            self.zero()
            loss.backward(retain_graph=False)
            self.step()

        if best:
            self.load_state_dict(best_state)

        return best_loss, best_loss_ # return summed and detailed loss

    def predict(self, X, to_numpy=True, transform=True ):
        """
        Create model predictions at the specified points.

        Parameters
        ----------
        X : np.ndarray
            An array of shape (N, input_dim) containing the coordinates at which to evaluate
            this neural field.
        to_numpy : bool
            True if the results should be cast to a numpy array rather than a `torch.Tensor`.
        transform : bool
            If True, any defined transform function is applied before encoding and evaluating the field for `x`.

        Returns
        --------
        S : An array of shape (N,1) containig the predicted scalar values
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor( X, device=curlew.device, dtype=curlew.dtype)
        S = self(X, transform=transform)
        if to_numpy:
            return S.cpu().detach().numpy()
        return S

    def forward(self, x: torch.Tensor, transform=True) -> torch.Tensor:
        """
        Forward pass of the network to create a scalar value or property estimate.

        If random Fourier features are enabled, the input is first encoded accordingly.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N, input_dim), where N is the batch size.
        transform : bool
            If True, any defined transform function is applied before encoding and evaluating the field for `x`.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, output_dim), representing the scalar potential.
        """

        # apply transform if needed
        if transform and self.transform is not None:
            x = self.transform(x)

        # encode position as Fourier features if needed
        if self.use_rff:
            x = self._encode_rff(x)

        # Pass through all layers and return
        out = self.mlp( x )

        return out

    def bind( self, C ):
        """
        Bind a CSet to this NF ready for loss computation and training.
        """
        self.C = C.torch() # make a copy

        # setup deltas for numerical if not yet defined
        C=self.C # shortand for our copy
        if C.grid is not None:
            if C.delta is None:
                # initialise differentiation step if needed
                C.delta = np.linalg.norm( C.grid.coords()[0,:] - C.grid.coords()[1,:] ) # / 2

            if C._offset is None:
                C._offset = []
                for i in range(self.input_dim):
                    o = [0]*self.input_dim
                    o[i] = C.delta
                    C._offset.append( torch.tensor( o, device=curlew.device, dtype=curlew.dtype) )

    def set_rate(self, lr=1e-2 ):
        """
        Update the learning rate for this NF's optimiser.
        """
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def init_optim(self, lr=1e-2):
        """
        Initialise optimiser used for this MLP. This should only be called
        (or re-called) once all parameters have been created.

        Parameters
        ------------
        lr : float
            The learning rate to use for the underlying ADAM optimiser.
        mlp : bool
            True (default) if the mlp layers will be included in the optimiser. If False,
            only other learnable parameters (e.g., fault slip) will be included.
        """
        self.optim = optim.Adam(self.mlp.parameters(), lr=lr)
        #self.optim = optim.SGD( self.parameters(), lr=lr, momentum=0 )

    def zero(self):
        """
        Zero gradients in the optimiser for this NF.
        """
        self.optim.zero_grad()

    def step(self):
        """
        Step the optimiser for this NF.
        """
        self.optim.step()

    def loss(self, transform=True) -> torch.Tensor:
        """
        Compute the loss associated with this neural field given its current state. The `transform` argument
        specifies if constraints need to be transformed from modern to paleo-coordinates before computing loss.
        """
        if self.C is None:
            assert False, "Scalar field has no constraints"

        C = self.C # shorter
        H = self.H
        total_loss = 0
        value_loss = 0
        grad_loss = 0
        ori_loss = 0
        thick_loss = 0
        mono_loss = 0
        flat_loss = 0
        iq_loss = 0

        # LOCAL LOSS FUNCTIONS
        # -----------------------------
        # [ N.B. positions are all in un-transformed coordinates! :-) ]
        # Value Loss
        if (C.vp is not None) and (C.vv is not None) and (isinstance(H.value_loss, str) or (H.value_loss > 0)):
            v_pred = self(C.vp, transform=transform)
            value_loss = self.loss_func( v_pred, C.vv[:,None] )

        # Gradient loss
        # [ N.B. positions (and thus gradients) are in un-transformed coordinates ]
        if (C.gp is not None) and (isinstance(H.grad_loss, str) or (H.grad_loss > 0)):
            gv_pred = self.compute_gradient(C.gp, normalize=True, transform=transform) # compute gradient direction 
            grad_loss = self.loss_func(gv_pred, C.gv) # orientation + younging direction

        # Orientation loss
        # [ N.B. positions (and thus gradients) are in un-transformed coordinates ]
        if (C.gop is not None) and (isinstance(H.ori_loss, str) or (H.ori_loss > 0)):
            gv_pred = self.compute_gradient(C.gop, normalize=True, transform=transform) # compute gradient direction 
            ori_loss = torch.clamp( torch.mean( 1 - torch.abs( self.closs(gv_pred, C.gov ) ) ), min=1e-6 ) # N.B.: Orientation loss on its own fits a bit too well, numerical precision crashes avoided with the clamp - AVK

        # GLOBAL LOSS FUNCTIONS
        # -------------------------------
        if C.grid is not None:
            if transform:
                gridL = C.grid.draw(self.transform) # specify transform
            else:
                gridL = C.grid.draw() # no transform
            
            if  isinstance(H.thick_loss, str) or isinstance(H.mono_loss, str) or isinstance(H.flat_loss, str) or \
                (H.thick_loss > 0) or (H.mono_loss > 0) or (H.flat_loss > 0):

                # numerically compute the hessian of our scalar field from the gradient vectors
                # to compute the divergence of the normalised field and so penalise bubbles (local maxima and minima)
                #hess = torch.zeros((gridL.shape[0], self.input_dim, self.input_dim), device=curlew.device, dtype=curlew.dtype)
                ndiv = torch.zeros((gridL.shape[0]), device=curlew.device, dtype=curlew.dtype)
                mnorm = 0
                for j in range(self.input_dim):
                    # compute hessian
                    grad_pos = self.compute_gradient(gridL + C._offset[j], normalize=False, transform=False)
                    grad_neg = self.compute_gradient(gridL - C._offset[j], normalize=False, transform=False)
                    #for i in range(self.input_dim):
                    #    hess[:, i, j] = (grad_pos[:, i] - grad_neg[:, i])/(2*C.delta)

                    # compute and accumulate average gradient
                    pnorm = torch.norm( grad_pos, dim=-1 )[:,None]
                    nnorm = torch.norm( grad_neg, dim=-1 )[:,None]
                    mnorm = mnorm + torch.mean(pnorm).item() + torch.mean(nnorm).item()

                    # compute divergence of normalised gradient field
                    if isinstance(H.mono_loss, str) or (H.mono_loss > 0):
                        grad_pos = grad_pos / pnorm
                        grad_neg = grad_neg / nnorm
                        ndiv = ndiv + (grad_pos[:,j] - grad_neg[:,j])/(2*C.delta)

                    # compute the percentage deviation in the gradient (at all the points where we evaluated it)
                    if isinstance(H.thick_loss, str) or (H.thick_loss > 0):
                        thick_loss = thick_loss + torch.mean( (1 - (pnorm) / torch.clip( torch.mean( pnorm ), 1e-8, torch.inf ) )**2 )
                        thick_loss = thick_loss + torch.mean( (1 - (nnorm) / torch.clip( torch.mean( nnorm ), 1e-8, torch.inf ) )**2 )

                # compute derived thickness and monotonocity loss
                if isinstance(H.mono_loss, str) or (H.mono_loss > 0):
                    mono_loss = torch.mean(ndiv**2) # (normalised) divergence should be close to 0
                if isinstance(H.thick_loss, str) or (H.thick_loss > 0):
                    # thick_loss = torch.mean( torch.linalg.det(hess)**2 ) # determinant should be close to 0 [ breaks in 2D, as the trace and determinant can't both be 0 unless all is 0!]
                    thick_loss = thick_loss / (2*self.input_dim)

                # Flatness Loss --  gradients parallel to trend
                if (isinstance(H.flat_loss, str) or (H.flat_loss > 0)) and (C.trend is not None):
                    if transform:
                        gv_at_grid_p = self.compute_gradient(gridL, normalize=True, transform=self.transform) # this requires gradients relative to modern coordinates! 
                    else:
                        gv_at_grid_p = self.compute_gradient(gridL, normalize=True, transform=False)
                    flat_loss = torch.mean((gv_at_grid_p - C.trend[None,:])**2) # "younging" direction
                    #flat_loss = (1 - self.closs( gv_at_grid_p, C.trend )).mean() # orientation only

                # store the mean gradient, as it can be handy if we want to scale our field to have an (average) gradient of 1
                # (Note that we need to get rid of the gradient here to prevent
                #  messy recursion during back-prop)
                self.mnorm = mnorm / (2*self.input_dim)

        # inequality losses
        if (C.iq is not None) and (isinstance(H.iq_loss, str) or (H.iq_loss > 0)):
            ns = C.iq[0] # number of samples
            for start,end,iq in C.iq[1]:
                # sample N random pairs to evaluate inequality
                six = torch.randint(0, start.shape[0], (ns,), dtype=int, device=curlew.device)
                eix = torch.randint(0, end.shape[0], (ns,), dtype=int, device=curlew.device)

                # evaluate value at these points
                start = self( start[ six, : ], transform=transform )
                end = self( end[ eix, : ], transform=transform )
                delta = start - end

                # compute loss according to the specific inequality
                if '=' in iq:
                    iq_loss = iq_loss + torch.mean(delta**2) # basically MSE
                elif '<' in iq:
                    iq_loss = iq_loss + torch.mean(torch.clamp(delta,0,torch.inf)**2)
                elif '>' in iq:
                    iq_loss = iq_loss + torch.mean(torch.clamp(delta,-torch.inf, 0)**2)

        if H.use_dynamic_loss_weighting:
            # Dynamically adjust task weights based on the inverse of real-time loss values.
            if value_loss > 0:
                H.value_loss = 1 / value_loss.item()
            if grad_loss > 0:
                H.grad_loss = 1 / grad_loss.item()
            if ori_loss > 0:
                H.ori_loss = 1 / ori_loss.item()
            if thick_loss > 0:
                H.thick_loss = 1 / thick_loss.item()
            if mono_loss > 0:
                H.mono_loss = 1 / mono_loss.item()
            if flat_loss > 0:
                H.flat_loss = 1 / flat_loss.item()
            if iq_loss > 0:
                H.iq_loss = 1 / iq_loss.item()
        else:
            # Initialise the weights if they are not defined
            if isinstance(H.value_loss, str) & (value_loss > 0):
                H.value_loss = float(H.value_loss) * (1/value_loss).item()
            if isinstance(H.grad_loss, str) & (grad_loss > 0):
                H.grad_loss = float(H.grad_loss) * (1/grad_loss).item()
            if isinstance(H.ori_loss, str) & (ori_loss > 0):
                H.ori_loss = float(H.ori_loss) * (1/ori_loss).item()
            if isinstance(H.thick_loss, str) & (thick_loss > 0):
                H.thick_loss = float(H.thick_loss) * (1/thick_loss).item()
            if isinstance(H.mono_loss, str) & (mono_loss > 0):
                H.mono_loss = float(H.mono_loss) * (1/mono_loss).item()
            if isinstance(H.flat_loss, str) & (flat_loss > 0):
                H.flat_loss = float(H.flat_loss) * (1/flat_loss).item()
            if isinstance(H.iq_loss, str) & (iq_loss > 0):
                H.iq_loss = float(H.iq_loss) * (1/iq_loss).item()

        # aggregate losses (and store individual parts for debugging)
        out = { self.name : [0,{}] }
        for alpha, loss, name in zip( [H.value_loss, H.grad_loss, H.ori_loss, H.thick_loss, H.mono_loss, H.flat_loss, H.iq_loss ],
                                      [value_loss,   grad_loss,   ori_loss,   thick_loss, mono_loss,   flat_loss, iq_loss ],
                                     ['value_loss', 'grad_loss', 'ori_loss', 'thick_loss', 'mono_loss', 'flat_loss', 'iq_loss'] ):
            if (alpha is not None) and (loss > 0):
                total_loss = total_loss + alpha*loss
                if isinstance(loss, torch.Tensor):
                    out[self.name][1][name] = (alpha*loss.item(), loss.item())
        out[self.name][0] = total_loss.item()

        # done! 
        return total_loss, out

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
            proj = coords @ (self.weight_matrix / self.length_scales[i]) + self.bias_vector
            cos_part = torch.cos(proj)
            sin_part = torch.sin(proj)
            outputs.append(torch.cat([cos_part, sin_part], dim=-1))
        return torch.cat(outputs, dim=-1)

    def compute_gradient(self, coords: torch.Tensor, normalize: bool = True, transform=True, return_value=False) -> torch.Tensor:
        """
        Compute the gradient of the scalar potential with respect to the input coordinates.

        Parameters
        ----------
        coords : torch.Tensor
            A tensor of shape (N, input_dim) representing the input coordinates.
        normalize : bool, optional
            If True, the gradient is normalized to unit length per sample.
        transform : bool
            If True, any defined transform function is applied before encoding and evaluating the field for `coords`.
        return_value : bool, optional
            If True, both the gradient and the scalar value at the evaluated points are returned.

        Returns
        -------
        torch.Tensor
            A tensor of shape (N, input_dim) representing the gradient of the scalar potential at each coordinate.
        torch.Tensor, optional
            A tensor of shape (N, 1) giving the scalar value at the evaluated points, if `return_value` is True.
        """
        coords.requires_grad_(True)

        # Forward pass to get the scalar potential
        potential = self.forward(coords, transform=transform).sum(dim=-1)  # sum in case output_dim > 1
        grad_out = torch.autograd.grad(
            outputs=potential,
            inputs=coords,
            grad_outputs=torch.ones_like(potential),
            create_graph=True,
            retain_graph=True
        )[0]

        if normalize:
            norm = torch.norm(grad_out, dim=-1, keepdim=True) + 1e-8
            grad_out = grad_out / norm

        if return_value:
            return grad_out, potential
        else:
            return grad_out


# import other child classes for easy access
from curlew.fields.analytical import AF, ALF
