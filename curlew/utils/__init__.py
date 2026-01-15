
"""
A series of utility functions that can come in handy when using `curlew`. This includes some minimalist(ish) classess
for data handling and visualisation.
"""
import numpy as np
from tqdm import tqdm
import torch

def get_colors(inp, colormap="viridis", normalize=True, vmin=None, vmax=None):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except:
        assert False, "Please install `matplotlib` to use get_colors."

    colormap = mpl.colormaps[colormap]
    if normalize:
        if vmin is None:
            vmin=np.min(inp)
        if vmax is None:
            vmax=np.max(inp)

    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))[:, :3]

def batchEval( array, function, batch_size = 10000, vb=False, **kwargs):
    """
    Evaluate the specified function in batches to save memory. This can be used to evaluate models on large datasets using M.predict(...) or M.classify(...).

    Parameters
    ----------
    array : np.ndarray
        The data to evaluate.
    function : callable
        The function to evaluate on the data. This should be a method of the model, e.g. M.predict or M.classify.
    batch_size : int, optional
        The size of each batch. Default is 10000.
    vb : bool, optional
        True (default) if a progress bar should be created using tqdm.

    **kwargs : keyword arguments
        Additional keyword arguments to pass to the function.
    """
    # Calculate the number of batches
    num_batches = int(np.ceil(array.shape[0] / batch_size))
    if num_batches > 1:
        # Initialize an empty list to store the results
        results = []

        # Loop over each batch
        loop = range(num_batches)
        if vb: loop = tqdm(loop, desc="Evaluating")
        for i in loop:
            # Get the start and end indices for the current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, array.shape[0])

            # Get the current batch of data
            batch_data = array[start_idx:end_idx]

            # Evaluate the function on the current batch and append the result to the list
            results.append(function(batch_data, **kwargs))

        # Concatenate all results into a single array
        from curlew.geology.geofield import Geode # needs to be here to avoid circular import
        if isinstance(results[0], Geode):
            return Geode.concat( results ) # concatenate and return
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        elif isinstance(results[0], torch.Tensor):
            return torch.concatenate(results)

    else: # easy!
        return function(array, **kwargs)

