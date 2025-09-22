
"""
A series of utility functions that can come in handy when using `curlew`. This includes some minimalist(ish) classess
for data handling and visualisation.
"""
import numpy as np
from tqdm import tqdm

def get_colors(inp, colormap="viridis", normalize=True, vmin=None, vmax=None):
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except:
        assert False, "Please install `matplotlib` to use get_colors."

    colormap = mpl.colormaps[colormap]
    if normalize:
        vmin=np.min(inp)
        vmax=np.max(inp)

    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))[:, :3]

def batchEval( array, function, batch_size = 10000, vb=True, **kwargs):
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
    return np.concatenate(results)

def stackValues( pred, mn=0, mx=1):
    """
    Take an array of model predictions containing scalar values and structure IDs, scale them such 
    that the scalar fields vary between mn and mx for each structural field, and then add offsets so that 
    there are no overlaps between the structural fields. This can be useful for plotting.

    Parameters
    ----------
    pred : np.ndarray
        An array of shape (n, 2) where the first column contains scalar values and
        the second column contains structure IDs.
    mn : float, optional
        The minimum value to scale the scalar values to, by default 0. 
    mx : float, optional
        The maximum value to scale the scalar values to, by default 1.

    Returns
    -------
    np.ndarray
        A new array of the same shape as `pred`, where the scalar values are scaled to the range [mn, mx]
        for each unique structure ID, and offsets are added to ensure no overlaps between the structural fields.
    """

    # get the unique structure IDs
    ids = np.unique(pred[:,1])
    
    # create a new array to hold the stacked values
    stacked = np.zeros_like(pred)

    # loop over each structure ID
    for i, id in enumerate(ids):
        # get the indices of the current structure ID
        idx = np.where(pred[:,1] == id)[0]
        # scale the scalar values to the range [mn, mx]
        if np.max(pred[idx,0]) - np.min(pred[idx,0]) == 0:
            # if all values are the same, set them to mn
            scaled_values = np.full_like(pred[idx,0], mn)
        else:
            # scale the values to the range [mn, mx]        
            scaled_values = mn + (mx - mn) * (pred[idx,0] - np.min(pred[idx,0])) / (np.max(pred[idx,0]) - np.min(pred[idx,0]))
        
        # add an offset based on the index of the structure ID
        stacked[idx, 0] = scaled_values + i * (mx - mn)
        stacked[idx, 1] = id
    
    return stacked

from curlew.geology.SF import SF
from curlew.geology.interactions import faultOffset
from curlew.utils import batchEval
from curlew.io import savePLY
from tqdm import tqdm
