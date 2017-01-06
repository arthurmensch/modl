import numpy as np


def get_sub_slice(indices, sub_indices):
    """
    Safe indexer with nested slices.

    Parameters
    ----------
    indices: ndarray or slice
    sub_indices: ndarray or slice

    Returns
    -------
    result: np.array(indices[sub_indices])
    """

    if indices is None:
        if isinstance(sub_indices, slice):
            return np.arange(sub_indices.start, sub_indices.stop)
        else:
            return sub_indices
    elif isinstance(indices, slice):
        return np.arange(indices.start + sub_indices.start,
                         indices.start + sub_indices.stop)
    else:  # ndarray
        return indices[sub_indices]