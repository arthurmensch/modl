import numpy as np


def get_sub_slice(indices, sub_indices):
    if indices is None:
        return sub_indices
    elif isinstance(indices, slice):
        return np.arange(indices.start + sub_indices.start,
                         indices.start + sub_indices.stop)
    else:  # ndarray
        return indices[sub_indices]