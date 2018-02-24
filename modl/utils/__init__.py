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


def concatenated_cv(cvs):
    for this_train, this_test in cvs[0]:
        train = [this_train]
        test = [this_test]
        for cv in cvs[1:]:
            this_train, this_test = next(cv)
            train.append(this_train)
            test.append(this_test)
        yield np.concatenate(train), np.concatenate(test)
