from typing import List

import numpy as np
from joblib import Parallel, delayed


def amari_discrepency(D1: np.ndarray, D2: np.ndarray) -> float:
    """

    Parameters
    ----------
    D1: np.ndarray, shape (n_components, n_features)

    D2: np.ndarray, shape (n_components, n_features)

    Returns
    -------
    discrepency: Amari discrepency
    """
    C = (D1.dot(D2.T) / np.sqrt(np.sum(D1 ** 2, axis=1))[:, None]
         / np.sqrt(np.sum(D2 ** 2, axis=1))[None, :])
    return .5 * (np.mean(1 - C.max(axis=0)) + np.mean(1 - C.max(axis=1)))


def mean_amari_discrepency(dictionaries: List[np.ndarray], n_jobs=1):
    discrepencies = Parallel(n_jobs=n_jobs)(
        delayed(amari_discrepency)(D1, D2)
        for i, D1 in enumerate(dictionaries[:-1])
        for D2 in dictionaries[i + 1:])
    return np.mean(np.array(discrepencies)), np.std(np.array(discrepencies))



