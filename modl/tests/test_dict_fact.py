# Author: Arthur Mensch

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils import check_random_state

from modl.dict_fact import DictMF

rng_global = 0

backends = ['c', 'python']

var_reds = ['weight_based', 'sample_based', 'combo']


def generate_sparse_synthetic(n_samples=200,
                              square_size=4):
    rng = check_random_state(0)
    n_features = square_size ** 2
    half_size = square_size // 2
    Q = np.zeros((4, n_features))
    for i in range(2):
        for j in range(2):
            atom = np.zeros((square_size, square_size))
            atom[(half_size * i):(half_size * (i + 1)),
            (half_size * j): (half_size * (j + 1))] = 1
            Q[2 * i + j] = np.ravel(atom)
    code = rng.randn(n_samples, 4)
    X = code.dot(Q)
    return X, Q


def generate_synthetic(n_samples=200,
                       n_components=4, n_features=16,
                       dictionary_rank=None):
    rng = check_random_state(0)
    if dictionary_rank is None:
        Q = rng.randn(n_components, n_features)
    else:
        V = rng.randn(dictionary_rank, n_features)
        U = rng.randn(n_components, dictionary_rank)
        Q = U.dot(V)
    code = rng.randn(n_samples, n_components)
    X = code.dot(Q)
    return X, Q


# def test_compute_code():
#     X, Q = generate_synthetic()
#     P = compute_code(X, Q, alpha=1e-3)
#     Y = P.T.dot(Q)
#     assert_array_almost_equal(X, Y, decimal=2)

@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("var_red", var_reds)
def test_dict_mf_reconstruction(backend, var_red):
    if backend == 'c' and var_red == 'combo':
        return
    X, Q = generate_synthetic()
    dict_mf = DictMF(n_components=4, alpha=1e-4,
                     max_n_iter=300, l1_ratio=0,
                     backend=backend,
                     var_red=var_red,
                     random_state=rng_global, reduction=1)
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.T.dot(dict_mf.components_)
    assert_array_almost_equal(X, Y, decimal=1)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("var_red", var_reds)
def test_dict_mf_reconstruction_reduction(backend, var_red):
    if backend == 'c' and var_red == 'combo':
        return
    X, Q = generate_synthetic(n_features=20,
                              n_samples=400,
                              dictionary_rank=4)
    dict_mf = DictMF(n_components=4, alpha=1e-6,
                     max_n_iter=800, l1_ratio=0,
                     backend=backend,
                     var_red=var_red,
                     learning_rate=0.9,
                     random_state=rng_global, reduction=2)
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.T.dot(dict_mf.components_)
    rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
    assert (rel_error < 0.06)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("var_red", var_reds)
@pytest.mark.parametrize("replacement", [False, True])
def test_dict_mf_reconstruction_reproductible(backend, var_red, replacement):
    if backend == 'c' and var_red == 'combo':
        return
    X, Q = generate_synthetic(n_features=20,
                              n_samples=400,
                              dictionary_rank=4)
    dict_mf = DictMF(n_components=4, alpha=1e-6,
                     max_n_iter=800, l1_ratio=0,
                     backend=backend,
                     replacement=replacement,
                     var_red=var_red,
                     random_state=rng_global, reduction=2)
    dict_mf.fit(X)
    D1 = dict_mf.components_.copy()
    P1 = dict_mf.transform(X)

    dict_mf.fit(X)
    D2 = dict_mf.components_.copy()
    P2 = dict_mf.transform(X)
    assert_array_equal(D1, D2)
    assert_array_equal(P1, P2)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("var_red", var_reds)
def test_dict_mf_reconstruction_reduction_batch(backend, var_red):
    if backend == 'c' and var_red == 'combo':
        return
    X, Q = generate_synthetic(n_features=20,
                              n_samples=400,
                              dictionary_rank=4)
    dict_mf = DictMF(n_components=4, alpha=1e-6,
                     max_n_iter=400, l1_ratio=0,
                     backend=backend,
                     var_red=var_red,
                     random_state=3, batch_size=3,
                     coupled_subset=True,
                     learning_rate=0.9,
                     reduction=2, )
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.T.dot(dict_mf.components_)
    rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
    assert (rel_error < 0.06)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("var_red", var_reds)
def test_dict_mf_reconstruction_sparse_dict(backend, var_red):
    if backend == 'c' and var_red == 'combo':
        return
    X, Q = generate_sparse_synthetic(300, 4)
    rng = check_random_state(0)
    dict_init = Q + rng.randn(*Q.shape) * 0.01
    dict_mf = DictMF(n_components=4, alpha=1e-4, max_n_iter=400, l1_ratio=1,
                     dict_init=dict_init,
                     backend=backend,
                     var_red=var_red,
                     random_state=rng_global)
    dict_mf.fit(X)
    Q_rec = dict_mf.components_
    Q_rec /= np.sqrt(np.sum(Q_rec ** 2, axis=1))[:, np.newaxis]
    Q /= np.sqrt(np.sum(Q ** 2, axis=1))[:, np.newaxis]
    G = np.abs(Q_rec.dot(Q.T))
    recovered_maps = min(np.sum(np.any(G > 0.95, axis=1)),
                         np.sum(np.any(G > 0.95, axis=0)))
    assert (recovered_maps >= 4)
    P = dict_mf.transform(X)
    Y = P.T.dot(dict_mf.components_)
    # Much stronger
    # assert_array_almost_equal(X, Y, decimal=2)
