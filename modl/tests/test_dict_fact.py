# Author: Arthur Mensch

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from sklearn.utils import check_random_state

from modl.dict_fact import DictMF

rng_global = 0

backends = ['c', 'python']

var_reds = ['weight_based', 'sample_based']


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
    X, Q = generate_synthetic(n_features=20,
                              n_samples=400,
                              dictionary_rank=5)
    dict_mf = DictMF(n_components=4, alpha=1e-6,
                     max_n_iter=800, l1_ratio=0,
                     backend=backend,
                     var_red=var_red,
                     random_state=rng_global, reduction=2)
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.T.dot(dict_mf.components_)
    rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
    assert (rel_error < 0.06)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("var_red", var_reds)
def test_dict_mf_reconstruction_reduction_batch(backend, var_red):
    X, Q = generate_synthetic(n_features=20,
                              n_samples=400,
                              dictionary_rank=5)
    dict_mf = DictMF(n_components=5, alpha=1e-6,
                     max_n_iter=800, l1_ratio=0,
                     backend=backend,
                     var_red=var_red,
                     random_state=rng_global, batch_size=2,
                     reduction=2, )
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.T.dot(dict_mf.components_)
    rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
    assert (rel_error < 0.04)


# @pytest.mark.parametrize("backend", backends)
# @pytest.mark.parametrize("var_red", var_reds)
# def test_dict_mf_reconstruction_sparse(backend, var_red):
#     X, Q = generate_synthetic(n_features=20,
#                               n_samples=200,
#                               dictionary_rank=5)
#     sp_X = np.zeros((X.shape[0] * 2, X.shape[1]))
#     rng = check_random_state(0)
#     # Generate a sparse simple problem
#     for i in range(X.shape[0]):
#         perm = rng.permutation(X.shape[1])
#         even_range = perm[::2]
#         odd_range = perm[1::2]
#         sp_X[2 * i, even_range] = X[i, even_range]
#         sp_X[2 * i + 1, odd_range] = X[i, odd_range]
#     sp_X = sp.csr_matrix(sp_X)
#     dict_mf = DictMF(n_components=4, alpha=1e-6,
#                      learning_rate=0.75,
#                      max_n_iter=500, l1_ratio=0,
#                      backend=backend,
#                      var_red=var_red,
#                      random_state=rng_global)
#     dict_mf.fit(sp_X)
#     P = dict_mf.transform(X)
#     Y = P.T.dot(dict_mf.components_)
#     rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
#     assert (rel_error < 0.04)
#     # Much stronger
#     # assert_array_almost_equal(X, Y, decimal=2)


@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("var_red", var_reds)
def test_dict_mf_reconstruction_sparse_dict(backend, var_red):
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
