# Author: Arthur Mensch

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal

from modl.dict_fact import DictMF, compute_code

rng_global = np.random.RandomState(0)

def generate_sparse_synthetic(n_samples=200,
                       square_size=4):
    n_features = square_size ** 2
    half_size = square_size // 2
    Q = np.zeros((4, n_features))
    for i in range(2):
            for j in range(2):
                atom = np.zeros((square_size, square_size))
                atom[(half_size * i):(half_size * (i + 1)),
                (half_size * j): (half_size * (j + 1))] = 1
                Q[2 * i + j] = np.ravel(atom)
    code = rng_global.randn(n_samples, 4)
    X = code.dot(Q)
    return X, Q

def generate_synthetic(n_samples=200,
                       n_components=4, n_features=16,
                       dictionary_rank=None):
    Q = np.zeros((n_components, n_features))
    if dictionary_rank is None:
        Q = rng_global.randn(n_components, n_features)
    else:
        V = rng_global.randn(dictionary_rank, n_features)
        U = rng_global.randn(n_components, dictionary_rank)
        Q = U.dot(V)
    code = rng_global.randn(n_samples, n_components)
    X = code.dot(Q)
    return X, Q


def test_compute_code():
    X, Q = generate_synthetic()
    P = compute_code(X, Q, alpha=1e-3)
    Y = P.T.dot(Q)
    assert_array_almost_equal(X, Y, decimal=2)


def test_dict_mf_reconstruction():
    X, Q = generate_synthetic()
    dict_mf = DictMF(n_components=4, alpha=1e-4,
                     max_n_iter=200, l1_ratio=0,
                     random_state=rng_global, reduction=1)
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.T.dot(dict_mf.Q_)
    assert_array_almost_equal(X, Y, decimal=1)


def test_dict_mf_reconstruction_reduction():
    X, Q = generate_synthetic(n_features=20,
                              n_samples=200,
                              dictionary_rank=5)
    dict_mf = DictMF(n_components=4, alpha=1e-6,
                     max_n_iter=400, l1_ratio=0,
                     random_state=rng_global, reduction=2)
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.T.dot(dict_mf.Q_)
    rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
    assert(rel_error < 0.02)


def test_dict_mf_reconstruction_sparse():
    X, Q = generate_synthetic(n_features=20,
                              n_samples=200,
                              dictionary_rank=5)
    sp_X = np.zeros((X.shape[0] * 2, X.shape[1]))
    # Generate a sparse simple problem
    for i in range(X.shape[0]):
        perm = rng_global.permutation(X.shape[1])
        even_range = perm[::2]
        odd_range = perm[1::2]
        sp_X[2 * i, even_range] = X[i, even_range]
        sp_X[2 * i, odd_range] = X[i, odd_range]
    sp_X = sp.csr_matrix(sp_X)
    dict_mf = DictMF(n_components=4, alpha=1e-6,
                     max_n_iter=400, l1_ratio=0,
                     random_state=rng_global)
    dict_mf.fit(sp_X)
    P = dict_mf.transform(X)
    Y = P.T.dot(dict_mf.Q_)
    rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
    assert(rel_error < 0.02)
    # Much stronger
    # assert_array_almost_equal(X, Y, decimal=2)


def test_dict_mf_reconstruction_sparse_dict():
    X, Q = generate_sparse_synthetic(300, 4)
    dict_init = Q + rng_global.randn(*Q.shape) * 0.01
    dict_mf = DictMF(n_components=4, alpha=1e-2, max_n_iter=300, l1_ratio=1,
                     dict_init=dict_init,
                     random_state=rng_global)
    dict_mf.fit(X)
    Q_rec = dict_mf.Q_
    Q_rec /= np.sqrt(np.sum(Q_rec ** 2, axis=1))[:, np.newaxis]
    Q /= np.sqrt(np.sum(Q ** 2, axis=1))[:, np.newaxis]
    G = np.abs(Q_rec.dot(Q.T))
    recovered_maps = min(np.sum(np.any(G > 0.95, axis=1)),
                         np.sum(np.any(G > 0.95, axis=0)))
    assert (recovered_maps >= 4)
    P = compute_code(X, dict_mf.Q_, alpha=1e-3)
    Y = P.T.dot(dict_mf.Q_)
    assert_array_almost_equal(X, Y, decimal=2)
    # Much stronger
    assert_array_almost_equal(X, Y, decimal=2)
