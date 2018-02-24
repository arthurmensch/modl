# Author: Arthur Mensch

import numpy as np
import pytest
from modl.decomposition.dict_fact import DictFact
from numpy import linalg
from numpy.testing import assert_array_equal
from sklearn.linear_model import cd_fast
from sklearn.utils import check_random_state

rng_global = 0

solvers = ['masked', 'gram', 'average', 'full']

solver_dict = {
    'masked': {'Dx_agg': 'masked', 'G_agg': 'masked'},
    'gram': {'Dx_agg': 'masked', 'G_agg': 'full'},
    'average': {'Dx_agg': 'masked', 'G_agg': 'masked'},
    'full': {'Dx_agg': 'full', 'G_agg': 'full'},
}


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


@pytest.mark.parametrize("solver", solvers)
def test_dict_mf_reconstruction(solver):
    X, Q = generate_synthetic()
    dict_mf = DictFact(n_components=4,
                       code_alpha=1e-4,
                       n_epochs=5,
                       comp_l1_ratio=0,
                       G_agg=solver_dict[solver]['G_agg'],
                       Dx_agg=solver_dict[solver]['Dx_agg'],
                       random_state=rng_global, reduction=1)
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.dot(dict_mf.components_)
    rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
    assert (rel_error < 0.02)

@pytest.mark.parametrize("solver", solvers)
def test_dict_mf_reconstruction_reduction(solver):
    X, Q = generate_synthetic(n_features=20,
                              n_samples=400,
                              dictionary_rank=4)
    dict_mf = DictFact(n_components=4,
                       code_alpha=1e-4,
                       n_epochs=2,
                       comp_l1_ratio=0,
                       G_agg=solver_dict[solver]['G_agg'],
                       Dx_agg=solver_dict[solver]['Dx_agg'],
                       random_state=rng_global, reduction=2)
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.dot(dict_mf.components_)
    rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
    assert (rel_error < 0.02)


@pytest.mark.parametrize("solver", solvers)
def test_dict_mf_reconstruction_reproductible(solver):
    X, Q = generate_synthetic(n_features=20,
                              n_samples=400,
                              dictionary_rank=4)
    dict_mf = DictFact(n_components=4,
                       code_alpha=1e-4,
                       n_epochs=2,
                       comp_l1_ratio=0,
                       G_agg=solver_dict[solver]['G_agg'],
                       Dx_agg=solver_dict[solver]['Dx_agg'],
                       random_state=0, reduction=2)
    dict_mf.fit(X)
    D1 = dict_mf.components_.copy()
    P1 = dict_mf.transform(X)

    dict_mf.random_state = 0

    dict_mf.fit(X)
    D2 = dict_mf.components_.copy()
    P2 = dict_mf.transform(X)
    assert_array_equal(D1, D2)
    assert_array_equal(P1, P2)


@pytest.mark.parametrize("solver", solvers)
def test_dict_mf_reconstruction_reduction_batch(solver):
    X, Q = generate_synthetic(n_features=20,
                              n_samples=400,
                              dictionary_rank=4)
    dict_mf = DictFact(n_components=4,
                       code_alpha=1e-4,
                       n_epochs=2,
                       comp_l1_ratio=0,
                       G_agg=solver_dict[solver]['G_agg'],
                       Dx_agg=solver_dict[solver]['Dx_agg'],
                       random_state=rng_global, reduction=2,
                       batch_size=10)
    dict_mf.fit(X)
    P = dict_mf.transform(X)
    Y = P.dot(dict_mf.components_)
    rel_error = np.sum((X - Y) ** 2) / np.sum(X ** 2)
    assert (rel_error < 0.06)


@pytest.mark.parametrize("solver", solvers)
def test_dict_mf_reconstruction_sparse_dict(solver):
    X, Q = generate_sparse_synthetic(500, 4)
    rng = check_random_state(0)
    dict_init = Q  + rng.randn(*Q.shape) * 0.2
    dict_mf = DictFact(n_components=4, code_alpha=1e-2, n_epochs=2,
                       code_l1_ratio=0,
                       comp_l1_ratio=1,
                       dict_init=dict_init,
                       G_agg=solver_dict[solver]['G_agg'],
                       Dx_agg=solver_dict[solver]['Dx_agg'],
                       random_state=rng_global)
    dict_mf.fit(X)
    Q_rec = dict_mf.components_
    Q_rec /= np.sqrt(np.sum(Q_rec ** 2, axis=1))[:, np.newaxis]
    Q /= np.sqrt(np.sum(Q ** 2, axis=1))[:, np.newaxis]
    G = np.abs(Q_rec.dot(Q.T))
    recovered_maps = min(np.sum(np.any(G > 0.95, axis=1)),
                         np.sum(np.any(G > 0.95, axis=0)))
    assert (recovered_maps >= 4)


def enet_regression_multi_gram_(G, Dx, X, code, l1_ratio, alpha,
                                positive):
    batch_size = code.shape[0]
    if l1_ratio == 0:
        n_components = G.shape[1]
        for i in range(batch_size):
            G.flat[::n_components + 1] += alpha
            code[i] = linalg.solve(G[i], Dx[i])
            G.flat[::n_components + 1] -= alpha
    else:
        # Unused but unfortunate API
        random_state = check_random_state(0)
        for i in range(batch_size):
            cd_fast.enet_coordinate_descent_gram(
                code[i],
                alpha * l1_ratio,
                alpha * (
                    1 - l1_ratio),
                G[i], Dx[i], X[i], 100, 1e-2,
                random_state,
                False, positive)
    return code


def enet_regression_single_gram_(G, Dx, X, code, code_l1_ratio, code_alpha,
                                 code_pos):
    batch_size = code.shape[0]
    if code_l1_ratio == 0:
        n_components = G.shape[0]
        G = G.copy()
        G.flat[::n_components + 1] += code_alpha
        code[:] = linalg.solve(G, Dx.T).T
        G.flat[::n_components + 1] -= code_alpha
    else:
        # Unused but unfortunate API
        random_state = check_random_state(0)
        for i in range(batch_size):
            cd_fast.enet_coordinate_descent_gram(
                code[i],
                code_alpha * code_l1_ratio,
                code_alpha * (
                    1 - code_l1_ratio),
                G, Dx[i], X[i], 100, 1e-2,
                random_state,
                False, code_pos)
    return code
