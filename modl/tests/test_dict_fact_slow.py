import numpy as np
from sklearn.utils import check_random_state

from modl.dict_fact_slow import DictFactSlow


def test_dict_fact_slow():
    random_state = check_random_state(0)
    A = random_state.randn(1000, 10)
    B = random_state.randn(10, 100)
    X = A.dot(B)
    mf = DictFactSlow(n_components=10, reduction=1, n_epochs=2,
                      code_l1_ratio=0,
                      comp_l1_ratio=0, code_alpha=0)
    mf.fit(X)
    print('a')


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
