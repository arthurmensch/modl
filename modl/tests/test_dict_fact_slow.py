import numpy as np
from sklearn.utils import check_random_state

from modl.dict_fact_slow import DictFactSlow


def test_dict_fact_slow():
    random_state = check_random_state(0)
    A = random_state.randn(100, 10)
    B = random_state.randn(10, 1000)
    X = A.dot(B)
    mf = DictFactSlow(n_components=10, reduction=1, n_epochs=2,
                      code_l1_ratio=0,
                      D_l1_ratio=0, code_alpha=0)
    mf.fit(X)
    print('a')