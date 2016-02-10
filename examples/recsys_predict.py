# Author: Mathieu Blondel
# License: BSD
import datetime
import os
import time
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np

from modl.dict_completion import DictCompleter
from spira.cross_validation import train_test_split
from spira.datasets import load_movielens


def sqnorm(M):
    m = M.ravel()
    return np.dot(m, m)


class Callback(object):
    def __init__(self, X_tr, X_te, refit=False):
        self.X_tr = X_tr
        self.X_te = X_te
        self.obj = []
        self.rmse = []
        self.times = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()
        X_pred = mf.predict(self.X_tr)
        loss = sqnorm(X_pred.data - self.X_tr.data) / 2
        regul = mf.alpha * (sqnorm(mf.P_))
        self.obj.append(loss + regul)

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        print(rmse)
        self.rmse.append(rmse)

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


random_state = 0

mf = DictCompleter(n_components=30, alpha=1, verbose=10,
                   batch_size=10, normalize=True,
                   impute=False,
                   fit_intercept=True,
                   random_state=0,
                   learning_rate=1,
                   max_n_iter=10000,
                   backend='python')

X = load_movielens('100k')
X_tr, X_te = train_test_split(X, train_size=0.75,
                              random_state=random_state)
X_tr = X_tr.tocsr()
X_te = X_te.tocsr()
cb = Callback(X_tr, X_te)
mf.set_params(callback=cb)
mf.fit(X_tr)

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
output_dir = join(expanduser('~/output/dl_fast'), timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.figure()
plt.plot(cb.times, cb.rmse, label='MODL')
plt.legend()
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("RMSE")
plt.show()