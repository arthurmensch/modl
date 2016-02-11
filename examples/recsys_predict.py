# Author: Mathieu Blondel
# License: BSD
import datetime
import os
import time
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np

from modl.dict_completion import DictCompleter, csr_center_data
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
        self.rmse_tr = []
        self.times = []
        self.q = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()
        X_pred = mf.predict(self.X_tr)
        loss = sqnorm(X_pred.data - self.X_tr.data) / 2
        regul = mf.alpha * (sqnorm(mf.P_))
        self.obj.append(loss + regul)

        # if mf.normalize:
        #     if not hasattr(self, 'X_tr_c_'):
        #         self.X_tr_c_, _, _ = csr_center_data(self.X_tr)
        #     else:
        #         mf._refit(self.X_tr_c_)
        # else:
        #     mf._refit(self.X_tr)

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        print(rmse)
        X_pred = mf.predict(self.X_tr)
        rmse_tr = np.sqrt(np.mean((X_pred.data - self.X_tr.data) ** 2))

        self.rmse.append(rmse)
        self.rmse_tr.append(rmse_tr)
        self.q.append(mf.Q_[1, :10].copy())
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


random_state = 0

mf = DictCompleter(n_components=30, alpha=.5, verbose=10,
                   batch_size=10, normalize=True,
                   offset=0,
                   impute=False,
                   fit_intercept=False,
                   random_state=0,
                   learning_rate=1,
                   max_n_iter=5000,
                   backend='python',
                   debug=True)

X = load_movielens('1m')
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
plt.plot(np.arange(len(cb.rmse)), cb.rmse, label='MODL')
plt.plot(np.arange(len(cb.rmse_tr)), cb.rmse_tr, label='Train')

plt.legend()
plt.xlabel("CPU time")
# plt.xscale("log")
plt.ylabel("RMSE")
plt.figure()
plt.plot(np.arange(len(mf._loss_stat[1])), mf._loss_stat[1])
plt.plot(np.arange(len(mf._loss_stat[2])), mf._loss_stat[2])
plt.figure()
plt.plot(np.arange(len(cb.q)), cb.q)
plt.show()