# Author: Arthur Mensch
# License: BSD
import time

import matplotlib.pyplot as plt
import numpy as np

from modl._utils.cross_validation import train_test_split
from modl.datasets.movielens import load_movielens
from modl.dict_completion import DictCompleter

def sqnorm(M):
    m = M.ravel()
    return np.dot(m, m)


class Callback(object):
    """Utility class for plotting RMSE"""
    def __init__(self, X_tr, X_te):
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

mf = DictCompleter(n_components=30, alpha=1, verbose=5,
                   batch_size=60, detrend=True,
                   offset=0,
                   impute=True,
                   impute_lr=-1,
                   fit_intercept=True,
                   full_projection=True,
                   random_state=0,
                   learning_rate=1,
                   max_n_iter=20000,
                   backend='python')

# Need to download from spira
X = load_movielens('1m')
X_tr, X_te = train_test_split(X, train_size=0.75,
                              random_state=random_state)
X_tr = X_tr.tocsr()
X_te = X_te.tocsr()
cb = Callback(X_tr, X_te)
mf.set_params(callback=cb)
mf.fit(X_tr)

plt.figure()
plt.plot(cb.times, cb.rmse, label='Test')
plt.plot(cb.times, cb.rmse_tr, label='Train')

plt.legend()
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("RMSE")
plt.title('Prediction scores')

plt.figure()
plt.plot(np.arange(len(cb.q)), cb.q)
plt.xlabel('Time (relative)')
plt.ylabel('Feature value')
plt.title('Dictionary trajectory')
plt.show()