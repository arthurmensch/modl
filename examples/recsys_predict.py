# Author: Arthur Mensch
# Inspired from spira
# License: BSD
import time

import numpy as np

from modl._utils.cross_validation import train_test_split
from modl.datasets.recsys import load_movielens
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
        self.times = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        self.rmse.append(rmse)
        print('Test RMSE: ', rmse)
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


random_state = 0

mf = DictCompleter(n_components=30, alpha=.001, beta=0, verbose=3,
                   batch_size=1000, detrend=True,
                   offset=0,
                   projection='partial',
                   random_state=0,
                   learning_rate=.9,
                   n_epochs=5,
                   backend='python')

# Need to download from spira
X = load_movielens('10m')
X_tr, X_te = train_test_split(X, train_size=0.75,
                              random_state=random_state)

X_tr = X_tr.tocsr()
X_te = X_te.tocsr()
cb = Callback(X_tr, X_te)
mf.set_params(callback=cb)
t0 = time.time()
mf.fit(X_tr)
print('Final test RMSE:', mf.score(X_te))
print('Time : %.2f s' % (time.time() - t0))


import matplotlib.pyplot as plt
plt.figure()
plt.plot(cb.times, cb.rmse, label='Test')
plt.plot(cb.times, cb.rmse, label='Train')

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
