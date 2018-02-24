# Author: Arthur Mensch
# Inspired from spira
# License: BSD
import time

import numpy as np
from modl.datasets.recsys import load_movielens
from modl.decomposition.recsys import RecsysDictFact
from modl.utils.recsys.cross_validation import train_test_split

class Callback(object):
    """Utility class for plotting RMSE"""

    def __init__(self, X_tr, X_te):
        self.X_tr = X_tr
        self.X_te = X_te
        self.obj = []
        self.rmse = []
        self.rmse_tr = []
        self.times = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        # X_pred = mf.predict(self.X_tr)
        # rmse_tr = np.sqrt(np.mean((X_pred.data - self.X_tr.data) ** 2))
        self.rmse.append(rmse)
        # self.rmse_tr.append(rmse_tr)
        print('Test RMSE: ', rmse)
        # print('Train RMSE: ', rmse_tr)
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


random_state = 0

mf = RecsysDictFact(n_components=50, alpha=1, beta=.1, verbose=10,
                    batch_size=10, detrend=True,
                    random_state=0,
                    learning_rate=.95,
                    n_epochs=10)

X = load_movielens('1m')
X_tr, X_te = train_test_split(X, train_size=0.8,
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
# plt.plot(cb.times, cb.rmse_tr, label='Train')

plt.legend()
plt.xlabel("CPU time")
plt.xscale("log")
plt.ylabel("RMSE")
plt.title('Prediction scores')

plt.show()
