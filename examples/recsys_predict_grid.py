# Author: Arthur Mensch
# License: BSD
import json
import time
from os.path import expanduser

import numpy as np
from joblib import Parallel
from joblib import delayed

from modl.datasets.movielens import load_netflix
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
        regul = mf.alpha * (sqnorm(mf.code_))
        self.obj.append(loss + regul)

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        print(rmse)
        X_pred = mf.predict(self.X_tr)
        rmse_tr = np.sqrt(np.mean((X_pred.data - self.X_tr.data) ** 2))

        self.rmse.append(rmse)
        self.rmse_tr.append(rmse_tr)
        self.q.append(mf.D_[1, :10].copy())
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


random_state = 0

def main(n_jobs):
    alphas = Parallel(n_jobs=n_jobs)(delayed(single_run)(alpha) for alpha in np.logspace(-4, 2, 15))
    json.dump(alphas, open(expanduser('~/scores.json'), 'w+'))

def single_run(alpha):
    mf = DictCompleter(n_components=30, alpha=alpha, verbose=0,
                       batch_size=4000, detrend=True,
                       offset=0,
                       fit_intercept=True,
                       projection='partial',
                       random_state=0,
                       learning_rate=.8,
                       n_epochs=3,
                       backend='c')
    # Need to download from spira
    # X = load_movielens('10m')
    # X_tr, X_te = train_test_split(X, train_size=0.75,
    #                               random_state=random_state)
    X_tr, X_te = load_netflix()

    X_tr = X_tr.tocsr()
    X_te = X_te.tocsr()
    # cb = Callback(X_tr, X_te)
    # mf.set_params(callback=cb)
    t0 = time.time()
    mf.fit(X_tr)
    score = mf.score(X_te)
    print('Time : %.2f s' % (time.time() - t0))
    print('Score : % .4f' % score)
    return score


if __name__ == '__main__':
    main(15)