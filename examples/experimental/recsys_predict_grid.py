# Author: Arthur Mensch
# License: BSD
import json
import os
import time
from os.path import expanduser, join

import numpy as np
from joblib import Parallel
from joblib import delayed

from modl._utils.cross_validation import train_test_split
from modl.datasets.movielens import load_netflix, load_movielens
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
        # self.rmse_tr = []
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
        print('Train RMSE', rmse)
        # X_pred = mf.predict(self.X_tr)
        # rmse_tr = np.sqrt(np.mean((X_pred.data - self.X_tr.data) ** 2))

        self.rmse.append(rmse)
        # self.rmse_tr.append(rmse_tr)
        self.q.append(mf.D_[1, :10].copy())
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


learning_rate = 0.8
n_epochs = 20
output_dir = expanduser('~/output/modl/recsys')


def main(dataset, n_jobs):
    trace_folder = join(output_dir, dataset)

    if not os.path.exists(trace_folder):
        os.makedirs(trace_folder)

    alphas = np.logspace(-4, 2, 14)
    scores = Parallel(n_jobs=n_jobs)(delayed(single_run)(dataset,
                                                         idx, alpha) for idx,
                                                                         alpha
                                     in enumerate(alphas))
    results = {alphas: alphas, scores: scores}
    # scores = np.array(scores)
    # alphas = np.array(alphas)
    # argbest = np.argmin(scores)
    # results['best_score'] = scores[argbest]
    # results['best_alpha'] = alphas[argbest]
    json.dump(results, open(join(trace_folder, 'scores.json'), 'w+'))


def single_run(dataset, idx, alpha):
    random_state = 0

    trace_folder = join(output_dir, dataset)

    # Need to download from spira
    if dataset in ['100k', '1m', '10m']:
        X = load_movielens(dataset)
        X_tr, X_te = train_test_split(X, train_size=0.75,
                                      random_state=random_state)
    else:
        X_tr, X_te = load_netflix()

    batch_size = X_tr.shape[0] // 100

    mf = DictCompleter(n_components=30, alpha=alpha, verbose=4,
                       batch_size=batch_size, detrend=True,
                       offset=0,
                       fit_intercept=True,
                       projection='partial',
                       random_state=0,
                       learning_rate=learning_rate,
                       n_epochs=n_epochs,
                       backend='c')

    results = {'dataset': dataset,
               'alpha': alpha,
               'learning_rate': learning_rate,
               'batch_size': batch_size}

    X_tr = X_tr.tocsr()
    X_te = X_te.tocsr()
    cb = Callback(X_tr, X_te)
    mf.set_params(callback=cb)
    t0 = time.time()
    mf.fit(X_tr)

    results['times'] = cb.times
    results['rmse'] = cb.rmse

    # try:
    #     previous_results = json.load(open(join(trace_folder,
    #                                            'results.json'), 'r'))
    # except FileNotFoundError:
    #     previous_results = []
    # results = [results] + previous_results
    json.dump(results, open(join(trace_folder, 'results_%i.json' % idx), 'w+'))

    score = mf.score(X_te)
    print('Time : %.2f s' % (time.time() - t0))
    print('Score : % .4f' % score)
    return score


if __name__ == '__main__':
    # main('100k', 14)
    # main('1m', 14)
    # main('10m', 14)
    main('netflix', 14)