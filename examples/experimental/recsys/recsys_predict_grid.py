# Author: Arthur Mensch
# License: BSD
import itertools
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
        self.rmse = []
        self.times = []
        self.iters = []
        self.start_time = time.clock()
        self.test_time = 0

    def __call__(self, mf):
        test_time = time.clock()

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        self.iters.append(mf.n_iter_)
        print('Train RMSE', rmse)

        self.rmse.append(rmse)
        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


learning_rate = 0.8
output_dir = expanduser('~/output/modl/recsys')


def main(dataset, n_jobs):
    trace_folder = join(output_dir, dataset)

    if not os.path.exists(trace_folder):
        os.makedirs(trace_folder)

    output_folder = join(trace_folder, 'cv')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    alphas = np.logspace(-3, 2, 6)
    betas = np.logspace(-1, 2, 4)
    random_states = np.arange(2)

    n_random_states = len(random_states)
    n_epochs = 5 if dataset == 'netflix' else 10
    res = Parallel(n_jobs=n_jobs * 2)(delayed(single_run)(output_folder,
                                                      dataset,
                                                      idx, alpha,
                                                      beta,
                                                      n_epochs,
                                                      random_split=0,
                                                      random_state=random_state,
                                                      bench=False) for
                                  idx, (alpha, beta, random_state)
                                  in enumerate(
        itertools.product(alphas, betas, random_states)))
    cv_scores, alphas, betas = zip(*res)
    cv_scores = np.array(cv_scores)
    alphas = np.array(alphas)
    betas = np.array(betas)
    average_cv_scores = cv_scores[::n_random_states]
    for i in range(1, n_random_states):
        average_cv_scores += cv_scores[i::n_random_states]
    average_cv_scores /= n_random_states
    alphas = alphas[::n_random_states]
    betas = betas[::n_random_states]

    results = {'alphas': alphas.tolist(), 'betas': betas.tolist(),
               'cv_scores': average_cv_scores.tolist()}
    json.dump(results, open(join(trace_folder, 'cv_scores.json'), 'w+'))

    print('Computing benchmarks')

    output_folder = join(trace_folder, 'benchmarks')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    argbest = np.argmin(average_cv_scores)
    best_alpha = alphas[argbest]
    best_beta = betas[argbest]

    n_epochs = 5 if dataset == 'netflix' else 20
    res = Parallel(n_jobs=n_jobs)(delayed(single_run)(output_folder,
                                                      dataset,
                                                      idx,
                                                      best_alpha,
                                                      best_beta,
                                                      n_epochs,
                                                      random_state=idx,
                                                      bench=True) for
                                  idx, random_state in
                                  enumerate(range(10)))

    scores, alphas, betas, times, rmse, iters = zip(*res)

    score = np.mean(np.array(scores))
    times = np.mean(np.array(times), axis=0)
    rmse = np.mean(np.array(rmse), axis=0)
    iters = np.mean(np.array(iters), axis=0)
    results = {'alpha': best_alpha,
               'beta': best_beta,
               'score': score,
               'times': times.tolist(),
               'rmse': rmse.tolist(),
               'iter': iters.tolist()}
    json.dump(results, open(join(trace_folder, 'benchmark.json'), 'w+'))


def single_run(output_folder,
               dataset,
               idx,
               alpha,
               beta,
               n_epochs,
               random_split=0,
               random_state=0,
               bench=True):
    # Need to download from spira
    if dataset in ['100k', '1m', '10m']:
        X = load_movielens(dataset)
        X_tr, X_te = train_test_split(X, train_size=0.75,
                                      random_state=random_split)
    else:
        X_tr, X_te = load_netflix()

    batch_size = 1000 if dataset == 'netflix' else X_tr.shape[0] // 100

    mf = DictCompleter(n_components=30, alpha=alpha, verbose=4,
                       beta=beta,
                       batch_size=batch_size, detrend=True,
                       offset=0,
                       fit_intercept=True,
                       projection='partial',
                       random_state=random_state,
                       learning_rate=learning_rate,
                       n_epochs=n_epochs,
                       backend='c')
    results = {'dataset': dataset,
               'random_split': int(random_split),
               'alpha': alpha,
               'beta': beta,
               'learning_rate': learning_rate,
               'random_state': int(random_state),
               'batch_size': batch_size}

    X_tr = X_tr.tocsr()
    X_te = X_te.tocsr()
    if bench:
        cb = Callback(X_tr, X_te)
        mf.set_params(callback=cb)
    t0 = time.time()
    mf.fit(X_tr)
    t1 = time.time() - t0

    score = mf.score(X_te)

    results['score'] = score
    results['time'] = t1

    if bench:
        results['times'] = cb.times
        results['rmse'] = cb.rmse
        results['iter'] = cb.iters

    json.dump(results,
              open(join(output_folder, 'results_%s.json' % idx), 'w+'))

    print('Final time : %.2f s' % t1)
    print('Final score : % .4f' % score)

    if not bench:
        return score, alpha, beta
    else:
        return score, alpha, beta, cb.times, cb.rmse, cb.iters


def simple(dataset, alpha):
    output_folder = join(output_dir, dataset, 'simple')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    single_run(output_folder,
               dataset,
               0,
               alpha,
               20,
               random_split=0,
               random_state=0,
               bench=True)


def plot(dataset):
    import matplotlib.pyplot as plt
    output_folder = join(output_dir, dataset, 'benchmarks')

    fig, ax = plt.subplots(1, 1)

    files = os.listdir(output_folder)
    for file in files:
        this_bench = json.load(open(join(output_folder, file), 'r'))
        ax.plot(this_bench['times'], this_bench['rmse'])
        ax.set_xscale('log')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Test RMSE')
        ax.set_title(dataset)
    plt.show()


if __name__ == '__main__':
    main('100k', 15)
    main('1m', 15)
    main('10m', 15)
    # main('netflix', 14)
    # plot('netflix')
