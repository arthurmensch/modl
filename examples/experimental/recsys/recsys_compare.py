# Author: Mathieu Blondel, Arthur Mensch
# License: BSD
import copy
import json
import os
import time
from collections import OrderedDict
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from joblib import Parallel, delayed
from matplotlib import gridspec
from sklearn import clone

from modl._utils.cross_validation import ShuffleSplit, \
    cross_val_score
from modl.datasets.recsys import get_recsys_data
from modl.dict_completion import DictCompleter
from modl.externals.spira.matrix_fact import ExplicitMF

trace_dir = expanduser('~/output/modl/recsys_bias')

estimator_grid = {'cd': {'estimator': ExplicitMF(n_components=30,
                                                 detrend=True,
                                                 ),
                         'name': 'Coordinate descent'},
                  'dl': {'estimator': DictCompleter(n_components=30,
                                                    detrend=True,
                                                    projection='full',
                                                    fit_intercept=True,
                                                    backend='c'),
                         'name': 'Proposed online masked MF'},
                  'dl_partial': {'estimator': DictCompleter(n_components=30,
                                                            detrend=True,
                                                            projection='partial',
                                                            fit_intercept=True,
                                                            backend='c'),
                                 'name': 'Proposed algorithm'
                                         ' (with partial projection)'}
                  }


def _get_hyperparams():
    hyperparams = {'cd': {'100k': dict(max_iter=200),
                          "1m": dict(max_iter=300),
                          "10m": dict(max_iter=200),
                          "netflix": dict(max_iter=50)},
                   'dl': {
                       '100k': dict(learning_rate=0.85, n_epochs=30,
                                    batch_size=10),
                       '1m': dict(learning_rate=0.85, n_epochs=30,
                                  batch_size=60),
                       '10m': dict(learning_rate=0.85, n_epochs=60,
                                   batch_size=600),
                       'netflix': dict(learning_rate=0.9, n_epochs=25,
                                       batch_size=4000)}}
    hyperparams['dl_partial'] = hyperparams['dl']
    return hyperparams


def _get_cvparams():
    cvparams = {'cd': {'100k': dict(alpha=.1),
                       '1m': dict(alpha=.03),
                       '10m': dict(alpha=.04),
                       'netflix': dict(alpha=.1)},
                'dl': {'100k': dict(alpha=.1),
                       '1m': dict(alpha=.03),
                       '10m': dict(alpha=.04),
                       'netflix': dict(alpha=.1)},
                'dl_partial': {'100k': dict(alpha=.1),
                               '1m': dict(alpha=.03),
                               '10m': dict(alpha=.04),
                               'netflix': dict(alpha=.1)}
                }

    # Replace hard-coded cv params by parameters learned by CV
    for version in ['100k', '1m', '10m', 'netflix']:
        try:
            with open(
                    join(trace_dir, 'cross_val', 'results_%s.json' % version),
                    'r') as f:
                results = json.load(f)
        except IOError:
            continue
        for idx in results.keys():
            cvparams[idx][version] = results[idx]['best_param']
    return cvparams


alphas = {'netflix': np.logspace(-2, 1, 15),
          '10m': np.logspace(-2, 1, 15),
          '1m': np.logspace(-2, 1, 30)
          }
betas = [0]
learning_rates = np.linspace(0.75, 1, 10)

# Optional : cross val biases on intercept
# betas = np.logspace(-1, 2, 4)

def sqnorm(M):
    m = M.ravel()
    return np.dot(m, m)


class Callback(object):
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
        print('Test RMSE : ', rmse)
        self.rmse.append(rmse)

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)


def compare_learning_rate(version='100k', n_jobs=1, random_state=0):
    X_tr, X_te = get_recsys_data(version, random_state)
    mf = copy.deepcopy(estimator_grid['dl_partial']['estimator'])

    hyperparams = _get_hyperparams()
    cvparams = _get_cvparams()

    mf.set_params(**hyperparams['dl_partial'][version])
    mf.set_params(**cvparams['dl_partial'][version])
    mf.set_params(random_state=random_state)
    output_dir = join(trace_dir, 'learning_rate')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = {}
    res = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(single_learning_rate)(mf, learning_rate, X_tr, X_te) for
        learning_rate in learning_rates)

    for i, learning_rate in enumerate(learning_rates):
        results[learning_rate] = res[i]
    with open(join(output_dir, 'results_%s.json' % version), 'w+') as f:
        json.dump(results, f)


def single_learning_rate(mf, learning_rate, X_tr, X_te):
    mf = clone(mf)
    mf.set_params(learning_rate=learning_rate, verbose=5)
    cb = Callback(X_tr, X_te)
    mf.set_params(callback=cb)
    mf.fit(X_tr)
    return dict(time=cb.times,
                rmse=cb.rmse)


def cross_val(dataset='100k',
              random_state=0,
              n_jobs=1):
    results = copy.deepcopy(estimator_grid)

    X_tr, X_te = get_recsys_data(dataset, random_state)

    subdir = 'cross_val'
    output_dir = expanduser(join(trace_dir, subdir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    hyperparams = _get_hyperparams()

    for idx in results.keys():
        mf = results[idx]['estimator']
        mf.set_params(**hyperparams[idx][dataset])
        if idx in ['dl', 'dl_partial']:
            mf.set_params(n_epochs=5)
        else:
            mf.set_params(max_iter=40)
        mf.set_params(random_state=random_state)
        param_grid = [dict(alpha=alpha, beta=beta) for alpha in alphas[dataset]
                      for beta in betas]

        mf.verbose = 0
        mf.alpha = 0
        mf.beta = 0
        if dataset == 'netflix':
            # We don't perform nested cross val here
            res = Parallel(n_jobs=n_jobs,
                           verbose=10,
                           max_nbytes=None)(
                delayed(single_fit)(mf, X_tr, X_te,
                                    params) for params in param_grid)
        else:
            cv = ShuffleSplit(n_iter=3,
                              train_size=0.66,
                              random_state=0)
            res = Parallel(n_jobs=n_jobs,
                           verbose=10,
                           max_nbytes=None)(
                delayed(single_fit_nested)(mf, X_tr, cv, params)
                for params in
                param_grid)
        scores, params = zip(*res)
        scores = np.array(scores).mean(axis=1)
        best_score_arg = scores.argmin()
        best_param = params[best_score_arg]
        best_score = scores[best_score_arg]

        results[idx]['params'] = params
        results[idx]['scores'] = scores.tolist()

        results[idx]['best_param'] = best_param
        results[idx]['best_score'] = best_score
        results[idx].pop('estimator')

    with open(join(output_dir, 'results_%s.json' % dataset), 'w+') as f:
        json.dump(results, f)


def benchmark(dataset='100k',
              random_state=0,
              n_jobs=1):
    results = copy.deepcopy(estimator_grid)

    hyperparams = _get_hyperparams()
    cvparams = _get_cvparams()

    X_tr, X_te = get_recsys_data(dataset, random_state)

    subdir = 'benches'
    output_dir = expanduser(join(trace_dir, subdir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    indices = sorted(results.keys())
    for idx in indices:
        mf = results[idx]['estimator']
        mf.set_params(**hyperparams[idx][dataset])
        mf.set_params(**cvparams[idx][dataset])
        mf.set_params(random_state=random_state)
    res = Parallel(n_jobs=n_jobs,
                   max_nbytes=None)(
        delayed(single_fit_bench)(results[idx]['estimator'],
                                  X_tr, X_te)
        for idx in indices)
    times, rmses = zip(*res)
    for time, rmse, idx in zip(times, rmses, indices):
        results[idx]['timings'] = time
        results[idx]['rmse'] = rmse
        results[idx].pop('estimator')

    with open(join(output_dir, 'results_%s.json' % dataset), 'w+') as f:
        json.dump(results, f)


def single_fit_bench(mf, X_tr, X_te):
    mf = clone(mf)
    cb = Callback(X_tr, X_te)
    mf.set_params(callback=cb, verbose=5)
    mf.fit(X_tr)
    return cb.times, cb.rmse


def single_fit_nested(mf, X_tr, cv, params):
    mf = clone(mf)
    mf.set_params(**params)
    mf.set_params(verbose=3)
    score = cross_val_score(mf, X_tr, cv)
    return score, params


def single_fit(mf, X_tr, X_te, params):
    mf = clone(mf)
    mf.set_params(**params)
    mf.fit(X_tr)
    score = [mf.score(X_te)]
    return score, params

def plot_learning_rate():
    output_dir = join(trace_dir, 'learning_rate')
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.33)
    fig.subplots_adjust(top=0.99)
    fig.subplots_adjust(right=0.98)

    fig.set_figwidth(3.25653379549)
    fig.set_figheight(1.25)
    ax = {}
    gs = gridspec.GridSpec(1, 2)
    palette = sns.cubehelix_palette(10, start=0, rot=3, hue=1, dark=.3,
                                    light=.7,
                                    reverse=False)

    for j, version in enumerate(['10m', 'netflix']):
        with open(join(output_dir, 'results_%s.json' % version), 'r') as f:
            data = json.load(f)
        ax[j] = fig.add_subplot(gs[j])
        learning_rates = sorted(data, key=lambda t: float(t))
        for i, learning_rate in enumerate(learning_rates):
            this_data = data[str(learning_rate)]
            n_epochs = _get_hyperparams()['dl_partial'][version]['n_epochs']
            ax[j].plot(np.linspace(0, n_epochs, len(this_data['rmse'])),
                       this_data['rmse'],
                       label='%.2f' % float(learning_rate),
                       color=palette[i],
                       zorder=int(100 * float(learning_rate)))
            ax[j].set_xscale('log')
        sns.despine(fig, ax)

        ax[j].spines['left'].set_color((.6, .6, .6))
        ax[j].spines['bottom'].set_color((.6, .6, .6))
        ax[j].xaxis.set_tick_params(color=(.6, .6, .6), which='both')
        ax[j].yaxis.set_tick_params(color=(.6, .6, .6), which='both')
        ax[j].tick_params(axis='y', labelsize=6)

    ax[0].set_ylabel('RMSE on test set')
    ax[0].set_xlabel('Epoch', ha='left', va='top')
    ax[0].xaxis.set_label_coords(-.18, -0.055)

    ax[0].set_xlim([.1, 40])
    ax[0].set_xticks([1, 10, 40])
    ax[0].set_xticklabels(['1', '10', '40'])
    ax[1].set_xlim([.1, 25])
    ax[1].set_xticks([.1, 1, 10, 20])
    ax[1].set_xticklabels(['.1', '1', '10', '20'])

    ax[0].annotate('MovieLens 10M', xy=(.95, .9), ha='right',
                   xycoords='axes fraction', zorder=100)
    ax[1].annotate('Netflix', xy=(.95, .9), ha='right',
                   xycoords='axes fraction', zorder=100)

    ax[0].set_ylim([0.795, 0.877])
    ax[1].set_ylim([0.93, .999])
    ax[0].legend(ncol=4, loc='upper left', bbox_to_anchor=(-0.09, -.13),
                 fontsize=7, numpoints=1, columnspacing=.3, frameon=False)
    ax[0].annotate('Learning rate $\\beta$', xy=(1.6, -.38),
                   xycoords='axes fraction')
    ltext = ax[0].get_legend().get_texts()
    plt.setp(ltext, fontsize=7)

    plt.savefig(join(trace_dir, 'learning_rate.pdf'))


def plot_benchs():
    output_dir = join(trace_dir, 'benches')

    fig = plt.figure()

    fig.subplots_adjust(right=.9)
    fig.subplots_adjust(top=.905)
    fig.subplots_adjust(bottom=.12)
    fig.subplots_adjust(left=.06)
    fig.set_figheight(fig.get_figheight() * 0.66)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.5])

    ylims = {'100k': [.90, .96], '1m': [.864, .915], '10m': [.80, .868],
             'netflix': [.93, .99]}
    xlims = {'100k': [0.0001, 10], '1m': [0.1, 20], '10m': [1, 400],
             'netflix': [30, 3000]}

    names = {'dl_partial': 'Proposed \n(partial projection)',
             'dl': 'Proposed \n(full projection)',
             'cd': 'Coordinate descent'}
    zorder = {'cd': 10,
              'dl': 1,
              'dl_partial': 5}
    for i, version in enumerate(['1m', '10m', 'netflix']):
        try:
            with open(join(output_dir, 'results_%s.json' % version), 'r') as f:
                results = json.load(f)
        except IOError:
            continue

        ax_time = fig.add_subplot(gs[0, i])
        ax_time.grid()
        sns.despine(fig, ax_time)

        ax_time.spines['left'].set_color((.6, .6, .6))
        ax_time.spines['bottom'].set_color((.6, .6, .6))
        ax_time.xaxis.set_tick_params(color=(.6, .6, .6), which='both')
        ax_time.yaxis.set_tick_params(color=(.6, .6, .6), which='both')

        for tick in ax_time.xaxis.get_major_ticks():
            tick.label.set_fontsize(7)
            tick.label.set_color('black')
        for tick in ax_time.yaxis.get_major_ticks():
            tick.label.set_fontsize(7)
            tick.label.set_color('black')

        if i == 0:
            ax_time.set_ylabel('RMSE on test set')
        if i == 2:
            ax_time.set_xlabel('CPU time')
            ax_time.xaxis.set_label_coords(1.14, -0.06)

        ax_time.grid()
        palette = sns.cubehelix_palette(3, start=0, rot=.5, hue=1, dark=.3,
                                        light=.7,
                                        reverse=False)
        color = {'dl_partial': palette[2], 'dl': palette[1], 'cd': palette[0]}
        for idx in sorted(OrderedDict(results).keys()):
            this_result = results[idx]
            ax_time.plot(this_result['timings'], this_result['rmse'],
                         label=names[idx], color=color[idx],
                         linewidth=2,
                         linestyle='-' if idx != 'cd' else '--',
                         zorder=zorder[idx])
        if version == 'netflix':
            ax_time.legend(loc='upper left', bbox_to_anchor=(.65, 1.1),
                           numpoints=1,
                           frameon=False)
        ax_time.set_xscale('log')
        ax_time.set_ylim(ylims[version])
        ax_time.set_xlim(xlims[version])
        if version == '1m':
            ax_time.set_xticks([.1, 1, 10])
            ax_time.set_xticklabels(['0.1 s', '1 s', '10 s'])
        elif version == '10m':
            ax_time.set_xticks([1, 10, 100])
            ax_time.set_xticklabels(['1 s', '10 s', '100 s'])
        else:
            ax_time.set_xticks([100, 1000])
            ax_time.set_xticklabels(['100 s', '1000 s'])
        ax_time.annotate(
            'MovieLens %s' % version.upper() if version != 'netflix' else 'Netflix (140M)',
            xy=(.5 if version != 'netflix' else .4, 1),
            xycoords='axes fraction', ha='center', va='bottom')
    plt.savefig(join(trace_dir, 'bench.pdf'))


if __name__ == '__main__':
    cross_val('1m', n_jobs=15)
    benchmark('1m', n_jobs=3)
    cross_val('10m', n_jobs=15)
    benchmark('10m', n_jobs=3)
    cross_val('netflix', n_jobs=15)
    benchmark('netflix', n_jobs=3)
    compare_learning_rate('10m', n_jobs=10)
    compare_learning_rate('netflix', n_jobs=10)
    plot_benchs()
    plot_learning_rate()
