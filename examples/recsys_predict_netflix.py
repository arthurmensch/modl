# Author: Mathieu Blondel, Arthur Mensch
# License: BSD
import datetime
import json
import os
import time
from collections import OrderedDict
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from joblib import load, Parallel, delayed
from matplotlib import gridspec
from sklearn import clone
from spira.datasets import load_movielens
from spira.impl.dict_fact import csr_center_data
from spira.impl.matrix_fact import ExplicitMF

from modl._utils.cross_validation import train_test_split, ShuffleSplit, \
    cross_val_score
from modl.dict_completion import DictCompleter


def sqnorm(M):
    m = M.ravel()
    return np.dot(m, m)


class Callback(object):
    def __init__(self, X_tr, X_te, refit=False, record_tr=False):
        self.X_tr = X_tr
        self.X_te = X_te
        self.obj = []
        self.rmse = []
        self.times = []
        self.start_time = time.clock()
        self.test_time = 0
        self.refit = refit

    def __call__(self, mf):
        test_time = time.clock()
        if self.refit:
            if mf.normalize:
                if not hasattr(self, 'X_tr_c_'):
                    self.X_tr_c_, _, _ = csr_center_data(self.X_tr)
                else:
                    mf._refit_code(self.X_tr_c_)
            else:
                mf._refit_code(self.X_tr)
        if self.record_tr:
            X_pred = mf.predict(self.X_tr)
            loss = sqnorm(X_pred.data - self.X_tr.data) / 2
            regul = mf.alpha * (sqnorm(mf.code_))  # + sqnorm(mf.Q_))
            self.obj.append(loss + regul)

        X_pred = mf.predict(self.X_te)
        rmse = np.sqrt(np.mean((X_pred.data - self.X_te.data) ** 2))
        print('Train RMSE', rmse)
        self.rmse.append(rmse)

        self.test_time += time.clock() - test_time
        self.times.append(time.clock() - self.start_time - self.test_time)

def compare_learning_rate(version='100k', n_jobs=1, random_state=0):
    if version in ['100k', '1m', '10m']:
        X = load_movielens(version)
        X_tr, X_te = train_test_split(X, train_size=0.75,
                                      random_state=random_state)
        X_tr = X_tr.tocsr()
        X_te = X_te.tocsr()
    elif version is 'netflix':
        X_tr = load(expanduser('~/spira_data/nf_prize/X_tr.pkl'))
        X_te = load(expanduser('~/spira_data/nf_prize/X_te.pkl'))
    mf = DictCompleter(n_components=30, n_epochs=20 if version == '10m' else 7,
                       alpha=0.1373823795883263 if version == '10m' else 0.16681005372000587,
                       verbose=5,
                       batch_size=600 if version == '10m' else 4000,
                       detrend=True,
                       fit_intercept=True,
                       random_state=0,
                       learning_rate=.75,
                       backend='c')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                 '-%M-%S')
    subdir = 'learning_rate'
    output_dir = expanduser(join('~/output/recommender/', timestamp, subdir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = {}
    par_res = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(single_learning_rate)(mf, learning_rate, X_tr, X_te) for
        learning_rate in np.linspace(0.5, 1, 10))

    for i, learning_rate in enumerate(np.linspace(0.5, 1, 10)):
        results[learning_rate] = par_res[i]
    with open(join(output_dir, 'results_%s.json' % version), 'w+') as f:
        json.dump(results, f)


def single_learning_rate(mf, learning_rate, X_tr, X_te):
    mf = clone(mf)
    mf.set_params(learning_rate=learning_rate)
    cb = Callback(X_tr, X_te, refit=False)
    mf.set_params(callback=cb)
    mf.fit(X_tr)
    return dict(time=cb.times,
                rmse=cb.rmse)


def main(version='100k', n_jobs=1, random_state=0, cross_val=False):
    dl_params = {}
    dl_params['100k'] = dict(learning_rate=1, batch_size=10, offset=0, alpha=1)
    dl_params['1m'] = dict(learning_rate=.75, batch_size=60, offset=0,
                           alpha=.8)
    dl_params['10m'] = dict(learning_rate=.75, batch_size=600, offset=0,
                            alpha=3)
    dl_params['netflix'] = dict(learning_rate=.8, batch_size=4000, offset=0,
                                alpha=0.0016)
    cd_params = {'100k': dict(alpha=.1), '1m': dict(alpha=.03),
                 '10m': dict(alpha=.04),
                 'netflix': dict(alpha=.1)}

    if version in ['100k', '1m', '10m']:
        X = load_movielens(version)
        X_tr, X_te = train_test_split(X, train_size=0.75,
                                      random_state=random_state)
        X_tr = X_tr.tocsr()
        X_te = X_te.tocsr()
    elif version is 'netflix':
        X_tr = load(expanduser('~/spira_data/nf_prize/X_tr.pkl'))
        X_te = load(expanduser('~/spira_data/nf_prize/X_te.pkl'))

    cd_mf = ExplicitMF(n_components=60, max_iter=50, alpha=.1, normalize=True,
                       verbose=1, )
    dl_mf = DictCompleter(n_components=30, n_epochs=20, alpha=1.17, verbose=5,
                          batch_size=10000, detrend=True,
                          fit_intercept=True,
                          random_state=0,
                          learning_rate=.75,
                          backend='c')
    dl_mf_partial = DictCompleter(n_components=60, n_epochs=20, alpha=1.17,
                                  verbose=2,
                                  batch_size=10000, detrend=True,
                                  fit_intercept=True,
                                  random_state=0,
                                  learning_rate=.75,
                                  backend='c')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H'
                                                 '-%M-%S')
    if cross_val:
        subdir = 'benches_ncv'
    else:
        subdir = 'benches'
    output_dir = expanduser(join('~/output/recommender/', timestamp, subdir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alphas = np.logspace(-4, 2, 15)
    mf_list = [dl_mf_partial]
    dict_id = {cd_mf: 'cd', dl_mf: 'dl', dl_mf_partial: 'dl_partial'}
    names = {'cd': 'Coordinate descent', 'dl': 'Proposed online masked MF',
             'dl_partial': 'Proposed algorithm (with partial projection)'}

    if os.path.exists(join(output_dir, 'results_%s_%s.json' % (version,
                                                               random_state))):
        with open(join(output_dir, 'results_%s_%s.json' % (version,
                                                           random_state)),
                  'r') as f:
            results = json.load(f)
    else:
        results = {}

    for mf in mf_list:
        results[dict_id[mf]] = {}
        if not cross_val:
            if isinstance(mf, DictCompleter):
                mf.set_params(
                    learning_rate=dl_params[version]['learning_rate'],
                    batch_size=dl_params[version]['batch_size'],
                    alpha=dl_params[version]['alpha'])
            else:
                mf.set_params(alpha=cd_params[version]['alpha'])
        else:
            if isinstance(mf, DictCompleter):
                mf.set_params(
                    learning_rate=dl_params[version]['learning_rate'],
                    batch_size=dl_params[version]['batch_size'])
            if version != 'netflix':
                cv = ShuffleSplit(n_iter=3, train_size=0.66, random_state=0)
                mf_scores = Parallel(n_jobs=n_jobs, verbose=10, max_nbytes=None)(
                    delayed(single_fit)(mf, alpha, X_tr, cv) for alpha in
                    alphas)
            else:
                mf_scores = Parallel(n_jobs=n_jobs, verbose=10, max_nbytes=None)(
                    delayed(single_fit)(mf, alpha, X_tr, X_te,
                                        nested=False) for alpha in alphas)
            mf_scores = np.array(mf_scores).mean(axis=1)
            best_alpha_arg = mf_scores.argmin()
            best_alpha = alphas[best_alpha_arg]
            mf.set_params(alpha=best_alpha)

        cb = Callback(X_tr, X_te, refit=False)
        mf.set_params(callback=cb)
        mf.fit(X_tr)
        results[dict_id[mf]] = dict(name=names[dict_id[mf]],
                                    time=cb.times,
                                    rmse=cb.rmse)
        if cross_val:
            results[dict_id[mf]]['alphas'] = alphas.tolist()
            results[dict_id[mf]]['cv_alpha'] = mf_scores.tolist()
            results[dict_id[mf]]['best_alpha'] = mf.alpha

        with open(join(output_dir, 'results_%s_%s.json' % (version,
                                                           random_state)),
                  'w+') as f:
            json.dump(results, f)

        print('Done')


def single_fit(mf, alpha, X_tr, cv, nested=True):
    mf_cv = clone(mf)
    if isinstance(mf_cv, DictCompleter):
        mf_cv.set_params(n_epochs=2)
    else:
        mf_cv.set_params(max_iter=10)
    mf_cv.set_params(alpha=alpha)
    if nested:
        score = cross_val_score(mf_cv, X_tr, cv)
    else:
        X_te = cv
        mf_cv.fit(X_tr)
        score = [mf_cv.score(X_te)]
    return score


def plot_learning_rate(
        output_dir=expanduser('~/output/recommender/learning_rate')):
    with open(join(output_dir, 'results_netflix.json'), 'r') as f:
        data_netflix = json.load(f)
    with open(join(output_dir, 'results_10m.json'), 'r') as f:
        data_10m = json.load(f)
    min_time = 400
    for i, learning_rate in enumerate(
            sorted(data_netflix.keys(), key=lambda t: float(t))):
        this_data = data_netflix[learning_rate]
        min_time = min(this_data['time'][0], min_time)
    for i, learning_rate in enumerate(
            sorted(data_netflix.keys(), key=lambda t: float(t))):
        this_data = data_netflix[learning_rate]
        for j in range(len(this_data)):
            this_data['time'][j] -= this_data['time'][0] - min_time
    fig = plt.figure()
    # fig.subplots_adjust(right=0.7)
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
    for j, data in enumerate([data_10m, data_netflix]):
        ax[j] = fig.add_subplot(gs[j])
        # palette = sns.hls_palette(10, l=.4, s=.7)
        for i, learning_rate in enumerate(
                sorted(data.keys(), key=lambda t: float(t))):
            if float(learning_rate) > .6:
                this_data = data[learning_rate]
                ax[j].plot(np.linspace(0., 20, len(this_data['rmse'])),
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

    ax[0].set_xlim([.1, 20])
    ax[0].set_xticks([1, 10, 20])
    ax[0].set_xticklabels(['1', '10', '20'])
    ax[1].set_xlim([.1, 20])
    ax[1].set_xticks([.1, 1, 10, 20])
    ax[1].set_xticklabels(['.1', '1', '10', '20'])

    ax[0].annotate('MovieLens 10M', xy=(.95, .8), ha='right',
                   xycoords='axes fraction')
    ax[1].annotate('Netflix', xy=(.95, .8), ha='right',
                   xycoords='axes fraction')

    ax[0].set_ylim([0.795, 0.863])
    ax[1].set_ylim([0.93, 0.983])
    ax[0].legend(ncol=4, loc='upper left', bbox_to_anchor=(0., -.13),
                 fontsize=6, numpoints=1, columnspacing=.3, frameon=False)
    ax[0].annotate('Learning rate $\\beta$', xy=(1.6, -.38),
                   xycoords='axes fraction')
    ltext = ax[0].get_legend().get_texts()
    plt.setp(ltext, fontsize=7)
    plt.savefig(expanduser('~/output/icml/learning_rate.pdf'))


def plot_benchs(output_dir=expanduser('~/output/recommender/benches')):
    fig = plt.figure()

    fig.subplots_adjust(right=.9)
    fig.subplots_adjust(top=.915)
    fig.subplots_adjust(bottom=.12)
    fig.subplots_adjust(left=.08)
    fig.set_figheight(fig.get_figheight() * 0.66)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.5])

    ylims = {'100k': [.90, .96], '1m': [.865, .915], '10m': [.795, .868],
             'netflix': [.928, .99]}
    xlims = {'100k': [0.0001, 10], '1m': [0.1, 15], '10m': [1, 200],
             'netflix': [30, 4000]}

    names = {'dl_partial': 'Proposed \n(partial projection)',
             'dl': 'Proposed \n(full projection)',
             'cd': 'Coordinate descent'}
    for i, version in enumerate(['1m', '10m', 'netflix']):
        with open(join(output_dir, 'results_%s.json' % version), 'r') as f:
            data = json.load(f)
        ax_time = fig.add_subplot(gs[0, i])
        ax_time.grid()
        sns.despine(fig, ax_time)

        ax_time.spines['left'].set_color((.6, .6, .6))
        ax_time.spines['bottom'].set_color((.6, .6, .6))
        ax_time.xaxis.set_tick_params(color=(.6, .6, .6), which='both')
        # ax_time.tick_params(axis='x', which='major', pad=2)
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
            ax_time.xaxis.set_label_coords(1.12, -0.045)

        ax_time.grid()
        palette = sns.cubehelix_palette(3, start=0, rot=.5, hue=1, dark=.3,
                                        light=.7,
                                        reverse=False)
        color = {'dl_partial': palette[2], 'dl': palette[1], 'cd': palette[0]}
        for estimator in sorted(OrderedDict(data).keys()):
            this_data = data[estimator]
            ax_time.plot(this_data['time'], this_data['rmse'],
                         label=names[estimator], color=color[estimator],
                         linewidth=2,
                         linestyle='-' if estimator != 'cd' else '--')
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
    plt.savefig(expanduser('~/output/icml/rec_bench.pdf'))


if __name__ == '__main__':
    # compare_learning_rate('netflix', n_jobs=10)
    # plot_learning_rate()
    main('netflix', n_jobs=15, cross_val=True)
    # main('100k', n_jobs=1, cross_val=False)
    # main('1m', cross_val=True, n_jobs=15, random_state=0)
    # main('10m', n_jobs=15, cross_val=True, random_state=0)
    # for i in range(5):
    #     main('1m', cross_val=True, n_jobs=15, random_state=i)
    #     main('10m', n_jobs=15, random_state=i, cross_val=True)
    plot_benchs()
