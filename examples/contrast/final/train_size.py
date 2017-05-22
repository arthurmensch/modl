from os.path import expanduser

from matplotlib.cm import get_cmap
from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import numpy as np


client = MongoClient()
db = client.amensch

idx = pd.IndexSlice

import collections
def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def fetch_results():
    runs = db.predict_contrast_multi
    multi_dataset_res = runs.aggregate(
        [{"$match": {"info.master_id": {"$in": [825, 930]},
                     }},
         {'$project': {'dropout': '$config.dropout_latent',
                       'dropout_input': '$config.dropout_input',
                       'dropout_latent': '$config.dropout_latent',
                       'seed': '$config.seed',
                       'geometric_reduction': '$config.geometric_reduction',
                       'latent_dim': '$config.latent_dim',
                       'train_size': '$config.train_size.archi',
                       'score': '$info.score',
                       'lr': '$config.lr',
                       'mix_batch': '$config.mix_batch',
                       'depth_weight': '$config.depth_weight',
                       'alpha': '$config.alpha',
                       'time': '$config.start_time',
                       "datasets": "$config.datasets"}}
         ])
    multi_dataset_res = list(multi_dataset_res)
    for this_res in multi_dataset_res:
        if 'score' in this_res:
            new_score = flatten(this_res['score'])
            for key in new_score:
                this_res[key] = new_score[key]
            mean_score = np.mean(np.array(
                [list(this_res['score'][kind]['test'].values()) for kind in
                 ['dataset', 'task']]))
            this_res['mean_score'] = mean_score
            this_res.pop('score')
        if this_res['train_size'] is None:
            this_res['train_size'] = 'none'
        this_res['datasets'] = '__'.join(this_res['datasets'])
    df = pd.DataFrame(multi_dataset_res)
    df['latent_dim'] = df['latent_dim'].fillna('none')
    df = df.set_index([ 'datasets', 'train_size', 'seed'])
    df.sort_index(inplace=True)
    df_agg = df.groupby(level=['datasets', 'train_size']).aggregate(['mean', 'std'])

    total = {"archi": 39, 'brainomics': 47, 'camcan': 302}
    datasets = ['archi', 'brainomics', 'camcan']
    res = {}
    for dataset in datasets:
        no_transfer = df_agg.loc[dataset, 'dataset_test_%s' % dataset]
        transfer = df_agg.loc['%s__hcp' % dataset, 'dataset_test_%s' % dataset]
        train_sizes = transfer.index.get_level_values(
            'train_size').values.copy()
        train_sizes[-1] = total[dataset]
        index = pd.Index(train_sizes, name='train_size')
        this_res = pd.concat([no_transfer, transfer],
                             keys=['no_transfer', 'transfer'],
                             axis=1)
        this_res.index = index
        res[dataset] = this_res
    res = pd.concat([res[dataset] for dataset in datasets], keys=datasets,
                    names=['dataset'], axis=0)

    res.to_csv(expanduser('~/nips/train_size.csv'))


def plot():
    datasets = ['archi', 'brainomics', 'camcan']
    res = pd.read_csv(expanduser('~/nips/train_size.csv'), index_col=[0, 1],
                      header=[0, 1])
    dataset_name = {'archi': 'Archi', 'brainomics': 'Brainomics',
                    'camcan': 'CamCan'}
    fig, axes = plt.subplots(1, 3, figsize=(5.5015, 1.))
    fig.subplots_adjust(right=.98, left=0.08, bottom=.195, top=0.86)
    label = {# 'transfer_4': 'Transfer from all datasets',
             'transfer': 'Multi-dataset classification',
             'no_transfer': 'Single dataset classification'}
    flatui = get_cmap('Vega10').colors
    flatui = flatui[3:]
    for i, dataset in enumerate(datasets):
        data = res.loc[dataset]
        ax = axes[i]
        for j, exp in enumerate(['no_transfer', 'transfer']):
            if exp == 'transfer':
                if dataset == 'camcan':
                    exp_ = 'transfer_4'
                else:
                    exp_ = 'transfer_2'
            else:
                exp_ = exp
            y = data[(exp_, 'mean')]
            std = data[(exp_, 'std')]
            x = data.index.get_level_values('train_size').values
            ax.fill_between(x, y - .95 * std, y + .95 * std, alpha=0.3,
                            interpolate=True, linewidth=0,
                            color=flatui[j])
            ax.plot(x, y, label=label[exp], color=flatui[j])
        if i == 0:
            ax.set_ylabel('Test accuracy')
            ax.annotate('Train \n subjects', fontsize=8,
                        xy=(-0.25, -0.13), ha='left', va='center',
                        xycoords='axes fraction')
        ax.annotate(dataset_name[dataset], xy=(.5, 0.2), ha='center',
                    va='center', xycoords='axes fraction')
        if dataset == 'camcan':
            ax.set_xlim([0, 200])
            ax.set_xticks([5, 30, 100, 200])
            ax.set_ylim([0.35, 0.68])
        else:
            ax.set_xlim([2, x[-1]])
            ax.set_xticks(x)
            if dataset == 'archi':
                ax.set_ylim([.6, .86])
            else:
                ax.set_ylim([.6, .93])
            ax.set_yticks([0.7, 0.8, 0.9])
        if i == 2:
            ax.legend(bbox_to_anchor=(-2, .88), ncol=3, loc='lower left',
                      frameon=False)
        sns.despine(fig, ax=ax)
    plt.savefig(expanduser('~/nips/train_size.pdf'))

if __name__ == '__main__':
    plot()
