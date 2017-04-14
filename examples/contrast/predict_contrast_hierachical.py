import os
from os.path import join

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.externals.joblib import load
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from modl.datasets import get_data_dirs
from modl.hierarchical import make_model, init_tensorflow
from modl.model_selection import StratifiedGroupShuffleSplit

idx = pd.IndexSlice

predict_contrast = Experiment('predict_contrast_hierachical')
collection = predict_contrast.path

global_artifact_dir = join(get_data_dirs()[0], 'pipeline', 'contrast',
                           'prediction_hierachical')

# observer = MongoObserver.create(db_name='amensch', collection=collection)
# predict_contrast.observers.append(observer)


# @predict_contrast.config
# def config():
reduced_dir = join(get_data_dirs()[0], 'pipeline', 'contrast', 'reduced')

datasets = ['hcp', 'archi']
test_size = dict(hcp=0.1, archi=0.5, la5c=0.5, brainomics=0.5)
n_subjects = dict(hcp=None, archi=None, la5c=None, brainomics=None)
dataset_weight = dict(hcp=1, archi=1, la5c=1, brainomics=1)
train_size = None

validation = True

alpha = 0.0001
latent_dim = 200
activation = 'linear'
dropout_input = 0.25
dropout_latent = 0.9
batch_size = 100
optimizer = 'adam'
epochs = 100

depth_probs = [0., 0.5, 0.5]

# projection = True

n_jobs = 24
verbose = 2
seed = 10


# @predict_contrast.automain
# def run(alpha,
#         latent_dim,
#         n_subjects,
#         test_size,
#         train_size,
#         dropout_input,
#         dropout_latent,
#         activation,
#         datasets,
#         depth_probs,
#         reduced_dir,
#         optimizer,
#         batch_size,
#         epochs,
#         verbose,
#         n_jobs,
#         _run,
#         _seed):
artifact_dir = join(global_artifact_dir,
                    '10', '_artifacts')
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)

if verbose:
    print('Fetch data')

X = []
for dataset in datasets:
    this_reduced_dir = join(reduced_dir, dataset)
    this_X = load(join(this_reduced_dir, 'Xt.pkl'), mmap_mode='r')
    subjects = this_X.index.get_level_values('subject'). \
        unique().values.tolist()
    subjects = subjects[:n_subjects[dataset]]
    this_X = this_X.loc[idx[subjects]]
    X.append(this_X)
X = pd.concat(X, keys=datasets, names=['dataset'])
X.sort_index(inplace=True)

# Building numpy array
y = np.zeros((X.shape[0], 3), dtype=np.int32)
le_dict = {}
for i, label in enumerate(['dataset', 'task', 'contrast']):
    le = LabelEncoder()
    y[:, i] = le.fit_transform(X.index.get_level_values(label).values)
    le_dict[label] = le

label_pool = np.vstack({tuple(row) for row in y})
indices = np.argsort(label_pool[:, -1])
label_pool = label_pool[indices]
# d is only used at test time. 1: per dataset. 2: per task
d = np.ones(y.shape[0])

lbin = LabelBinarizer()
contrasts = y[:, -1]
contrasts_oh = lbin.fit_transform(contrasts)
n_features = X.shape[1]
x = X.values

dataset_length = X.iloc[:, 0].groupby(level='dataset').transform('count').values
sample_weight = 1. / dataset_length
sample_weight /= np.min(sample_weight)

# Cross validation folds
cv = StratifiedGroupShuffleSplit(stratify_levels='dataset',
                                 group_name='subject',
                                 test_size=test_size,
                                 train_size=train_size,
                                 n_splits=1,
                                 random_state=0)
train, test = next(cv.split(X))

init_tensorflow(n_jobs=n_jobs, debug=False)

model = make_model(n_features,
                   alpha=alpha,
                   latent_dim=latent_dim,
                   activation=activation,
                   dropout_input=dropout_input,
                   dropout_latent=dropout_latent,
                   optimizer=optimizer,
                   label_pool=label_pool,
                   seed=0,
                   depth_probs=depth_probs,
                   shared_supervised=False)
model.fit(x=[x[train], y[train], d[train]],
          y=contrasts_oh[train], sample_weight=sample_weight[train],
          batch_size=batch_size,
          epochs=epochs)
# use only y[:, :d] as input
y_pred_oh = model.predict(x=[x, y, d])

y_pred = lbin.inverse_transform(y_pred_oh)
true_contrast = le_dict['contrast'].inverse_transform(contrasts)
predicted_contrast = le_dict['contrast'].inverse_transform(y_pred)
prediction = pd.DataFrame({'true_label': true_contrast,
                           'predicted_label': predicted_contrast},
                          index=X.index)
prediction = pd.concat([prediction.iloc[train],
                        prediction.iloc[test]],
                       names=['fold'], keys=['train', 'test'])
prediction.to_csv('prediction.csv')
prediction.to_csv(join(artifact_dir, 'prediction.csv'))
model.save(join(model, 'model.keras'))

