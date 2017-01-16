import warnings
from os.path import join

import numpy as np
from task_predict_core import get_encodings, fit_lr, \
    get_components
from modl.datasets.hcp import fetch_hcp_rest, fetch_hcp_task
from modl.utils.system import get_cache_dirs, get_data_dirs
from nilearn.datasets import fetch_adhd
from sacred import Experiment
from sacred import Ingredient
from sklearn.externals.joblib import Memory
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

data_ing = Ingredient('data')

task_predict_ex = Experiment('task_predict', ingredients=[data_ing])


@data_ing.config
def config():
    n_subjects = 500
    test_size = 4


@task_predict_ex.config
def config():
    batch_size = 200
    learning_rate = 0.92
    method = 'masked'
    reduction = 10
    alpha = 1e-3
    n_epochs = 3
    verbose = 15
    n_jobs = 16
    smoothing_fwhm = 6
    n_components = 40
    seed = 20


@data_ing.capture
def get_datasets(n_subjects, test_size, _seed):
    rest_dataset = fetch_hcp_rest(n_subjects=n_subjects)
    mask_img = rest_dataset.mask
    data_dir = get_data_dirs()[0]
    rest_dataset = fetch_adhd(n_subjects=n_subjects)
    mask_img = join(data_dir, 'ADHD_mask', 'mask_img.nii.gz')
    task_dataset = fetch_hcp_task(n_subjects=n_subjects)
    train, test = next(
        ShuffleSplit(test_size=test_size, random_state=_seed).split(rest_dataset.func))
    return mask_img, rest_dataset, task_dataset, test, train


@task_predict_ex.automain
def run(alpha, batch_size, learning_rate, n_components, n_epochs, n_jobs,
        reduction, smoothing_fwhm, method, verbose, _seed):
    warnings.filterwarnings('ignore', module='sklearn.externals.joblib.logger',
                            category=DeprecationWarning)
    random_state = _seed
    # Rest and task are aligned
    mask_img, rest_dataset, task_dataset, test, train = get_datasets()
    train_rest_data = np.array(rest_dataset.func).tolist()
    # train_rest_data = reduce(add, train_rest_data)
    test_, train = train_test_split(np.arange(500), random_state=_seed,
                                    test_size=2)

    memory = Memory(cachedir=get_cache_dirs()[0], verbose=0)

    components = memory.cache(get_components)(
        alpha, batch_size, learning_rate, mask_img,
        n_components, n_epochs, n_jobs, reduction,
        smoothing_fwhm, method, train_rest_data, random_state, verbose)
    X_test, X_train, y_test, y_train = memory.cache(get_encodings)(
        components, mask_img, task_dataset, test, train, n_jobs)

    print(y_train)
    print(y_test)
    lr = memory.cache(fit_lr)(X_train, y_train)
    y_pred = lr.predict(X_test)

    score = np.sum(y_pred == y_test) / y_test.shape[0]
    return score