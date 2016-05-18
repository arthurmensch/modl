import fnmatch
import json
import os
from os.path import expanduser, join

import numpy as np
from nilearn.datasets import fetch_atlas_smith_2009
from nilearn.input_data import NiftiMasker
from sklearn.externals.joblib import delayed, Parallel
from sklearn.linear_model import Ridge

from modl.datasets.hcp import get_hcp_data

data_dir = expanduser('~/data')
n_test_records = 4


def objective_function(X, components, alpha=0.):
    """Score function based on explained variance

        Parameters
        ----------
        X: ndarray,
            Holds single subject data to be tested against components

        alpha: regularization

        Returns
        -------
        score: ndarray,
            Holds the score for each subjects. score is two dimensional if
            per_component = True
        """

    lr = Ridge(fit_intercept=False, alpha=alpha)
    lr.fit(components.T, X.T)
    residuals = X - lr.coef_.dot(components)
    return np.sum(residuals ** 2) + alpha * np.sum(lr.coef_ ** 2)


def load_data():
    with open(expanduser('~/data/HCP_unmasked/data.json'), 'r') as f:
        data = json.load(f)
        for this_data in data:
            this_data['array'] += '.npy'
        mask_img = expanduser('~/data/HCP_mask/mask_img.nii.gz')
    masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=4,
                         standardize=True)
    masker.fit()
    smith2009 = fetch_atlas_smith_2009()
    init = smith2009.rsn70
    dict_init = masker.transform(init)
    return masker, dict_init, sorted(data, key=lambda t: t['filename'])


def compute_objective_l1l2(X, masker, filename, alpha):
    print('Computing explained variance')
    components = masker.transform(filename)
    densities = np.sum(np.abs(components)) / np.sqrt(np.sum(components ** 2))
    exp_var = objective_function(X, components, alpha)
    return exp_var, densities


def analyse_dir(output_dir, X, masker):
    output_files = os.listdir(output_dir)
    records = []
    objectives = []
    l1l2s = []
    analysis = {}
    if os._exists(join(output_dir, 'analysis.json')):
        return
    try:
        with open(join(output_dir, 'results.json'), 'r') as f:
            results = json.load(f)
    except IOError:
        return

    reduction = int(results['reduction'])
    filenames = sorted(fnmatch.filter(output_files,
                                      'record_*.nii.gz'),
                       key=lambda t: int(t[7:-7]))
    timings = []
    for filename in filenames[::reduction]:
        record = int(filename[7:-7])
        timing = results['timings'][record]
        print('Record %i' % record)
        objective, density = compute_objective_l1l2(X, masker,
                                                    join(output_dir, filename),
                                                    alpha=results['alpha'])
        timings.append(timing)
        records.append(record)
        objectives.append(objective)
        l1l2s.append(density)

    order = np.argsort(np.array(records))
    objectives = np.array(objectives)[order].tolist()
    l1l2s = np.array(l1l2s)[order].tolist()
    records = np.array(records)[order].tolist()
    timings = np.array(timings)[order].tolist()
    analysis['records'] = records
    analysis['objectives'] = objectives
    analysis['densities'] = l1l2s
    analysis['timings'] = timings
    with open(join(output_dir, 'analysis.json'), 'w+') as f:
        json.dump(analysis, f)


def main(output_dir, n_jobs):
    dir_list = [join(output_dir, f) for f in os.listdir(output_dir) if
                os.path.isdir(join(output_dir, f))]

    mask, func_filenames = get_hcp_data(data_dir=data_dir, raw=True)

    masker = NiftiMasker(mask_img=mask, smoothing_fwhm=None,
                         standardize=False)
    masker.fit()

    test_data = func_filenames[(-n_test_records * 2)::2]

    n_samples, n_voxels = np.load(test_data[-1], mmap_mode='r').shape
    X = np.empty((n_test_records * n_samples, n_voxels))

    for i, this_data in enumerate(test_data):
        X[i * n_samples:(i + 1) * n_samples] = np.load(this_data,
                                                       mmap_mode='r')

    Parallel(n_jobs=n_jobs, verbose=1, temp_folder='/dev/shm')(
        delayed(analyse_dir)(dir_name, X, masker) for dir_name in dir_list)


if __name__ == '__main__':
    output_dir = expanduser('~/output/modl/hcp_new')
    main(output_dir, 3)
