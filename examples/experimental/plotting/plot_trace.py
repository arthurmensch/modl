import glob
import os
from os.path import join, expanduser

import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
from nilearn.plotting import plot_prob_atlas
from sklearn.externals.joblib import delayed, Parallel


def plot_4D_from_dir(output_dir, n_jobs=1):
    files = glob.glob(join(output_dir, '*.nii.gz'))
    Parallel(n_jobs=n_jobs, verbose=10)(delayed(plot_and_dump)(file) for file in files)


def plot_and_dump(file):
    dir, basename = os.path.split(file)
    basename = int(basename.replace(
        'record_', '').replace('.nii.gz', ''))
    fig = plt.figure()
    plot_prob_atlas(file, figure=fig)
    plt.savefig(join(dir, 'record_%04i.png' % basename))
    plt.savefig(join(dir, 'video_%04i.png' % (basename // 4)))
    plt.close(fig)

if __name__ == '__main__':
    plot_4D_from_dir(expanduser('~/output/modl/hcp'), n_jobs=40)
