from os.path import join

from modl.datasets import get_data_dirs
from modl.datasets.hcp import fetch_hcp, INTERESTING_CONTRASTS

n_jobs = 10
batch_size = 1200
dataset = fetch_hcp()
imgs = dataset.contrasts
interesting_con = list(INTERESTING_CONTRASTS.keys())
imgs = imgs.loc[(slice(None), slice(None), interesting_con), :]
mask = dataset.mask

dump_dir = join(get_data_dirs()[0], 'raw', 'hcp', 'task')


