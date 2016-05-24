from os.path import expanduser

from modl.datasets.hcp import prepare_hcp_raw_data

prepare_hcp_raw_data(data_dir=expanduser('~/data'))