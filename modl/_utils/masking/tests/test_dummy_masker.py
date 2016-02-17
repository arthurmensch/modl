import numpy as np
import pytest

from modl._utils.masking import DummyMasker
from modl.datasets.hcp import fetch_hcp_rest


@pytest.mark.slow
def test_dummy_masker():
    # Smoke test dummy masker
    data_dir = '/storage/data/HCP_unmasked'
    dummy_masker = DummyMasker(data_dir=data_dir)
    dummy_masker.fit()
    imgs = fetch_hcp_rest(data_dir='/storage/data', n_subjects=1).func
    data = dummy_masker.transform(imgs[:1])
    assert(len(data) == 1)
    single_data = data[0]
    print(single_data.shape)
    mask_img = dummy_masker.mask_img_
    mask_size = np.sum(mask_img.get_data() != 0)
    assert(mask_size == 212445)
    assert(single_data.shape == (1200, mask_size))
