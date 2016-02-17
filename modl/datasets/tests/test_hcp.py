from modl.datasets.hcp import DummyMasker, fetch_hcp_rest


def test_dummy_masker():
    # Smoke test dummy masker
    data_dir = '/storage/data/HCP_unmasked'
    dummy_masker = DummyMasker(data_dir=data_dir)
    dummy_masker.fit()
    imgs = fetch_hcp_rest(data_dir='/storage/data')
    data = dummy_masker.transform(imgs[0])
    data = dummy_masker.transform(imgs[:2])