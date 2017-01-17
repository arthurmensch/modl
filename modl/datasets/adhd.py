def fetch_adhd(n_subjects=30, data_dir=None, url=None, resume=True,
               modl_data_dir=None,
               mask_url=None,
               verbose=1):
    dataset = nilearn_fetch_adhd(n_subjects=n_subjects,
                                 data_dir=data_dir, url=url, resume=resume,
                                 verbose=verbose)
    modl_data_dir = get_data_dirs(modl_data_dir)[0]
    mask_data_dir = join(modl_data_dir, 'adhd')
    if mask_url is None:
        mask_url = 'http://amensch.fr/data/adhd/mask_img.nii.gz'
    _fetch_file(mask_url, mask_data_dir, resume=resume)
    mask_img = join(mask_data_dir, 'mask_img.nii.gz')
    return Bunch(func=dataset.func, confounds=dataset.confounds,
                 phenotypic=dataset.phenotypic, description=dataset.description,
                 mask=mask_img)
