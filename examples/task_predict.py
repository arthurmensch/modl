from modl import fMRIDictFact
from modl.datasets.hcp import fetch_hcp_task
from nilearn.input_data import MultiNiftiMasker

dataset = fetch_hcp_task()
imgs = [img for subject_imgs in dataset.func for img in subject_imgs]
contrasts = [contrast for subject_contrasts in dataset.contrasts
             for contrast in subject_contrasts]
mask_img = dataset.mask

masker = MultiNiftiMasker(smoothing_fwhm=3, mask_img=mask_img).fit()
dict_init = 'components.nii.zg'
dict_fact = fMRIDictFact(dict_init=dict_init, n_epochs=0).fit(dict_init)