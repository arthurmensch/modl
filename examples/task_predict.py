from modl import fMRIDictFact
from modl.datasets.fmri import load_rest_func

# dataset = fetch_hcp_task()
# imgs = [img for subject_imgs in dataset.func for img in subject_imgs]
# contrasts = [contrast for subject_contrasts in dataset.contrasts
#              for contrast in subject_contrasts]
# mask_img = dataset.mask
#
# masker = MultiNiftiMasker(smoothing_fwhm=3, mask_img=mask_img).fit()
dict_init = 'components.nii.gz'
train_data, test_data, mask = load_rest_func('adhd')
dict_fact = fMRIDictFact(dict_init=dict_init, n_epochs=0, mask=mask).fit()