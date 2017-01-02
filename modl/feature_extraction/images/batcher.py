import numpy as np
from sklearn.feature_extraction.image import extract_patches
from sklearn.utils import check_random_state, gen_batches

from ...feature_extraction.base import BaseBatcher
from .clean import clean_mask


class ImageBatcher(BaseBatcher):
    def __init__(self, patch_shape=(8,), batch_size=10, random_state=None,
                 clean=True,
                 max_samples=None):
        BaseBatcher.__init__(self, random_state=random_state,
                             batch_size=batch_size)
        self.patch_shape = patch_shape
        self.clean = clean
        self.max_samples = max_samples

    def prepare(self, image):
        if len(self.patch_shape) == 1:
            patch_shape = (self.patch_shape, self.patch_shape)
        if len(self.patch_shape) == 2:
            patch_shape = (self.patch_shape[0],
                                 self.patch_shape[1], image.shape[2])
        self.patches_ = extract_patches(image, patch_shape=patch_shape)
        self.random_state_ = check_random_state(self.random_state)

        if self.clean:
            mask = clean_mask(self.patches_, image)
        else:
            mask = np.ones(self.patches_.shape[:3], dtype=bool)
        # 3d index of each patch
        self.indices_3d_ = np.c_[np.where(mask)][:self.max_samples]
        n_samples = self.indices_3d_.shape[0]
        # 1d index of each patch
        self.indices_1d_ = np.arange(n_samples)

    def generate_once(self):
        n_samples = self.indices_3d_.shape[0]
        permutation = self.random_state_.permutation(n_samples)
        self.indices_1d_ = self.indices_1d_[permutation]
        self.indices_3d_ = self.indices_3d_[permutation]
        batches = gen_batches(n_samples, self.batch_size)
        for batch in batches:
            these_indices_3d = list(self.indices_3d_[batch].T)
            these_indices_1d = self.indices_1d_[batch]
            yield self.patches_[these_indices_3d], these_indices_1d

    @property
    def n_samples_(self):
        return self.indices_3d_.shape[0]