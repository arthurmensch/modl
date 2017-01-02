import numpy as np
from sklearn.feature_extraction.image import extract_patches
from sklearn.utils import check_random_state, gen_batches

from ...feature_extraction.base import BaseBatcher
from .clean import clean_mask


class ImageBatcher(BaseBatcher):
    def __init__(self, patch_shape=(8,), batch_size=10, random_state=None,
                 clean=True,
                 normalize=False,
                 center=False,
                 max_samples=None):
        BaseBatcher.__init__(self, random_state=random_state,
                             batch_size=batch_size)
        self.patch_shape = patch_shape
        self.clean = clean
        self.max_samples = max_samples

        self.normalize = normalize
        self.center = center

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
        self.indices = np.c_[np.where(mask)]
        n_samples = self.indices.shape[0]
        selection = self.random_state_.permutation(n_samples)[:self.max_samples]
        self.n_samples_ = selection.shape[0]
        self.indices = self.indices[selection]
        self.indices_1d = np.arange(n_samples)

    def generate_once(self):
        batches = gen_batches(self.n_samples_, self.batch_size)
        for batch in batches:
            these_indices_3d = list(self.indices[batch].T)
            patches = self.patches_[these_indices_3d]
            if self.center:
                patches -= np.mean(patches, axis=(1, 2))[:,
                           np.newaxis, np.newaxis, :]
            if self.normalize:
                std = np.sqrt(np.sum(patches ** 2,
                                     axis=(1, 2)))
                std[std == 0] = 1
                patches /= std[:, np.newaxis, np.newaxis, :]
            batch_size = patches.shape[0]
            patches = patches.reshape((batch_size, -1))
            yield patches, self.indices_1d[batch]

    def shuffle(self):
        n_samples = self.indices.shape[0]
        permutation = self.random_state_.permutation(n_samples)
        self.indices = self.indices[permutation]
        self.indices_1d = self.indices_1d[permutation]