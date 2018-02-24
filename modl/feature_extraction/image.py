import numpy as np
from ..input_data.image import clean_mask, fill
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.image import extract_patches
from sklearn.utils import check_random_state


class LazyCleanPatchExtractor(BaseEstimator):
    def __init__(self, patch_size=None,
                 random_state=None,
                 max_patches=None):
        """
        Patch extractor that handles images with partial data,
        represented by -1. Extracted patches are fully known. Patches are
        materialized in memory on demand, using transform or partial_transform
        (useful for large images e.g., hyperspectral images).

        Parameters
        ----------
        patch_size: (int, int)
            Size of the patches that will be extracted
        random_state: np.random.RandomState, int or None,
            Randomness control
        max_patches: int or None,
            Maximum number of patches to extract
        """

        self.patch_size = patch_size
        self.max_patches = max_patches

        self.random_state = random_state

    def fit(self, X, y=None):
        self.random_state = check_random_state(self.random_state)
        i_h, i_w, n_channels = X.shape
        if self.patch_size is None:
            patch_size = i_h // 10, i_w // 10
        else:
            patch_size = self.patch_size
        patch_shape = (patch_size[0], patch_size[1], n_channels)
        self.patches_ = extract_patches(X, patch_shape=patch_shape)

        clean = np.all(X != -1)
        if not clean:
            self.indices_3d = clean_mask(self.patches_, X)
        else:
            self.indices_3d = fill(*self.patches_.shape[:3])
        n_samples = self.indices_3d.shape[0]
        selection = self.random_state.permutation(n_samples)[:self.max_patches]
        self.indices_3d = self.indices_3d[selection]

        return self

    def partial_transform(self, X=None, batch=None):
        if X is not None:
            self.fit(X)
        if batch is None:
            return self.transform()
        elif isinstance(batch, int):
            batch = slice(0, batch)
        these_indices = list(self.indices_3d[batch].T)
        patches = self.patches_[these_indices]
        return patches

    def transform(self, X=None):
        if X is not None:
            self.fit(X)
        patches = self.patches_[list(self.indices_3d.T)]
        return patches

    def shuffle(self, permutation=None):
        if permutation is None:
            n_samples = self.indices_3d.shape[0]
            permutation = self.random_state.permutation(n_samples)
        self.indices_3d = self.indices_3d[permutation]

    @property
    def n_patches_(self):
        return self.indices_3d.shape[0]

    @property
    def patch_shape_(self):
        return self.patches_.shape[-3:]
