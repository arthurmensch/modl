from math import sqrt

from modl.preprocessing.image import scale_patches
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, gen_batches

from .dict_fact import DictFact
from .feature_extraction.image import LazyCleanPatchExtractor


class ImageDictFact(BaseEstimator):
    methods = {'masked': {'G_agg': 'masked', 'Dx_agg': 'masked'},
               'dictionary only': {'G_agg': 'full', 'Dx_agg': 'full'},
               'gram': {'G_agg': 'masked', 'Dx_agg': 'masked'},
               # 1st epoch parameters
               'average': {'G_agg': 'average', 'Dx_agg': 'average'},
               'reducing ratio': {'G_agg': 'masked', 'Dx_agg': 'masked'}}

    settings = {'dictionary learning': {'comp_l1_ratio': 0,
                                        'code_l1_ratio': 1,
                                        'comp_pos': False,
                                        'code_pos': False,
                                        'with_std': True,
                                        'with_mean': True},
                'NMF': {'comp_l1_ratio': 0,
                        'code_l1_ratio': 1,
                        'comp_pos': True,
                        'code_pos': True,
                        'with_std': True,
                        'with_mean': False}}

    def __init__(self, method='masked',
                 setting='dictionary learning',
                 patch_size=(8, 8),
                 batch_size=100,
                 buffer_size=None,
                 n_components=50,
                 alpha=0.1,
                 learning_rate=0.92,
                 reduction=10,
                 n_epochs=1,
                 random_state=None,
                 callback=None,
                 max_patches=None,
                 verbose=0,
                 n_threads=1,
                 ):
        self.n_threads = n_threads
        self.verbose = verbose
        self.callback = callback
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.reduction = reduction
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.n_components = n_components
        self.batch_size = batch_size
        self.method = method
        self.setting = setting
        self.patch_size = patch_size
        self.buffer_size = buffer_size
        self.max_patches = max_patches

    def fit(self, image, y=None):
        self.random_state = check_random_state(self.random_state)

        method = ImageDictFact.methods[self.method]
        G_agg = method['G_agg']
        Dx_agg = method['Dx_agg']

        setting = ImageDictFact.settings[self.setting]
        comp_l1_ratio = setting['comp_l1_ratio']
        code_l1_ratio = setting['code_l1_ratio']
        comp_pos = setting['comp_pos']
        code_pos = setting['code_pos']
        with_std = setting['with_std']
        with_mean = setting['with_mean']

        if self.buffer_size is None:
            buffer_size = self.batch_size * 10
        else:
            buffer_size = self.buffer_size

        self.dict_fact_ = DictFact(n_epochs=self.n_epochs,
                                   random_state=self.random_state,
                                   n_components=self.n_components,
                                   comp_l1_ratio=comp_l1_ratio,
                                   learning_rate=self.learning_rate,
                                   comp_pos=comp_pos,
                                   code_pos=code_pos,
                                   batch_size=self.batch_size,
                                   G_agg=G_agg,
                                   Dx_agg=Dx_agg,
                                   reduction=self.reduction,
                                   code_alpha=self.alpha,
                                   code_l1_ratio=code_l1_ratio,
                                   callback=self._callback,
                                   verbose=self.verbose,
                                   n_threads=self.n_threads)

        if self.verbose:
            print('Preparing patch extraction')
        patch_extractor = LazyCleanPatchExtractor(
            patch_size=self.patch_size, max_patches=self.max_patches,
            random_state=self.random_state)
        patch_extractor.fit(image)

        n_patches = patch_extractor.n_patches_
        self.patch_shape_ = patch_extractor.patch_shape_

        if self.verbose:
            print('Fitting dictionary')
        init_patches = patch_extractor.partial_transform(batch=
                                                         self.n_components)
        init_patches = _flatten_patches(init_patches, with_std=with_std,
                                        with_mean=with_mean, copy=False)
        self.dict_fact_.prepare(n_samples=n_patches, X=init_patches)
        for i in range(self.n_epochs):
            if self.verbose:
                print('Epoch %i' % (i + 1))
            if i >= 1:
                if self.verbose:
                    print('Shuffling dataset')
                permutation = self.dict_fact_.shuffle()
                patch_extractor.shuffle(permutation)
            buffers = gen_batches(n_patches, buffer_size)
            if self.method == 'gram' and i == 2:
                self.dict_fact_.set_params(G_agg='full', Dx_agg='average')
            if self.method == 'reducing ratio':
                reduction = 1 + (self.reduction - 1) / sqrt(i + 1)
                self.dict_fact_.set_params(reduction=reduction)
            for j, buffer in enumerate(buffers):
                buffer_size = buffer.stop - buffer.start
                patches = patch_extractor.partial_transform(batch=buffer)
                patches = _flatten_patches(patches, with_mean=with_mean,
                                           with_std=with_std, copy=False)
                self.dict_fact_.partial_fit(patches, buffer)
        return self

    def transform(self, patches):
        with_std = ImageDictFact.settings[self.setting]['with_std']
        with_mean = ImageDictFact.settings[self.setting]['with_mean']

        patches = _flatten_patches(patches, with_mean=with_mean,
                                   with_std=with_std, copy=True)
        return self.dict_fact_.transform(patches)

    def score(self, patches):
        with_std = ImageDictFact.settings[self.setting]['with_std']
        with_mean = ImageDictFact.settings[self.setting]['with_mean']

        patches = _flatten_patches(patches, with_mean=with_mean,
                                   with_std=with_std, copy=True)
        return self.dict_fact_.score(patches)

    @property
    def n_iter_(self):
        # Property for callback purpose
        return self.dict_fact_.n_iter_

    @property
    def components_(self):
        # Property for callback purpose
        components_shape = (self.n_components,) + self.patch_shape_
        return self.dict_fact_.components_.reshape(
            components_shape)

    def _callback(self, *args):
        if self.callback is not None:
            self.callback(self)


def _flatten_patches(patches, with_mean=True,
                     with_std=True, copy=False):
    n_patches = patches.shape[0]
    patches = scale_patches(patches, with_mean=with_mean,
                            with_std=with_std, copy=copy)
    patches = patches.reshape((n_patches, -1))
    return patches
