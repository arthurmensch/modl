from keras.models import Sequential
from keras.layers import Dense, Activation

from modl.datasets import fetch_adhd
from modl.utils.system import get_cache_dirs
from nilearn.input_data import MultiNiftiMasker
from sklearn.externals.joblib import Memory

data = fetch_adhd()

imgs = data.rest

memory = Memory(cachedir=get_cache_dirs()[0])

masker = MultiNiftiMasker(mask_img=data.mask,
                          smoothing_fwhm=6,
                          standardize=True,
                          detrend=True,
                          memory=memory).fit()

n_voxels = masker.mask_img_.get_shape[3]

n_components = 32

model = Sequential()
model.add(Dense(n_components,
                name='projection',
                input_shape=(None, n_voxels)))
model.add(Activation('relu'))
model.add(Dense(n_voxels,
                name='reconstruction'))
model.compile(optimizer='adagrad',
              loss='mse')

for i, img in enumerate(imgs):
    X = masker.transform(X)
    loss = model.train_on_batch(X, X)
    print('Record %i, loss %.4f' % (img, loss))


