from time import time

import matplotlib.pyplot as plt
from numpy.random import RandomState
from sklearn.datasets import fetch_olivetti_faces

from modl.dict_fact import DictMF

n_row, n_col = 3, 4
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

###############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)


###############################################################################
# Plot a sample of the input data

plot_gallery("First centered Olivetti faces", faces_centered[:n_components])

###############################################################################
# Do the estimation and plot it

center = True
estimator = DictMF(n_components=n_components, batch_size=3,
                   reduction=3, l1_ratio=0, alpha=0.001, max_n_iter=1000,
                   full_projection=True,
                   verbose=3)
# For comparison
# estimator = MiniBatchSparsePCA(n_components=n_components, alpha=0.1,
#                                n_iter=1000, batch_size=3,
#                                random_state=rng)
name = 'MODL'

print("Extracting the top %d %s..." % (n_components, name))
t0 = time()
data = faces
if center:
    data = faces_centered
estimator.fit(data)
train_time = (time() - t0)
print("done in %0.3fs" % train_time)
if hasattr(estimator, 'cluster_centers_'):
    components_ = estimator.cluster_centers_
else:
    components_ = estimator.components_
if hasattr(estimator, 'noise_variance_'):
    plot_gallery("Pixelwise variance",
                 estimator.noise_variance_.reshape(1, -1), n_col=1,
                 n_row=1)
plot_gallery('%s - Train time %.1fs' % (name, train_time),
             components_[:n_components])

plt.show()
