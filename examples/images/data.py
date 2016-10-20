from tempfile import NamedTemporaryFile, TemporaryFile

import atexit
from joblib import Memory
from modl.datasets import get_data_dirs
import os
from os.path import join, expanduser

from modl.datasets.images import load_images
from sacred.ingredient import Ingredient

data_ing = Ingredient('data')


@data_ing.config
def config():
    source = 'aviris'
    scale = 1
    gray = False


@data_ing.capture
def load_data(source, scale, gray):
    return load_images(source, scale=scale,
                       gray=gray, memory=Memory(cachedir='None'))