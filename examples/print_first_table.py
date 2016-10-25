from os.path import expanduser
from tempfile import NamedTemporaryFile

import gridfs
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns
from bson import ObjectId
from matplotlib.lines import Line2D
from modl.plotting.fmri import display_maps
from modl.plotting.images import plot_patches
from pymongo import MongoClient
from sacred.experiment import Experiment

import matplotlib.patches as patches

plot_ex = Experiment('plot')


@plot_ex.config
def config():
    name = 'compare_reductions'


def get_connections(sub_db):
    # client = MongoClient('localhost', 27017)
    client = MongoClient('localhost', 27018)
    db = client[sub_db]
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs