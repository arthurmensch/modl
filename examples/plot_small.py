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

@plot_ex.named_config
def adhd():
    sub_db = 'fmri'
    exp_name = 'compare_adhd'
    name = 'compare_adhd'
    status = 'COMPLETED'
    ylim_zoom = [1e-1, 2e-1]
    ylim = [21000, 31000]
    xlim = [10, 1000]
    xlim_zoom = [100, 1000]
    AB_agg = 'full'
    oid = None

@plot_ex.capture
def get_connections(sub_db):
    client = MongoClient('localhost', 27017)
    # client = MongoClient('localhost', 27018)
    db = client[sub_db]
    fs = gridfs.GridFS(db, collection='default')
    return db.default.runs, fs


@plot_ex.automain
def plot(exp_name, oid, status, xlim, ylim, ylim_zoom,
         xlim_zoom, AB_agg, name):
    db, fs = get_connections()

    exp = db.find({'status': status}).sort('_id', -1)[0]
    fig, ax = plt.subplots(1, 1)
    ax.plot(exp['info']['time'], exp['info']['score'])
    plt.savefig('plot_small.pdf')