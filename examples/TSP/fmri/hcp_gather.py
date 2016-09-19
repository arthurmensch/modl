import json
import pandas as pd

import matplotlib.pyplot as plt

import os

n_exp = len(os.listdir())

results = []
for i in range(n_exp):
    exp = json.load(open('experiment_%i/experiment.json' % i, 'r'))
    res = json.load(open('experiment_%i/callback.json' % i, 'r'))
    results.append(dict(**exp, **res))

df = pd.DataFrame(results)
fig, ax = plt.subplots(111)
for reduction, this_df in df.groupby('reduction'):
    for line in this_df.iterrows():
        ax.plot(df.ix[i, 'time'], df.ix[i, ''], label = df.idx[i, 'reduction'])
