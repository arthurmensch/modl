#!/usr/bin/env bash

python plot.py with adhd
python plot.py with hcp
python plot.py with aviris

python plot.py with adhd AB_agg='async'
python plot.py with hcp AB_agg='async'
python plot.py with aviris AB_agg='async'