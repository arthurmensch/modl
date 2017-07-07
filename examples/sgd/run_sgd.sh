#!/usr/bin/env bash

python decompose_rest_sgd.py with step_size=0.00001 &
python decompose_rest_sgd.py with step_size=0.0001 &
python decompose_rest_sgd.py with step_size=0.001 &
python decompose_rest_sgd.py with step_size=0.01 &
python decompose_rest_sgd.py with step_size=0.1 &
python decompose_rest_sgd.py with step_size=1 &
