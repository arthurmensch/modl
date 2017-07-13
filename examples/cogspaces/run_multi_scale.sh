#!/usr/bin/env bash

python decompose_rest.py with n_components=16 &
python decompose_rest.py with n_components=64 &
python decompose_rest.py with n_components=512 &