# MODL: Massive Online Dictionary Learning

[![Travis](https://travis-ci.org/arthurmensch/modl.svg?branch=master)](https://travis-ci.org/arthurmensch/modl)
[![Coveralls](https://coveralls.io/repos/github/arthurmensch/modl/badge.svg?branch=master)](https://coveralls.io/github/arthurmensch/modl?branch=master)

This python package ([webpage](https://github.com/arthurmensch/modl)) allows to perform sparse / dense matrix factorization
 on fully-observed/missing data very efficiently, by leveraging random subsampling with online learning.
It is able to factorize matrices of terabyte scale with hundreds of components in the latent space in a few hours. 

The stochastic-subsampled online matrix factorization (SOMF) algorithm is an order or magnitude faster than online matrix factorization  (OMF) on large datasets.

![Benchmark](assets/compare.jpg)

It provides [https://github.com/scikit-learn/scikit-learn](scikit-learn) compatible estimators that fully implements the stochastic-subsampled
online matrix factorization (SOMF) algorithms. This package allows to reproduce the experiments and figures from the papers cited in reference.

## Installing from source with pip

Installation from source is simple. In a command prompt:

```
git clone https://github.com/arthurmensch/modl.git
cd modl
pip install -r requirements.txt
pip install .
cd $HOME
py.test --pyargs modl
```

*This package is only tested with Python 3.5+ !*

## Core code

The package essentially provides three estimators:

- `DictFact`, that computes a matrix factorization from Numpy arrays
- `fMRIDictFact`, that computes sparse spatial maps from fMRI images
- `ImageDictFact`, that computes a patch dictionary from an image
- `RecsysDictFact`, that allows to predict score from a collaborative filtering approach


## Examples

### fMRI decomposition

A fast running example that decomposes a small dataset of resting-fmri data into a 70 components map is provided

```
python examples/decompose_fmri.py
```

It can be adapted for running on the 2TB HCP dataset, by changing the source parameter into 'hcp' (you will need to download the data first)

### Hyperspectral images

A fast running example that extracts the patches of a HD image can be run from

```
python examples/decompose_image.py
```

It can be adapted to run on AVIRIS data, changing the image source into 'aviris' in the file.

### Recommender systems

Our core algorithm can be run to perform collaborative filtering very efficiently:

```
python examples/recsys_compare.py
```

You will need to download datasets beforehand:

```
make download-movielens1m
make download-movielens10m
```

## Contributions

Please feel free to report any issue and propose improvements on github.

## References

This package implements the two following papers:

>Arthur Mensch, Julien Mairal, Bertrand Thirion, Gaël Varoquaux.
[Stochastic Subsampling for Factorizing Huge Matrices](https://hal.archives-ouvertes.fr/hal-01431618v1), in IEEE Transactions on Signal Processing, 2018

>Arthur Mensch, Julien Mairal, Bertrand Thirion, Gaël Varoquaux.
[Dictionary Learning for Massive Matrix Factorization](https://hal.archives-ouvertes.fr/hal-01308934v2), in Proceedings of the International Conference
 on Machine Learning, 2016
 

### Related projects

  - [spira](https://github.com/mblondel/spira) is a python library to perform collaborative filtering based on coordinate descent. It serves as the baseline for recsys experiments - we hard included it for simplicity.
  - [scikit-learn](https://github.com/scikit-learn/scikit-learn) is a python library for machine learning. It serves as the basis of this project.
  - [nilearn](https://github.com/nilearn/nilearn) is a neuro-imaging library that we wrap in our fMRI related estimators.
  - [cogspaces](https://cogspaces.github.io) uses the dictionaries learned on large fMRI datasets to perform multi-study decoding.

## Authors

Licensed under simplified BSD.

Arthur Mensch, 2015 - present

