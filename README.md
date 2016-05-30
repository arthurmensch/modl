# MODL: Massive Online Dictionary Learning

[![Travis](https://travis-ci.org/arthurmensch/modl.svg?branch=master)](https://travis-ci.org/arthurmensch/modl)
[![Coveralls](https://coveralls.io/repos/github/arthurmensch/modl/badge.svg?branch=master)](https://coveralls.io/github/arthurmensch/modl?branch=master)

This package implements our ICML'16 paper:

>Arthur Mensch, Julien Mairal, Bertrand Thirion, Gaël Varoquaux.
Dictionary Learning for Massive Matrix Factorization. International Conference
 on Machine Learning, Jun 2016, New York, United States. 2016

It allows to perform sparse / dense matrix factorization on fully-observed/missing data very efficiently, by leveraging random sampling with online learning.

Reference paper is available on [HAL](https://hal.archives-ouvertes.fr/hal-01308934) / [arxiv](http://arxiv.org/abs/1605.00937). This package allows to reproduce the
 experiments and figures from the papers.

More importantly, it provides [https://github.com/scikit-learn/scikit-learn](scikit-learn) compatible
 estimators that fully implements the proposed algorithms.

## Installing from source with pip

Installation from source is simple In a command prompt:

```
git clone https://github.com/arthurmensch/modl.git
cd modl
pip install -r requirements.txt
pip install .
cd $HOME
py.test --pyargs modl
```

## Examples

Two simple examples runs out-of-the box. Those are a good basis for understanding the API of `modl` estimators.
  - ADHD (rfMRI) sparse decomposition, relying on [nilearn](https://github.com/nilearn/nilearn)
  ```
  python examples/adhd_decompose.py
  ```
  - Movielens (User/Movie ratings) prediction
   ```
  python examples/recsys_predict.py
  ```

For Movielens example, you will need to download the dataset, from [spira repository](https://github.com/mblondel/spira).
```
make download-movielens10m
```

## Experiments

### Recommender systems

Recommender systems experiments can be reproduced running the following command in the root repository.

```
python examples/experimental/recsys/recsys_compare.py
```

You will need to download datasets beforehand:

```
make download-movielens1m
make download-movielens10m
make download-netflix
```

### HCP decomposition

You will need to retrieve the S500 release of the [HCP dataset](http://www.humanconnectome.org/data/) in some way
 beforehand. You may use the public S3 bucket, order filled hard-drives, or download it directly.

Edit `$HCPLOCATION` in the `Makefile` and run
```
make hcp
```
to create symlinks and download a useful mask.

The HCP experiment can be reproduced as such:
```
# unmask data
python examples/experiment/fmri/hcp_prepare.py
# compare methods
python examples/experiment/fmri/hcp_compare.py
# analyse convergence
python examples/experiment/fmri/hcp_analysis.py
# plot results
python examples/experiment/fmri/hcp_plot.py
```

By default, results will be available in `$HOME/output/modl`

## References

Related projects :
  - [spira](https://github.com/mblondel/spira) is a python library to perform collaborative filtering based on coordinate descent. It serves as the baseline for recsys experiments.
  - [scikit-learn](https://github.com/scikit-learn/scikit-learn) is a python library for machine learning. It serves as the basis of this project.
  - [nilearn](https://github.com/nilearn/nilearn) is a neuro-imaging library that we wrap in our fMRI related estimators.

## Author

Arthur Mensch, 2015-