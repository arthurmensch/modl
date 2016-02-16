PYTHON ?= python
CYTHON ?= cython
PYTEST ?= py.test --pyargs
DATADIR=$(HOME)/modl_data

# Compilation...

CYTHONSRC= $(wildcard modl/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.c)

inplace:
	$(PYTHON) setup.py build_ext -i

in: inplace

all: cython inplace

cython: $(CSRC)

clean:
	rm -f modl/impl/*.html
	rm -f `find modl -name "*.pyc"`
	rm -f `find modl -name "*.so"`
	rm -rf htmlcov
	rm -rf build
	rm -rf coverage .coverage
	rm -rf .cache

%.c: %.pyx
	$(CYTHON) $<

# Tests...
#
test-code:
	$(PYTEST) --pyargs modl

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) --pyargs --cov=modl modl

test: test-code

# Datasets...
# from Mathieu Blondel
datadir:
	mkdir -p $(DATADIR)

download-movielens100k: datadir
	./download.sh http://www.mblondel.org/data/movielens100k.tar.bz2
	tar xvfj movielens100k.tar.bz2
	mv movielens100k $(DATADIR)

download-movielens1m: datadir
	./download.sh http://www.mblondel.org/data/movielens1m.tar.bz2
	tar xvfj movielens1m.tar.bz2
	mv movielens1m $(DATADIR)

download-movielens10m: datadir
	./download.sh http://www.mblondel.org/data/movielens10m.tar.bz2
	tar xvfj movielens10m.tar.bz2
	mv movielens10m $(DATADIR)
