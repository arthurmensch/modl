PYTHON ?= python
CYTHON ?= cython
PYTEST ?= py.test --pyargs
DATADIR=$(HOME)/modl_data
MRIDATADIR=$(HOME)/data

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
	rm -f $(CSRC)
	rm -f `find modl -name "*.cpp"`
	rm -f `find modl -name "*.pyx"  | sed s/.pyx/.html/g`
	rm -rf htmlcov
	rm -rf build
	rm -rf coverage .coverage
	rm -rf .cache
	rm -rf modl.egg-info
	rm -rf dist

# Tests...
#
test-code:
	$(PYTEST) --pyargs modl

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) --pyargs --cov=modl modl

test: test-code

datadir:
	mkdir -p $(DATADIR)
	mkdir -p $(MRIDATADIR)

download-hcp-extra: datadir
	./misc/download.sh http://www.amensch.fr/data/hcp_extra.tar.bz2
	tar xvfj netflix.tar.bz2
	mv movielens10m $(MRIDATADIR)
