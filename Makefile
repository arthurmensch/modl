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

install-coverage:
	$(PYTHON) setup.py build_ext -D CYTHON_TRACE -D CYTHON_TRACE_NOGIL install

all: cython inplace

cython: $(CSRC)

clean:
	rm -f modl/impl/*.html
	rm -f `find modl -name "*.pyc"`
	rm -f `find modl -name "*.so"`
	rm -f $(CSRC)
	rm -f `find modl -name "*.cpp"`
	rm -f `find modl -name "*.pyx"  | sed s/.pyx/.html/g`
	rm -f `find modl -name "*.pyx"  | sed s/.pyx/.c/g`
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

download-data: datadir
	./misc/download.sh http://www.amensch.fr/data/modl_data.tar.bz2
	tar xvfj modl_data.tar.bz2
	mv modl_data/* $(DATADIR)
	rmdir modl_data
	rm modl_data.tar.bz2
