PYTHON ?= python
PYTEST ?= py.test --pyargs
DATADIR=$(HOME)/modl_data

# Compilation...

in: inplace

inplace: clean-obj
	$(PYTHON) setup.py build_ext -i

inplace-coverage: clean-obj
	$(PYTHON) setup.py build_ext -i -D CYTHON_TRACE -D CYTHON_TRACE_NOGIL

install-coverage: clean-obj
	$(PYTHON) setup.py build_ext -D CYTHON_TRACE -D CYTHON_TRACE_NOGIL install

install: clean-obj
	$(PYTHON) setup.py install

# Flush objects when macro change
clean-obj:
	rm -rf dist
	rm -f `find modl -name "*.so"`

clean:
	rm -f `find modl -name "*.so"`
	rm -f `find modl -name "*.pyx"  | sed s/.pyx/.html/g`
	rm -f `find modl -name "*.pyx"  | sed s/.pyx/.c/g`
	rm -f `find modl -name "*.pyx"  | sed s/.pyx/.cpp/g`
	rm -rf htmlcov
	rm -rf build
	rm -rf coverage .coverage
	rm -rf .cache
	rm -rf modl.egg-info
	rm -rf dist

# Tests
#
test:
	$(PYTEST) --pyargs modl

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) --pyargs --cov=modl modl

# Data
#
datadir:
	mkdir -p $(DATADIR)

download-data: datadir
	./misc/download.sh http://www.amensch.fr/data/modl_data.tar.bz2
	tar xvfj modl_data.tar.bz2
	mv modl_data/* $(DATADIR)
	rmdir modl_data
	rm modl_data.tar.bz2
