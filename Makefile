PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
DATADIR=$(HOME)/spira_data

# Compilation...

CYTHONSRC= $(wildcard spira/impl/*.pyx)
CSRC= $(CYTHONSRC:.pyx=.c)

inplace:
	$(PYTHON) setup.py build_ext -i

all: cython inplace

cython: $(CSRC)

clean:
	rm -f spira/impl/*.html
	rm -f `find spira -name "*.pyc"`
	rm -f `find spira -name "*.so"`

%.c: %.pyx
	$(CYTHON) $<

# Tests...
#
test-code: in
	$(NOSETESTS) -s spira

test-coverage:
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=spira spira

test: test-code test-doc

# Datasets...
#
#
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
