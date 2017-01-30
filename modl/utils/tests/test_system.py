# Adapted from nilearn
import os
import shutil
from tempfile import mkdtemp

import pytest
from numpy.testing import assert_equal

from modl.utils.system import get_cache_dirs
from modl.datasets import get_data_dirs


@pytest.fixture(scope="module")
def tmpdir():
    # create temporary dir
    tmp = mkdtemp()
    yield tmp
    if tmp is not None:
        shutil.rmtree(tmp)


def test_get_data_dir(tmpdir):
    # testing folder creation under different environments, enforcing
    # a custom clean install
    os.environ.pop('NILEARN_DATA', None)
    os.environ.pop('NILEARN_SHARED_DATA', None)

    expected_base_dir = os.path.expanduser('~/modl_data')
    data_dir = get_data_dirs()[0]
    assert_equal(data_dir, expected_base_dir)

    expected_base_dir = os.path.join(tmpdir, 'modl_data')
    os.environ['MODL_DATA'] = expected_base_dir
    data_dir = get_data_dirs()[0]
    assert_equal(data_dir, expected_base_dir)

    expected_base_dir = os.path.join(tmpdir, 'mdoln_shared_data')
    os.environ['MODL_SHARED_DATA'] = expected_base_dir
    data_dir = get_data_dirs()[0]
    assert_equal(data_dir, expected_base_dir)

    expected_base_dir = os.path.join(tmpdir, 'modl_data')
    os.environ.pop('MODL_DATA', None)
    os.environ.pop('MODL_SHARED_DATA', None)
    data_dir = get_data_dirs(expected_base_dir)[0]
    assert_equal(data_dir, expected_base_dir)


def test_get_data_dir(tmpdir):
    # testing folder creation under different environments, enforcing
    # a custom clean install
    os.environ.pop('CACHE', None)
    os.environ.pop('SHARED_CACHE', None)

    expected_base_dir = os.path.expanduser('~/cache')
    data_dir = get_cache_dirs()[0]
    assert_equal(data_dir, expected_base_dir)

    expected_base_dir = os.path.join(tmpdir, 'cache')
    os.environ['CACHE'] = expected_base_dir
    data_dir = get_cache_dirs()[0]
    assert_equal(data_dir, expected_base_dir)

    expected_base_dir = os.path.join(tmpdir, 'cache_shared')
    os.environ['SHARED_CACHE'] = expected_base_dir
    data_dir = get_cache_dirs()[0]
    assert_equal(data_dir, expected_base_dir)

    expected_base_dir = os.path.join(tmpdir, 'cache')
    os.environ.pop('CACHE', None)
    os.environ.pop('SHARED_CACHE', None)
    data_dir = get_cache_dirs(expected_base_dir)[0]
    assert_equal(data_dir, expected_base_dir)