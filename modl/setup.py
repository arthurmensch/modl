from distutils.extension import Extension

import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('modl', parent_package, top_path)

    extensions = [
        Extension('modl.dict_fact_fast',
                  sources=['modl/dict_fact_fast.pyx'],
                  include_dirs=[numpy.get_include()],
                  ),
        Extension('modl.recsys_fast',
                  sources=['modl/recsys_fast.pyx'],
                  include_dirs=[numpy.get_include()],
                  ),
    ]
    config.ext_modules += extensions

    config.add_subpackage('tests')
    config.add_subpackage('utils')
    config.add_subpackage('datasets')
    config.add_subpackage('plotting')
    config.add_subpackage('feature_extraction')
    config.add_subpackage('preprocessing')

    config.ext_modules = cythonize(config.ext_modules)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
