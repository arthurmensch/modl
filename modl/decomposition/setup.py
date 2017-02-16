from distutils.extension import Extension

import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('components', parent_package, top_path)

    extensions = [
        Extension('modl.components.dict_fact_fast',
                  sources=['modl/components/dict_fact_fast.pyx'],
                  include_dirs=[numpy.get_include()],
                  ),
        Extension('modl.components.recsys_fast',
                  sources=['modl/components/recsys_fast.pyx'],
                  include_dirs=[numpy.get_include()],
                  ),
    ]
    config.ext_modules += extensions

    config.add_subpackage('tests')

    config.ext_modules = cythonize(config.ext_modules, nthreads=4)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
