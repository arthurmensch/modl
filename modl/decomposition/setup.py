from distutils.extension import Extension

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('decomposition', parent_package, top_path)

    extensions = [
        Extension('modl.decomposition.dict_fact_fast',
                  sources=['modl/decomposition/dict_fact_fast.pyx'],
                  include_dirs=[numpy.get_include()],
                  ),
        Extension('modl.decomposition.recsys_fast',
                  sources=['modl/decomposition/recsys_fast.pyx'],
                  include_dirs=[numpy.get_include()],
                  ),
    ]
    config.ext_modules += extensions

    config.add_subpackage('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
