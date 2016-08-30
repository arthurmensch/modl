from distutils.extension import Extension

import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('new', parent_package, top_path)

    extensions = [Extension('modl/new/dict_fact_fast_new',
                            sources=['modl/new/dict_fact_fast_new.pyx'],
                            include_dirs=[numpy.get_include(),
                                          'modl/new/randomkit'],
                            # extra_compile_args=['-fopenmp'],
                            # extra_link_args=['-fopenmp']
                            )]
    config.ext_modules += cythonize(extensions)

    config.add_subpackage('randomkit')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
