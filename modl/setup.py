from distutils.extension import Extension

import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('modl', parent_package, top_path)

    extensions = [Extension('modl/dict_fact_old_fast',
                            sources=['modl/dict_fact_old_fast.pyx'],
                            include_dirs=[numpy.get_include(),
                                          'modl/_utils/randomkit'],
                            extra_compile_args=['-fopenmp'],
                            extra_link_args=['-fopenmp']
                            ),
                  Extension('modl/dict_fact_fast',
                            sources=['modl/dict_fact_fast.pyx'],
                            ),
                  ]
    config.ext_modules += cythonize(extensions)

    config.add_subpackage('tests')
    config.add_subpackage('_utils')
    config.add_subpackage('datasets')
    config.add_subpackage('plotting')
    config.add_subpackage('streaming')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
