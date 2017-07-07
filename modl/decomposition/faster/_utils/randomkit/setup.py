import sys
import numpy

from distutils.extension import Extension

from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('randomkit', parent_package, top_path)
    libs = []
    if sys.platform == 'win32':
        libs.append('Advapi32')

    extensions = [Extension('modl.decomposition.faster._utils.'
                            'randomkit.random_fast',
                            sources=['modl/decomposition/faster/_utils'
                                     '/randomkit/random_fast.pyx',
                                     'modl/decomposition/faster/_utils'
                                     '/randomkit/randomkit.c',
                                     'modl/decomposition/faster/_utils'
                                     '/randomkit/distributions.c',
                                     ],
                            include_dirs=[numpy.get_include(),
                                          'modl/decomposition/faster/_utils'
                                          '/randomkit'],
                            )]
    config.add_subpackage('tests')

    config.ext_modules += extensions

    config.add_data_files('modl/decomposition/faster/_utils'
                          '/randomkit/randomkit.h')
    config.add_data_files('modl/decomposition/faster/_utils'
                          '/randomkit/distributions.h')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
