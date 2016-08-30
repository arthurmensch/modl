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

    extensions = [Extension('modl/new/randomkit/random_fast',
                            sources=['modl/new/randomkit/random_fast.pyx',
                                     'modl/new/randomkit/randomkit.c'],
                            include_dirs=[numpy.get_include(),
                                          'modl/new/randomkit'],
                            )]
    config.ext_modules += cythonize(extensions)

    config.add_subpackage('tests')
    config.add_data_files('randomkit.h')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
