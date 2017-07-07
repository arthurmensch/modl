import sys
import numpy

from distutils.extension import Extension


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('randomkit', parent_package, top_path)
    libs = []
    if sys.platform == 'win32':
        libs.append('Advapi32')

    extensions = [Extension('modl.utils.randomkit.random_fast',
                            sources=['modl/utils/randomkit/random_fast.pyx',
                                     'modl/utils/randomkit/randomkit.c',
                                     'modl/utils/randomkit/distributions.c',
                                     ],
                            language="c++",
                            include_dirs=[numpy.get_include(),
                                          'modl/utils/randomkit'],
                            ),
                  Extension('modl.utils.randomkit.sampler',
                            sources=['modl/utils/randomkit/sampler.pyx'],
                            language="c++",
                            include_dirs=[numpy.get_include()]
                            )]
    config.ext_modules += extensions

    config.add_subpackage('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup, Extension

    setup(**configuration(top_path='').todict())
