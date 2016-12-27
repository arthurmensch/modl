from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('streaming', parent_package, top_path)

    extensions = [
                  Extension('modl/streaming/images_fast',
                            sources=['modl/streaming/images_fast.pyx'],
                            include_dirs=[numpy.get_include(),
                                          'modl/_utils/randomkit'],
                            ),
                  ]
    config.ext_modules += cythonize(extensions)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
