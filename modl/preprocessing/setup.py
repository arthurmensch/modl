from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('preprocessing', parent_package, top_path)

    extensions = [
                  Extension('modl.preprocessing.image_fast',
                            sources=['modl/preprocessing/image_fast.pyx'],
                            include_dirs=[numpy.get_include()]
                            ),
                  ]
    config.ext_modules += cythonize(extensions)

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
