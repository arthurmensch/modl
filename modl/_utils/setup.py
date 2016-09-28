from distutils.extension import Extension

import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('_utils', parent_package, top_path)

    extensions = [Extension('modl/_utils/enet_proj_fast',
                            sources=['modl/_utils/enet_proj_fast.pyx'],
                            include_dirs=[numpy.get_include()],
                            )]
    config.ext_modules += cythonize(extensions)

    config.add_subpackage('tests')
    config.add_subpackage('system')
    config.add_subpackage('randomkit')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
