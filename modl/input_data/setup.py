from distutils.extension import Extension

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('input_data', parent_package, top_path)

    extensions = [
                  Extension('modl.input_data.image_fast',
                            sources=['modl/input_data/image_fast.pyx'],
                            include_dirs=[numpy.get_include()]
                            ),
                  ]
    config.add_subpackage('tests')
    config.add_subpackage('fmri')

    config.ext_modules += extensions

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
