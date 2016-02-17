import numpy

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('_utils', parent_package, top_path)

    config.add_extension('enet_proj_fast',
                         sources=['enet_proj_fast.c'],
                         include_dirs=[numpy.get_include()])

    config.add_subpackage('tests')
    config.add_subpackage('system')
    config.add_subpackage('masking')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
