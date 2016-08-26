import numpy

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('modl', parent_package, top_path)
    config.add_extension('dict_fact_fast',
                         sources=['dict_fact_fast.c'],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=['-fopenmp'],
                         extra_link_args=['-fopenmp']
                         )

    config.add_subpackage('tests')
    config.add_subpackage('_utils')
    config.add_subpackage('datasets')
    config.add_subpackage('externals')


    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
