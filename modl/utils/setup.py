def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('utils', parent_package, top_path)

    config.add_subpackage('math')
    config.add_subpackage('tests')
    config.add_subpackage('randomkit')
    config.add_subpackage('recsys')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
