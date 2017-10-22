def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('modl', parent_package, top_path)

    config.add_subpackage('utils')
    config.add_subpackage('datasets')
    config.add_subpackage('plotting')
    config.add_subpackage('feature_extraction')
    config.add_subpackage('decomposition')
    config.add_subpackage('input_data')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(**configuration(top_path='').todict())
