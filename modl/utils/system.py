import os


def get_cache_dirs(cache_dir=None):
    """ Returns the directories in which modl stores its cache

    This is typically useful for the end-user to check where the cache is stored.

    Parameters
    ----------
    cache_dir: string, optional
        Path of the cache directory. Used to force cache storage in a specified
        location. Default: None

    Returns
    -------
    paths: list of strings
        Paths of the dataset directories.

    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :
    1. the keyword argument data_dir
    2. the global environment variable SHARED_CACHE
    3. the user environment variable CACHE
    4. modl_data in the user home folder
    """

    paths = []

    # Check data_dir which force storage in a specific location
    if cache_dir is not None:
        paths.extend(cache_dir.split(os.pathsep))

    # If data_dir has not been specified, then we crawl default locations
    if cache_dir is None:
        global_data = os.getenv('SHARED_CACHE')
        if global_data is not None:
            paths.extend(global_data.split(os.pathsep))

        local_data = os.getenv('CACHE')
        if local_data is not None:
            paths.extend(local_data.split(os.pathsep))

        paths.append(os.path.expanduser('~/cache'))
    return paths


def get_output_dir(data_dir=None):
    """ Returns the directories in which cogspaces store results.

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    paths: list of strings
        Paths of the dataset directories.

    Notes
    -----
    This function retrieves the datasets directories using the following
    priority :
    1. the keyword argument data_dir
    2. the global environment variable MODL_SHARED_DATA
    3. the user environment variable MODL_DATA
    4. modl_data in the user home folder
    """

    paths = []

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        return data_dir.split(os.pathsep)
    else:
        # If data_dir has not been specified, then we crawl default locations
        output_dir = os.getenv('OUTPUT_MODL_DIR')
        if output_dir is not None:
            return output_dir.split(os.pathsep)
    return os.path.expanduser('~/output/modl')



