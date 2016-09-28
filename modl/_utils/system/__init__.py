import os

def get_cache_dirs(cache_dir=None):
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
