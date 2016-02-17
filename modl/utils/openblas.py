import ctypes
from ctypes.util import find_library

# enforces priority of hand-compiled OpenBLAS library over version in /usr/lib
# that comes from Ubuntu repos
try_paths = ['/opt/OpenBLAS/lib/libopenblas.so',
             '/lib/libopenblas.so',
             '/usr/lib/libopenblas.so.0',
             find_library('openblas')]
openblas_lib = None
for libpath in try_paths:
    try:
        openblas_lib = ctypes.cdll.LoadLibrary(libpath)
        break
    except OSError:
        continue
if openblas_lib is None:
    raise EnvironmentError('Could not locate an OpenBLAS shared library', 2)


def set_num_threads(n):
    """
    Set the current number of threads used by the OpenBLAS server
    """
    openblas_lib.openblas_set_num_threads(int(n))


def num_threads(n):
    """
    Set the OpenBLAS thread context:

        print "Before ", get_num_threads()

        with num_threads(n):
            print "In thread context: ", get_num_threads()

        print "After ", get_num_threads()

    """
    return ThreadContext(n)


class ThreadContext(object):

    def __init__(self, num_threads):
        self._old_num_threads = get_num_threads()
        self.num_threads = num_threads

    def __enter__(self):
        set_num_threads(self.num_threads)

    def __exit__(self, *args):
        set_num_threads(self._old_num_threads)


# these features were added very recently:
# <https://github.com/xianyi/OpenBLAS/commit/65a847cd361d33b4a65c10d13eefb11eb02f04d7>
try:
    openblas_lib.openblas_get_num_threads()
    def get_num_threads():
        """Get the current number of threads used by the OpenBLAS server
        """
        return openblas_lib.openblas_get_num_threads()
except AttributeError:
    def get_num_threads():
        """Dummy function (symbol not present in %s), returns -1.
        """ % openblas_lib._name
        return -1
    pass

try:
    openblas_lib.openblas_get_num_procs()
    def get_num_procs():
        """Get the total number of physical processors
        """
        return openblas_lib.openblas_get_num_procs()
except AttributeError:
    def get_num_procs():
        """Dummy function (symbol not present in %s), returns -1.
        """ % openblas_lib._name
        return -1
    pass