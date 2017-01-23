import sys

from nilearn._utils.niimg import _safe_get_data
from sklearn.externals.joblib.func_inspect import filter_args
from sklearn.externals.joblib.hashing import NumpyHasher, Hasher


class NibabelHasher(NumpyHasher):
    def __init__(self, hash_name='md5', coerce_mmap=False):
        """
            Parameters
            ----------
            hash_name: string
                The hash algorithm to be used
            coerce_mmap: boolean
                Make no difference between np.memmap and np.ndarray
                objects.
        """
        NumpyHasher.__init__(self, hash_name=hash_name,
                             coerce_mmap=coerce_mmap)
        import nibabel as nibabel
        self.nibabel = nibabel

    def save(self, obj):
        if isinstance(obj, self.nibabel.Nifti1Image):
            # if obj.get_filename() is not None:
            #     obj = (obj.get_filename(), 'HASHED')
            # else:
            obj = (_safe_get_data(obj), obj.affine, obj.header, 'HASHED')
        NumpyHasher.save(self, obj)


def our_hash(obj, hash_name='md5', coerce_mmap=False):
    """ Quick calculation of a hash to identify uniquely Python objects
        containing numpy arrays.


        Parameters
        -----------
        hash_name: 'md5' or 'sha1'
            Hashing algorithm used. sha1 is supposedly safer, but md5 is
            faster.
        coerce_mmap: boolean
            Make no difference between np.memmap and np.ndarray
    """
    if 'nibabel' in sys.modules and 'numpy' in sys.modules:
        hasher = NibabelHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    elif 'numpy' in sys.modules:
        hasher = NumpyHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    else:
        hasher = Hasher(hash_name=hash_name)
    return hasher.hash(obj)


def our_get_argument_hash(self, *args, **kwargs):
    return our_hash(filter_args(self.func, self.ignore,
                                args, kwargs),
                    coerce_mmap=(self.mmap_mode is not None))


import sklearn.externals.joblib.memory as original_memory

original_memory.MemorizedFunc._get_argument_hash = our_get_argument_hash

Memory = original_memory.Memory
