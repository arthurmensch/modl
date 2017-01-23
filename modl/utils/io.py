import functools
import nibabel
from nibabel import Nifti1Image as NibabelNifti1Image
from nibabel import load as nibabel_load
import numpy as np
import os

def load(filename, **kwargs):
    img = nibabel_load(filename, **kwargs)
    img.__class__ = Nifti1Image
    return img

class Nifti1Image(NibabelNifti1Image):
    def __getstate__(self):
        if self.get_filename() is not None:
            filename = self.get_filename()
            stat = os.stat(filename)
            last_modified = stat.st_mtime
            state = {'filename': filename,
                     'last_modified': last_modified}
        else:
            state = {'dataobj': np.asarray(self._dataobj),
                     'header': self.header,
                     'affine': self.affine,
                     'extra': self.extra}
        return state

    def __setstate__(self, state):
        if 'filename' in state:
            print('unpickling %s' % state['filename'])
            new_self = Nifti1Image.from_filename(state['filename'])
        else:
            new_self = Nifti1Image(dataobj=state['dataobj'],
                                   affine=state['affine'],
                                   header=state['header'],
                                   extra=state['extra'])
        self.__dict__ = new_self.__dict__


def monkey_patch_nifti_image(safe=True):
    nibabel.load = load