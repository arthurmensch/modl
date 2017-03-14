from tempfile import NamedTemporaryFile

from keras.engine import Model
from keras.models import load_model


def __setstate__(self, state):
    binary = state.pop('binary')
    with NamedTemporaryFile(dir='/tmp') as f:
        f.write(binary)
        new_self = load_model(f.name)
        self.__dict__ = new_self.__dict__


def __getstate__(self):
    with NamedTemporaryFile(dir='/tmp') as f:
        self.save(f.name)
        data = f.read()
    state = {'binary': data}
    return state

Model.__setstate__ = __setstate__
Model.__getstate__ = __getstate__
