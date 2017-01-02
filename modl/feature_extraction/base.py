class BaseBatcher(object):
    def __init__(self, batch_size=10, random_state=None):
        self.batch_size = batch_size
        self.random_state = random_state

    def prepare(self, data_source):
        pass

    def generate_once(self):
        return
        yield

    def generate_single(self):
        return next(self.generate_once())
