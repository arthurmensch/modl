try:
    import mkl

    class ThreadContext(object):

        def __init__(self, num_threads):
            self._old_num_threads = mkl.get_max_threads()
            self.num_threads = num_threads

        def __enter__(self):
            mkl.set_num_threads(self.num_threads)

        def __exit__(self, *args):
            mkl.set_num_threads(self._old_num_threads)

except ImportError:
    class ThreadContext(object):
        def __init__(self, num_threads):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

def num_threads(n):
    """
    Set the mkl thread context:

        print "Before ", get_num_threads()

        with num_threads(n):
            print "In thread context: ", get_num_threads()

        print "After ", get_num_threads()

    """
    return ThreadContext(n)