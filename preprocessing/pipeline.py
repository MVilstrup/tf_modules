from extensions.assertions.checks import *
from collections import defaultdict

class Pipeline(object):

    def __init__(self, input_tensor, name="Pipeline"):
        assert tfTensor(input_tensor)
        self.name = name
        self.input = input_tensor
        self.pipeline = []

    def apply(self, ops, *args, **kargs):
        self.pipeline.append((ops, args, kargs))

    def compute(self):
        with tf.name_scope(self.name):
            _input = self.input
            for op, args, kargs in self.pipeline:
                _input = op(_input, *args, **kargs)

            return _input

    def split(self, name=None):
        new = self.copy()
        if name is not None:
            new.name = name

        return self, new

    def copy(self):
        new = Pipeline(self.input, name="{}_{}_split".format(self.name, len(self.pipeline)))
        new.pipeline = self.pipeline
        return new
