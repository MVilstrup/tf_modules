import random
import numpy as np
import tensorflow as tf
from typing import Tuple

NPTuple = Tuple[np.ndarray]

class Generator:
    def __init__(self, config: object):
        # We make this class callable, which allows us to simply give it to tensorflow
        self.config = config

    def sample(self) -> NPTuple:
        raise NotImplementedError("Sample() -> tuple should be defined when inheriting this class")

    def dtype(self) -> list:
        string = """\
        dtype() should be defined when inheriting this class.
        Returns: a tuple of tensorflow types, which is used by the dataset api to prepare for the data
            e.g. (tf.float32, tf.int32, tf.int32)
        """
        raise NotImplementedError(string)

    def shape(self) -> list:
        shapes = []
        for d in self.sample():
            if isinstance(d, np.ndarray):
                shapes.append(tf.TensorShape(d.shape))
            else:
                string = """\
                Shape expects a list/tuple of numpy arrays.
                Either convert all data to numpy arrays, or overwrite this method
                """
                raise ValueError(string)

        return tuple(shapes)

    def __call__(self, *args, **kwargs):
        return self

    def __next__(self):
        return self.sample()

    def __iter__(self):
        return self
