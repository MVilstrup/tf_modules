import random
import numpy as np
import tensorflow as tf
from typing import Tuple

NPTuple = Tuple[np.ndarray]

class Generator:
    def __init__(self, config: object, distribution: list = None):
        # We make this class callable, which allows us to simply give it to tensorflow
        #self.__class__ = type(self.__class__.__name__, (self.__class__,), {})
        #self.__class__.__call__ = lambda x: self.generate()
        self.config = config
        self.stacked = []

        if distribution is not None:
            self.sample_distribution(distribution)

    def sample_distribution(self, distribution: list) -> None:
        assert sum(distribution) == 1., "Distribution should sum to 1, not {}".format(sum(distribution))
        assert len(distribution) == self.config.num_classes, "Distribution should match number of categories"
        assert isinstance(distribution, list), "Distribution should be a list of floats"

        # Stack each percentage, to create ranges from 0 to 1 for each category
        self.stacked = [sum(distribution[:i]) + d for i, d in enumerate(distribution)]

    def sample(self, category: int) -> NPTuple:
        raise NotImplementedError("Sample(category: int) -> tuple should be defined when inheriting this class")

    def _sample(self) -> tuple:
        # If no distribution has been defined,
        if self.sample:
            return self.sample(random.choice(range(self.config.num_classes)))

        # Choose a random value, which will act as the choice of action
        # This is done by checking which range the random value is in between.
        rnd = random.random()
        for i, val in enumerate(self.stacked):
            if rnd < val:
                return self.sample(i)

    def dtype(self) -> list:
        string = """\
        dtype() should be defined when inheriting this class.
        Returns: a tuple of tensorflow types, which is used by the dataset api to prepare for the data
            e.g. (tf.float32, tf.int32, tf.int32)
        """
        raise NotImplementedError(string)

    def shape(self) -> list:
        shapes = []
        for d in self._sample():
            if isinstance(d, np.ndarray):
                shapes.append(tf.TensorShape(d.shape))
            else:
                string = """\
                Shape expects a list/tuple of numpy arrays. 
                Either convert all data to numpy arrays, or overwrite this method
                """
                raise ValueError(string)

        return shapes

    def __call__(self, *args, **kwargs):
        return self

    def __next__(self):
        return self._sample()

    def __iter__(self):
        return self
