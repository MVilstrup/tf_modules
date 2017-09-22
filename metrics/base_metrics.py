
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import time

from extensions.utils import *
from extensions.assertions.checks import *

class BaseMetrics(object):

    def __init__(self, config):
        self.config = config

        self.train_scope = "train"
        self.validation_scope = "validation"
        self.test_scope = "test"
        self.collections = [self.train_scope,
                            self.validation_scope,
                            self.test_scope]

        self.training_ops = []
        self.validation_ops = []
        self.test_ops = []

        self.ops = {}
        self.ops[self.train_scope] = self.training_ops
        self.ops[self.validation_scope] = self.validation_ops
        self.ops[self.test_scope] = self.test_ops

        # Ensure the config is as expected
        self._base_assertions()
        self._config_assertions()

    def step(self, sess, step, feed_dict, is_training=True, is_testing=False):
        if not hasattr(self, "train_summary"):
            self._initialize_summaries()

        if is_testing:
            summaries, _ = sess.run([self.test_summary, self.test_ops],
                                    feed_dict=feed_dict)
            self._write(self.test_scope, summaries, step)
        elif is_training:
            summaries, _ = sess.run([self.train_summary, self.training_ops],
                                    feed_dict=feed_dict)
            self._write(self.train_scope, summaries, step)
        else:
            summaries, _ = sess.run([self.validation_summary, self.validation_ops],
                                    feed_dict=feed_dict)
            self._write(self.validation_scope, summaries, step)

    def _initialize_summaries(self):
        self.train_summary = tf.summary.merge_all(self.train_scope)
        self.validation_summary = tf.summary.merge_all(self.validation_scope)
        self.test_summary = tf.summary.merge_all(self.test_scope)

    def _write(self, scope, summaries, step):
        writer = "{}_writer".format(scope)
        graph = self.config.graph if scope == self.train_scope else None
        if not hasattr(self, writer):
            # Create writers which will be used to log the summaries
            setattr(self, writer, self._create_writer(scope, graph))

        current_writer = getattr(self, writer)
        current_writer.add_summary(summaries, step)
        current_writer.flush()

    def add_scalar(self, name, tensor, collections=None):
        if collections is None:
            collections = self.collections

        found = False
        for c in collections:
            if c in self.ops:
                self.ops[c].append(tensor)
                found = True
        if not found:
            msg = "{} is not a valid scope, only 'train', 'test' and 'validation' is allowed".format(scope)
            raise ValueError(msg)

        tf.summary.scalar(name, tensor, collections=collections)

    def add_histogram(self, name, tensor, collections=None):
        pass

    def add_image(self, name, image, collections=None):
        pass

    def _add_to_all(self, op):
        for c in self.collections:
            self.ops[c].append(op)

    def _create_writer(self, scope, graph=None):
        return tf.summary.FileWriter('{}/{}_{}'.format(self.config.log_dir,
                                                        scope,
                                                        self.config.name),
                                                        graph)

    def __add__(self, other):
        if not isinstance(other, BaseMetrics):
            raise ValueError("Can only add two metrics together")

        for scope, ops in other.ops.items():
            if scope in self.ops:
                self.ops[scope] += ops

        return self

    def _config_assertions(self):
        print("You can include a config assertion in your metrics class to validate configuration")

    def _base_assertions(self):
        assertions = {"log_dir": existingStr,
                      "name": existingStr,
                      "graph": tfGraph}
        self.config.assertions(assertions)
