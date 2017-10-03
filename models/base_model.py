#!/usr/bin/python3

import tensorflow as tf

slim = tf.contrib.slim

from tf_modules.utils import *
from tf_modules.assertions.checks import (optionalStr,
                                         optionalTensor,
                                         tfVariable)

class BaseModelMeta(type):
    def __call__(cls, *args, **kwargs):
        error = lambda x,y,z: "The {} method should return {} not {}".format(x, y, z)

        instance = super().__call__(*args, **kwargs)
        return instance

class BaseModel(metaclass=BaseModelMeta):

    def __init__(self, config):
        #Create the global step for monitoring the learning_rate and training.
        self.config = config
        self._base_assertions()
        self._config_assertions()

    def _base_assertions(self):
        # List the assertions for the configuration
        assertions = {"save_path": optionalStr,
                      "log_dir": optionalStr,
                      "global_step": tfVariable}
        self.config.assertions(assertions)

    def _config_assertions(self):
        print("You can include configuration assertions to you model by defining a _config_assertions() method")
        return None

    def save(self, sess):
        saver = tf.train.Saver()
        mkpath(self.config.save_path)
        saver.save(sess, self.config.save_path)
        print("{} saved in file: {}".format(self.config.model_name, self.config.save_path))

    def load(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess,  self.config.checkpoint_name)
        print("{} loaded from file: {}".format(self.config.model_name, self.config.checkpoint_name))

    def visualize(self):
        with tf.Session(graph=self.config.graph) as sess:
            print("Initializing variables")
            sess.run(tf.global_variables_initializer())

            print("Creating Graph")
            tmp_def = rename_nodes(sess.graph_def, lambda s:"/".join(s.split('_', 1)))
            show_graph(tmp_def)
