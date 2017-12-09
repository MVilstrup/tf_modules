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
        self.name = self.config.model_name

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
        print("{} saved in file: {}".format(self.name, self.config.save_path))

    def load(self, sess, checkpoint=None):
        checkpoint = checkpoint if checkpoint else self.config.checkpoint_file

        if hasattr(self, "trainable_layers") and self.trainable_layers:
            # We save the intersection of the trainable layers and the trainable variables
            trainable = list(set(self.trainable_layers) & set(tf.trainable_variables()))
            print('trainable layers:', len(trainable))
            saver = tf.train.Saver(var_list=trainable)
        else:
            saver = tf.train.Saver()

        if 'ckpt' in checkpoint:
            saver.restore(sess, checkpoint)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))

        print("{} loaded from file: {}".format(self.name, checkpoint))

    def visualize(self):
        with tf.Session(graph=self.config.graph) as sess:
            print("Initializing variables")
            sess.run(tf.global_variables_initializer())

            print("Creating Graph")
            tmp_def = rename_nodes(sess.graph_def, lambda s:"/".join(s.split('_', 1)))
            show_graph(tmp_def)
