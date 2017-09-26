#!/usr/bin/python3

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
slim = tf.contrib.slim
import time

from tf_modules.utils import *
from tf_modules.models.base_model import BaseModel
from tf_modules.assertions.checks import *

class TrainableModelMeta(type):
        def __call__(cls, *args, **kwargs):
            instance = super().__call__(*args, **kwargs)
            instance._loss()
            instance._train()
            return instance

class AddMeta(metaclass=TrainableModelMeta): pass

class TrainableModel(resolve(BaseModel, AddMeta)):

    def __init__(self, config):
        BaseModel.__init__(self, config)
        AddMeta.__init__(self)

        #Create the global step for monitoring the learning_rate and training.
        self.config = config
        self._trainable_assertions()

    def _loss(self):
        raise ValueError("_loss() function should be created when inheriting TrainableModel")

    def _train(self):
        raise ValueError("_train() function should be created when inheriting TrainableModel")

    def _metric_step(self, sess, step, feed_dict, is_training, is_testing=False):
        msg = """
        _metric_step(sess, feed_dict, is_training, is_testing=False)
        Should be created when inheriting TrainableModel
        """
        print(msg)

    def predictions(self):
        raise ValueError("predictions() function should be created when inheriting TrainableModel")

    def targets(self):
        raise ValueError("targets() function should be created when inheriting TrainableModel")

    def _trainable_assertions(self):
        # List the assertions for the configuration
        assertions = {"is_training": tensorOrBool,
                      "log_dir": existingStr,
                      "global_step": tfVariable}
        self.config.assertions(assertions)

    def step(self, sess, feed_dict, is_training=None):
        if is_training is not None:
            training = is_training
        elif tfTensor(self.config.is_training):
            training = sess.run(self.config.is_training, feed_dict=feed_dict)
        else:
            training = self.config.is_training

        if training:
            self._train_step(sess, feed_dict)
        else:
            self._evaluate_step(sess, feed_dict)

    def _train_step(self, sess, feed_dict):
        _ , step = sess.run([self._train(), self.config.global_step], feed_dict=feed_dict)

        # Every fifth step we make summaries
        if step % 5 == 0:
            self._metric_step(sess, step, feed_dict, is_training=True, is_testing=False)


    def _evaluate_step(self, sess, feed_dict):
        step = sess.run(self.config.global_step, feed_dict=feed_dict)
        self._metric_step(sess, step, feed_dict, is_training=False, is_testing=False)

    def _check(self, sess):
        # Check to see if the model should initiate early stopping
        pass
