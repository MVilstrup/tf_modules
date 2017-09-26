#!/usr/bin/python3

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim
import time

from tf_modules.utils import *
from tf_modules.assertions.checks import *
from tf_modules.metrics.base_metrics import BaseMetrics


class ClassificationMetrics(BaseMetrics):
    def __init__(self, config, predictions, targets):
        BaseMetrics.__init__(self, config)

        with tf.name_scope('Classification-Metrics'):
            self.predictions = tf.argmax(predictions, 1)
            self.targets = tf.argmax(targets, 1)

        self.validation_ops += [self.predictions, self.targets]
        self.test_ops += [self.predictions, self.targets]

        self.create_confusion_metrics()
        self.create_standard_metrics()

    def create_confusion_metrics(self):
        # Create a confusion matrix pr scope
        with tf.variable_scope("Metrics--Confusion-Matrix"):
            for scope in self.collections:
                batch_confusion = tf.confusion_matrix(self.targets,
                                                      self.predictions,
                                                      num_classes=self.config.num_classes,
                                                      name='batch_confusion')
                # Create an accumulator variable to hold the counts
                confusion = tf.Variable(tf.zeros([self.config.num_classes,
                                                  self.config.num_classes],
                                        dtype=tf.int32),
                                        name='confusion')

                # Create the update op for doing a "+=" accumulation on the batch
                confusion_update = confusion.assign(confusion + batch_confusion)

                # Cast counts to float so tf.summary.image renormalizes to [0,255]
                matrix = [1, self.config.num_classes, self.config.num_classes, 1]
                confusion_image = tf.reshape(tf.cast(confusion, tf.float32), matrix)

                tf.summary.image(scope.title(), confusion_image, collections=[scope])

                self.ops[scope].append(confusion_update)

    def reset(self):
        pass

    def create_standard_metrics(self):
        with tf.name_scope("Metric---Accuracy"):
            for scope in self.collections:
                # Calculate accuracy and error.
                accuracy, accuracy_up = tf.metrics.accuracy(self.targets, self.predictions)
                # Add accuracy to TensorBoard.
                tf.summary.scalar(scope.title(), accuracy, collections=[scope])
                tf.summary.histogram(scope.title(), accuracy, collections=[scope])
                self.ops[scope].append(accuracy_up)

        with tf.name_scope("Metric---Mean-Per-Class-Accuracy"):
            for scope in self.collections:
                # Calculate accuracy and error.
                mpk, mpk_up = tf.metrics.mean_per_class_accuracy(self.targets,
                                                                 self.predictions,
                                                                 self.config.num_classes)
                # Add mpk to TensorBoard.
                tf.summary.scalar(scope.title(), mpk, collections=[scope])
                tf.summary.histogram(scope.title(), mpk, collections=[scope])
                self.ops[scope].append(mpk_up)

    def _config_assertions(self):
        assertions = {"num_classes": positiveInt}
        self.config.assertions(assertions)
