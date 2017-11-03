#!/usr/bin/python3

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

import multiprocessing as mp
import shutil
import inspect

from tf_modules.utils import *
from tf_modules.assertions.assertions import AssertionClass
from random import randint
import os

class Config(AssertionClass):

    def __init__(self,
                 name,
                 num_classes,
                 batch_size,
                 epoch_amount,
                 num_checkpoints=0,
                 log_dir=None,
                 save_folder=None):
        AssertionClass.__init__(self)

        self.model_name = name
        self.graph = tf.get_default_graph()
        self.global_step = get_or_create_global_step()
        self.is_training = tf.placeholder(tf.bool)

        self.num_classes = num_classes
        self.num_checkpoints = num_checkpoints

        self.epoch_amount = epoch_amount
        self.batch_size = batch_size

        self.log_dir = log_dir
        self.save_folder = save_folder
        self.save_path = "{}{}.ckpt".format(save_folder, name)
        tf.reset_default_graph()


    def add_data(self, train_data=[], validation_data=[], test_data=[], multiplier=1):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

        self.train_steps      = int(len(train_data)      * multiplier / self.batch_size) if self.batch_size > 0 else 0
        self.validation_steps = int(len(validation_data) * multiplier / self.batch_size) if self.batch_size > 0 else 0
        self.test_steps       = int(len(test_data)       * multiplier / self.batch_size) if self.batch_size > 0 else 0

        self.total_batches = self.train_steps * self.epoch_amount

        self.evaluate_at = int(self.train_steps / self.validation_steps) if self.validation_steps > 0 else None
        self.save_at = int(self.total_batches / self.num_checkpoints) if self.num_checkpoints > 0 else None

    def clear_log(self):
        if os.path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)
            os.mkdir(self.log_dir)

    def _attrs(self, max_length = 100):

        tuples = []
        for attr, value in sorted(self.__dict__.items()):
            if not str(attr).startswith('_') and not str(attr).endswith('_'):
                length = 0
                try:
                    length = len(value)
                except:
                    length = len(str(value))
                if length > max_length:
                    value = '{} (size: {})'.format(type(value).__name__, pretty_num(len(value)))
                if isinstance(value, int) or isinstance(value, float):
                    value = pretty_num(value)

                tuples.append((attr, value))

        return tuples

    def _funcs(self):
        _str = "\n\nFunctions:\n"
        methods = [m for m, x in inspect.getmembers(self, predicate=inspect.ismethod)]

        for f in methods:
            if not f.startswith("_"):
                count =  getattr(self, f).__code__.co_argcount
                args = [a for a in getattr(self, f).__code__.co_varnames[:count] if not "self" == str(a)]
                args = ", ".join(args)
                _str += pretty("\t{}({})".format(f, args))
        return _str

    @property
    def __weakref__(self):
        return self.__wrapped__.__weakref__

    def __str__(self):
        _str = "Attributes:\n"
        for attr, value in self._attrs():
            _str += pretty("\t{}:".format(attr), 25) + "{}".format(value)

        _str += self._funcs()
        return _str

    def _reset_learning_rate(self):
        if not hasattr(self, "lr_type"):
            return

        if self.lr_type == "exponential":
            self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                             self.global_step,
                                                             self.decay_steps,
                                                             self.decay_rate,
                                                             staircase=self.staircase,
                                                             name="learning-rate")

        elif self.lr_type == "decayed":
            self.learning_rate = tf.train.inverse_time_decay(self.initial_learning_rate,
                                                              self.global_step,
                                                              self.decay_rate,
                                                              name="learning-rate")
        elif self.lr_type != "standard":
            raise ValueError("Learning rate not set")


    def reset_tf_variables(self):
        if hasattr(self, "graph"):
            del self.graph
        self.graph = tf.get_default_graph()

        if hasattr(self, "global_step"):
            del self.global_step
        self.global_step = get_or_create_global_step()

        if hasattr(self, "learning_rate") and not isinstance(self.learning_rate, float):
            del self.learning_rate
        self._reset_learning_rate()

        if hasattr(self, "is_training") and not isinstance(self.is_training, bool):
            del self.is_training
            self.is_training = tf.placeholder(tf.bool)

        if not hasattr(self, "random_seed"):
            self.random_seed = randint(0, 999)

        tf.set_random_seed(self.random_seed)


    def set_exponential_learning_rate(self, initial_learning_rate, decay_times, decay_rate, staircase=False):
        self.lr_type = "exponential"
        self.decay_steps = self.total_batches / decay_times
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,
                                                         self.global_step,
                                                         self.decay_steps,
                                                         self.decay_rate,
                                                         staircase=self.staircase,
                                                         name="learning-rate")

    def set_decayed_learning_rate(self, initial_learning_rate, decay_times, decay_rate):
        self.lr_type = "decayed"
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = self.total_batches / decay_times
        self.decay_rate = decay_rate
        self.learning_rate = tf.train.inverse_time_decay(self.initial_learning_rate,
                                                          self.global_step,
                                                          self.decay_steps,
                                                          self.decay_rate,
                                                          name="learning-rate")

    def set_learning_rate(self, learning_rate):
        self.lr_type = "standard"
        self.learning_rate = learning_rate

    def set_name(self, name):
        self.model_name = name
        self.save_path = "{}{}".format(self.save_folder, self.model_name)


    def log(self, sess):
        tuples = self._attrs()



