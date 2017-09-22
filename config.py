#!/usr/bin/python3

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

import multiprocessing as mp
import shutil
import copy
import inspect

from extensions.utils import *
from extensions.assertions.assertions import AssertionClass
from random import randint

class Config(AssertionClass):

    def __init__(self,
                 name,
                 num_classes,
                 batch_size,
                 epoch_amount,
                 num_checkpoints,
                 log_dir=None,
                 save_folder=None):
        AssertionClass.__init__(self)

        self.name = name
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

    def add_data_files(self, train_files=[], validation_files=[], test_files=[]):
        self.train_files = train_files
        self.validation_files = validation_files
        self.test_files = test_files

        self.train_steps      = int(len(train_files) / self.batch_size) if self.batch_size > 0 else 0
        self.validation_steps = int(len(validation_files) / self.batch_size) if self.batch_size > 0 else 0
        self.test_steps       = int(len(test_files) / self.batch_size) if self.batch_size > 0 else 0

        self.total_batches = self.train_steps * self.epoch_amount

        self.evaluate_at = int(self.train_steps / self.validation_steps) if self.validation_steps > 0 else None
        self.save_at = int(self.total_batches / self.num_checkpoints) if self.num_checkpoints > 0 else None

    def clear_log(self):
        if os.path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)
            os.mkdir(self.log_dir)

    def _attrs(self, max_length = 100):
        _str = "Attributes:\n"

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
                _str += pretty("\t{}:".format(attr), 25) + "{}".format(value)
        return _str

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
        _str = self._attrs()
        _str += self._funcs()
        return _str

    def _reset_learning_rate(self):
        if not hasattr(self, "lr_type"):
            raise ValueError("Learning rate not set")

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
        if hasattr(self, "global_step"):
            del self.global_step
        if hasattr(self, "learning_rate"):
            del self.learning_rate
        if hasattr(self, "is_training"):
            del self.is_training

        self.graph = tf.get_default_graph()

        if not hasattr(self, "random_seed"):
            self.random_seed = randint(0, 999)

        tf.set_random_seed(self.random_seed)
        self.global_step = get_or_create_global_step()
        self._reset_learning_rate()
        self.is_training = tf.placeholder(tf.bool)


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
        self.name = name
        self.save_path = "{}{}".format(self.save_folder, self.name)


    def clean_files(self):
        with mp.Pool(mp.cpu_count()) as pool:
            print("Checking Training Files")
            train_removed = pool.map(remove_if_bad, train_files)
            print("Removed from Train", len([f for f in train_removed if not f]))

            print("Checking Validation Files")
            validation_removed = pool.map(remove_if_bad, validation_files)
            print("Removed from Validation", len([f for f in validation_removed if not f]))

            print("Checking Test Files")
            test_removed = pool.map(remove_if_bad, test_files)
            print("Removed from Test", len([f for f in test_removed if not f]))
