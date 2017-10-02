import tensorflow as tf
from tf_modules.assertions.checks import *

from tf_modules.models import BaseModel
import numpy as np

class VGG19Old(BaseModel):

    def __init__(self, config, should_be_extended=False):
        BaseModel.__init__(self, config)

        if config.use_initial_weights:
            self.data_dict = np.load(config.weight_file, encoding='latin1').item()
        else:
            self.data_dict = {}

        self.var_dict = {}
        self.trainable = []
        self.should_be_extended = should_be_extended

        with tf.name_scope("VGG19"):
            self.inputs_placeholder = tf.placeholder(tf.float32,
                                          shape=[None, config.width, config.height, config.channels],
                                          name="train-input")

            if not should_be_extended:
                self.targets_placeholder = tf.placeholder(tf.int32,
                                              shape=[None, config.num_classes],
                                              name="train-labels")

            rgb_scaled = self.inputs_placeholder

            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

            VGG_MEAN = [103.939, 116.779, 123.68]
            bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                                            green - VGG_MEAN[1],
                                            red - VGG_MEAN[2]])

            self.conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
            self.pool1   = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
            self.pool2   = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
            self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
            self.pool3   = self.max_pool(self.conv3_4, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
            self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
            self.pool4   = self.max_pool(self.conv4_4, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
            self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
            self.pool5   = self.max_pool(self.conv5_4, 'pool5')

            self.fc6     = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
            self.relu6   = tf.nn.relu(self.fc6)

            self.relu6 = tf.cond(self.config.is_training,
                                 lambda: tf.nn.dropout(self.relu6, config.dropout),
                                 lambda: self.relu6)

            self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)
            self.relu7 = tf.cond(self.config.is_training,
                                 lambda: tf.nn.dropout(self.relu7, config.dropout),
                                 lambda: self.relu7)

            self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")

            if not should_be_extended:
                self.logits = self.fc_layer(self.fc8, 1000, config.num_classes, "logits")
                self.preds = tf.nn.softmax(self.logits, name="prob")

    def trainable_layers(self):
        return set(self.trainable)

    def extension_layer(self):
        return self.fc8

    def inputs(self):
        return self.inputs_placeholder

    def predictions(self):
        if hasattr(self, "preds"):
            return self.preds
        return None

    def targets(self):
        if hasattr(self, "targets_placeholder"):
            return self.targets_placeholder
        return None

    def avg_pool(self, bottom, name):
        self.trainable.append(name)
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        self.trainable.append(name)
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        self.trainable.append(name)
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        self.trainable.append(name)
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        checks = [self.data_dict is not None,
                  name in self.data_dict]

        if self.config.reset_trainable:
            checks.append(name not in self.config.trainable)

        if self.should_be_extended:
            checks.append(name != 'fc8')

        if False not in checks:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        var = tf.Variable(value, name=var_name)
        self.var_dict[(name, idx)] = var

        assert var.get_shape() == initial_value.get_shape()

        return var

    def _config_assertions(self):
        # List the assertions for the configuation of the model
        assertions = {"dropout":  percentage,
                      "height": positiveInt,
                      "width":    positiveInt,
                      "num_classes": positiveInt,
                      "trainable": strList,
                      "is_training": tfTensor,
                      "channels": lambda x: isinstance(x, int) and x == 3}

        self.config.assertions(assertions)
        self.config.condition(cond=("use_initial_weights", isTrue),
                              assertion=("weight_file",  optionalStr))
        self.config.condition(cond=("trainable", nonEmptyList),
                              assertion=("reset_trainable",  boolean))
