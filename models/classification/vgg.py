"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import re
from tf_modules.models import BaseModel
from tf_modules.assertions.checks import *

slim = tf.contrib.slim


def vgg_arg_scope(weight_decay=0.0005):
    """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

class VGG(BaseModel):
    def __init__(self, config, name, url):
        self.model_path = 'model'

        self.name = name
        self.url = url
        self.config = config

        self._config_assertions()

        if not self.config.checkpoint_file:
            self.config.checkpoint_file = '{}/{}.ckpt'.format(self.model_path, self.name)

        if self.config.use_initial_weights:
            self.maybe_download_weights()

    def maybe_download_weights(self):
        if not os.path.exists(self.config.checkpoint_file):
            if not os.path.isdir(self.model_path):
                os.mkdir(self.model_path)

            print('Downloading Original Weights')

            file_name = self.url.split('/')[-1]
            os.system("curl -O '{}'".format(self.url))
            os.system("tar -xf '{}'".format(file_name))
            os.rename('{}.ckpt'.format(self.name), self.config.checkpoint_file)
            os.remove(file_name)

    def extend(self):
        return self.last_layer

    def endpoints(self):
        return self.all_endpoints

    def restore(self, should_be_extended):
        # Define the few layers which we would not like to restore
        random_layers = self.config.initialize_randomly

        if should_be_extended:
            random_layers.append('fc8')

        regex = "(" + "|".join(random_layers) + ")"
        names = lambda s: re.findall(regex, s)
        exclude = [v.name for v in tf.trainable_variables() if len(names(v.name)) >= 1]
        slim.get_variables_to_restore(exclude=exclude)

    def _config_assertions(self):
        # List the assertions for the configuation of the model
        assertions = {"dropout": percentage,
                      "num_classes": positiveInt,
                      "is_training": tfTensor,
                      'initialize_randomly': strList,
                      "use_initial_weights": boolean}

        self.config.assertions(assertions)


class VGG16(VGG):
    def __init__(self, config, inputs, should_be_extended=True, reuse=None, spatial_squeeze=True,
                 scope='vgg_16', fc_conv_padding='VALID'):
        """Oxford Net VGG 16-Layers version D Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
            To use in classification mode, resize input to 224x224.

        Args:
            config: a configuration object
            inputs: a tensor of size [batch_size, height, width, channels].
            spatial_squeeze: whether or not should squeeze the spatial dimensions of the
                             outputs. Useful to remove unnecessary dimensions for classification.
            scope: Optional scope for the variables.
            fc_conv_padding: the type of padding to use for the fully connected layer
                             that is implemented as a convolutional layer. Use 'SAME' padding if you
                             are applying the network in a fully convolutional manner and want to
                             get a prediction map downsampled by a factor of 32 as an output.
                             Otherwise, the output prediction map will be (input / 32) - 6 in case of
                             'VALID' padding.

        Returns:
            the last op containing the log predictions and end_points dict.
        """
        VGG.__init__(self, config, name='vgg_16', url='http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz')
        self.spatial_squeeze = spatial_squeeze
        self.scope = scope
        self.fc_conv_padding = fc_conv_padding

        # Define the VGG16 model
        self.build(inputs)

        # Restore all the weights
        self.restore(should_be_extended)

    def build(self, inputs):
        with slim.arg_scope(vgg_arg_scope()):
            with tf.variable_scope(self.scope, 'vgg_16', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'
                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    # Use conv2d instead of fully_connected layers.
                    net = slim.conv2d(net, 4096, [7, 7], padding=self.fc_conv_padding, scope='fc6')
                    net = slim.dropout(net, self.config.dropout, is_training=self.config.is_training,
                                       scope='dropout6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = slim.dropout(net, self.config.dropout, is_training=self.config.is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, 1000, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                    if self.spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                        end_points[sc.name + '/fc8'] = net

                    self.last_layer = net
                    self.all_endpoints = end_points

class VGG19(VGG):
    def __init__(self, config, inputs, should_be_extended=True, reuse=None, spatial_squeeze=True,
                 scope='vgg_19', fc_conv_padding='VALID'):
        """Oxford Net VGG 19-Layers.

        Note: All the fully_connected layers have been transformed to conv2d layers.
            To use in classification mode, resize input to 224x224.

        Args:
            config: a configuration object
            inputs: a tensor of size [batch_size, height, width, channels].
            spatial_squeeze: whether or not should squeeze the spatial dimensions of the
                             outputs. Useful to remove unnecessary dimensions for classification.
            scope: Optional scope for the variables.
            fc_conv_padding: the type of padding to use for the fully connected layer
                             that is implemented as a convolutional layer. Use 'SAME' padding if you
                             are applying the network in a fully convolutional manner and want to
                             get a prediction map downsampled by a factor of 32 as an output.
                             Otherwise, the output prediction map will be (input / 32) - 6 in case of
                             'VALID' padding.

        Returns:
            the last op containing the log predictions and end_points dict.
        """
        VGG.__init__(self, config, name='vgg_19', url='http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz')
        self.spatial_squeeze = spatial_squeeze
        self.scope = scope
        self.reuse = reuse
        self.inputs = inputs
        self.fc_conv_padding = fc_conv_padding

        # Define the VGG19 model
        self.build(inputs)

        # Restore all the weights
        self.restore(should_be_extended)

    def build(self, inputs):
        with slim.arg_scope(vgg_arg_scope()):
            with tf.variable_scope(self.scope, 'vgg_19', [inputs]) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                    outputs_collections=end_points_collection):
                    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    # Use conv2d instead of fully_connected layers.
                    net = slim.conv2d(net, 4096, [7, 7], padding=self.fc_conv_padding, scope='fc6')
                    net = slim.dropout(net, self.config.dropout, is_training=self.config.is_training,
                                       scope='dropout6')
                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = slim.dropout(net, self.config.dropout, is_training=self.config.is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, self.config.num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                    if self.spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                        end_points[sc.name + '/fc8'] = net

                    self.last_layer, self.end_points = net, end_points