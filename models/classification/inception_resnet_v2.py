# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition of the Inception Resnet V2 architecture.

As described in http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_modules.assertions.checks import *

from tf_modules.models import BaseModel
import tensorflow as tf

slim = tf.contrib.slim


def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 35x35 resnet block."""
  with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
    with tf.variable_scope('Branch_2'):
      tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
      tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
      tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 17x17 resnet block."""
  with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                  scope='Conv2d_0b_1x7')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                  scope='Conv2d_0c_7x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
  """Builds the 8x8 resnet block."""
  with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
    with tf.variable_scope('Branch_0'):
      tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
    with tf.variable_scope('Branch_1'):
      tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
      tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                  scope='Conv2d_0b_1x3')
      tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                  scope='Conv2d_0c_3x1')
    mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
    up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                     activation_fn=None, scope='Conv2d_1x1')
    net += scale * up
    if activation_fn:
      net = activation_fn(net)
  return net


class InceptionResnetV2(BaseModel):

    def __init__(self, config, should_be_extended=True):
        """Creates the Inception Resnet V2 model.

        Args:
        config: a Configuration class with all the necessary configurations
        should_be_extended: boolean indicating whether the model should be extended.
        """
        BaseModel.__init__(self, config)

        self.end_points = {}

        self.config = config

        self.inputs_placeholder = tf.placeholder(tf.float32,
                                                 shape=[None, config.width, config.height, config.channels],
                                                 name="inputs")
        if not should_be_extended:
            self.targets_placeholder = tf.placeholder(tf.int32,
                                                      shape=[None, config.num_classes],
                                                      name="labels")

        with slim.arg_scope(self.arg_scope()):
            self.build(should_be_extended)

        self.restore()

    def build(self, should_be_extended):
        with tf.variable_scope(scope, 'InceptionResnetV2', [self.inputs_placeholder], reuse=self.config.reuse):

            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=self.config.is_training):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                              stride=1, padding='SAME'):
                    # 149 x 149 x 32
                    net = slim.conv2d(self.inputs_placeholder, 32, 3, stride=2,
                                      padding='VALID', scope='Conv2d_1a_3x3')
                    self.end_points['Conv2d_1a_3x3'] = net
                    # 147 x 147 x 32
                    net = slim.conv2d(net, 32, 3, padding='VALID',
                                      scope='Conv2d_2a_3x3')
                    self.end_points['Conv2d_2a_3x3'] = net
                    # 147 x 147 x 64
                    net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                    self.end_points['Conv2d_2b_3x3'] = net
                    # 73 x 73 x 64
                    net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                          scope='MaxPool_3a_3x3')
                    self.end_points['MaxPool_3a_3x3'] = net
                    # 73 x 73 x 80
                    net = slim.conv2d(net, 80, 1, padding='VALID',
                                      scope='Conv2d_3b_1x1')
                    self.end_points['Conv2d_3b_1x1'] = net
                    # 71 x 71 x 192
                    net = slim.conv2d(net, 192, 3, padding='VALID',
                                      scope='Conv2d_4a_3x3')
                    self.end_points['Conv2d_4a_3x3'] = net
                    # 35 x 35 x 192
                    net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                          scope='MaxPool_5a_3x3')
                    self.end_points['MaxPool_5a_3x3'] = net

                    # 35 x 35 x 320
                    with tf.variable_scope('Mixed_5b'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5,
                                                    scope='Conv2d_0b_5x5')
                        with tf.variable_scope('Branch_2'):
                            tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
                            tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3,
                                                    scope='Conv2d_0b_3x3')
                            tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3,
                                                    scope='Conv2d_0c_3x3')
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME',
                                                     scope='AvgPool_0a_3x3')
                            tower_pool_1 = slim.conv2d(tower_pool, 64, 1,
                                                   scope='Conv2d_0b_1x1')
                            net = tf.concat(axis=3, values=[tower_conv, tower_conv1_1,
                                          tower_conv2_2, tower_pool_1])

                    self.end_points['Mixed_5b'] = net
                    net = slim.repeat(net, 10, block35, scale=0.17)

                    # 17 x 17 x 1024
                    with tf.variable_scope('Mixed_6a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 384, 3, stride=2, padding='VALID',
                                                 scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                                        scope='Conv2d_0b_3x3')
                            tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                                    stride=2, padding='VALID',
                                                    scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                        net = tf.concat(axis=3, values=[tower_conv, tower_conv1_2, tower_pool])

                    self.end_points['Mixed_6a'] = net
                    net = slim.repeat(net, 20, block17, scale=0.10)

                    # Auxillary tower
                    with tf.variable_scope('AuxLogits'):
                        aux = slim.avg_pool2d(net, 5, stride=3, padding='VALID',
                                            scope='Conv2d_1a_3x3')
                        aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
                        aux = slim.conv2d(aux, 768, aux.get_shape()[1:3],
                                        padding='VALID', scope='Conv2d_2a_5x5')
                        aux = slim.flatten(aux)
                        aux = slim.fully_connected(aux, self.config.num_classes, activation_fn=None,
                                                 scope='Logits')
                        self.end_points['AuxLogits'] = aux

                    with tf.variable_scope('Mixed_7a'):
                        with tf.variable_scope('Branch_0'):
                            tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                                   padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_1'):
                            tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_2'):
                            tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
                            tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                                    scope='Conv2d_0b_3x3')
                            tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                                    padding='VALID', scope='Conv2d_1a_3x3')
                        with tf.variable_scope('Branch_3'):
                            tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                                     scope='MaxPool_1a_3x3')
                        net = tf.concat(axis=3, values=[tower_conv_1, tower_conv1_1,
                                          tower_conv2_2, tower_pool])

                    self.end_points['Mixed_7a'] = net

                    net = slim.repeat(net, 9, block8, scale=0.20)
                    net = block8(net, activation_fn=None)

                    net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
                    self.end_points['Conv2d_7b_1x1'] = net

                    with tf.variable_scope('Logits'):
                        self.end_points['PrePool'] = net
                        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                            scope='AvgPool_1a_8x8')

                        self.embedding = slim.flatten(net)

                        if not should_be_extended:
                            net = slim.dropout(self.embedding, self.config.dropout,
                                               is_training=self.config.is_training,
                                               scope='Dropout')

                            self.end_points['PreLogitsFlatten'] = net

                            logits = slim.fully_connected(net, self.config.num_classes, activation_fn=None,
                                                        scope='Logits')
                            self.end_points['Logits'] = logits
                            self.preds = tf.nn.softmax(logits, name='Predictions')
                            self.end_points['Predictions'] = self.preds


    def arg_scope(self, weight_decay=0.00004, batch_norm_decay=0.9997,
                  batch_norm_epsilon=0.001):
        """Yields the scope with the default parameters for inception_resnet_v2.

        Args:
        weight_decay: the weight decay for weights variables.
        batch_norm_decay: decay for the moving average of batch_norm momentums.
        batch_norm_epsilon: small float added to variance to avoid dividing by zero.

        Returns:
        a arg_scope with the parameters needed for inception_resnet_v2.
        """
        # Set weight_decay for weights in conv2d and fully_connected layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          biases_regularizer=slim.l2_regularizer(weight_decay)):

            batch_norm_params = {
                'decay': batch_norm_decay,
                'epsilon': batch_norm_epsilon,
            }
            # Set activation_fn and parameters for batch_norm.
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params) as scope:
                return scope

    def restore(self):
        exclude = []

        if self.should_be_extended:
            exclude += ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']

        if self.config.reset_trainable:
            exclude += self.config.trainable

        # Define the few layers which we would not like to restore
        regex = "(" + "|".join(exclude) + ")"
        names = lambda s: re.findall(regex, s)
        exclude = [v.name for v in tf.trainable_variables() if len(names(v.name)) >= 1]
        variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

    def maybe_download_weights(self):
        if not self.config.use_initial_weights:
            return

        if not self.config.checkpoint_file:
            model_path ='model'
            checkpoint_file = '{}/inception_resnet_v2.ckpt'.format(model_path)

        if not os.path.exists(checkpoint_file):
            if not os.path.isdir(model_path):
                os.mkdir(model_path)

            os.system("curl -O 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'")
            os.system("tar -xf 'inception_resnet_v2_2016_08_30.tar.gz'")
            os.rename('inception_resnet_v2_2016_08_30.ckpt', checkpoint_file)
            os.remove('inception_resnet_v2_2016_08_30.tar.gz')

    def trainable_layers(self):
        return set(list(self.end_points.keys()))

    def extension_layer(self):
        return self.embedding

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

    def _config_assertions(self):
        # List the assertions for the configuation of the model
        assertions = {"dropout":  percentage,
                      "height": positiveInt,
                      "width":    positiveInt,
                      "num_classes": positiveInt,
                      "is_training": tfTensor,
                      "trainable": strList,
                      "trainable": lambda l: len([x for x in l if x in self.end_points]) == len(l),
                      "channels": lambda x: isinstance(x, int) and x == 3}

        self.config.assertions(assertions)
        self.config.condition(cond=("use_initial_weights", isTrue),
                              assertion=("checkpoint_file",  optionalStr))
        self.config.condition(cond=("trainable", nonEmptyList),
                              assertion=("reset_trainable",  boolean))
