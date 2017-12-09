#!/usr/bin/python3

import tensorflow as tf
import os

slim = tf.contrib.slim

from tf_modules.utils import *
from tf_modules.assertions.checks import (optionalStr,
                                          optionalTensor,
                                          tfVariable)

# class BaseModel#Meta(type):
#    def __call__(cls, *args, **kwargs):
#        error = lambda x,y,z: "The {} method should return {} not {}".format(x, y, z)
#
#        instance = super().__call__(*args, **kwargs)
#        return instance

class Model:
    def __init__(self, config, name, url=None):
        # Create the global step for monitoring the learning_rate and training.
        self.config = config
        self.name = name
        self.url = url
        self.name = name
        self.all_vars = {}
        self.interal_vars = {}
        self.last_layer = None

    def info(self):
        for key, op in self.all_vars.items():
            print(key, op.name, tf.shape(op), op.type)
            print("_" * 80)

    def save(self, sess, path):
        saver = tf.train.Saver()
        mkpath(path)
        saver.save(sess, path)
        print("{} saved in file: {}".format(self.name, path))

    def load_ckpt(self, sess, checkpoint):
        if hasattr(self, "trainable_layers") and self.trainable_layers:
            saver = tf.train.Saver(self.trainable_layers)
        else:
            saver = tf.train.Saver()

        if 'ckpt' in checkpoint:
            saver.restore(sess, checkpoint)
        else:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
            
        print("{} loaded from file: {}".format(self.name, checkpoint))

    def load_graph(self, filename, show=False):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.gfile.GFile(filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="frozen_" + self.name)

            if show:
                print('Graph contains the following Nodes:')
                for op in graph.get_operations():
                    print(" - ", op.name)

        return graph

    def initialize(self, sess, checkpoint, exclude=[]):
        self._maybe_download_weights(checkpoint)

        vars = []
        for key, var in self.interal_vars.values():
            if key not in exclude:
                vars.append(var)

        saver = tf.train.Saver(vars)
        saver.restore(sess, checkpoint)

    def _maybe_download_weights(self, checkpoint):
        if self.url is None:
            return

        path = '/'.join(checkpoint.split('/')[:-1])
        if not os.path.exists(checkpoint):
            mkpath(path)

            print('Downloading Original Weights')

            file_name = self.url.split('/')[-1]
            os.system("curl -O '{}'".format(self.url))
            os.system("tar -xf '{}'".format(file_name))
            os.rename('{}.ckpt'.format(self.name), checkpoint)
            os.remove(file_name)

    def internal_layer(self, name, func, input, *args, **kwargs):
        assert isinstance(name, str)

        self.last_layer = func(input, *args, **kwargs)

        self.internal_vars[name] = self.last_layer
        self.all_vars[name] = self.last_layer

        return self.last_layer

    def layer(self, name, func, input, *args, **kwargs):
        assert isinstance(name, str)

        self.last_layer = func(input, *args, **kwargs)

        self.all_vars[name] = self.last_layer

        return self.last_layer

    def freeze(self, model_dir, output_node_names=[]):
        """Extract the sub graph defined by the output nodes and convert
        all its variables into constant
        Args:
            model_dir: the root folder containing the checkpoint state file
            output_node_names: a string, containing all the output node's names,
                                comma separated
        """
        if not tf.gfile.Exists(model_dir):
            raise AssertionError(
                "Export directory doesn't exists. Please specify an export "
                "directory: %s" % model_dir)

        if not output_node_names:
            output_node_names = self.all_vars.keys()

        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path

        # We precise the file fullname of our freezed graph
        absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
        output_graph = absolute_model_dir + "/frozen_{}.pb".format(self.name)

        # We clear devices to allow TensorFlow to control on which device it will load operations
        clear_devices = True

        # We start a session using a temporary fresh Graph
        with tf.Session(graph=tf.Graph()) as sess:
            # We import the meta graph in the current default Graph
            saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

            # We restore the weights
            saver.restore(sess, input_checkpoint)

            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                output_node_names.split(",")  # The output node names are used to select the usefull nodes
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

        return output_graph_def

    def visualize(self):
        with tf.Session(graph=self.config.graph) as sess:
            print("Initializing variables")
            sess.run(tf.global_variables_initializer())

            print("Creating Graph")
            tmp_def = rename_nodes(sess.graph_def, lambda s: "/".join(s.split('_', 1)))
            show_graph(tmp_def)
