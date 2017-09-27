#!/usr/bin/python3

import tensorflow as tf
#from tensorflow.estimator import ModeKeys

from tf_modules.utils import *
from tf_modules.assertions.checks import (optionalStr,
                                         optionalTensor,
                                         tfVariable)


class Modes(object):
  """Standard names for model modes.
  The following standard keys are defined:
  * `TRAIN`: training mode.
  * `EVAL`: evaluation mode.
  * `PREDICT`: inference mode.
  * `EXTEND`: Extensions mode.
  """
  TRAIN = 'train'
  EVAL = 'eval'
  PREDICT = 'pred'
  EXTEND = 'extend'


class BaseModelMeta(type):
    def __call__(cls, *args, **kwargs):
        error = lambda x,y,z: "The {} method should return {} not {}".format(x, y, z)
        instance = super().__call__(*args, **kwargs)

        methods = [("extend()", instance.extension_layer())]
        for name, result in methods:
            assert(optionalTensor(result)), error(name, "a Tensor", type(result))

        return instance

class BaseModel(metaclass=BaseModelMeta):

    def __init__(self, model_dir, config, params):
        #Create the global step for monitoring the learning_rate and training.
        self.config = config
        self._base_assertions()
        self._config_assertions()

    def extension_layer(self):
        output = """
        extension_layer()-method should be created when inheriting BaseModel

        Info: This is the next to final layer in the model. Typically it is the
        layer just before the final layer called the "logits".

        This method is used to make the model easier to extend.
        """
        raise ValueError(output)


    def evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None, name=None):
        pass

    def export_savedmodel(export_dir_base, serving_input_receiver_fn, assets_extra=None, as_text=False, checkpoint_path=None):
        pass

    def predict(self, input_fn, predict_keys=None, hooks=None, checkpoint_path=None):
        pass

    def train(self, hooks=None, steps=None, max_steps=None):
        pass

    def extend(self):
        pass

    def estimator(self):
        pass

    def _base_assertions(self):
        pass
        ## List the assertions for the configuration
        #assertions = {"save_path": optionalStr,
        #              "log_dir": optionalStr,
        #              "global_step": tfVariable}
        #self.config.assertions(assertions)

    def _config_assertions(self):
        print("You can include configuration assertions to you model by defining a _config_assertions() method")
        return None

    def load(self, sess, checkpoint):
        saver = tf.train.Saver()
        saver.restore(sess,  checkpoint)
        print("{} loaded from file: {}".format(self.config.model_name, checkpoint))

    def visualize(self):
        with tf.Session(graph=self.config.graph) as sess:
            print("Initializing variables")
            sess.run(tf.global_variables_initializer())

            print("Creating Graph")
            tmp_def = rename_nodes(sess.graph_def, lambda s:"/".join(s.split('_', 1)))
            show_graph(tmp_def)
