#!/usr/bin/python3
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
from IPython.display import clear_output, Image, display, HTML
import matplotlib
import matplotlib.pyplot as plt
import imghdr
from scipy.misc import imread
from subprocess import call
import os

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def.
    Created by Alex Mordvintsev
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
    """

    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def


def rename_nodes(graph_def, rename_func):
    """
    Created by Alex Mordvintsev
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
    """

    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def


def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph.
    Created by Alex Mordvintsev
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
    """


    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    #strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(graph_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


def plot_decision_boundary(pred_func, X, y):
    #from https://github.com/dennybritz/nn-from-scratch/blob/master/nn-from-scratch.ipynb
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    yy = yy.astype('float32')
    xx = xx.astype('float32')
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])[:,0]
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    # plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=-y, cmap=plt.cm.Spectral)

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def import_file(full_path_to_module):
    try:
        import os
        module_dir, module_file = os.path.split(full_path_to_module)
        module_name, module_ext = os.path.splitext(module_file)
        save_cwd = os.getcwd()
        os.chdir(module_dir)
        module_obj = __import__(module_name)
        module_obj.__file__ = full_path_to_module
        globals()[module_name] = module_obj
        os.chdir(save_cwd)
    except:
        raise ImportError

def remove_if_bad(image):
    file_type = imghdr.what(image)
    if file_type == 'jpeg':
        if len(imread(image).shape) == 3:
            return True

    # If the image does not live up to the expectations, remove it
    os.remove(image)
    return False

def mkpath(path):
    if os.path.exists(path):
        return

    path = os.path.abspath(path).split('/')
    for i, p in enumerate(path):
        if '.' in p:
            break

        curr = "/".join(path[:i] + [p])
        if curr and not os.path.isdir(curr):
            os.mkdir(curr)

def maybe_download_inpception(location_folder):
    checkpoint_name = '{}/inception_resnet_v2.ckpt'.format(location_folder)

    if not os.path.exists(checkpoint_name):
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        call(["curl", "-O", 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'])
        call(["tar", "-xf", 'inception_resnet_v2_2016_08_30.tar.gz'])
        os.rename('inception_resnet_v2_2016_08_30.ckpt', '{}'.format(checkpoint_name))
        os.remove('inception_resnet_v2_2016_08_30.tar.gz')

    return checkpoint_name

def resolve(*classes):
    metaclass = tuple(set(type(cls) for cls in classes))
    metaclass = metaclass[0] if len(metaclass)==1 \
                else type("_".join(mcls.__name__ for mcls in metaclass), metaclass, {})   # class M_C
    return metaclass("_".join(cls.__name__ for cls in classes), classes, {})

def pretty(_str, space=20):
    return "\n" + _str + " " * (space - len(_str))

def pretty_num(num):
    if num < 1:
        return str(num).replace('.', ',')

    _str = ""
    for divisor in [1e12, 1e9, 1e6, 1e3, 1e0]:
        if divisor > num:
            continue

        amount = int(num / divisor)
        _str += "{0:03d}.".format(amount) if _str else "{}.".format(amount)
        num = num - (amount * divisor)

    return _str[:-1]


def detect_overridden(cls, obj):
  common = cls.__dict__.keys() & obj.__class__.__dict__.keys()
  diff = [m for m in common if cls.__dict__[m] != obj.__class__.__dict__[m]]
