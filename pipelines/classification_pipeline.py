#!/usr/bin/python3

import tensorflow as tf
import multiprocessing as mp
#import preprocessing
from tf_modules.assertions.checks import *

class ClassificationPipeline:

    def __init__(self, inputs, targets, config):

        # Create references to all the placeholders
        self.inputs = inputs
        self.outputs = targets
        self.is_training = config.is_training
        self.config = config

        self._config_assertions()

        # Define Training Pipeline
        preprocess_treads = max(2, int(mp.cpu_count() / 2))
        self.train_images, self.train_labels = self._pipeline(config.train_files,
                                                              config.num_classes,
                                                              config.batch_size,
                                                              preprocess_treads,
                                                              name="Train",
                                                              num_epochs=config.epoch_amount,
                                                              is_training=True)

        # Define Validation Pipeline
        preprocess_treads = min(2, max(1, int(mp.cpu_count() / 4)))
        self.eval_images, self.eval_labels = self._pipeline(config.validation_files,
                                                            config.num_classes,
                                                            config.batch_size,
                                                            preprocess_treads,
                                                            name="Eval",
                                                            num_epochs=config.epoch_amount,
                                                            is_training=False)

    def _pipeline(self, filenames, num_classes, batch_size, preprocess_treads, name, num_epochs, is_training):
        with tf.name_scope("{}_pipeline".format(name)):
            # Make a queue of file names including all the JPEG images files in the relative
            # image directory.
            filename_queue = tf.train.string_input_producer(filenames)

            # Read an entire image file which is required since they're JPEGs, if the images
            # are too large they could be split in advance to smaller files or use the Fixed
            # reader to split up the file.
            image_reader = tf.WholeFileReader()

            # Read a whole file from the queue, the first returned value in the tuple is the
            # filename which we are ignoring.
            file_name, image_file = image_reader.read(filename_queue)

            # Retrive the cvaorrect label from the
            path = tf.string_split([file_name], '/')

            path_length = tf.cast(path.dense_shape[1],tf.int32)

            # Get the parent directory, this is the second last folder in the path
            label = path.values[path_length - tf.constant(2, dtype=tf.int32)]
            label = tf.string_to_number(label,out_type=tf.int32) # Convert the label into an int
            label = tf.one_hot(label, num_classes, dtype=tf.int32)

            # Decode the image as a JPEG file, this will turn it into a Tensor which we can
            # then use in training.
            image = tf.image.decode_jpeg(image_file, channels=self.config.channels)

            #Perform the correct preprocessing for this image depending if it is training or evaluating
            #image = preprocessing.preprocess_image(image,
            #                                       self.config.height,
            #                                       self.config.width,
            #                                       is_training)

            image = tf.reshape(image, (self.config.height,
                                       self.config.width,
                                       self.config.channels))

            # Take each image into a mini-batch
            min_queue_size = int(16 * batch_size)
            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=preprocess_treads,
                capacity=min_queue_size * batch_size,
                min_after_dequeue=min_queue_size)

            return images, labels

    def train_data(self, sess):
        image_batch, target_batch = sess.run([self.train_images, self.train_labels])
        return {self.inputs: image_batch,
                self.outputs: target_batch,
                self.is_training: True}

    def validation_data(self, sess):
        image_batch, target_batch = sess.run([self.eval_images, self.eval_labels])
        return {self.inputs: image_batch,
                self.outputs: target_batch,
                self.is_training: False}

    def _config_assertions(self):
        # List assertions of the configuration
        assertions = {"batch_size": positiveInt,
                      "epoch_amount": positiveInt,
                      "train_files": strList,
                      "validation_files": strList,
                      "num_classes": positiveInt,
                      "width": positiveInt,
                      "height": positiveInt,
                      "channels": positiveInt}
        self.config.assertions(assertions)
