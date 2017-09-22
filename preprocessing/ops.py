import tensorflow as tf

def convert_to_class(file_name, level=2):
    # Retrive the correct label from the
    path = tf.string_split([file_name], '/')
    path_length = tf.cast(path.dense_shape[1],tf.int32)
    # Get the parent directory, this is the second last folder in the path
    label = path.values[path_length - tf.constant(level, dtype=tf.int32)]
    label = tf.string_to_number(label, out_type=tf.int32) # Convert the label into an int
    return label
