


def matrix_distance(batch, matrix):
    p1 = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(batch), 1), 1),
                   tf.ones(shape=(1, matrix.get_shape()[0])))

    p2 = tf.transpose(tf.matmul(
        tf.reshape(tf.reduce_sum(tf.square(matrix), 1), shape=[-1, 1]),
        tf.ones(shape=(batch.get_shape()[0], 1)),
        transpose_b=True))

    result = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(batch, matrix, transpose_b=True))
    return result


def pairwise_eucledian_distance(batch, matrix):
    r = tf.reduce_sum(batch * matrix, 1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    result = r - 2 * tf.matmul(batch, matrix, transpose_b=True) + tf.transpose(r)
    return result
