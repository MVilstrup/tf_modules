from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.summary import summary

def contrastive_loss(left, right, y, margin, extra=False, scope="constrastive_loss"):
    """
    Loss for Siamese networks as described in the paper:
        `Learning a Similarity Metric Discriminatively, with Application to Face
         Verification <http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf>`_ by Chopra et al.`

    math:
        \frac{1}{2} [y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2], d = \Vert l - r \Vert_2

    Args:
        left  (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y     (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.
        margin    (float): horizon for negative examples (y==0).
        extra      (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: constrastive_loss (averaged over the batch), (and optionally average_pos_dist, average_neg_dist)
    """
    with tf.name_scope(scope):
        y = tf.cast(y, tf.float32)

        delta = tf.reduce_sum(tf.square(left - right), 1)
        delta_sqrt = tf.sqrt(delta + 1e-10)

        match_loss = delta
        missmatch_loss = tf.square(tf.nn.relu(margin - delta_sqrt))

        loss = tf.reduce_mean(0.5 * (y * match_loss + (1 - y) * missmatch_loss))

        if extra:
            num_pos = tf.count_nonzero(y)
            num_neg = tf.count_nonzero(1 - y)
            pos_dist = tf.where(tf.equal(num_pos, 0), 0.,
                                tf.reduce_sum(y * delta_sqrt) / tf.cast(num_pos, tf.float32),
                                name="pos-dist")
            neg_dist = tf.where(tf.equal(num_neg, 0), 0.,
                                tf.reduce_sum((1 - y) * delta_sqrt) / tf.cast(num_neg, tf.float32),
                                name="neg-dist")
            return loss, pos_dist, neg_dist
        else:
            return loss


def siamese_cosine_loss(left, right, y, scope="cosine_loss"):
    """Loss for Siamese networks (cosine version).
    Same as contrastive_loss but with different similarity measurement.

    math:
        [\frac{l \cdot r}{\lVert l\rVert \lVert r\rVert} - (2y-1)]^2

    Args:
        left  (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y     (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.

    Returns:
        tf.Tensor: cosine-loss as a scalar tensor.
    """

    def l2_norm(t, eps=1e-12):
        with tf.name_scope("l2_norm"):
            return tf.sqrt(tf.reduce_sum(tf.square(t), 1) + eps)

    with tf.name_scope(scope):
        y = 2 * tf.cast(y, tf.float32) - 1
        pred = tf.reduce_sum(left * right, 1) / (l2_norm(left) * l2_norm(right) + 1e-10)

        return tf.nn.l2_loss(y - pred) / tf.cast(tf.shape(left)[0], tf.float32)


def triplet_loss(anchor, positive, negative, margin, extra=False, scope="triplet_loss"):
    """
    Loss for Triplet networks as described in the paper:
        `FaceNet: A Unified Embedding for Face Recognition and Clustering
         <https://arxiv.org/abs/1503.03832> by Schroff et al.`

    Learn embeddings from an anchor point and a similar input (positive) as
    well as a not-similar input (negative).
    Intuitively, a matching pair (anchor, positive) should have a smaller relative distance
    than a non-matching pair (anchor, negative).

    math:
        \max(0, m + \Vert a-p\Vert^2 - \Vert a-n\Vert^2)

    Args:
        anchor   (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        margin       (float): horizon for negative examples
        extra         (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    def triplet(anchor, positive, negative, extra):
        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

        loss = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))

        if extra:
            pos_dist = tf.reduce_mean(tf.sqrt(d_pos + 1e-10), name='pos-dist')
            neg_dist = tf.reduce_mean(tf.sqrt(d_neg + 1e-10), name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss

    with tf.name_scope(scope):
        if extra:
            loss1, pos_dist1, neg_dist1 = triplet(anchor, positive, negative, extra)
            loss2, pos_dist2, neg_dist2 = triplet(positive, anchor, negative, extra)
            return tf.cond(loss1 > loss2,
                           lambda: (loss1, pos_dist1, neg_dist1),
                           lambda: (loss2, pos_dist2, neg_dist2))
        else:
            loss1 = triplet(anchor, positive, negative, extra)
            loss2 = triplet(positive, anchor, negative, extra)
            return tf.maximum(loss1, loss2)


def decov_loss(xs):
    """Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
    'Reducing Overfitting In Deep Networks by Decorrelating Representation'
    """
    x = tf.reshape(xs, [int(xs.get_shape()[0]), -1])
    m = tf.reduce_mean(x, 0, True)
    z = tf.expand_dims(x - m, 2)
    corr = tf.reduce_mean(tf.matmul(z, tf.transpose(z, perm=[0, 2, 1])), 0)
    corr_frob_sqr = tf.reduce_sum(tf.square(corr))
    corr_diag_sqr = tf.reduce_sum(tf.square(tf.diag_part(corr)))
    loss = 0.5 * (corr_frob_sqr - corr_diag_sqr)
    return loss


def soft_triplet_loss(anchor, positive, negative, extra=True, scope="soft_triplet_loss"):
    """Loss for triplet networks as described in the paper:
        `Deep Metric Learning using Triplet Network
        <https://arxiv.org/abs/1412.6622> by Hoffer et al.`

    It is a softmax loss using `(anchor-positive)^2` and `(anchor-negative)^2` as logits.

    Args:
        anchor   (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        extra         (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    eps = 1e-10
    with tf.name_scope(scope):
        d_pos = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive), 1) + eps)
        d_neg = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative), 1) + eps)

        logits = tf.stack([d_pos, d_neg], axis=1)
        ones = tf.ones_like(tf.squeeze(d_pos), dtype="int32")

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ones))

        if extra:
            pos_dist = tf.reduce_mean(d_pos, name='pos-dist')
            neg_dist = tf.reduce_mean(d_neg, name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss


def center_loss(features, labels, num_classes, decay=0.95, scope="center_loss"):
    """Center-Loss as described in the paper:
        `A Discriminative Feature Learning Approach for Deep Face Recognition
        <http://ydwen.github.io/papers/WenECCV16.pdf> by Wen et al.`

    Args:
        features (tf.Tensor): features produced by the network
        label     (tf.Tensor): ground-truth label for each feature
        num_classes     (int): number of different classes
        alpha         (float): learning rate for updating the centers
        extra          (bool): also return the centers and their update-Op

    Returns:
        tf.Tensor: center loss
    """
    with tf.variable_scope(scope):
        feature_amount = features.get_shape()[1]
        centers = tf.get_variable('centers',
                                  [num_classes, feature_amount],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)

        # We calculat the difference between the centers and the features
        diff = (centers_batch - features)

        # We calculate the amount of times we saw the centers as a regularizer
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        # We divide the difference by the amount of times we saw the center
        diff = diff / tf.cast((1 + appear_times), tf.float32)

        # We add decay to decrease the update time
        diff = (1 - decay) * diff

        # we scatter the centers to their new locations
        centers = tf.scatter_sub(centers, labels, diff)

        loss = tf.reduce_mean(tf.square(features - centers_batch))
        return loss, centers

def pairwise_distance(feature, squared=False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
    feature: 2-D Tensor of size [number of data, feature dimension].
    squared: Boolean, whether or not to square the pairwise distances.
    Returns:
    pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = math_ops.add(
      math_ops.reduce_sum(
          math_ops.square(feature),
          axis=[1],
          keep_dims=True),
      math_ops.reduce_sum(
          math_ops.square(
              array_ops.transpose(feature)),
          axis=[0],
          keep_dims=True)) - 2.0 * math_ops.matmul(
              feature, array_ops.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = math_ops.sqrt(
        pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = math_ops.multiply(
      pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

    num_data = array_ops.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
      array_ops.ones([num_data]))
    pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.
    Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = math_ops.reduce_min(data, dim, keep_dims=True)
    masked_maximums = math_ops.reduce_max(
      math_ops.multiply(
          data - axis_minimums, mask), dim, keep_dims=True) + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.
    Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = math_ops.reduce_max(data, dim, keep_dims=True)
    masked_minimums = math_ops.reduce_min(
      math_ops.multiply(
          data - axis_maximums, mask), dim, keep_dims=True) + axis_maximums
    return masked_minimums


def triplet_semihard_loss(labels, embeddings, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
    The loss encourages the positive distances (between a pair of embeddings with
    the same labels) to be smaller than the minimum negative distance among
    which are at least greater than the positive distance plus the margin constant
    (called semi-hard negative) in the mini-batch. If no such negative exists,
    uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.
    Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
      be l2 normalized.
    margin: Float, margin term in the loss definition.
    Returns:
    triplet_loss: tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = pairwise_distance(embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    # Compute the mask.
    pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
    mask = math_ops.logical_and(
      array_ops.tile(adjacency_not, [batch_size, 1]),
      math_ops.greater(
          pdist_matrix_tile, array_ops.reshape(
              array_ops.transpose(pdist_matrix), [-1, 1])))
    mask_final = array_ops.reshape(
      math_ops.greater(
          math_ops.reduce_sum(
              math_ops.cast(
                  mask, dtype=dtypes.float32), 1, keep_dims=True),
          0.0), [batch_size, batch_size])
    mask_final = array_ops.transpose(mask_final)

    adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    mask = math_ops.cast(mask, dtype=dtypes.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = array_ops.reshape(
      masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = array_ops.transpose(negatives_outside)

    # negatives_inside: largest D_an.
    negatives_inside = array_ops.tile(
      masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
    semi_hard_negatives = array_ops.where(
      mask_final, negatives_outside, negatives_inside)

    loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

    mask_positives = math_ops.cast(
      adjacency, dtype=dtypes.float32) - array_ops.diag(
          array_ops.ones([batch_size]))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = math_ops.reduce_sum(mask_positives)

    triplet_loss = math_ops.truediv(
      math_ops.reduce_sum(
          math_ops.maximum(
              math_ops.multiply(loss_mat, mask_positives), 0.0)),
      num_positives,
      name='triplet_semihard_loss')

    return triplet_loss



def lifted_struct_loss(labels, embeddings, margin=1.0):
    """Computes the lifted structured loss.
    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than any negative distances (between a
    pair of embeddings with different labels) in the mini-batch in a way
    that is differentiable with respect to the embedding vectors.
    See: https://arxiv.org/abs/1511.06452.
    Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should not
      be l2 normalized.
    margin: Float, margin term in the loss definition.
    Returns:
    lifted_loss: tf.float32 scalar.
    """
    # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
    lshape = array_ops.shape(labels)
    assert lshape.shape == 1
    labels = array_ops.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pairwise_distances = pairwise_distance(embeddings)

    # Build pairwise binary adjacency matrix.
    adjacency = math_ops.equal(labels, array_ops.transpose(labels))
    # Invert so we can select negatives only.
    adjacency_not = math_ops.logical_not(adjacency)

    batch_size = array_ops.size(labels)

    diff = margin - pairwise_distances
    mask = math_ops.cast(adjacency_not, dtype=dtypes.float32)
    # Safe maximum: Temporarily shift negative distances
    #   above zero before taking max.
    #     this is to take the max only among negatives.
    row_minimums = math_ops.reduce_min(diff, 1, keep_dims=True)
    row_negative_maximums = math_ops.reduce_max(
      math_ops.multiply(
          diff - row_minimums, mask), 1, keep_dims=True) + row_minimums

    # Compute the loss.
    # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
    #   where m_i is the max of alpha - negative D_i's.
    # This matches the Caffe loss layer implementation at:
    #   https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp  # pylint: disable=line-too-long

    max_elements = math_ops.maximum(
      row_negative_maximums, array_ops.transpose(row_negative_maximums))
    diff_tiled = array_ops.tile(diff, [batch_size, 1])
    mask_tiled = array_ops.tile(mask, [batch_size, 1])
    max_elements_vect = array_ops.reshape(
      array_ops.transpose(max_elements), [-1, 1])

    loss_exp_left = array_ops.reshape(
      math_ops.reduce_sum(math_ops.multiply(
          math_ops.exp(
              diff_tiled - max_elements_vect),
          mask_tiled), 1, keep_dims=True), [batch_size, batch_size])

    loss_mat = max_elements + math_ops.log(
      loss_exp_left + array_ops.transpose(loss_exp_left))
    # Add the positive distance.
    loss_mat += pairwise_distances

    mask_positives = math_ops.cast(
      adjacency, dtype=dtypes.float32) - array_ops.diag(
          array_ops.ones([batch_size]))

    # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
    num_positives = math_ops.reduce_sum(mask_positives) / 2.0

    lifted_loss = math_ops.truediv(
      0.25 * math_ops.reduce_sum(
          math_ops.square(
              math_ops.maximum(
                  math_ops.multiply(loss_mat, mask_positives), 0.0))),
      num_positives,
      name='liftedstruct_loss')
    return lifted_loss
