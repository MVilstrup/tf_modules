import tensorflow as tf


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

    with tf.name_scope(scope):
        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

        loss = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))

        if extra:
            pos_dist = tf.reduce_mean(tf.sqrt(d_pos + 1e-10), name='pos-dist')
            neg_dist = tf.reduce_mean(tf.sqrt(d_neg + 1e-10), name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
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


def center_loss(embedding, labels, num_classes, alpha=0.1, extra=True, scope="center_loss"):
    """Center-Loss as described in the paper:
        `A Discriminative Feature Learning Approach for Deep Face Recognition
        <http://ydwen.github.io/papers/WenECCV16.pdf> by Wen et al.`

    Args:
        embedding (tf.Tensor): features produced by the network
        label     (tf.Tensor): ground-truth label for each feature
        num_classes     (int): number of different classes
        alpha         (float): learning rate for updating the centers
        extra          (bool): also return the centers and their update-Op

    Returns:
        tf.Tensor: center loss
    """
    with tf.name_scope(scope):
        feature_amount = embedding.get_shape()[1]
        centers = tf.get_variable('centers',
                                  [num_classes, feature_amount],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        loss = tf.nn.l2_loss(embedding - centers_batch)

        # Calculate the updates to the centers
        diff = centers_batch - embedding

        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])

        diff = alpha * (diff / tf.cast((1 + appear_times), tf.float32))

        centers_update_op = tf.scatter_sub(centers, labels, diff)

        if extra:
            return loss, centers_batch, centers_update_op
        else:
            return loss
