import tensorflow as tf


class NegativePenaltyCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, p_indices:list, name='negative_penalty_categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.p_indices = [[p_index] for p_index in p_indices]

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.accuracy = _get_accuracy(y_true, y_pred, sample_weight, self.p_indices)

    def result(self):
        return self.accuracy


class NegativePenaltySparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, p_indices:list, name='negative_penalty_sparse_categorical_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.p_indices = [[p_index] for p_index in p_indices]

    def update_state(self, y_true, y_pred, sample_weight=None):
        num_classes = y_pred.shape[-1]
        y_true = tf.squeeze(tf.one_hot(y_true, num_classes), axis=1)
        self.accuracy = _get_accuracy(y_true, y_pred, sample_weight, self.p_indices)

    def result(self):
        return self.accuracy
    

def _get_accuracy(y_true, y_pred, sample_weight, p_indices:list):
    batch_size = y_true.shape[0]
    y_true = tf.cast(y_true, tf.float32)
    # compute accuracy for positive samples in a batch
    positive_sample_weights = tf.cast(
        tf.reduce_any(
            tf.transpose(tf.math.equal(tf.math.argmax(y_true, axis=1), p_indices), perm=(1, 0)), axis=1
        ), 
        dtype=tf.float32
    )  # 1.0: postive sample, 0.0: negative sample
    positive_sample_values = tf.where(
        tf.math.argmax(y_true, axis=1) ==  tf.math.argmax(y_pred, axis=1), 1.0, 0.0
    )
    positive_sample_values = positive_sample_weights * positive_sample_values
    # compute accuracy for negative samples in a batch
    negative_sample_weights = tf.where(positive_sample_weights == 1.0, 0.0, 1.0)  # 1.0: negative sample, 0.0: postive sample
    negative_sample_values = tf.cast(
        tf.reduce_all(
            tf.transpose(tf.math.not_equal(tf.math.argmax(y_pred, axis=1), p_indices), perm=(1, 0)), axis=1
        ), 
        dtype=tf.float32
    )
    negative_sample_values = negative_sample_weights * negative_sample_values
    # combine positive values and negative values
    positive_sample_values = tf.cast(positive_sample_values, tf.bool)
    negative_sample_values = tf.cast(negative_sample_values, tf.bool)
    values = tf.math.logical_or(positive_sample_values, negative_sample_values)
    values = tf.cast(values, tf.float32)
    if sample_weight is not None:
        values = tf.math.multiply(values, tf.squeeze(sample_weight, axis=1))
    else:
        sample_weight = tf.repeat([[1.0]], batch_size, axis=0)
        values = tf.math.multiply(values, tf.squeeze(sample_weight, axis=1))
    accuracy = tf.math.reduce_sum(values, axis=None) / tf.math.reduce_sum(sample_weight, axis=None)
    return accuracy
