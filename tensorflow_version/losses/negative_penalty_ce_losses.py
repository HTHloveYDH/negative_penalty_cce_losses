import tensorflow as tf
import tensorflow.keras.backend as K


class NegativePenaltyCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_num:int, p_indices:list, alpha=1.0, penalty_scale=None, \
                 reduction=tf.keras.losses.Reduction.AUTO, \
                 name='negative_penalty_categorical_crossentropy'):
        super(NegativePenaltyCategoricalCrossentropy, self).__init__(reduction=reduction, name=name)
        self.p_indices = [[p_index] for p_index in p_indices]
        self.alpha = alpha
        self.penalty_scale = float(len(p_indices)) if penalty_scale is None else penalty_scale
        self.penalty_label = _get_penalty_label(class_num, p_indices)
    
    def call(self, y_true, y_pred):
        losses = _get_losses(y_true, y_pred, self.p_indices, self.penalty_label, self.alpha, self.penalty_scale)
        return losses
    

class NegativePenaltySparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, class_num:int, p_indices:list, alpha=1.0, penalty_scale=None, \
                 reduction=tf.keras.losses.Reduction.AUTO, \
                 name='negative_penalty_sparse_categorical_crossentropy'):
        super(NegativePenaltySparseCategoricalCrossentropy, self).__init__(reduction=reduction, name=name)
        self.p_indices = [[p_index] for p_index in p_indices]
        self.alpha = alpha
        self.penalty_scale = float(len(p_indices)) if penalty_scale is None else penalty_scale
        self.penalty_label = _get_penalty_label(class_num, p_indices)

    def call(self, y_true, y_pred):
        num_classes = y_pred.shape[-1]
        y_true = tf.keras.utils.to_categorical(y_true, num_classes)
        losses = _get_losses(y_true, y_pred, self.p_indices, self.penalty_label, self.alpha, self.penalty_scale)
        return losses
    

def _get_losses(y_true, y_pred, p_indices:list, penalty_label:list, alpha:float, penalty_scale:float):
    batch_size = y_true.shape[0]
    y_true = tf.cast(y_true, tf.float32)
    # cce loss part for positive samples
    cce_loss_sample_weights = tf.cast(
        tf.reduce_any(
            tf.transpose(tf.math.equal(tf.math.argmax(y_true, axis=1), p_indices), perm=(1, 0)), axis=1
        ), 
        dtype=tf.float32
    )
    cce_losses = K.categorical_crossentropy(y_true, y_pred, from_logits=False)  # shape: (batch_size,)
    cce_losses = cce_loss_sample_weights * cce_losses
    # penalty loss part for negative samples
    y_penalty = tf.repeat(tf.expand_dims(tf.constant(penalty_label), axis=0), batch_size, axis=0)
    y_penalty = tf.cast(y_penalty, tf.float32)
    penalty_loss_sample_weights = tf.where(cce_loss_sample_weights == 1.0, 0.0, 1.0)  # 1.0: negative sample, 0.0: postive sample
    # option 1
    # penalty_losses = 1 / K.categorical_crossentropy(y_penalty, y_pred, from_logits=False)  # shape: (batch_size,)
    # option 2
    # penalty_losses = K.categorical_crossentropy(y_penalty, 1.0 - y_pred, from_logits=False)  # shape: (batch_size,)
    # option 3
    penalty_losses = -tf.math.reduce_sum(
        y_penalty * tf.math.log(tf.clip_by_value(1.0 - y_pred, K.epsilon(), 1.0 - K.epsilon())), 
        axis=-1
    )
    # scale penalty_losses
    penalty_losses = penalty_losses / penalty_scale
    penalty_losses = penalty_loss_sample_weights * penalty_losses
    # total loss
    losses = cce_losses + alpha * penalty_losses
    return losses


def _get_penalty_label(class_num:int, p_indices:list):
    penalty_label = [1 if i in p_indices else 0 for i in range(0, class_num)]
    return penalty_label