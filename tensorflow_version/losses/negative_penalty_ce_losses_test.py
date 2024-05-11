import tensorflow as tf
import tensorflow.keras.backend as K

from negative_penalty_ce_losses import NegativePenaltyCategoricalCrossentropy, \
    NegativePenaltySparseCategoricalCrossentropy


if __name__ == '__main__':
    # for NegativePenaltyCategoricalCrossentropy
    # loss1
    cce_loss_sample_weights = tf.where(tf.math.argmax([[1.0, 0.0], [1.0, 0.0]], axis=1) < 1, 1.0, 0.0)
    cce_losses = K.categorical_crossentropy([[1.0, 0.0], [1.0, 0.0]], [[0.8, 0.2], [0.7, 0.3]], from_logits=False)  # shape: (batch_size,)
    cce_losses = cce_loss_sample_weights * cce_losses
    penalty_loss_sample_weights = tf.where(tf.math.argmax([[0.0, 1.0], [0.0, 1.0]], axis=1) < 1, 0.0, 1.0)
    # penalty_losses = K.categorical_crossentropy([[1.0, 0.0], [1.0, 0.0]], [[0.2, 0.8], [0.3, 0.7]], from_logits=False)  # shape: (batch_size,)
    penalty_losses = K.categorical_crossentropy([[1.0, 0.0], [1.0, 0.0]], [[1 - 0.2, 1 - 0.8], [1 - 0.3, 1 - 0.7]], from_logits=False)  # shape: (batch_size,)
    penalty_losses = penalty_losses / 1.0
    penalty_losses = penalty_loss_sample_weights * penalty_losses
    # loss1 = tf.math.reduce_sum(cce_losses + 1 / penalty_losses) / 4
    loss1 = tf.math.reduce_sum(cce_losses + penalty_losses) / 4
    # loss2
    loss2 = NegativePenaltyCategoricalCrossentropy(class_num=2, p_indices=[0], alpha=1.0)(
        tf.constant([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), 
        tf.constant([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]])
    )
    is_succeed = abs((loss1 - loss2).numpy()) < 0.0001
    print(f'loss1: {loss1.numpy()}', f'loss2: {loss2.numpy()}', f'loss1 - loss2: {(loss1 - loss2).numpy()}')
    print('NegativePenaltyCategoricalCrossentropy test', 'succeeded' if is_succeed else 'failed')

    # for NegativePenaltySparseCategoricalCrossentropy
    # loss1
    num_classes = tf.constant([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]]).shape[-1]
    y_true = tf.keras.utils.to_categorical(tf.constant([[0], [0]]), num_classes)
    y_true = tf.cast(y_true, tf.float32)
    cce_loss_sample_weights = tf.where(tf.math.argmax(y_true, axis=1) < 1, 1.0, 0.0)
    cce_losses = K.categorical_crossentropy([[1.0, 0.0], [1.0, 0.0]], [[0.8, 0.2], [0.7, 0.3]], from_logits=False)  # shape: (batch_size,)
    cce_losses = cce_loss_sample_weights * cce_losses
    y_true = tf.keras.utils.to_categorical(tf.constant([[1], [1]]), num_classes)
    y_true = tf.cast(y_true, tf.float32)
    penalty_loss_sample_weights = tf.where(tf.math.argmax([[0.0, 1.0], [0.0, 1.0]], axis=1) < 1, 0.0, 1.0)
    # penalty_losses = K.categorical_crossentropy([[1.0, 0.0], [1.0, 0.0]], [[0.2, 0.8], [0.3, 0.7]], from_logits=False)  # shape: (batch_size,)
    penalty_losses = K.categorical_crossentropy([[1.0, 0.0], [1.0, 0.0]], [[1 - 0.2, 1 - 0.8], [1 - 0.3, 1 - 0.7]], from_logits=False)  # shape: (batch_size,)
    penalty_losses = penalty_losses / 1.0
    penalty_losses = penalty_loss_sample_weights * penalty_losses
    # loss1 = tf.math.reduce_sum(cce_losses + 1 / penalty_losses) / 4
    loss1 = tf.math.reduce_sum(cce_losses + penalty_losses) / 4
    # loss2
    loss2 = NegativePenaltySparseCategoricalCrossentropy(class_num=2, p_indices=[0], alpha=1.0)(
        tf.constant([[0], [0], [1], [1]]), 
        tf.constant([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]])
    )
    is_succeed = abs((loss1 - loss2).numpy()) < 0.0001
    print(f'loss1: {loss1.numpy()}', f'loss2: {loss2.numpy()}', f'loss1 - loss2: {(loss1 - loss2).numpy()}')
    print('NegativePenaltySparseCategoricalCrossentropy test', 'succeeded' if is_succeed else 'failed')