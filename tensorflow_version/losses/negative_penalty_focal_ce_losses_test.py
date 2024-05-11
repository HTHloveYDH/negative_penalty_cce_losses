import tensorflow as tf
import tensorflow.keras.backend as K

from negative_penalty_focal_ce_losses import NegativePenaltyCategoricalFocalCrossentropy, \
    NegativePenaltySparseCategoricalFocalCrossentropy


if __name__ == '__main__':
    # for NegativePenaltyCategoricalFocalCrossentropy
    # loss1
    cce_loss_sample_weights = tf.where(tf.math.argmax([[1.0, 0.0], [1.0, 0.0]], axis=1) < 1, 1.0, 0.0)
    cce_losses = K.categorical_focal_crossentropy(
        [[1.0, 0.0], [1.0, 0.0]], [[0.8, 0.2], [0.7, 0.3]], 1.0, 2.0, from_logits=False
    )  # shape: (batch_size,)
    cce_losses = cce_loss_sample_weights * cce_losses
    penalty_loss_sample_weights = tf.where(tf.math.argmax([[0.0, 1.0], [0.0, 1.0]], axis=1) < 1, 0.0, 1.0)
    # penalty_losses = K.categorical_focal_crossentropy(
    #     [[1.0, 0.0], [1.0, 0.0]], [[0.2, 0.8], [0.3, 0.7]], 1.0, 2.0, from_logits=False
    # )  # shape: (batch_size,)
    penalty_losses = K.categorical_focal_crossentropy(
        [[1.0, 0.0], [1.0, 0.0]], [[1 - 0.2, 1 - 0.8], [1 - 0.3, 1 - 0.7]], 1.0, 2.0, from_logits=False
    )  # shape: (batch_size,)
    penalty_losses = penalty_losses / 1.0
    penalty_losses = penalty_loss_sample_weights * penalty_losses
    # loss1 = tf.math.reduce_sum(cce_losses + 1 / penalty_losses) / 4
    loss1 = tf.math.reduce_sum(cce_losses + penalty_losses) / 4
    # loss2
    loss2 = NegativePenaltyCategoricalFocalCrossentropy(2, [0], 1.0, 1.0, 2.0)(
        tf.constant([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]), 
        tf.constant([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]])
    )
    is_succeed = abs((loss1 - loss2).numpy()) < 0.0001
    print(f'loss1: {loss1.numpy()}', f'loss2: {loss2.numpy()}', f'loss1 - loss2: {(loss1 - loss2).numpy()}')
    print('NegativePenaltyCategoricalFocalCrossentropy test', 'succeeded' if is_succeed else 'failed')

    # for NegativePenaltySparseCategoricalFocalCrossentropy
    # loss1
    num_classes = tf.constant([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]]).shape[-1]
    y_true = tf.keras.utils.to_categorical(tf.constant([[0], [0]]), num_classes)
    y_true = tf.cast(y_true, tf.float32)
    cce_loss_sample_weights = tf.where(tf.math.argmax(y_true, axis=1) < 1, 1.0, 0.0)
    cce_losses = K.categorical_focal_crossentropy(
        [[1.0, 0.0], [1.0, 0.0]], [[0.8, 0.2], [0.7, 0.3]], 1.0, 2.0, from_logits=False
    )  # shape: (batch_size,)
    cce_losses = cce_loss_sample_weights * cce_losses
    y_true = tf.keras.utils.to_categorical(tf.constant([[1], [1]]), num_classes)
    y_true = tf.cast(y_true, tf.float32)
    penalty_loss_sample_weights = tf.where(tf.math.argmax([[0.0, 1.0], [0.0, 1.0]], axis=1) < 1, 0.0, 1.0)
    # penalty_losses = K.categorical_focal_crossentropy(
    #     [[1.0, 0.0], [1.0, 0.0]], [[0.2, 0.8], [0.3, 0.7]], 1.0, 2.0, from_logits=False
    # )  # shape: (batch_size,)
    penalty_losses = K.categorical_focal_crossentropy(
        [[1.0, 0.0], [1.0, 0.0]], [[1 - 0.2, 1 - 0.8], [1 - 0.3, 1 - 0.7]], 1.0, 2.0, from_logits=False
    )  # shape: (batch_size,)
    penalty_losses = penalty_losses / 1.0
    penalty_losses = penalty_loss_sample_weights * penalty_losses
    # loss1 = tf.math.reduce_sum(cce_losses + 1 / penalty_losses) / 4
    loss1 = tf.math.reduce_sum(cce_losses + penalty_losses) / 4
    # loss2
    loss2 = NegativePenaltySparseCategoricalFocalCrossentropy(2, [0], 1.0, 1.0, 2.0)(
        tf.constant([[0], [0], [1], [1]]), 
        tf.constant([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]])
    )
    is_succeed = abs((loss1 - loss2).numpy()) < 0.0001
    print(f'loss1: {loss1.numpy()}', f'loss2: {loss2.numpy()}', f'loss1 - loss2: {(loss1 - loss2).numpy()}')
    print('NegativePenaltySparseCategoricalFocalCrossentropy test', 'succeeded' if is_succeed else 'failed')