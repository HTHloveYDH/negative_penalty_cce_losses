import tensorflow as tf


from negative_penalty_accuracy_metrics import NegativePenaltyCategoricalAccuracy, \
    NegativePenaltySparseCategoricalAccuracy

if __name__ == '__main__':
    # for NegativePenaltyCategoricalAccuracy
    npca = NegativePenaltyCategoricalAccuracy(p_indices=[0])
    npca.update_state(tf.constant([[1, 0], [0, 1]]), tf.constant([[0.9, 0.1], [0.7, 0.3]]), None)
    acc = npca.result()
    is_succeed = (acc.numpy() - 0.5) < 0.0001
    print(f'acc: {acc.numpy()}')
    print('NegativePenaltyCategoricalAccuracy test', 'succeeded' if is_succeed else 'failed')

    # for NegativePenaltySparseCategoricalAccuracy
    npsca = NegativePenaltySparseCategoricalAccuracy(p_indices=[0])
    npsca.update_state(tf.constant([[0], [1]]), tf.constant([[0.9, 0.1], [0.7, 0.3]]), None)
    acc = npsca.result()
    is_succeed = abs((acc.numpy() - 0.5)) < 0.0001
    print(f'acc: {acc.numpy()}')
    print('NegativePenaltySparseCategoricalAccuracy test', 'succeeded' if is_succeed else 'failed')

