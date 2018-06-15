import tensorflow as tf

def weights_biases(in_shape, out_shape, stddev):
    return tf.Variable(initial_value = tf.truncated_normal((in_shape, out_shape), stddev = stddev)), tf.Variable(initial_value = tf.zeros((out_shape,)))