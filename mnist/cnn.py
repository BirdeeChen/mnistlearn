import random
import tensorflow as tf
import numpy as np
from common import read_data_set
from common.cnnconfig import *

# 1st step: preparing data
trImages = read_data_set(TRAIN_IMAGES)
trLabels = read_data_set(TRAIN_LABELS, one_hot = True)
tsImages = read_data_set(TEST_IMAGES)
tsLabels = read_data_set(TEST_LABELS, one_hot = True)
num_samples = trImages.shape[0]
# reshape to apply convolution
trImages = np.reshape(trImages, (-1, INPUT_HEIGHT, INPUT_WIDTH, 1))
tsImages = np.reshape(tsImages, (-1, INPUT_HEIGHT, INPUT_WIDTH, 1))
# 2nd step: define func to initialize filter or max_pool kernel
def init_filter(shape):
    return tf.Variable(tf.random_normal(shape, stddev = STDDEV))
# 3rd step: define convolution model
def conv_model(train_set, filter_1st, filter_2nd, filter_3rd, full_connect, w_out, filter_strides, mp_kernel, mp_strides):
    # 1st convolution
    conv1 = tf.nn.conv2d(input = train_set, filter = filter_1st, strides = filter_strides, padding = 'SAME')
    print ('conv1.shape', conv1.shape)
    # apply non-linear transformation
    conv1 = tf.nn.relu(conv1)
    # apply max_pooling
    conv1 = tf.nn.max_pool(value = conv1, ksize = mp_kernel, strides = mp_strides, padding = 'SAME')
    print ('conv1.shape after max_pooling', conv1.shape)
    # apply drop-out goes here
    # conv1 = tf.nn.dropout(conv1, keep_prob = 0.8)
    # 2nd convolution
    conv2 = tf.nn.conv2d(input = conv1, filter = filter_2nd, strides = filter_strides, padding = 'SAME')
    print ('conv2.shape', conv2.shape)
    # apply non-liner transformation
    conv2 = tf.nn.relu(conv2)
    # apply max_pooling
    conv2 = tf.nn.max_pool(value = conv2, ksize = mp_kernel, strides = mp_strides, padding = 'SAME')
    print ('conv2.shape after max_pooling', conv2.shape)
    # apply drop_out goes here
    # conv2 = tf.nn.dropout(conv2, keep_prob = 0.8)
    # 3rd convolution
    conv3 = tf.nn.conv2d(input = conv2, filter = filter_3rd, strides = filter_strides, padding = 'SAME')
    print ('conv3.shape', conv3.shape)
    # apply non_linear transformation
    conv3 = tf.nn.relu(conv3)
    # apply max_pooling
    conv3 = tf.nn.max_pool(value = conv3, ksize = mp_kernel, strides = mp_strides, padding = 'SAME')
    print ('conv3.shape after max_pooling', conv3.shape)
    # apply drop_out here
    # conv3 = tf.nn.dropout(conv3, keep_prob = 0.8)
    # reshape the full connect layer
    fc_layer = tf.reshape(conv3, shape = [-1, full_connect.shape.as_list()[0]])
    # apply non_linear transformation
    fc_layer = tf.nn.relu(tf.matmul(fc_layer, full_connect))
    # drop_out goes here
    # fc_layer = tf.nn.dropout(fc_layer, keep_prob = 0.8)
    result = tf.matmul(fc_layer, w_out)
    return result
# 4th step: define filter and kernel
filter_1st = init_filter([FILTER_HEIGHT, FILTER_WIDTH, INPUT_CHANNEL, OUT_CHANNEL_1ST])
filter_2nd = init_filter([FILTER_HEIGHT, FILTER_WIDTH, OUT_CHANNEL_1ST, OUT_CHANNEL_2ND])
filter_3rd = init_filter([FILTER_HEIGHT, FILTER_WIDTH, OUT_CHANNEL_2ND, OUT_CHANNEL_3RD])
full_connect = init_filter([OUT_CHANNEL_3RD * 4 * 4, FC_CHANNEL]) # 4 * 4需要根据实际filter、max_pool kernel及strides 推算
w_out = init_filter([FC_CHANNEL, N_OUTPUT])
filter_strides = [1, STRIDE_HEIGHT, STRIDE_WIDTH, 1] # according to 'NHWC' data format
mp_kernel = [1, KERNEL_HEIGHT, KERNEL_WIDTH, 1]
mp_strides = [1, K_STRIDE_HEIGHT, K_STRIDE_WIDTH, 1]
# 5th step: define placeholders
X = tf.placeholder(dtype = tf.float32, shape = [None, INPUT_HEIGHT, INPUT_WIDTH, 1])
y_ = tf.placeholder(dtype = tf.float32, shape = [None, N_OUTPUT])
# 6th step: define prediction
ylogits = conv_model(X, filter_1st, filter_2nd, filter_3rd, full_connect, w_out, filter_strides, mp_kernel, mp_strides)
y = tf.nn.softmax(logits = ylogits)
# 7th step: define cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = ylogits, labels = y_))
# 8th stpe: define prediction accuracy
is_correct = tf.equal(tf.argmax(y, axis = 1), tf.argmax(y_, axis = 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, dtype = tf.float32))
# 9th step: define train optimizer
train_op = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE).minimize(cost)
# 10th step: add summary
tf.summary.scalar('Cost', cost)
tf.summary.scalar('Prediction Accuracy', accuracy)
summary_op = tf.summary.merge_all()
# 11th step: define
init = tf.global_variables_initializer()
# 12th step: start session
with tf.Session() as sess:
    # run parameter initialize
    sess.run(init)
    # define writer
    writer = tf.summary.FileWriter(LOG_PATH, graph = tf.get_default_graph())
    # run batched stochastic optimization
    for epoch in range(EPOCHS):
        # sample_batches is the group of start-end index
        batches_index = zip(range(0, num_samples, BATCH_SIZE), range(BATCH_SIZE, num_samples + 1, BATCH_SIZE))
        # randomly arrange the sample batches index
        stochastic_batches = [ix for ix in batches_index]
        random.shuffle(stochastic_batches) # open stochastic_batches will not influence the result much, but cause cost to oscillate
        num_batches = len(stochastic_batches)
        # start training
        for start, end in stochastic_batches:
            _, c, summary = sess.run([train_op, cost, summary_op], feed_dict = {X : trImages[start : end], y_ : trLabels[start : end]})
            writer.add_summary(summary, epoch * num_batches + start / BATCH_SIZE)
        print ('Epoch', epoch, 'Cost', c)
    print ('Optimization Done!')
    # calculate the prediction accuracy
    # Directily using this prediction accuracy will cause out of memory error
    # print ('Prediction Accuracy', sess.run(accuracy, feed_dict = {X : tsImages, y_ : tsLabels}))
    sess.close()


