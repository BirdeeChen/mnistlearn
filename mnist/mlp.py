import random
import tensorflow as tf
from common import read_data_set, weights_biases
from common.mlpconfig import *

# 1st step: prepare data
trImages = read_data_set(TRAIN_IMAGES)
trLabels = read_data_set(TRAIN_LABELS, one_hot = True)
tsImages = read_data_set(TEST_IMAGES)
tsLabels = read_data_set(TEST_LABELS, one_hot = True)
num_samples = trImages.shape[0]
# print (trImage.shape, trLabel.shape, tsImage.shape, tsLabel.shape)
# 2nd step: define placeholders
X = tf.placeholder(tf.float32, shape = [None, N_FEATURES])
y_ = tf.placeholder(tf.float32, shape = [None, N_OUTPUT])
# 3rd step: define variables
W1, B1 = weights_biases(N_FEATURES, HIDDEN_1ST, 0.1)
W2, B2 = weights_biases(HIDDEN_1ST, HIDDEN_2ND, 0.1)
W3, B3 = weights_biases(HIDDEN_2ND, HIDDEN_3RD, 0.1)
W4, B4 = weights_biases(HIDDEN_3RD, HIDDEN_4TH, 0.1)
W_out, B_out = weights_biases(HIDDEN_4TH, N_OUTPUT, 0.1)
# 4th step: apply activation fucntion
Y1 = tf.sigmoid(tf.matmul(X, W1) + B1) # dropout may be applied here
Y2 = tf.sigmoid(tf.matmul(Y1, W2) + B2)  # dropout may be applied here
Y3 = tf.sigmoid(tf.matmul(Y2, W3) + B3)  # dropout may be applied here
Y4 = tf.sigmoid(tf.matmul(Y3, W4) + B4)  # dropout may be applied here
y_out = tf.matmul(Y4, W_out) + B_out  # dropout may be applied here
# 5th step: apply softmax to normalize the probability
y = tf.nn.softmax(y_out)
# 6th step: define the cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_out, labels = y_))
# 7th step: define correction measurement
is_correct = tf.equal(tf.argmax(y, axis = 1), tf.argmax(y_, axis = 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, dtype = tf.float32))
# 8th step: define training optimizer
train_op = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE).minimize(cost)
# 9th step: define initializer
init_op = tf.global_variables_initializer()
# 10th step: add summary
tf.summary.scalar('Cost', cost)
tf.summary.scalar('Prediction Accuracy', accuracy)
# 11th step: define summary_op
summary_op = tf.summary.merge_all()
# 12th step: define tf session
with tf.Session() as sess:
    # run parameter initialize
    sess.run(init_op)
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
    print ('Prediction Accuracy', sess.run(accuracy, feed_dict = {X : tsImages, y_ : tsLabels}))
    sess.close()
        


