import random
import tensorflow as tf
from common import read_data_set
from common.config import *

# 1st step: preparing data
trImages = read_data_set(TRAIN_IMAGES)
trLabels = read_data_set(TRAIN_LABELS, one_hot=True)
tsImages = read_data_set(TEST_IMAGES)
tsLabels = read_data_set(TEST_LABELS, one_hot=True)
num_samples = trImages.shape[0]
# print (trImages.shape, trLabels.shape, tsImages.shape, tsLabels.shape)
#2nd step: define placeholders
X = tf.placeholder(tf.float32, shape=[None, N_FEATURES])#images to train or test
y_ = tf.placeholder(tf.float32, shape=[None, N_OUTPUT])#labels to train or test
#3rd step: define weight and bias
W = tf.Variable(initial_value = tf.zeros((N_FEATURES, N_OUTPUT)))
B = tf.Variable(initial_value = tf.zeros((N_OUTPUT,))) # no need to specify num_samples, cause tf will do the broadcasting
#4th step: define how output is calculated, the predicted y, using softmax to normalize(1 hidden layer)
y_logits = tf.sigmoid(tf.matmul(X, W) + B)
# y_logits = tf.matmul(X, W) + B
y = tf.nn.softmax(y_logits)
#5th step: define the cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_logits, labels = y_))
#6th step: define the accuracy measurement formula
is_correct = tf.equal(tf.argmax(y, axis = 1), tf.argmax(y_, axis = 1)) # tf.argmax axis = 1 means compare over columns
accuracy = tf.reduce_mean(tf.cast(is_correct, dtype = tf.float32))
#7th step: define the optimizer
train_op = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE).minimize(cost)
#8th step: add visualization
tf.summary.scalar('Cost', cost)
tf.summary.scalar('Prediction Accuracy', accuracy)
summary_op = tf.summary.merge_all()
#9th step: variable initialization
init = tf.global_variables_initializer()
#10 step: start tf session
with tf.Session() as sess:
    # run initialization
    sess.run(init)
    # log_writer
    writer = tf.summary.FileWriter(LOG_PATH, graph = tf.get_default_graph())
    # apply batched stochastic gradient descent
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
