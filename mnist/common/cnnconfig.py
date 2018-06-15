# define constant variables
# dataset directory
TRAIN_IMAGES = 'temp/train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'temp/train-labels-idx1-ubyte.gz'
TEST_IMAGES = 'temp/t10k-images-idx3-ubyte.gz'
TEST_LABELS = 'temp/t10k-labels-idx1-ubyte.gz'
# picture size
INPUT_HEIGHT = 28
INPUT_WIDTH = 28
INPUT_CHANNEL = 1
# output size
N_OUTPUT = 10
# train parameters
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 100
# log file path
LOG_PATH = 'mnist_log_cnn'
# filter kernel dimension, 3 layers of convolution
FILTER_HEIGHT = 3
FILTER_WIDTH = 3
OUT_CHANNEL_1ST = 32
OUT_CHANNEL_2ND = 64
OUT_CHANNEL_3RD = 128
FC_CHANNEL = 625
# filter Strides dimension
STRIDE_HEIGHT = 1
STRIDE_WIDTH = 1
# max_pooling kernel dimension
KERNEL_HEIGHT = 2
KERNEL_WIDTH = 2
# max_pooling stride dimension
K_STRIDE_HEIGHT = 2
K_STRIDE_WIDTH = 2
# parameter initialize stddev
STDDEV = 0.01

