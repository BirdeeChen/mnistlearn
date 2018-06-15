# define constant variables
# dataset directory
TRAIN_IMAGES = 'temp/train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'temp/train-labels-idx1-ubyte.gz'
TEST_IMAGES = 'temp/t10k-images-idx3-ubyte.gz'
TEST_LABELS = 'temp/t10k-labels-idx1-ubyte.gz'
# picture size
N_FEATURES = 28 * 28
# output size
N_OUTPUT = 10
# train parameters
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 100
# log file path
LOG_PATH = 'mnist_log_MLP'
# Hidden layers dimension
HIDDEN_1ST = 200
HIDDEN_2ND = 100
HIDDEN_3RD = 60
HIDDEN_4TH = 30
