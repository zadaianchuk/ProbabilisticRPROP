import tensorflow as tf

# Global constants describing MNIST
NUM_CLASSES = 10
IMAGE_SIZE = 784

def logistic_regression(images):
    with tf.variable_scope('fc1') as scope:
        W = weight_variable([IMAGE_SIZE,NUM_CLASSES])
        b = bias_variable([NUM_CLASSES])
        y = tf.matmul(images,W) + b

    return y

def conv_network(x):
    with tf.name_scope("conv_layer1"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)


    with tf.name_scope("conv_layer2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])


    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # add Dropout
    # I need to give additional tensor as input and
    # therefore can't just use different graph not changing the experiment function
    # to fix just create right feed_dict

    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return  y_conv


# to constract variables
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W")

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="b")

# to apply conv and pool layers
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
