import tensorflow as tf

# Global constants describing MNIST
NUM_CLASSES = 10
IMAGE_SIZE = 784

def logistic_regression(images):
    with tf.variable_scope('fc1') as scope:
        W = weight_variable([IMAGE_SIZE,NUM_CLASSES])
        b = bias_variable([NUM_CLASSES])
        y = tf.matmul(images,W) + b

        # tf.summary.histogram("weights", W, collections=['hist'])
        # tf.summary.histogram("bias", b, collections=['hist'])
        # tf.summary.histogram("activations",y, collections=['hist'])
    return y

def conv_network(x):
    with tf.name_scope("conv_layer1"):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # tf.summary.histogram("weights", W_conv1, collections=['hist'])
        # tf.summary.histogram("bias", b_conv1, collections=['hist'])
        # tf.summary.histogram("activations",h_conv1, collections=['hist'])

    with tf.name_scope("conv_layer2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

        # tf.summary.histogram("weights", W_conv1, collections=['hist'])
        # tf.summary.histogram("bias", b_conv1, collections=['hist'])
        # tf.summary.histogram("activations",h_conv1, collections=['hist'])

    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # tf.summary.histogram("weights", W_conv1, collections=['hist'])
        # tf.summary.histogram("bias", b_conv1, collections=['hist'])
        # tf.summary.histogram("activations",h_conv1, collections=['hist'])

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

        # tf.summary.histogram("weights", W_conv1, collections=['hist'])
        # tf.summary.histogram("bias", b_conv1, collections=['hist'])
        # tf.summary.histogram("activations",h_conv1, collections=['hist'])
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

# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# import argparse
# import os
# import re
# import sys
# import tarfile
#
# from six.moves import urllib
#
# from importlib import reload
# import cifar10_input
#
# # Global constants describing the CIFAR-10 data set.
# IMAGE_SIZE = cifar10_input.IMAGE_SIZE
# NUM_CLASSES = cifar10_input.NUM_CLASSES
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
#
# DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
# TOWER_NAME = 'tower'
#
# class Flags():
#     def __init__(self, batch_size=32, data_dir='/tmp/cifar10_data', use_fp16=False):
#         self.batch_size=batch_size
#         self.data_dir = data_dir
#         self.use_fp16 = use_fp16
#
# FLAGS = Flags()
#
# # use for CIFAR-10
# def inference(images):
#     """Build the CIFAR-10 model.
#
#     Args:
#       images: Images returned from distorted_inputs() or inputs().
#
#     Returns:
#       Logits.
#     """
#     # We instantiate all variables using tf.get_variable() instead of
#     # tf.Variable() in order to share variables across multiple GPU training runs.
#     # If we only ran this model on a single GPU, we could simplify this function
#     # by replacing all instances of tf.get_variable() with tf.Variable().
#     #
#     # conv1
#     with tf.variable_scope('conv1') as scope:
#         kernel = _variable_with_weight_decay('weights',
#                                              shape=[5, 5, 3, 64],
#                                              stddev=5e-2,
#                                              wd=0.0)
#         conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv1 = tf.nn.relu(pre_activation, name=scope.name)
#         _activation_summary(conv1)
#
#     # pool1
#     pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
#                            padding='SAME', name='pool1')
#     # norm1
#     norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                       name='norm1')
#
#     # conv2
#     with tf.variable_scope('conv2') as scope:
#         kernel = _variable_with_weight_decay('weights',
#                                              shape=[5, 5, 64, 64],
#                                              stddev=5e-2,
#                                              wd=0.0)
#         conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
#         biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
#         pre_activation = tf.nn.bias_add(conv, biases)
#         conv2 = tf.nn.relu(pre_activation, name=scope.name)
#         _activation_summary(conv2)
#
#     # norm2
#     norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
#                       name='norm2')
#     # pool2
#     pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
#                            strides=[1, 2, 2, 1], padding='SAME', name='pool2')
#
#     # local3
#     with tf.variable_scope('local3') as scope:
#         # Move everything into depth so we can perform a single matrix multiply.
#         reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
#         dim = reshape.get_shape()[1].value
#         weights = _variable_with_weight_decay('weights', shape=[dim, 384],
#                                               stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
#         local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
#         _activation_summary(local3)
#
#     # local4
#     with tf.variable_scope('local4') as scope:
#         weights = _variable_with_weight_decay('weights', shape=[384, 192],
#                                               stddev=0.04, wd=0.004)
#         biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
#         local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
#         _activation_summary(local4)
#
#     # linear layer(WX + b),
#     # We don't apply softmax here because
#     # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
#     # and performs the softmax internally for efficiency.
#     with tf.variable_scope('softmax_linear') as scope:
#         weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
#                                               stddev=1/192.0, wd=0.0)
#         biases = _variable_on_cpu('biases', [NUM_CLASSES],
#                                   tf.constant_initializer(0.0))
#         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
#         _activation_summary(softmax_linear)
#
#     return softmax_linear
#
# def _activation_summary(x):
#     """Helper to create summaries for activations.
#
#     Creates a summary that provides a histogram of activations.
#     Creates a summary that measures the sparsity of activations.
#
#     Args:
#       x: Tensor
#     Returns:
#       nothing
#     """
#     # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#     # session. This helps the clarity of presentation on tensorboard.
#     tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
#     tf.summary.histogram(tensor_name + '/activations', x)
#     tf.summary.scalar(tensor_name + '/sparsity',
#                                          tf.nn.zero_fraction(x))
#
#
# def _variable_on_cpu(name, shape, initializer):
#     """Helper to create a Variable stored on CPU memory.
#
#     Args:
#       name: name of the variable
#       shape: list of ints
#       initializer: initializer for Variable
#
#     Returns:
#       Variable Tensor
#     """
#     with tf.device('/cpu:0'):
#         dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
#         var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
#     return var
#
#
# def _variable_with_weight_decay(name, shape, stddev, wd):
#     """Helper to create an initialized Variable with weight decay.
#
#     Note that the Variable is initialized with a truncated normal distribution.
#     A weight decay is added only if one is specified.
#
#     Args:
#       name: name of the variable
#       shape: list of ints
#       stddev: standard deviation of a truncated Gaussian
#       wd: add L2Loss weight decay multiplied by this float. If None, weight
#           decay is not added for this Variable.
#
#     Returns:
#       Variable Tensor
#     """
#     dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
#     var = _variable_on_cpu(
#         name,
#         shape,
#         tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
#     if wd is not None:
#         weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#         tf.add_to_collection('losses', weight_decay)
#     return var
