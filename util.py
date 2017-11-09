class Data():
    def __init__(self, parameters,loss,train_accuracy=None,test_accuracy=None,sign_changes=None,full_loss=None):
        self.full_loss = full_loss
        self.parameters=parameters
        self.loss=loss
        self.train_accuracy=train_accuracy
        self.test_accuracy=test_accuracy
        self.sign_changes=sign_changes


#change the list of dict to the dict of lists
from collections import defaultdict

def get_dict_of_lists(list_of_dicts):
    dd = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            dd[key].append(value)
    return dd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    def f1(): return 0.5 * tf.square(residual)
    def f2(): return delta * residual - 0.5 * tf.square(delta)
    return tf.cond(residual < delta, f1, f2)

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
