import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from importlib import reload
import argparse
import pickle

import rprop_optimizer as opts
import util
import models
reload(opts)
reload(util)
reload(models)


class model_SGD:
    """ Build the SGD graph without specific model """
    def __init__(self, image_size, num_classes, batch_size, learning_rate):
        self.image_size = image_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.opt_name="SGD"

    def _create_placeholders(self):
        """ Define the placeholders for input and output """
        with tf.name_scope("data"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.image_size],name='x')
            self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes],name='y_')

    def _create_inference_model(self):
        """Needed to be defined in children classes"""
        pass

    def _create_loss(self):
        """ Define the loss function """
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                                               logits=self.y),
                                       name = "loss" )

    def _create_optimizer(self):
        """ Define optimizer """
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
            # self.optimizer , self.sign_changes = opts.RPROPOptimizer(self.lr).minimize(self.loss)

    def _create_summaries(self):
        # building phase
        with tf.name_scope("streaming"):
            accuracy, acc_update_op = tf.metrics.accuracy(tf.argmax(self.y_,1), tf.argmax(self.y,1))
            loss,loss_op=tf.metrics.mean (self.loss)
            self.stream_fetches = {
                'accuracy': accuracy,
                'acc_op': acc_update_op,
                'loss': loss,
                'loss_op': loss_op
            }
        with tf.name_scope("summaries"):
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            with tf.name_scope("fast"):
                loss_sum = tf.summary.scalar("loss", self.loss, collections=['fast'])
                acc_sum = tf.summary.scalar("accuracy", self.accuracy, collections=['fast'] )
                self.fast_summary = tf.summary.merge([loss_sum,acc_sum])
                # because you have several summaries, we should merge them all
                # into one op to make it easier to manage
            with tf.name_scope("slow_test"):
                with tf.control_dependencies([self.stream_fetches['acc_op']]):
                    mean_accuracy=tf.summary.scalar('mean_accuracy', self.stream_fetches["accuracy"])
                    mean_loss=tf.summary.scalar('mean_loss', self.stream_fetches["loss"])
                # loss_sum2 = tf.summary.scalar("loss", self.loss, collections=['slow'])
                # acc_sum2 = tf.summary.scalar("accuracy", self.accuracy, collections=['slow'] )
                self.slow_test_summary = tf.summary.merge([mean_loss,mean_accuracy])
            with tf.name_scope("slow_train"):
                with tf.control_dependencies([self.stream_fetches['acc_op']]):
                    mean_accuracy=tf.summary.scalar('mean_accuracy', self.stream_fetches["accuracy"])
                    mean_loss=tf.summary.scalar('mean_loss', self.stream_fetches["loss"])
                # loss_sum2 = tf.summary.scalar("loss", self.loss, collections=['slow'])
                # acc_sum2 = tf.summary.scalar("accuracy", self.accuracy, collections=['slow'] )
                self.slow_train_summary = tf.summary.merge([mean_loss,mean_accuracy])

            with tf.name_scope("hist"):
                # Create summaries to visualize weights
                hist_list=[]
                for var in tf.trainable_variables():
                    print(var.name)
                    hist_list.append(tf.summary.histogram(var.name, var, collections=['hist']))
                self.hist_summary=tf.summary.merge(hist_list)

    def build_graph(self):
        """ Build the graph for our model """
        tf.reset_default_graph() #!!!
        self._create_placeholders()
        self._create_inference_model()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

class model_RPROP(model_SGD):
    """ Build the graph for RPROP without specific model """
    def __init__(self, image_size, num_classes, batch_size, delta_0, learning_rate):
        model_SGD.__init__(self,image_size, num_classes, batch_size, learning_rate)
        self.opt_name="RPROP"
        self._delta_0 = delta_0

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.name_scope('optimizer'):
            self.optimizer , self.tensors_for_summaries = opts.RPROPOptimizer(self.lr).minimize(self.loss)

    def _create_summaries(self):
        model_SGD._create_summaries(self)
        with tf.name_scope("summaries"):
            with tf.name_scope("fast"):
                hist_list_delta = []
                delta_hists = []
                for (i,delta) in enumerate(self.tensors_for_summaries['delta']):
                    delta_sum = tf.summary.histogram("delta_hist/"+str(i), delta)

                    value_range = [tf.reduce_min(delta),0.99*tf.reduce_max(delta)]
                    nbins=tf.constant(100)
                    delta_hist_counts = tf.histogram_fixed_width(values=delta, value_range=value_range, nbins=nbins, dtype=tf.float32)
                    delta_hist = {"counts":delta_hist_counts, "range": value_range, "nbins":nbins }

                    delta_hists.append(delta_hist)
                    hist_list_delta.append(delta_sum)

                self.delta_hists = delta_hists
                switch_sum = tf.summary.scalar("switch", self.tensors_for_summaries['sign']["switch"], collections=['fast'])
                self.fast_summary = tf.summary.merge([self.fast_summary, switch_sum])

class model_ProbRPROP(model_SGD):
    """ Build the graph for RPROP """
    def __init__(self, image_size, num_classes, batch_size, delta_0, learning_rate = 1):
        model_SGD.__init__(self, image_size, num_classes, batch_size, learning_rate)
        self.opt_name="ProbRPROP"
        self._delta_0 = delta_0
    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.name_scope('optimizer'):
            self.optimizer , self.tensors_for_summaries = opts.ProbRPROPOptimizer(delta_0=self._delta_0).minimize(self.loss)
    def _create_summaries(self):
        model_SGD._create_summaries(self)
        with tf.name_scope("summaries"):
            with tf.name_scope("fast"):
                mu_power_sum = tf.summary.scalar("mu_power",self.tensors_for_summaries['mu_power'])
                switch_sum = tf.summary.scalar("switch", self.tensors_for_summaries['sign']["switch"], collections=['fast'])
                switch_prob_sum = tf.summary.scalar("switch_prob>0.75", self.tensors_for_summaries['sign']["switch_prob"], collections=['fast'])
                hist_list_prob = []
                hist_list_snr = []
                hist_list_delta = []
                delta_hists = []
                for (i,prob) in enumerate(self.tensors_for_summaries['prob']):
                    prob_sum = tf.summary.histogram("switch_hist/"+str(i), prob)
                    hist_list_prob.append(prob_sum)
                for (i,snr) in enumerate(self.tensors_for_summaries['snr']):
                    snr_sum = tf.summary.histogram("snr_hist/"+str(i), snr)
                    hist_list_snr.append(snr_sum)

                for (i,delta) in enumerate(self.tensors_for_summaries['delta']):
                    delta_sum = tf.summary.histogram("delta_hist/"+str(i), delta)

                    value_range = [tf.reduce_min(delta),0.99*tf.reduce_max(delta)]
                    nbins=tf.constant(100)
                    delta_hist_counts = tf.histogram_fixed_width(values=delta, value_range=value_range, nbins=nbins, dtype=tf.float32)
                    delta_hist = {"counts":delta_hist_counts, "range": value_range, "nbins":nbins }

                    delta_hists.append(delta_hist)
                    hist_list_delta.append(delta_sum)

                self.delta_hists = delta_hists
                self.fast_summary = tf.summary.merge([mu_power_sum,self.fast_summary, switch_sum,switch_prob_sum, hist_list_prob, hist_list_snr,hist_list_delta])

class LR_SGD(model_SGD):
    """ Build the graph for SGD LR """
    def __init__(self, image_size, num_classes, batch_size, learning_rate):
        model_SGD.__init__(self,image_size, num_classes, batch_size, learning_rate)
        self.name="LR"
    def _create_inference_model(self):
        with tf.name_scope("inference"):
            self.y=models.logistic_regression(self.x)

class CNN_SGD(model_SGD):
    """ Build the graph for SGD CNN """
    def __init__(self, image_size, num_classes, batch_size, learning_rate):
        model_SGD.__init__(self,image_size, num_classes, batch_size, learning_rate)
        self.name="CNN"
    def _create_inference_model(self):
        with tf.name_scope("inference"):
            self.y=models.conv_network(self.x)


class LR_RPROP(model_RPROP):
    """ Build the graph for SGD CNN """
    def __init__(self, image_size, num_classes, batch_size, delta_0, learning_rate =1 ):
        model_RPROP.__init__(self,image_size, num_classes, batch_size, delta_0, learning_rate)
        self.name="LR"
    def _create_inference_model(self):
        with tf.name_scope("inference"):
            self.y=models.logistic_regression(self.x)

class CNN_RPROP(model_RPROP):
    """ Build the graph for SGD CNN """
    def __init__(self, image_size, num_classes, batch_size, delta_0, learning_rate =1 ):
        model_RPROP.__init__(self,image_size, num_classes, batch_size, delta_0, learning_rate)
        self.name="CNN"
    def _create_inference_model(self):
        with tf.name_scope("inference"):
            self.y=models.logistic_regression(self.x)


class LR_ProbRPROP(model_ProbRPROP):
    """ Build the graph for ProbRPOP LR"""
    def __init__(self, image_size, num_classes, batch_size, delta_0, learning_rate =1):
        model_ProbRPROP.__init__(self,image_size, num_classes, batch_size, delta_0, learning_rate)
        self.name="LR"

    def _create_inference_model(self):
        with tf.name_scope("inference"):
            self.y=models.logistic_regression(self.x)

class CNN_ProbRPROP(model_ProbRPROP):
    """ Build the graph for ProbRPROP CNN """
    def __init__(self, image_size, num_classes, batch_size, delta_0, learning_rate = 1):
        model_ProbRPROP.__init__(self,image_size, num_classes, batch_size,  delta_0, learning_rate)
        self.name="CNN"

    def _create_inference_model(self):
        with tf.name_scope("inference"):
            self.y=models.conv_network(self.x)
