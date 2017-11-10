""" RPROP training MNIST
"""

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from importlib import reload
import argparse
import pickle

from full_graph import LR_ProbRPROP, CNN_ProbRPROP
import util
reload(util)

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model", type=str,
                    help="type of the model")
parser.add_argument("-n","--num_steps", type=int,
                    help="itterations")
parser.add_argument("-eval","--eval_every", type=int,
                    help="Calculates summaries every $eval_every steps")
parser.add_argument("-bs","--batch_size", type=int,
                    help="train the model with given batch size")
parser.add_argument("-del_0","--delta", type=float,
                    help="learning_rate")
parser.add_argument("-r","--random_seed", type=int,
                    help="Fix Reandom_seed to reproduce the experiment")
parser.add_argument("--num_val_examples", type=int, default=20000)
args = parser.parse_args()

MODEL_NAME="LR"
if args.model:
    MODEL_NAME = args.model
#Train
NUM_TRAIN_STEPS=1000
if args.num_steps:
    NUM_TRAIN_STEPS = args.num_steps
EVAL_STEP = 10
if args.eval_every:
    EVAL_STEP = args.eval_every

#Oprimization
DELTA=0.001
if args.delta:
    DELTA = args.delta
BATCH_SIZE=100
if args.batch_size:
    BATCH_SIZE=args.batch_size
RANDOM_SEED=random.randint(1,1000)
if args.random_seed:
    RANDOM_SEED = args.random_seed
NUM_VAL_STEPS = int(args.num_val_examples // BATCH_SIZE)




mnist=input_data.read_data_sets('MNIST_data', one_hot=True)
batch_gen=mnist.train
# Global constants describing MNIST
NUM_CLASSES = 10
IMAGE_SIZE = 784

def train_model(model, batch_gen, num_train_steps,eval_every):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps

        dir_to_save="./summary_for_delta2/"+model.name+"/"
        unique_str=model.opt_name+", delta_0=" + str(model._delta_0) + ", batch_size=" + str(model.batch_size)
        writer = tf.summary.FileWriter(dir_to_save+unique_str, sess.graph)

        # initial_step = model.global_step.eval()
        delta_hists_100 = []
        for index in range(NUM_TRAIN_STEPS):
            batch = batch_gen.next_batch(model.batch_size)
            feed_dict_batch={model.x: batch[0], model.y_: batch[1]}
            loss_batch, _, fast_summary, hist_summary= sess.run([model.loss, model.optimizer,
            model.fast_summary,model.hist_summary],
                                              feed_dict=feed_dict_batch)
            if index<100:
                delta_hists  = sess.run(model.delta_hists,feed_dict=feed_dict_batch)
                delta_hists_100.append(delta_hists)
            writer.add_summary(fast_summary, global_step=index)
            writer.add_summary(hist_summary, global_step=index)
            if (index + 1) % eval_every == 0:
                print('Average loss at step {}: {:5.3f}'.format(index, total_loss / eval_every))
                # when testing the classifier
                with tf.name_scope("streaming"):
                    # clear counters for a fresh evaluation
                    sess.run(tf.local_variables_initializer())
                for _ in range(NUM_VAL_STEPS+1):
                    batch = mnist.train.next_batch(BATCH_SIZE)
                    stream,slow_summary_train= sess.run([model.stream_fetches,model.slow_train_summary], {model.x: batch[0], model.y_: batch[1]})
                total_loss=stream["loss"]
                writer.add_summary(slow_summary_train, global_step=index)

                with tf.name_scope("streaming"):
                    # clear counters for a fresh evaluation
                    sess.run(tf.local_variables_initializer())
                for _ in range(NUM_VAL_STEPS+1):
                    batch = mnist.test.next_batch(BATCH_SIZE)
                    stream,slow_summary_test= sess.run([model.stream_fetches,model.slow_test_summary], {model.x: batch[0], model.y_: batch[1]})
                    total_loss=stream["loss"]
                writer.add_summary(slow_summary_test, global_step=index)
        with open(dir_to_save+unique_str+'_delta_hists.pickle', 'wb') as f:
            pickle.dump(delta_hists_100, f)

def main():

    models ={"LR":LR_ProbRPROP, "CNN":CNN_ProbRPROP}
    model = models[MODEL_NAME](IMAGE_SIZE, NUM_CLASSES, BATCH_SIZE, delta_0=DELTA)
    model.build_graph()
    train_model(model, batch_gen, NUM_TRAIN_STEPS, EVAL_STEP)

if __name__ == '__main__':
    main()
