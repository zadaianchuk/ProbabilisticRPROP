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

from full_graph import LR_SGD, CNN_SGD
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
parser.add_argument("-lr","--learning_rate", type=float,
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
LEARNING_RATE=0.005
if args.learning_rate:
    LEARNING_RATE = args.learning_rate
BATCH_SIZE=128
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
    # saver = tf.train.Saver() #to_save_variables

    # initial_step = 0
    # util.make_dir('checkpoints')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
        # if that checkpoint exists, restore from checkpoint
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps

        dir_to_save="summary/"+model.name+"/"
        unique_str=model.opt_name+", lr=" + str(model.lr) + ", batch_size=" + str(model.batch_size)
        writer = tf.summary.FileWriter(dir_to_save+unique_str, sess.graph)

        # initial_step = model.global_step.eval()
        for index in range(NUM_TRAIN_STEPS):
            batch = batch_gen.next_batch(model.batch_size)
            feed_dict_batch={model.x: batch[0], model.y_: batch[1]}
            loss_batch, _, fast_summary, hist_summary  = sess.run([model.loss, model.optimizer,
            model.fast_summary,model.hist_summary],
                                              feed_dict=feed_dict_batch)

            writer.add_summary(fast_summary, global_step=index)
            writer.add_summary(hist_summary, global_step=index)
            if (index + 1) % eval_every == 0:
                print('Average loss at step {}: {:5.3f}'.format(index, total_loss / eval_every))
                # total_loss = 0.0
                # loss = []
                # accuracy = []
                # when testing the classifier
                with tf.name_scope("streaming"):
                    # clear counters for a fresh evaluation
                    sess.run(tf.local_variables_initializer())
                for _ in range(NUM_VAL_STEPS+1):
                    batch = mnist.train.next_batch(BATCH_SIZE)
                    stream,slow_summary_train= sess.run([model.stream_fetches,model.slow_train_summary], {model.x: batch[0], model.y_: batch[1]})
                    total_loss=stream["loss"]
                #   loss.append(l)
                #   accuracy.append(a)
                # loss = np.mean(loss) #try cast
                # accuracy = np.mean(accuracy)
                # print ("train: loss", loss, "acc", accuracy)

                # sum_accuracy = tf.Summary()
                # sum_loss = tf.Summary()
                # sum_accuracy.value.add(tag="summaries/evaluation/train_accuracy", simple_value=sum_accuracy)
                # sum_loss.value.add(tag="summaries/evaluation/train_loss" , simple_value=sum_loss)
                # writer.add_summary(sum_loss, global_step=index)
                # writer.add_summary(sum_accuracy, global_step=index)
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
                # loss = []
                # accuracy = []
                # for _ in range(NUM_VAL_STEPS):
                #   batch = mnist.test.next_batch(BATCH_SIZE)
                #   l, a = sess.run([model.loss, model.accuracy], {model.x: batch[0], model.y_: batch[1]})
                #   loss.append(l)
                #   accuracy.append(a)
                # loss = np.mean(loss)
                # accuracy = np.mean(accuracy)
                # print ("test: loss", loss, "acc", accuracy)
                #
                # sum_accuracy = tf.Summary()
                # sum_loss = tf.Summary()
                # sum_accuracy.value.add(tag="summaries/evaluation/test_accuracy", value=sum_accuracy)
                # sum_loss.value.add(tag="summaries/evaluation/test_loss" , value=sum_loss)
                # writer.add_summary(sum_loss, global_step=index)
                # writer.add_summary(sum_accuracy, global_step=index)

                # # saver.save(sess, 'checkpoints/LR/RPROP', index)
                # feed_dict_train = {model.x: mnist.train.images, model.y_: mnist.train.labels} #big batch of data
                # feed_dict_test = {model.x: mnist.test.images, model.y_: mnist.test.labels}
                #
                # #calculate our summaries
                # loss_train,slow_summary_train = sess.run([model.loss, model.slow_train_summary],
                #                                   feed_dict=feed_dict_train)
                # accuracy_test,slow_summary_test = sess.run([model.accuracy, model.slow_test_summary],
                #                                   feed_dict=feed_dict_test)
                #
                # writer.add_summary(slow_summary_test, global_step=index)
                # writer.add_summary(slow_summary_train, global_step=index)

def main():
    models ={"LR":LR_SGD, "CNN":CNN_SGD}
    model = models[MODEL_NAME](IMAGE_SIZE, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE)
    model.build_graph()
    train_model(model, batch_gen, NUM_TRAIN_STEPS, EVAL_STEP)

if __name__ == '__main__':
    main()
