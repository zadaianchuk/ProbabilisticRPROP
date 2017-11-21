# -*- coding: utf-8 -*-
"""
Run MomentumOptimizer on a test problem.

Runs tensorflow's built-in momentum optimizer (without Nesterov acceleration)
on a test problem from the tfobs package.
  - All necessary parameters and options are passed via the command line.
  - At regularly-spaced checkpoints during training, loss and accuracy are
    evaluated on training and test data.
  - All results are saved to pickle files and logged with tensorflow
    summaries. (The latter can be switched off.) We also log all tensorflow
    summaries created elsewhere, as long as they are tagged with the key
    "per_iteration".

Usage:

    python run_momentum.py <dataset>.<problem> --all_the_command_line_arguments

Execute python run_momentum.py --help to see a description for the command line
arguments. The command line arguments separate into the following categories:
  - arguments needed to set up the problem: "test_problem" (the first (and only)
    positional argument), "batch_size", "data_dir", "random_seed"
  - The basic mechanis: "num_steps", "checkpoint_interval", "eval_size"
  - Learning rate: "lr" for constant learning rates; "lr_sched_steps" and
    "lr_sched_vals" for step-wise schedules; other options might be added later
  - Optimizer parameters other than learning rates: only the momentum parameter
    "mu" in this case, but there might be more for other optimizers
  - Logging options: "train_log_interval", "saveto", "logdir", "nologs"
"""

import tensorflow as tf
import pickle
import argparse
import importlib
import time
import os

import tfobs
import probabilistic_rprop_optimizer3 as opts
reload(opts)

# ------- Parse Command Line Arguments ----------------------------------------
parser = argparse.ArgumentParser(description="Run MomentumOptimizer on a tfobs "
    "test problem.")
parser.add_argument("test_problem",
    help="Name of the test_problem (e.g. 'cifar10.cifar10_3c3d'")
parser.add_argument("--data_dir",
    help="Path to the base data dir. If not set, tfobs uses its default.")
parser.add_argument("--bs", "--batch_size", required=True, type=int,
    help="The batch size (positive integer).")

# Optimizer hyperparams other than learning rate
parser.add_argument("--delta_0",  type=float, default=0.0001,
    help="Initial value of step size")
parser.add_argument("--delta_min",  type=float, default=10**(-9),
    help="Min value of step size")
parser.add_argument("--delta_max",  type=float, default=0.05,
    help="Max value of step size")
parser.add_argument("--eta_minus",  type=float, default=0.5,
    help="Decrease of step size")
parser.add_argument("--eta_plus",  type=float, default=1.2,
    help="Increase of step size")
parser.add_argument("--eta_type",  type=str, default="linear",
    help="Type of eta function")
parser.add_argument("-MNS","--MAKE_NEG_STEP", dest='MAKE_NEG_STEP', action='store_true',
    help="If True we do make smaller step in case of sign switch")
parser.set_defaults(MAKE_NEG_STEP=True)
parser.add_argument("-NNS","--NO_NEG_STEP", dest='MAKE_NEG_STEP', action='store_false')

parser.add_argument("-UME","--USE_MINIBATCH_ESTIMATE", dest='USE_MINIBATCH_ESTIMATE', action='store_true',
    help="If True we use mini-batch estimate of varience")
parser.set_defaults(USE_MINIBATCH_ESTIMATE=True)
parser.add_argument("-UMA","--USE_MOVING", dest='USE_MINIBATCH_ESTIMATE', action='store_false')

parser.add_argument("-SS","--SOFT_SIGN",  dest='SOFT_SIGN', action='store_true',
    help="If True we do variance-adapted direction")
parser.set_defaults(SOFT_SIGN=True)
parser.add_argument("-NSS","--NO_SOFT_SIGN",  dest='SOFT_SIGN', action='store_false',
    help="If True we do variance-adapted direction")

parser.add_argument("--lr", type=float, default=1,
    help="Constant learning rate (positive float) to use. To set a learning "
    "rate *schedule*, do *not* set '--lr' and use '--lr_sched_steps' "
    "and '--lr_sched_values' instead.")
parser.add_argument("--p_min", type=float, default=0.25,
    help="The probability for theshold: if the probability of sign change is less"
    "than this we change to the case where sign of the product is positive one"
    "If the probability of sign change is larger then 1-p_min we change the  to the case where sign of the product is negative one"
    "if the probability of sign change is between p then 1-p_min we change the  to the case where sign of the product is zero")
parser.add_argument("--lr_sched_steps", nargs="+", type=int,
    help="One or more step numbers (positive integers) that mark learning rate "
    "changes, e.g., '--lr_sched_steps 2000 4000 5000' to change the learning "
    "rate after 2k, 4k and 5k steps. The corresponding learning rate values "
    "(!number of lr_sched_steps plus one!) have to be passed via "
    "'--lr_sched_vals'.")
parser.add_argument("--lr_sched_vals", nargs="+", type=float,
    help="Learning rate values in the learning rate schedule. Use in "
    "concurrence with '--lr_sched_steps'. Number of learning rate values "
    "must be one more than lr_sched_steps. The first value is the initial "
    "learning rate.")

# Number of steps and checkpoint interval
parser.add_argument("-N", "--num_steps", required=True, type=int,
                    help="Total number of training steps.")
parser.add_argument("-C", "--checkpoint_interval", required=True, type=int,
    help="Interval of training steps at which to evaluate on the test set and "
    "on a larger chunk of the training set.")
parser.add_argument("--eval_size", type=int, default=10000,
    help="Number of data points used for evaluation at checkpoints. This should "
    "usually be the test set size (the default is 10000, the test set size of "
    "MNIST, CIFAR). We evaluate on floor(eval_size/batch_size) batches. "
    "The number is the same for test and training evaluation.")

# Random seed
parser.add_argument("-r", "--random_seed", type=int, default=42,
    help="An integer to set as tensorflow's random seed.")

# Logging
parser.add_argument("--train_log_interval", type=int, default=10,
    help="The interval of steps at which the mini-batch training loss is "
    "logged. Set to 1 to log every training step. Default is 10.")
parser.add_argument("--print_train_iter", action="store_const",
    const=True, default=False,
    help="Add this flag to print training loss to stdout at each logged "
    "training step.")
parser.add_argument("--saveto",
    help="Folder for saving the results file. If not specified, defaults to "
    "'results/<test_problem>.' The directory will be created if it does not "
    "already exist.")
parser.add_argument("--logdir",
    help="Folder for tensorboard logging. Defaults to 'tflogs/<test_problem>'. "
    "The directory will be created if it does not already exist. This will be "
    "ignored if the '--nologs' flag is set.")
parser.add_argument("--nologs", action="store_const", const=True, default=False,
    help="Add this flag to switch off tensorflow logging.")

args = parser.parse_args()
# -----------------------------------------------------------------------------

# Create an identifier for this experiment
name = args.test_problem.split(".")[-1]
name += "__ProbRPROP"
name += "__bs_" + str(args.bs)
name += "__delta_0_" + tfobs.run_utils.float2str(args.delta_0)
name += "__delta_min_" + tfobs.run_utils.float2str(args.delta_min)
name += "__delta_max_" + tfobs.run_utils.float2str(args.delta_max)
name += "__p_min_" + str(args.p_min)
name += "__eta_minus_" + str(args.eta_minus)
name += "__eta_plus_" + str(args.eta_plus)
name += "__eta_type_" + args.eta_type
name += "__N_" + str(args.num_steps)
name += "__MNS_" + str(args.MAKE_NEG_STEP)
name += "__UME_" + str(args.USE_MINIBATCH_ESTIMATE)
name += "__SOFT_" + str(args.SOFT_SIGN)
name += "__seed_" + str(args.random_seed)
# Set the data dir
if args.data_dir is not None:
  tfobs.dataset_utils.set_data_dir(args.data_dir)

# Number of evaluation iterations
num_eval_iters = args.eval_size//args.bs

# Set up test problem
test_problem = importlib.import_module("tfobs."+args.test_problem)
tf.reset_default_graph()
tf.set_random_seed(args.random_seed) # Set random seed
losses, regularization_term, variables, phase, accuracy = test_problem.set_up(
    batch_size=args.bs)
loss = tf.reduce_mean(losses)
if regularization_term is not None:
  loss = loss + regularization_term

# Learning rate tensor; constant or schedule
global_step = tf.Variable(0, trainable=False)
learning_rate = tfobs.run_utils.make_learning_rate_tensor(global_step, args)

# Set up optimizer
opt =  opts.ProbRPROPOptimizer(delta_0=args.delta_0,
             delta_min=args.delta_min, delta_max=args.delta_max,
             eta_minus=args.eta_minus, eta_plus=args.eta_plus, p_min = args.p_min, eta_type = args.eta_type)
step = opt.minimize(losses, var_list=variables, global_step=global_step, USE_MINIBATCH_ESTIMATE=args.USE_MINIBATCH_ESTIMATE, MAKE_NEG_STEP=args.MAKE_NEG_STEP, SOFT_SIGN=args.SOFT_SIGN)

# Lists for tracking stuff
# train_<quantity>[i] is <quantity> after training for train_steps[i] steps
# checkpoint_<quantity>[i] is <quantity> after training for checkpoint_steps[i] steps
train_steps = []
train_losses = []
checkpoint_steps = []
checkpoint_train_losses = []
checkpoint_train_accuracies = []
checkpoint_test_losses = []
checkpoint_test_accuracies = []

# Tensorboard summaries
if not args.nologs:
  train_loss_summary = tf.summary.scalar("training_loss", loss,
      collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
  train_acc_summary = tf.summary.scalar("training_accuracy", accuracy,
      collections=[tf.GraphKeys.SUMMARIES, "per_iteration"])
  per_iteration_summaries = tf.summary.merge_all(key="per_iteration")
  logdir = args.logdir or "tflogs"
  logdir = os.path.join(logdir, args.test_problem.split(".")[-1], name)
  summary_writer = tf.summary.FileWriter(logdir)

# ------- start of train looop --------
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for n in range(args.num_steps+1):

  # Evaluate if applicable
  if n%args.checkpoint_interval==0 or n==args.num_steps:

    print "********************************"
    print "CHECKPOINT (", n, "of", args.num_steps, "steps )"

    checkpoint_steps.append(n)

    # Evaluate on training data
    train_loss_, train_acc_ = 0.0, 0.0
    for _ in range(num_eval_iters):
      l_, a_ = sess.run([loss, accuracy], {phase: "train_eval"})
      train_loss_ += l_
      train_acc_ += a_
    train_loss_ /= float(num_eval_iters)
    train_acc_ /= float(num_eval_iters)

    # Evaluate on test set
    test_loss_, test_acc_ = 0.0, 0.0
    for _ in range(num_eval_iters):
      l_, a_  = sess.run([loss, accuracy], {phase: "test"})
      test_loss_ += l_
      test_acc_ += a_
    test_loss_ /= float(num_eval_iters)
    test_acc_ /= float(num_eval_iters)

    # Append results to listsMNS
    checkpoint_train_losses.append(train_loss_)
    checkpoint_train_accuracies.append(train_acc_)
    checkpoint_test_losses.append(test_loss_)
    checkpoint_test_accuracies.append(test_acc_)

    # Log results to tensorflow summaries
    if not args.nologs:
      summary = tf.Summary()
      summary.value.add(tag="checkpoint_train_loss", simple_value=train_loss_)
      summary.value.add(tag="checkpoint_train_acc", simple_value=train_acc_)
      summary.value.add(tag="checkpoint_test_loss", simple_value=test_loss_)
      summary.value.add(tag="checkpoint_test_acc", simple_value=test_acc_)
      summary_writer.add_summary(summary, n)

    print "TRAIN: loss", train_loss_, "acc", train_acc_
    print "TEST: loss", test_loss_, "acc", test_acc_
    print "********************************"

    # Break from train loop after the last round of evaluation
    if n==args.num_steps:
      break

  # Training step, with logging if we hit the train_log_interval
  if n%args.train_log_interval!=0:
    _ = sess.run(step, {phase: "train"})
  else: # if n%args.train_log_interval==0:
    if not args.nologs:
      _, loss_, per_iter_summary_ = sess.run([step, loss, per_iteration_summaries],
                                             {phase: "train"})
      summary_writer.add_summary(per_iter_summary_, n)
    else:
      _, loss_ = sess.run([step, loss], {phase: "train"})
    train_steps.append(n)
    train_losses.append(loss_)
    if args.print_train_iter:
      print "Step", n, ": loss", loss_

sess.close()
# ------- end of train looop --------

# Put logged stuff into results dict
results = {
  "args" : args,
  "train_steps" : train_steps,
  "train_losses" : train_losses,
  "checkpoint_steps" : checkpoint_steps,
  "checkpoint_train_losses" : checkpoint_train_losses,
  "checkpoint_train_accuracies" : checkpoint_train_accuracies,
  "checkpoint_test_losses" : checkpoint_test_losses,
  "checkpoint_test_accuracies" : checkpoint_test_accuracies
  }

# Dump the results dict into a pickle file
folder = args.saveto or os.path.join("results", args.test_problem.split(".")[-1])
if not os.path.exists(folder):
  os.makedirs(folder)
fname = "results__"+name+"__"+time.strftime("%Y-%m-%d-%H-%M-%S")
with open(os.path.join(folder, fname+".pickle"), "w") as f:
  pickle.dump(results, f)
