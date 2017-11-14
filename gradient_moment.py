# -*- coding: utf-8 -*-
"""
Computation of *moments* of gradients through tensorflow operations.

Tensorflow is typically used for empircal risk minimzation with gradient-based
optimization methods. That is, we want to adjust trainable variables ``W``,
such as to minimize an objective quantity, called ``LOSS``, of the form

    LOSS(W) = (1/m) * sum{i=1:m}[ loss(W, x_i) ]

That is the mean of individual losses induced by ``m`` training data points
``x_i``. Consquently, the gradient of ``LOSS`` w.r.t. the variables ``W`` is
the mean of individual gradients ``dloss(W, x_i)``. These individual gradients
are not computed separately when we call ``tf.gradients`` on the aggregate
``LOSS``. Instead, they are implicitly aggregated by the operations in the
backward graph. This batch processing is crucial for the computational
efficiency of the gradient computation.

This module provides functionality to compute the (element-wise) second moment
of gradients, i.e., the quantity

    MOM(W) = (1/m) * sum{i=1:m}[ dloss(w, x_i)**2 ]

without giving up the efficiency of batch processing. For a detailed 
explanation, see the note [1]. Applications of this are the computation of the 
gradient variance estimate in [2] and [3].

[1] https://drive.google.com/open?id=0B0adgqwcMJK5aDNaQ2Q4ZmhCQzA

[2] M. Mahsereci and P. Hennig. Probabilistic line searches for stochastic
optimization. In Advances in Neural Information Processing Systems 28, pages
181-189, 2015.
http://papers.nips.cc/paper/5753-probabilistic-line-searches-for-stochastic-optimization

[3] L. Balles, J. Romero and P. Hennig. Coupling Adaptive Batch Sizes with
Learning Rates. In Thirty-Third Conference on Uncertainty in Artificial
Intelligence (UAI).
https://arxiv.org/abs/1612.05086.
"""

import tensorflow as tf
from tensorflow.python.ops import gen_array_ops

VALID_TYPES = ["MatMul", "Conv2D", "Add"]
REGULARIZATION_TYPES = ["L2Loss"]

def grads_and_grad_moms(losses, var_list):
  """Compute the gradients and gradient moments of ``loss`` w.r.t. to the
  variables in ``var_list``
  
  Inputs:
      :losses: The 1D tensor containing the m scalar losses, one for each
        example in the batch.
      :var_list: The list of variables, for which the gradients and gradient
        moments are to be computed.
  
  Returns:
      :grads: The gradients of ``loss`` w.r.t. the variables in ``var_list``
          as computed by ``tf.gradients(loss, var_list)``.
      :grad_moms: The gradient moments for each variable in ``var_list``.
      :loss: The mean loss ``loss=tf.reduce_mean(losses)
      :batch_size: A tensor containing the batch size."""
  
  # Make sure that var_list is a duplicate-free list of tf Variables/Tensors
  if not isinstance(var_list, list) or \
     not all([isinstance(v, (tf.Variable, tf.Tensor)) for v in var_list]):
    raise TypeError("var_list should be a list of tf.Variable/tf.Tensor")
  if len(set(var_list))!=len(var_list):
    raise ValueError("var_list contains duplicates.")
  
  # Make sure that losses is a rank 1 tensor
  if not isinstance(losses, tf.Tensor):
    raise TypeError("'losses' needs to be a tf.Tensor")
  if len(losses.shape.as_list())!=1:
    raise ValueError("'losses' has to be a rank-1 (i.e. vectorial) tensor"
        "containing one loss for each example in the mini-batch.")
  
  # Convert variables to tensors
  vs = [tf.convert_to_tensor(v) for v in var_list]
  num_vars = len(vs)
  
  # Extract dtype of losses
  dtype = losses.dtype
  
  # Extract batch size and mean loss
  loss = tf.reduce_mean(losses)
  batch_size = tf.cast(tf.size(losses), dtype, name="batch_size")
  
  # Collect all consumer operations and their outputs
  consumers = []
  consumer_outs = []
  for v in vs:
    v_consumers = [op for op in v.consumers() if op.type not in REGULARIZATION_TYPES]
    if len(v_consumers) > 1:
      raise ValueError("Variable {} is consumed by more than one operation.".format(v.name))
    op = v_consumers[0]
    if op.type in VALID_TYPES:
      consumers.append(op)
      consumer_outs.extend(op.outputs) # TODO: op.outputs is a one-elt list for all valid types!? 
    else:
      raise ValueError("Variable {} is consumed by an operation of type {}, "
          "for which I don't how to compute the gradient moment. "
          "Allowed consumer operation types are {}. "
          "Do not add regularization operations.".format(op.type, str(VALID_TYPES)))
  
  # Use tf.gradients to compute gradients w.r.t. the variables, while also
  # retrieving gradients w.r.t. the outputs
  all_grads = tf.gradients(loss, vs+consumer_outs)
  v_grads = all_grads[0:num_vars]
  out_grads = all_grads[num_vars::]
    
  # Compute the gradient moment for each variable, given its consumer op and
  # the gradient w.r.t. the output thereof
  with tf.name_scope("grad_moms"):
    grad_moms = [_GradMom(o, v, out_grad, batch_size)
                for o, v, out_grad in zip(consumers, vs, out_grads)]
  
  return v_grads, grad_moms, loss, batch_size

def _GradMom(op, v, out_grad, batch_size):
  """Wrapper function for the operation type-specific GradMom functions below.
  
  Inputs:
      :op: A tensorflow operation of type in VALID_TYPES.
      :v: The read-tensor of the trainable variable consumed by this operation.
      :out_grad: The tensor containing the gradient w.r.t. to the output of
          the op (as computed by ``tf.gradients``).
      :batch_size: Batch size ``m`` (constant integer or scalar int tf.Tensor)"""
  
  with tf.name_scope(op.name+"_grad_mom"):
    if op.type == "MatMul":
      return _MatMulGradMom(op, v, out_grad, batch_size)
    elif op.type == "Conv2D":
      return _Conv2DGradMom(op, v, out_grad, batch_size)
    elif op.type == "Add":
      return _AddGradMom(op, v, out_grad, batch_size)
    else:
      raise ValueError("Don't know how to compute gradient moment for "
          "variable {}, consumed by operation of type {}".format(v.name,
          op.type))

def _MatMulGradMom(op, W, out_grad, batch_size):
  """Computes gradient moment for a weight matrix through a MatMul operation.
  
  Assumes ``Z=tf.matmul(A, W)``, where ``W`` is a d1xd2 weight matrix, ``A``
  are the nxd1 activations of the previous layer (n being the batch size).
  ``out_grad`` is the gradient w.r.t. ``Z``, as computed by ``tf.gradients()``.
  No transposes in the MatMul operation allowed.
  
  Inputs:
      :op: The MatMul operation
      :W: The weight matrix (the tensor, not the variable)
      :out_grad: The tensor of gradient w.r.t. to the output of the op
      :batch_size: Batch size n (constant integer or scalar int tf.Tensor)"""
  
  assert op.type == "MatMul"
  t_a, t_b = op.get_attr("transpose_a"), op.get_attr("transpose_b")
  assert W is op.inputs[1] and not t_a and not t_b
  
  A = op.inputs[0]
  out_grad_squ = tf.square(out_grad)
  A_squ = tf.square(A)
  return tf.multiply(batch_size, tf.matmul(A_squ, out_grad_squ, transpose_a=True))

def _Conv2DGradMom(op, f, out_grad, batch_size):
  """Computes gradient moment for the filter of a Conv2D operation.
  
  Assumes ``Z=tf.nn.conv2d(A, f)``, where ``f`` is a ``[h_f, w_f, c_in, c_out]``
  convolution filter and ``A`` are the ``[n, h_in, w_in, c_in]`` activations of
  the previous layer (``n`` being the batch size). ``out_grad`` is the gradient
  w.r.t. ``Z``, as computed by ``tf.gradients()``.
  
  Inputs:
      :op: The Conv2D operation
      :f: The filter (the tensor, not the variable)
      :out_grad: The tensor of gradient w.r.t. to the output of the op
      :batch_size: Batch size ``n`` (constant integer or scalar int tf.Tensor)"""
  
  assert op.type == "Conv2D"
  assert f is op.inputs[1]
  
  strides = op.get_attr("strides")
  padding = op.get_attr("padding")
  use_cudnn = op.get_attr("use_cudnn_on_gpu")
  data_format = op.get_attr("data_format")
  # TODO: What to do with those last two attributes?
  assert data_format=="NHWC"
  
#  if strides!=[1, 1, 1, 1]:
#    raise Warning("Convolutions with stride >1 might cause problems for the"
#                  "gradient moment computation for some combinations of"
#                  "filter size, input size, and stride. Sorry!")
  
  inp = op.inputs[0]
  h_f, w_f, c_in, c_out = f.get_shape().as_list()
  inp_patches = tf.extract_image_patches(inp, ksizes=[1, h_f, w_f, 1 ],
                                         strides=strides, rates=[1, 1, 1, 1],
                                         padding=padding)
  f_grads_batch = tf.einsum("ijkp,ijkq->ipq", inp_patches, out_grad)
  f_grad_mom = tf.multiply(batch_size, tf.reduce_sum(tf.square(f_grads_batch),
                                                     axis=0))
  return tf.reshape(f_grad_mom, [h_f, w_f, c_in, c_out])

def _AddGradMom(op, b, out_grad, batch_size):
  """Computes gradient moment for a bias variable through an Add operation.
  
  Assumes ``Zz = tf.add(Z, b)``, where ``b`` is a bias parameter and ``Z`` is
  a ``[n, ?]`` tensor (``n`` being the batch size). Broadcasting for all kinds
  of shapes of ``Z`` (e.g. ``[n, d_in]`` or ``[n, h_in, w_in, c_in]`` are
  supported. ``out_grad`` is the gradient w.r.t. ``Z``, as computed by
  ``tf.gradients()``.
  
  Inputs:
      :op: The Add operation
      :b: The bias parameter (the tensor, not the variable)
      :out_grad: The tensor of gradient w.r.t. to the output of the op
      :batch_size: Batch size ``n`` (constant integer or scalar int tf.Tensor)"""
  
  assert op.type == "Add"
  
  # Get the OTHER input of the op, i.e., the tensor Z that b is added to
  if b is op.inputs[0]:
    Z = op.inputs[1]
  elif b is op.inputs[1]:
    Z = op.inputs[0]
  else:
    raise ValueError("b is not an input of op.")
  
  # Extract shapes of b and Z, get the reduction indices from broadcast_gradient_args
  shape_b = tf.shape(b)
  shape_Z = tf.shape(Z)
  rb, _ = gen_array_ops._broadcast_gradient_args(shape_b, shape_Z)
  
  # Sum out_grad accross these reduction indices, excluding the batch dimension,
  # to get the individual gradients
  b_grads_batch = tf.reduce_sum(out_grad, rb[1:])  
  return tf.multiply(batch_size, tf.reduce_sum(tf.square(b_grads_batch), axis=0))
