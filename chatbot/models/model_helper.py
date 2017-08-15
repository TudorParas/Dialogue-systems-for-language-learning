# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for building models."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

import utils.misc_utils as utils


def get_device_str(device_id, num_gpus):
    """Return a device string for multi-GPU setup."""
    if num_gpus == 0:
        return "/cpu:0"
    device_str_output = "/gpu:%d" % (device_id % num_gpus)
    return device_str_output


def create_embeddings(vocab_size, embedding_size, dtype=tf.float32, scope=None):
    """
    Returns a word embedding matrix
    """
    with tf.variable_scope(scope or "embeddings", dtype=dtype) as scope:
        embedding = tf.get_variable(name="embedding", shape=[vocab_size, embedding_size],
                                    dtype=dtype)

    return embedding


def _single_cell(unit_type, num_units, forget_bias, dropout, residual_connection=False,
                  device_str=None, verbose=True):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    # forget_bias not meaningful when the unit is of type GRU

    # Cell Type
    if unit_type == "lstm":
        if verbose:
            utils.print_out("LSTM, forget_bias=%g" % forget_bias, new_line=False)
        single_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units,
            forget_bias=forget_bias)
    elif unit_type == "gru":
        if verbose:
            utils.print_out("  GRU", new_line=False)
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        if verbose:
            utils.print_out("  Layer Normalized LSTM, forget_bias=%g" % forget_bias,
                        new_line=False)
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))
        if verbose:
            utils.print_out("  %s, dropout=%g " % (type(single_cell).__name__, dropout),
                        new_line=False)

    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell)
        if verbose:
            utils.print_out("  %s" % type(single_cell).__name__, new_line=False)

    # Device Wrapper
    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)
        if verbose:
            utils.print_out("  %s, device=%s" %
                        (type(single_cell).__name__, device_str), new_line=False)

    return single_cell


def _cell_list(unit_type, num_units, num_layers, num_residual_layers,
               forget_bias, dropout, mode, num_gpus, base_gpu=0, verbose=True):
    """Create a list of RNN cells."""
    # Multi-GPU
    cell_list = []
    for i in range(num_layers):
        if verbose:
            utils.print_out("  cell %d" % i, new_line=False)
        dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0  # Disable dropout outside training.
        single_cell = _single_cell(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            residual_connection=(i >= num_layers - num_residual_layers),  # Apply residual wrapper to last layers.
            device_str=get_device_str(i + base_gpu, num_gpus),  # Parallelize computation over GPUs
            verbose=verbose  # Whether to print to stdout
        )
        if verbose:
            utils.print_out("")  # create new line
        cell_list.append(single_cell)

    return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode, num_gpus, base_gpu=0, verbose=True):
    """Create multi-layer RNN cell.

    Args:
      unit_type: string representing the unit type, i.e. "lstm".
      num_units: the depth of each unit.
      num_layers: number of cells.
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_layers=4` and `num_residual_layers=2`, the last 2 RNN
        cells in the returned list will be wrapped with `ResidualWrapper`.
      forget_bias: the initial forget bias of the RNNCell(s).
      dropout: floating point value between 0.0 and 1.0:
        the probability of dropout.  this is ignored if `mode != TRAIN`.
      mode: either tf.contrib.learn.TRAIN/EVAL/INFER
      num_gpus: The number of gpus to use when performing round-robin
        placement of layers.
      base_gpu: The gpu device id to use for the first RNN cell in the
        returned list. The i-th RNN cell will use `(base_gpu + i) % num_gpus`
        as its device id.

    Returns:
      An `RNNCell` instance.
    """
    cell_list = _cell_list(unit_type=unit_type,
                           num_units=num_units,
                           num_layers=num_layers,
                           num_residual_layers=num_residual_layers,
                           forget_bias=forget_bias,
                           dropout=dropout,
                           mode=mode,
                           num_gpus=num_gpus,
                           base_gpu=base_gpu,
                           verbose=verbose)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)


def gradient_clip(gradients, params, max_gradient_norm):
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

    # Define summary
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
        tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary


def load_model(model, ckpt, session, name, verbose=True):
    # Load the model from checkpoint
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    if verbose:
        utils.print_out(
        "  loaded %s model parameters from %s, time %.2fs" %
        (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name, verbose=True):
    """Create model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name, verbose)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        utils.print_out("  created %s model with fresh parameters, time %.2fs" %
                        (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step


def compute_perplexity(model, sess, name):
    """Compute perplexity of the output of the model.

    Args:
      model: model for compute perplexity.
      sess: tensorflow session to use.
      name: name of the batch.

    Returns:
      The perplexity of the eval outputs.
    """
    total_loss = 0
    total_predict_count = 0
    start_time = time.time()

    while True:
        try:
            loss, predict_count, batch_size = model.eval(sess)
            total_loss += loss * batch_size
            total_predict_count += predict_count
        except tf.errors.OutOfRangeError:
            break

    perplexity = utils.safe_exp(total_loss / total_predict_count)
    utils.print_time("  eval %s: perplexity %.2f" % (name, perplexity),
                     start_time)
    return perplexity
