"""
A dialogue system meant to be used for language learning.

This is based on Google Neural Machine Tranlation model
https://github.com/tensorflow/nmt
which is based on Thang Luong's thesis on
Neural Machine Translation: https://github.com/lmthang/thesis

And on the paper Building End-To-End Dialogue Systems
Using Generative Hierarchical Neural Network Models:
https://arxiv.org/pdf/1507.04808.pdf

Created by Tudor Paraschivescu for the Cambridge UROP project
"Dialogue systems for language learning"

The hierarchical model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from chatbot.models.base_model import BaseModel

import utils.misc_utils as utils
from chatbot.models import model_helper

utils.check_tensorflow_version(version="1.3.0")


class HierarchicalModel(BaseModel):
    """
    Sequence-to-sequence hierarchical model.

    This class implements a multi-layer recurrent neural network as encoder,
    a multi-layer recurrent neural network as a context encoder
    and a multi-layer recurrent neural network decoder.
    """

    def _build_encoder(self, hparams):
        """Build an encoder"""
        encoder_num_layers = hparams.num_layers
        encoder_num_residual_layers = hparams.num_residual_layers

        context_num_layers = hparams.context_num_layers
        context_num_residual_layers = hparams.context_num_residual_layers

        iterator = self.iterator
        sources = iterator.source  # shape=[batch_size, dialogue_len src_max_len]
        sources = tf.transpose(sources, perm=[1, 0, 2])  # shape=[dialogue_len, batch_size, src_max_len]

        sequence_lengths = tf.transpose(iterator.source_sequence_length)  # shape=[dialogue_len, batch_size]

        with tf.variable_scope("encoder") as encoder_scope:
            dtype = encoder_scope.dtype
            if self.verbose:
                utils.print_out(" Building encoder cell: num_layers = %d, num_residual_layers=%d" %
                                (encoder_num_layers, encoder_num_residual_layers))
            # Build the encoder cell. Decided to leave the default base gpu
            encoder_cell = self._build_encoder_cell(hparams,
                                                    encoder_num_layers,
                                                    encoder_num_residual_layers)
            if self.verbose:
                utils.print_out(" Building context cell: num_layers = %d, num_residual_layers=%d" %
                                (encoder_num_layers, encoder_num_residual_layers))
            context_cell = self._build_encoder_cell(hparams,
                                                    context_num_layers,
                                                    context_num_residual_layers)

            # Initialize the state using the current batch size
            current_batch_size = tf.shape(sources)[1]
            initial_state = context_cell.zero_state(current_batch_size, dtype=dtype)

            # Define the body and the condition for the while loop
            def body(context_state, counter):
                source = tf.gather(sources, counter)

                if self.time_major:
                    source = tf.transpose(source)  # [max_time, batch_size]

                seq_len = tf.gather(sequence_lengths, counter, name='get_current_source')
                encoder_emb_inp = tf.nn.embedding_lookup(
                    self.embeddings, source)
                # Create RNN. Performs fully dynamic unrolling of inputs
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    inputs=encoder_emb_inp,
                    sequence_length=seq_len,
                    dtype=dtype,
                    time_major=self.time_major,
                )
                # The encoder_state is a tuple. (cell state, memory state), aka (c, h).
                # Use the cell state as input.
                context_input = encoder_state[0]

                output, next_state = context_cell(inputs=context_input, state=context_state, scope="context")

                return [next_state, tf.add(counter, 1, name='increment_counter')]

            def condition(context_state, counter):
                return tf.less(counter, tf.shape(sources)[0], name='condition')

            # Initialize the counter
            counter = tf.Variable(0, name='counter', trainable=False, dtype=tf.int32)

            # Create the while loop, filling the encoder_states list
            final_context_state, _ = tf.while_loop(cond=condition, body=body,
                                                   loop_vars=[initial_state, counter])

        return final_context_state

    def _build_decoder_cell(self, hparams, encoder_state):
        """Build an RNN cell that can be used by decoder."""
        # We only make use of encoder_outputs in attention-based models


        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers
        decoder_cell = model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            verbose=self.verbose
        )

        # For beam search, we need to replicate encoder infos beam_width times
        if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
            # Tile them along the batch_size. [batch_size, etc.] to [batch_size * multiplier, etc]
            # by copying each t[i], i in [0, batch_size - 1] 'multiplier' times
            decoder_initial_state = tf.contrib.seq2seq.tile_batch(
                encoder_state, multiplier=hparams.beam_width
            )
        else:
            decoder_initial_state = encoder_state

        return decoder_cell, decoder_initial_state
