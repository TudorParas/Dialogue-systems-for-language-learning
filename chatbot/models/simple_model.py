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

"""Basic sequence-to-sequence model with dynamic RNN support."""

# tp423: Added more comments, changed couple of embeddings to one embedding matrix
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from chatbot.models.base_model import BaseModel

import utils.misc_utils as utils
from chatbot.models import model_helper

utils.check_tensorflow_version()


class SimpleModel(BaseModel):
    """
    Sequence-to-sequence dynamic model.

    This class implements a multi-layer recurrent neural network as encoder,
    and a multi-layer recurrent neural network decoder.
    """

    def _build_encoder(self, hparams):
        """Build an encoder"""
        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers

        iterator = self.iterator
        # The source ids
        source = iterator.source  # [batch_size, max_time]
        if self.time_major:
            source = tf.transpose(source)  # [max_time, batch_size]

        with tf.variable_scope("encoder") as encoder_scope:
            dtype = encoder_scope.dtype
            # Look up embedding, emp_inp: [source shape, num_units],
            # or transposed if not time major?
            encoder_emb_inp = tf.nn.embedding_lookup(
                self.embeddings, source)

            # Encoder_outpus: [source shape, num_units]
            if hparams.encoder_type == "uni":
                if self.verbose:
                    utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                                (num_layers, num_residual_layers))
                # Build the encoder cell. Decided to leave the default base gpu
                cell = self._build_encoder_cell(hparams,
                                                num_layers,
                                                num_residual_layers)
                # Create RNN. Performs fully dynamic unrolling of inputs
                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=encoder_emb_inp,
                    sequence_length=iterator.source_sequence_length,
                    dtype=dtype,
                    time_major=self.time_major
                )
            elif hparams.encoder_type == "bi":
                # Split the layers between the cells
                num_bi_layers = int(num_layers / 2)
                num_bi_residual_layers = int(num_residual_layers / 2)
                if self.verbose:
                    utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                                (num_bi_layers, num_bi_residual_layers))

                encoder_outputs, bi_encoder_state = self._build_bidirectional_rnn(
                    inputs=encoder_emb_inp,
                    sequence_length=iterator.source_sequence_length,
                    dtype=dtype,
                    hparams=hparams,
                    num_bi_layers=num_bi_layers,
                    num_bi_residual_layers=num_bi_residual_layers
                )
                # bi_encoder_state == (output_state_fw, output_state_bw)
                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])
                        encoder_state.append(bi_encoder_state[1][layer_id])
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
        return encoder_outputs, encoder_state

    def _build_bidirectional_rnn(self, inputs, sequence_length,
                                 dtype, hparams,
                                 num_bi_layers,
                                 num_bi_residual_layers,
                                 base_gpu=0):
        """Create and call biddirectional RNN cells.

        Args:
          num_residual_layers: Number of residual layers from top to bottom. For
            example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
            layers in each RNN cell will be wrapped with `ResidualWrapper`.
          base_gpu: The gpu device id to use for the first forward RNN layer. The
            i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
            device id. The `base_gpu` for backward RNN cell is `(base_gpu +
            num_bi_layers)`.

        Returns:
          The concatenated bidirectional output and the bidirectional RNN cell"s
          state.
        """
        # Construct forward and backward cells
        fw_cell = self._build_encoder_cell(hparams,
                                           num_layers=num_bi_layers,
                                           num_residual_layers=num_bi_residual_layers,
                                           base_gpu=base_gpu)
        bw_cell = self._build_encoder_cell(hparams,
                                           num_layers=num_bi_layers,
                                           num_residual_layers=num_bi_residual_layers,
                                           base_gpu=base_gpu + num_bi_residual_layers)  # Account for fw_cell

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=dtype,
            time_major=self.time_major
        )
        # bi_outputs is a tuple (output_fw, output_bw), which have shapes [source shape, num_units].
        # To turn them into a single output we will concat them along the last dimension
        encoder_outputs = tf.concat(values=bi_outputs, axis=-1)

        return encoder_outputs, bi_state

    def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                            source_sequence_length):
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