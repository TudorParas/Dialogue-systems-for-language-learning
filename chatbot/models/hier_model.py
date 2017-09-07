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

Stub script for the hierarchical model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

import utils.misc_utils as utils
from chatbot.models import model_helper
from utils import end2end_iterator_utils

utils.check_tensorflow_version()


class HierarchicalModel(object):
    """Hierarchical seq2seq model. Contains a context state which keeps track of dialogue"""

    def __init__(self,
                 hparams,
                 mode,
                 iterator,
                 vocab_table,
                 verbose=True,
                 ids_to_words=None,
                 scope=None):
        """
        Create the model

        :param hparams: Hyperparameter configurations.
        :param mode: TRAIN/EVAL/INFERENCE
        :param iterator: Dataset Iterator that feeds data.
        :param vocab_table: Lookup table mapping source words to ids.
        :param ids_to_words: Lookup table mapping ids to target words. Only
        required in INFER mode. Defaults to None.
        :param scope: scope of the model

        """
        if not isinstance(iterator, end2end_iterator_utils.End2EndBatchedInput):
            raise ValueError("Iterator has to be an instance of the End2EndBatchedInput class")

        self.iterator = iterator
        self.mode = mode
        self.vocab_table = vocab_table
        self.verbose = verbose

        self.vocab_size = hparams.vocab_size
        self.num_layers = hparams.num_layers
        self.num_gpus = hparams.num_gpus
        # Boolean dictating the shape format of the inputs and outputs Tensors.
        # If true, these Tensors must be shaped [max_time, batch_size, depth].
        # If false, these Tensors must be shaped [batch_size, max_time, depth]
        self.time_major = hparams.time_major
        # Create the initializer and set it as the default one. When we'll initialize a variable we'll use this.
        initializer = tf.random_uniform_initializer(minval=-hparams.initial_weight, maxval=hparams.initial_weight,
                                                    seed=hparams.random_seed)
        tf.get_variable_scope().set_initializer(initializer=initializer)
        self.init_embeddings(hparams, scope)
        # Get the batch-size by testing the size of the first batch of dialogues
        self.batch_size = tf.size(self.iterator.dialogue_length)

        # Create the densely-connected layer which will compute the output from the RNN cell.
        # We use this because we have freedom in the dimensionality of the input space like this.
        with tf.variable_scope(scope or "build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(
                    units=hparams.vocab_size, use_bias=False, name="output_projection")

        # Train graph
        logits, loss, sample_ids = self.build_graph(hparams, scope)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = loss
            # Used at determining words per second
            self.word_count = tf.reduce_sum(
                self.iterator.source_sequence_length) + tf.reduce_sum(
                self.iterator.target_sequence_length)
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss = loss
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            self.infer_logits = logits
            self.sample_ids = sample_ids

            # Get the corresponding words for the output ids.
            self.sample_words = [ids_to_words.lookup(
                tf.to_int64(sample_id)
            )for sample_id in self.sample_ids]

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            # Count the number of predicted words for computing perplexity.
            self.predict_count = tf.reduce_sum(
                self.iterator.target_sequence_length
            )

        # Initialize the global step.
        self.global_step = tf.Variable(0, trainable=False)

        params = tf.trainable_variables()

        # Gradients and SGD update operation for training the model.
        # Arrange for the embedding vars to appear at the beginning.
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            print("  start_decay_step=%d, learning_rate=%g, decay_steps %d,"
                  "decay_factor %g" % (hparams.start_decay_step, hparams.learning_rate,
                                       hparams.decay_steps, hparams.decay_factor))

            if hparams.optimizer == "sgd":
                self.learning_rate = tf.cond(
                    pred=self.global_step < hparams.start_decay_step,  # We do not apply decay yet
                    true_fn=lambda: tf.constant(hparams.learning_rate),
                    false_fn=lambda: tf.train.exponential_decay(  # lr = lr * decay_rate ^ (global_step / decay_steps)
                        learning_rate=hparams.learning_rate,
                        global_step=self.global_step - hparams.start_decay_step,
                        # Variable using in computing the decay
                        decay_steps=hparams.decay_steps,
                        decay_rate=hparams.decay_factor,
                        staircase=True  # Decay the rate at discrete intervals
                    ),
                    name="learning_rate"
                )

                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif hparams.optimizer == "adam":
                if float(hparams.learning_rate) > 0.001:
                    raise ValueError("! High Adam learning rate %g" % hparams.learning_rate)

                self.learning_rate = tf.constant(hparams.learning_rate)
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            else:
                raise ValueError("Current build supports only sgd or adam, %s is not supported" % hparams.optimizer)
            # Compute gradients for clipping, thus avoiding exploding gradients
            gradients = tf.gradients(
                ys=self.train_loss,  # Tensor or list of tensors to be differentiated
                xs=params,  # Tensor or list of tensors to use for differentiation
                colocate_gradients_with_ops=hparams.colocate_gradients_with_ops
            )
            # Clip them :)
            clipped_gradients, gradient_norm_summary = model_helper.gradient_clip(
                gradients=gradients,
                params=params,
                max_gradient_norm=hparams.max_gradient_norm
            )
            # Create the update op by applying the grads
            self.update = optimizer.apply_gradients(
                grads_and_vars=zip(clipped_gradients, params),
                global_step=self.global_step
            )

            # Train summary
            self.train_summary = tf.summary.merge([tf.summary.scalar("lr", self.learning_rate),
                                                   tf.summary.scalar("total_loss", self.train_loss)]
                                                  + gradient_norm_summary)  # Contains grad_norm and clipped gradient

        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            # Nonexistent in the model without attention
            self.infer_summary = self._get_infer_summary(hparams)

        # Saver
        self.saver = tf.train.Saver(tf.global_variables())

        # Print out the trainable variables
        if self.verbose:
            utils.print_out("# Trainable variables")
            for param in params:
                utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                                  param.op.device))

    def init_embeddings(self, hparams, scope):
        """Create the embeddings matrix"""
        self.embeddings = model_helper.create_embeddings(vocab_size=self.vocab_size,
                                                         embedding_size=hparams.num_units,
                                                         scope=scope)

    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        # Run one training iteration
        return sess.run([self.update,
                         self.train_loss,
                         self.predict_count,
                         self.train_summary,
                         self.global_step,
                         self.word_count,
                         self.batch_size])

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,
                         self.predict_count,
                         self.batch_size])

    def build_graph(self, hparams, scope=None):
        """Build the graph

        Creates a hierarchical sequence-to-sequence model with dynamic RNN decoder API.
        Args:
          hparams: Hyperparameter configurations.
          scope: VariableScope for the created subgraph; default "hier_seq2seq".

        Returns:
          A tuple of the form (logits, loss, final_context_state),
          where:
            logits: float32 Tensor [batch_size x num_decoder_symbols].
            loss: the total loss / batch_size.
            final_context_state: The final state of decoder RNN.

        Raises:
          ValueError: if encoder_type differs from mono and bi, or
            attention_option is not (luong | scaled_luong |
            bahdanau | normed_bahdanau).
        """
        if self.verbose:
            utils.print_out("# creating %s graph" % self.mode)

        dtype = tf.float32
        num_layers = hparams.num_layers
        num_gpus = hparams.num_gpus
        with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
            # Create the context cell and initialize the context cell to all zeros
            context_cell, context_state = self._build_context_cell(hparams,
                                                                   num_layers=hparams.context_num_layers,
                                                                   num_residual_layers=hparams.context_num_residual_layers)

            # We are going to unstack the dialogs along the temporal, aka dialog_len axis. This will split
            # it into batches of utterances at different time steps
            sources = tf.unstack(self.iterator.source, axis=1)  # shape=[batch_size, src_max_len] list
            target_inputs = tf.unstack(self.iterator.target_input, axis=1)  # shape=[batch_size, tgt_max_len] list
            target_outputs = tf.unstack(self.iterator.target_output, axis=1)  # shape=[batch_size, tgt_max_len] list
            target_weights = tf.unstack(self.iterator.target_weights, axis=1)  # shape=[batch_size, tgt_max_len] list
            src_seq_lengths = tf.unstack(self.iterator.source_sequence_length, axis=1)  # shape = [batch_size] list
            tgt_seq_lengths = tf.unstack(self.iterator.target_sequence_length, axis=1)  # shape = [batch_size] list

            # Initialize a list of predictions from the model which we'll use from computing the loss
            model_logits = []
            # Initialize a list of sample ids which represents the output ids
            sample_ids = []
            # Get the maximum dialogue length of the batch so we know how many times we loop.
            max_dialogue = len(target_inputs)
            for index in range(max_dialogue):
                current_source = sources[index]
                current_target_input = target_inputs[index]
                current_target_output = target_outputs[index]
                current_src_len = src_seq_lengths[index]
                current_tgt_len = tgt_seq_lengths[index]

                # Generate the vector encoding from the input
                source_encoding = self._get_encoding(hparams, current_source, current_src_len)
                # Update the context state using the source
                context_state = self._update_context(context_state, context_cell, state_vector=source_encoding)

                # Decoder
                current_logits, current_sample_id, _ = self._build_decoder(
                    context_state, current_target_input, current_src_len, current_tgt_len, hparams)
                if self.mode != tf.contrib.learn.ModeKeys.INFER:
                    # If the mode is Train or Eval we feed the correct output in
                    # Generate an encoding for the target output
                    out_encoding = self._get_encoding(hparams, source=current_target_output)
                else:
                    # If the mode is inference we feed its output back in
                    # ToDo: See whether it is correct to feed the sample_id in
                    # Generate an encoding for the decoder output
                    out_encoding = self._get_encoding(hparams, source=current_sample_id)
                    # Update the context state using the encoding of the outputs
                context_state = self._update_context(context_state, context_cell, state_vector=out_encoding)
                # Append the logits to the model_logits for computing loss later
                model_logits.append(current_logits)
                # Apped the sample id to the sample ids list
                sample_ids.append(current_sample_id)

            # Loss
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                # Compute it on the same gpu as the last cell
                with tf.device(model_helper.get_device_str(num_layers - 1, num_gpus)):
                    loss = self._compute_loss(model_logits, target_outputs, target_weights)
            else:
                # Cannot compute loss because we have no target outputs
                loss = None

        return model_logits, loss, sample_ids

    def _build_context_cell(self, hparams, num_layers, num_residual_layers, base_gpu=0):
        """Build a multi-layer RNN cell that which is out context."""
        cell = model_helper.create_rnn_cell(
            unit_type=hparams.context_unit_type,
            num_units=hparams.context_num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            verbose=self.verbose,
            base_gpu=base_gpu
        )
        # Initialize the state to be zeros.
        # It has this shape because the input is shaped [batch_size, encoder_cell.state_size]
        state = tf.zeros([hparams.batch_size, hparams.context_num_units])

        return cell, state

    def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                            base_gpu=0):
        """Build a multi-layer RNN cell that can be used by encoder."""

        return model_helper.create_rnn_cell(
            unit_type=hparams.unit_type,
            num_units=hparams.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=hparams.forget_bias,
            dropout=hparams.dropout,
            num_gpus=hparams.num_gpus,
            mode=self.mode,
            base_gpu=base_gpu)

    def _get_encoding(self, hparams, source, source_sequence_length=None):
        """It generates the state vector for the source"""

        num_layers = hparams.num_layers
        num_residual_layers = hparams.num_residual_layers

        # The source ids
        if self.time_major:
            source = tf.transpose(source)  # [max_time, batch_size]

        with tf.variable_scope("encoder") as encoder_scope:
            dtype = encoder_scope.dtype
            # Look up embedding, emp_inp: [source shape, num_units],
            # or transposed if not time major?
            encoder_emb_inp = tf.nn.embedding_lookup(
                self.embeddings, source)

            # Encoder_outputs: [source shape, num_units]
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
                    sequence_length=source_sequence_length,
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

                bi_encoder_state = self._build_bidirectional_rnn(
                    inputs=encoder_emb_inp,
                    sequence_length=source_sequence_length,
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
        # We only care about the final state
        return encoder_state

    def _build_bidirectional_rnn(self, inputs,
                                 dtype, hparams,
                                 num_bi_layers,
                                 num_bi_residual_layers,
                                 sequence_length=None,
                                 base_gpu=0):
        """Create and call biddirectional RNN cells.

        Args:
          num_bi_residual_layers: Number of residual layers from top to bottom. For
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

        return bi_state

    def _update_context(self, context_state, context_cell, state_vector):
        """Update the encoding cell by feeding it the state vector"""
        output, next_state = context_cell.call(inputs=state_vector, state=context_state)
        return next_state

    def _build_decoder_cell(self, hparams, context_state):
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
                context_state, multiplier=hparams.beam_width
            )
        else:
            # Make a copy
            decoder_initial_state = tf.Variable(context_state.inititalized_value())

        return decoder_cell, decoder_initial_state

    def _build_decoder(self, context_state, target_input, source_sequence_length,
                       target_sequence_length, hparams):
        """Build and run a RNN decoder with a final projection layer.

        Args:
          context_state: The state of the context thus far in the dialogue
          target_input: What we're going to feed in the decoder
          source_sequence_length: Lengths of the source sequences at this time step
          target_sequence_length: Lengths of the target sequences at this time step
          hparams: The Hyperparameters configurations.

        Returns:
          A tuple of final logits and final decoder state:
            logits: size [time, batch_size, vocab_size] when time_major=True.
        """

        sos_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.sos)),
                         tf.int32)
        eos_id = tf.cast(self.vocab_table.lookup(tf.constant(hparams.eos)),
                         tf.int32)

        num_layers = hparams.num_layers
        num_gpus = hparams.num_gpus

        # maximum_iterations: The maximum decoding steps.
        if hparams.tgt_max_len_infer:
            maximum_iterations = hparams.tgt_max_len_infer
            # We only print here as the other one is a tensor
            if self.verbose:
                utils.print_out("  Using maximum iterations %d" % maximum_iterations)
        else:
            # If it it not provided then make it `decoding_length_factor' as big as any un-padded input
            # Src_seq_len: the lengths of the un-padded inputs.
            max_encoder_length = tf.reduce_max(source_sequence_length)
            maximum_iterations = tf.to_int32(tf.round(
                tf.to_float(max_encoder_length) * hparams.decoding_length_factor
            ))

        # Build the decoder
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(
                hparams=hparams,
                context_state=context_state
            )

            # Train or eval
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                target_input = target_input  # shape=[batch_size, max_time]

                # Time should be the 0 axis
                if self.time_major:
                    target_input = tf.transpose(target_input)  # shape=[max_time, batch_size]
                # shape=[max_time, batch_size, num_units]?. First 2 axes determined by target_input shape?
                decoder_emb_inputs = tf.nn.embedding_lookup(
                    params=self.embeddings,
                    ids=target_input
                )

                # A helper for use during training.  Only reads inputs.
                # Returned sample_ids are the argmax of the RNN output logits.
                helper = tf.contrib.seq2seq.TrainingHelper(
                    inputs=decoder_emb_inputs,
                    sequence_length=target_sequence_length,
                    time_major=self.time_major
                )
                # Added ability to switch when doing the output layer. Uncomment for effects
                if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                    # The decoder. We are going to apply the output layer to everything at the end
                    basic_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=cell,
                        helper=helper,
                        initial_state=decoder_initial_state
                    )
                else:
                    # The decoder for evaluation. We will apply the output layer per timestep because OOM.
                    basic_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=cell,
                        helper=helper,
                        initial_state=decoder_initial_state,
                        output_layer=self.output_layer
                    )

                # Dynamic decoding: Calls initialize() once and step() repeatedly on the Decoder object.
                outputs, final_context_state, final_seq_lengths = tf.contrib.seq2seq.dynamic_decode(
                    decoder=basic_decoder,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope
                )

                sample_id = outputs.sample_id

                if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                    # Note: there's a subtle difference here between train and inference.
                    # We could have set output_layer when create my_decoder
                    #   and shared more code between train and inference.
                    # We chose to apply the output_layer to all timesteps for speed:
                    #   10% improvements for small models & 20% for larger ones.
                    # If memory is a concern, we should apply output_layer per timestep.
                    device_id = num_layers if num_layers < num_gpus else num_layers - 1
                    with tf.device(model_helper.get_device_str(device_id, num_gpus)):
                        logits = self.output_layer(outputs.rnn_output)
                else:
                    # Evaluation
                    logits = outputs.rnn_output

            # Inference
            else:
                # Number of nodes it explores at each iteration
                beam_width = hparams.beam_width
                length_penalty_weight = hparams.length_penalty_weight
                # Start the responses with the sos token
                start_tokens = tf.fill([self.batch_size], sos_id)
                end_token = eos_id

                # Create the decoder based on whether we use beam search
                # Bugged in current version, 1.2.1 gpu
                if beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=cell,
                        embedding=self.embeddings,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=beam_width,
                        output_layer=self.output_layer,  # Optional layer to apply to the RNN output.
                        length_penalty_weight=length_penalty_weight  # Float weight to penalize length. Disabled with 0
                    )
                else:
                    # The decoding becomes a greedy search
                    # Uses the argmax of the output (treated as logits) and passes the result through
                    # an embedding layer to get the next input.
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding=self.embeddings,
                        start_tokens=start_tokens,
                        end_token=end_token
                    )
                    # Returns (outputs, next_state, next_inputs, finished)
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=cell,
                        helper=helper,
                        initial_state=decoder_initial_state,
                        output_layer=self.output_layer  # Applied per timestep
                    )

                # Dynamic decoding
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=my_decoder,
                    maximum_iterations=maximum_iterations,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=decoder_scope
                )

                if beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id

        return logits, sample_id, final_context_state

    def _compute_loss(self, logits, target_outputs, target_weights):
        """
        Compute the loss for the model
        :param logits: List of the logits at every time step
        :param target_outputs: List of the target outputs at every time step
        :param target_weights: List of the target weights at every time step
        :return: The loss
        """

        total_loss = tf.constant(0, dtype=tf.float32)
        # Iterate over the outputs
        for index in range(len(logits)):

            current_logits = logits[index]
            current_tgt_outputs = target_outputs[index]
            current_tgt_weights = target_weights[index]
            if self.time_major:
                current_tgt_outputs = tf.transpose(current_tgt_outputs)  # shape=[max_time, batch_size]
                current_tgt_weights = tf.transpose(current_tgt_weights)

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=current_tgt_outputs,
                logits=current_logits
            )

            # Multiply it by the weights in order to invalidate the effect of the padding
            # Get the average loss for the batch
            loss = tf.reduce_sum(
                cross_entropy * current_tgt_weights) / tf.to_float(self.batch_size)
            # Add them to the total loss
            total_loss = tf.add(total_loss, loss)

        return total_loss

    def _get_infer_summary(self, hparams):
        """Placeholder function. Summary nonexistent in the model without attention"""
        return tf.no_op()

    def infer(self, sess):
        """Run a session and get the inference output"""
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            raise ValueError("Mode need to be inference to get the inference")
        return sess.run([
            self.infer_logits, self.infer_summary, self.sample_ids, self.sample_words
        ])

    def decode(self, sess):
        """Decode a batch.

        Args:
          sess: tensorflow session to use.

        Returns:
          A tuple consiting of outputs, infer_summary.
            outputs: of size [batch_size, time]
        """
        # infer_summary is nonexistent in the model without attention
        _, infer_summary, _, sample_words = self.infer(sess)

        # make sure outputs is of shape [batch_size, time]
        if self.time_major:
            sample_words = sample_words.transpose()
        return sample_words, infer_summary
