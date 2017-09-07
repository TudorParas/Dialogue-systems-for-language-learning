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

For loading data into the end to end model.
"""
from __future__ import division
from __future__ import print_function
import collections

import tensorflow as tf


class End2EndBatchedInput(collections.namedtuple("BatchedInput",
                                                 ("initializer",
                                                  "source",
                                                  "target_input",
                                                  "target_output",
                                                  "target_weights",
                                                  "source_sequence_length",
                                                  "target_sequence_length",
                                                  "dialogue_length"))):  # ToDo: find a less confusing name
    pass


def get_infer_iterator(dataset, vocab_table, batch_size,
                       src_reverse, eos, eou, src_max_len=None,
                       dialogue_max_len=None):
    """
    Returns an iterator for the inference graph which does not contain target data.
    We do not use buckets for inference. The iterator will iterate over tensors of shape
    [batch_size, dialogue_max_len, utt_max_len]
    :param dataset: Consists of a sequence of utterances from the same dialogue to which the bot responds.
    :param vocab_table: Word to index mappings in the form of a tf HashTable.
    :param batch_size: The number of consecutive elements of this dataset to combine in a single batch.
    :param src_reverse: Whether to reverse the inputs (makes the beginning of the input
                    have a bigger impact on the response
    :param eos: The end of sentence string
    :param eou: End of utterance. This is the token which marks end of utterances in the input file
    :param src_max_len: Maximum accepted length for utterance. Bigger inputs will be truncated.
    :param dialogue_max_len: Maximum accepted length for the dialogue. Responses considered as well, for consistency
    """

    # Make dialogue_max_len represent only how many user utterances there are
    # We use the eos token to pad the data
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)
    # Split the dialogs into individual utterances.
    dataset = dataset.map(lambda dialogue: tf.string_split([dialogue], eou).values)  # shape=[dialogue_length, ]

    # Cut the dialogue short if we have a max length
    if dialogue_max_len:
        dataset = dataset.map(lambda dialogue: dialogue[:dialogue_max_len])
    # Tokenize the utterances. This step also pads the matrices up to the length of the longest
    # utterance in the matrix
    dataset = dataset.map(lambda dialogue: string_split_dense(
        dialogue, pad=eos))  # shape=[dialogue_length, utterance_length]
    if src_max_len:
        dataset = dataset.map(lambda dialogue: dialogue[:, :src_max_len])
    # Get the integers mappings from the vocab_table. We also need to explicitly cast it
    dataset = dataset.map(lambda dialogue: tf.cast(vocab_table.lookup(dialogue), tf.int32))
    # Reverse the utterances if so states. We do so by reversing along the columns
    if src_reverse:
        dataset = dataset.map(lambda dialogue: tf.reverse(dialogue, axis=[1]))
    # Get the length of the biggest utterance in the dialogue. Used to cap the length of the response
    dataset = dataset.map(lambda dialogue: (dialogue, tf.shape(dialogue)[0]))
    # Get the weights for the utterances (places which are padded are 0, the rest are 1)
    dataset = dataset.map(lambda dialogue, dialogue_length: (dialogue,
                                                             tf.cast(tf.not_equal(dialogue, eos_id), tf.int32),
                                                             dialogue_length))
    # Compute sequence lengths from the weights
    dataset = dataset.map(lambda dialogue, weights, dialogue_length: (dialogue,
                                                                      tf.cast(tf.count_nonzero(weights, axis=1),
                                                                              tf.int32),  # cast bc batch complain
                                                                      dialogue_length))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(tf.TensorShape([None, None]),  # src
                           tf.TensorShape([None]),  # src_len
                           tf.TensorShape([])),  # dialogue length
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(eos_id,  # src
                            0,
                            0))  # src_len -- unused

    dataset = batching_func(dataset)
    iterator = dataset.make_initializable_iterator()
    (ids, utt_length, diag_len) = iterator.get_next()
    return End2EndBatchedInput(
        initializer=iterator.initializer,
        source=ids,  # shape=[batch_size, dialogue_max_len, src_max_len]
        target_input=None,
        target_output=None,
        target_weights=None,
        source_sequence_length=utt_length,  # shape=[batch_size, utterance_max_len]
        target_sequence_length=None,
        dialogue_length=diag_len  # shape=[batch_size]
    )


def get_iterator(src_dataset,
                 tgt_dataset,
                 vocab_table,
                 batch_size,
                 sos,
                 eos,
                 eou,
                 src_reverse,
                 random_seed,
                 num_dialogue_buckets,
                 dialogue_max_len=None,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_threads=4,
                 output_buffer_size=None,
                 skip_count=None):
    """
    Create iterator for the training or evaluation graph.
    :param src_dataset: The dataset consisting of utterances separated by eou
    :param tgt_dataset: The dataset consisting of responses separated by eou
    :param sos: The 'start of string' string.
    :param eos: The 'end of string' string
    :param eou: End of utterance. This is the token which marks end of utterances in the input file
    :param src_reverse: Whether to reverse the input.
    :param random_seed: Seed used to fuel a pseudo-random number generator.
    :param num_dialogue_buckets: Number of buckets in which we put data of similar dialogue length.
    :param dialogue_max_len: Maximum length of the dialogue. A utterance-response pair counts as 1.
    :param num_threads: The number of threads to use for processing elements in parallel.
    :param output_buffer_size: The number of elements from this dataset from which the new dataset will sample
    :param skip_count: The number of elements of this dataset that should be skipped to form the new dataset.
    :return: An instance of BatchedIterator.
    """
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000
    # Get the ids
    sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)
    # Zip them together to create a single dataset
    dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))
    # Split the dialogs into individual utterances.
    dataset = dataset.map(lambda src, tgt: (tf.string_split([src], eou).values,  # shape=[dialogue_length, ]
                                            tf.string_split([tgt], eou).values),
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)


    # Cut the dialogue short to max length
    if dialogue_max_len:
        dataset = dataset.map(lambda src, tgt: (src[:dialogue_max_len], tgt[:dialogue_max_len]),
                              num_threads=num_threads,
                              output_buffer_size=output_buffer_size)
    # Skip the first skip_count elements.
    if skip_count is not None:
        dataset = dataset.skip(count=skip_count)
    # Shuffle the dataset.
    dataset = dataset.shuffle(output_buffer_size, random_seed)

    # Tokenize the utterances. This step also pads the matrices up to the length of the longest
    # utterance in the matrix
    dataset = dataset.map(lambda src, tgt: (string_split_dense(src, pad=eos),
                                            string_split_dense(tgt, pad=eos)),
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)  # shape=[dialogue_length, utterance_length] tuple
    # Apply the truncation for src length and target length if necessary
    if src_max_len:
        dataset = dataset.map(lambda src, tgt: (src[:, :src_max_len], tgt),
                              num_threads=num_threads,
                              output_buffer_size=output_buffer_size)
    if tgt_max_len:
        dataset = dataset.map(lambda src, tgt: (src, tgt[:, :tgt_max_len]),
                              num_threads=num_threads,
                              output_buffer_size=output_buffer_size)
    # Reverse the source
    if src_reverse:
        dataset = dataset.map(lambda src, tgt: (tf.reverse(src, axis=[1]), tgt))
    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer. We do it after the split because
    # we don't want to pad the target up to the size of the source
    dataset = dataset.map(lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                                            tf.cast(vocab_table.lookup(tgt), tf.int32)),
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    # We are going to use the pad value provided by tensorflow.
    dataset = dataset.map(lambda src, tgt: (src,
                                            tf.pad(tgt,  # target input
                                                   paddings=[[0, 0], [1, 0]],  # Pad one column before the matrix
                                                   constant_values=sos_id),  # use this to pad
                                            tf.pad(tgt,  # target output, the input shifted
                                                   paddings=[[0, 0], [0, 1]],  # Pad one column after the matrix
                                                   constant_values=eos_id)),
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)
    # Get the length of the biggest utterance in the dialogue for the source and the target respectively.
    # Get the length of the dialogue. Source is always longer than target
    dataset = dataset.map(lambda src, tgt_in, tgt_out: (src,
                                                        tgt_in,
                                                        tgt_out,
                                                        tf.shape(tgt_in)[0]),  # len of the exchange, a pair counts as 1
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)
    # Get the weights for the source and the target. The source weights will be used to compute sequence lengths.
    dataset = dataset.map(lambda src, tgt_in, tgt_out, diag_len: (src,
                                                                  tgt_in,
                                                                  tgt_out,
                                                                  tf.cast(tf.not_equal(src, eos_id), tf.int32),
                                                                  tf.cast(tf.not_equal(tgt_in, eos_id), tf.int32),
                                                                  diag_len),
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)
    # Get the sequence lengths.
    dataset = dataset.map(lambda src, tgt_in, tgt_out, src_weights, tgt_weights, diag_len: (
        src,
        tgt_in,
        tgt_out,
        tgt_weights,
        tf.cast(tf.count_nonzero(src_weights, axis=1), tf.int32),
        tf.cast(tf.count_nonzero(tgt_weights, axis=1), tf.int32),
        diag_len
    ))

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...).
    # Shape becomes [batch_size, dialogue_max_length, utterance_max_len]
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(tf.TensorShape([None, None]),  # src
                           tf.TensorShape([None, None]),  # tgt_input
                           tf.TensorShape([None, None]),  # tgt_output
                           tf.TensorShape([None, None]),  # tgt_weights
                           tf.TensorShape([None]),  # src_len
                           tf.TensorShape([None]),  # tgt_len
                           tf.TensorShape([])),  # dialogue length
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(eos_id,  # src
                            eos_id,  # tgt_input
                            eos_id,  # tgt_output
                            0,  # tgt_weights
                            0,  # src_len
                            0,  # tgt_len
                            0))  # diag_len -- unused

    if num_dialogue_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, unused_4, unused_5, usused_6, dialogue_len):
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            # If there is a max length find the width so that we equally split data in buckets.
            # Calculate bucket_width by maximum source sequence length.
            if dialogue_max_len:
                bucket_width = (src_max_len + num_dialogue_buckets - 1) // num_dialogue_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target sentence.
            bucket_id = dialogue_len // bucket_width

            return tf.to_int64(tf.minimum(num_dialogue_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)

        # Maps each consecutive elements in this dataset to a key using key_func to at
        # most window_size elements matching the same key.
        dataset = dataset.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size
        )
    else:
        dataset = batching_func(dataset)

    iterator = dataset.make_initializable_iterator()
    (src, tgt_in, tgt_out, tgt_weights, src_seq_len, tgt_seq_len, dialogue_len) = iterator.get_next()
    return End2EndBatchedInput(
        initializer=iterator.initializer,
        source=src,  # shape=[batch_size, ceil(dialogue_max_len / 2), src_max_len]
        target_input=tgt_in,  # shape=[batch_size, floor(dialogue_max_len / 2), tgt_max_len]
        target_output=tgt_out,  # shape=[batch_size, floor(dialogue_max_len / 2), tgt_max_len]
        target_weights=tgt_weights,  # shape=[batch_size, floor(dialogue_max_len / 2), tgt_max_len]
        source_sequence_length=src_seq_len,  # shape=[batch_size, dialogue_max_length]
        target_sequence_length=tgt_seq_len,  # shape=[batch_size, dialogue_max_length]
        dialogue_length=dialogue_len  # shape=[batch_size]
    )


def string_split_dense(dialogue, pad, delimiter=' '):
    """Takes a dialogue and returns a tuple of the form [dialogue_len, utterance_max_len]
    where utt_max_len returns the length of the biggest utterance"""
    sparse = tf.string_split(dialogue, delimiter=delimiter)
    dense = tf.sparse_tensor_to_dense(sparse, default_value=pad)

    return dense
