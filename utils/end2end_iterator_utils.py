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


class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length",
                                           "dialogue_length"))):  # ToDo: find a less confusing name
    pass

def get_infer_iterator(dataset, vocab_table, batch_size,
                       src_reverse, eos, eou, utt_max_len=None,
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
    :param utt_max_len: Maximum accepted length for utterance. Bigger inputs will be truncated.
    :param dialogue_max_len: Maximum accepted length for the dialogue. bigger dialogs will be truncated.
    """

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
    if utt_max_len:
        dataset = dataset.map(lambda dialogue: dialogue[:, :utt_max_len])
    # Get the integers mappings from the vocab_table. We also need to explicitly cast it
    dataset = dataset.map(lambda dialogue: tf.cast(vocab_table.lookup(dialogue), tf.int32))
    # Reverse the utterances if so states. We do so by reversing along the columns
    if src_reverse:
        dataset = dataset.map(lambda dialogue: tf.reverse(dialogue, axis=[1]))
    # Get the length of the biggest utterance in the dialogue. Used to cap the length of the response
    dataset = dataset.map(lambda dialogue: (dialogue, tf.shape(dialogue)[1], tf.shape(dialogue)[0]))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(tf.TensorShape([None, None]),  # src
                           tf.TensorShape([]),  # src_len
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
    return BatchedInput(
        initializer=iterator.initializer,
        source=ids,
        target_input=None,
        target_output=None,
        source_sequence_length=utt_length,
        target_sequence_length=None,
        dialogue_length=diag_len
    )

def get_iterator(dataset,
                 vocab_table,
                 batch_size,
                 sos,
                 eos,
                 eou,
                 src_reverse,
                 random_seed,
                 num_utterance_buckets,
                 dialogue_max_len=None,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_threads=4,
                 output_buffer_size=None,
                 skip_count=None):
    """
    Create iterator for the training or evaluation graph.
    :param dataset: The dataset consisting of alternating utterances and responses. The user starts and the DS
        will be trained to predict the response.
    :param sos: The 'start of string' string.
    :param eos: The 'end of string' string
    :param eou: End of utterance. This is the token which marks end of utterances in the input file
    :param src_reverse: Whether to reverse the input.
    :param random_seed: Seed used to fuel a pseudo-random number generator.
    :param num_utterance_buckets: Number of buckets in which we put data of similar utterance length.
    :param dialogue_max_len: Maximum length of the dialogue. A utterance-response pair counts as 2.
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
    # Split the dialogs into individual utterances.
    dataset = dataset.map(lambda dialogue: tf.string_split([dialogue], eou).values)  # shape=[dialogue_length, ]

    # ToDo: Remove the fist line from each dialogue and append it at the end. In this way we don't lose
    # half of the data

    # Cut the dialogue short to max length
    if dialogue_max_len:
        dataset = dataset.map(lambda dialogue: dialogue[:dialogue_max_len])
    # Skip the first skip_count elements.
    if skip_count is not None:
        dataset = dataset.skip(count=skip_count)
    # Shuffle the dataset.
    dataset = dataset.shuffle(output_buffer_size, random_seed)

    # We will split the data into source and target, i.e. what the user says and what we want the system to predict.
    # We need a predetermined dialogue_max_len
    dataset = dataset.map(lambda dialogue: (tf.strided_slice(dialogue, begin=[0],
                                                             end=[tf.shape(dialogue)[0]], strides=[2]),
                                            tf.strided_slice(dialogue, begin=[1],
                                                             end=[tf.shape(dialogue)[0]], strides=[2])),
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)

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
                                                        tf.shape(src)[1],
                                                        tf.shape(tgt_in)[1],
                                                        tf.shape(tgt_in)[0]),  # len of the exchange, a pair counts as 1
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)

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
                           tf.TensorShape([]),  # src_len
                           tf.TensorShape([]),
                           tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(eos_id,  # src
                            eos_id,  # tgt_input
                            eos_id,  # tgt_output
                            0,  # src_len -- unused
                            0,
                            0))  # tgt_len -- unused

    # ToDo: implement bucketing by the length of the dialogue as well
    if num_utterance_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len, unused_4):
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            # If there is a max length find the width so that we equally split data in buckets.
            # Calculate bucket_width by maximum source sequence length.
            if src_max_len:
                utt_bucket_width = (src_max_len + num_utterance_buckets - 1) // num_utterance_buckets
            else:
                utt_bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target sentence.
            utt_bucket_id = tf.maximum(src_len // utt_bucket_width, tgt_len // utt_bucket_width)

            return tf.to_int64(tf.minimum(num_utterance_buckets, utt_bucket_id))

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
    (src, tgt_in, tgt_out, src_seq_len, tgt_seq_len, dialogue_len) = iterator.get_next()
    return BatchedInput(
        initializer=iterator.initializer,
        source=src,
        target_input=tgt_in,
        target_output=tgt_out,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len,
        dialogue_length=dialogue_len
    )



def string_split_dense(dialogue, pad, delimiter=' '):
    """Takes a dialogue and returns a tuple of the form [dialogue_len, utterance_max_len]
    where utt_max_len returns the length of the biggest utterance"""
    sparse = tf.string_split(dialogue, delimiter=delimiter)
    dense = tf.sparse_tensor_to_dense(sparse, default_value=pad)

    return dense

