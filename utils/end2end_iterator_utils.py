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
                                           "target_sequence_length"))):
    pass

def get_infer_iterator(dataset, vocab_table, batch_size,
                       src_reverse, eos, eou, utt_max_len=None,
                       dialogue_max_len=None):
    """
    Returns an iterator for the inference graph which does not contain target data.
    We do not use buckets for inference. The iterator will iterate over tensors of shape
    [batch_size, dialogue_max_len, utt_max_len]
    :param dataset: Data which we'll be working with.
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
    dataset = dataset.map(lambda dialogue: (dialogue, tf.shape(dialogue)[1]))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(tf.TensorShape([None, None]),  # src
                           tf.TensorShape([])),  # src_len
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(eos_id,  # src
                            0))  # src_len -- unused

    dataset = batching_func(dataset)
    iterator = dataset.make_initializable_iterator()
    (ids, length) = iterator.get_next()
    return BatchedInput(
        initializer=iterator.initializer,
        source=ids,
        target_input=None,
        target_output=None,
        source_sequence_length=length,
        target_sequence_length=None
    )


def string_split_dense(dialogue, pad, delimiter=' '):
    """Takes a dialogue and returns a tuple of the form [dialogue_len, utterance_max_len]
    where utt_max_len returns the length of the biggest utterance"""
    sparse = tf.string_split(dialogue, delimiter=delimiter)
    dense = tf.sparse_tensor_to_dense(sparse, default_value=pad)

    return dense

