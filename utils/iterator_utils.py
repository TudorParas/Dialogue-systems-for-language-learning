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

"""For loading data into NMT models."""

# tp423 - Added comments

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
# Acts as an interface from Java, used for checking classes.
class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "target_output",
                                           "source_sequence_length",
                                           "target_sequence_length"))):
    pass


def get_infer_iterator(dataset, vocab_table, batch_size,
                       src_reverse, eos, src_max_len):
    """
    Returns an iterator for the inference graph which does not contain target data.
    We do not use buckets for inference.
    :param dataset: Data which we'll be working with.
    :param vocab_table: Word to index mappings in the form of a tf HashTable.
    :param batch_size: The number of consecutive elements of this dataset to combine in a single batch.
    :param src_reverse: Whether to reverse the inputs (makes the beginning of the input
                    have a bigger impact on the response
    :param eos: The end of sentence string
    :param src_max_len: Maximum accepted length. Bigger inputs will be truncated.
    """
    # Get the id for the eos token. We will use this to pad the data
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)
    # Tokenize the input data by applying split. For better tokenization data is expected to
    # be tokenized in the preprocessing phase.
    dataset = dataset.map(lambda line: tf.string_split([line]).values)

    if src_max_len:
        dataset = dataset.map(lambda line: line[:src_max_len])
    # This map converts a vector of strings to a vector of integers
    dataset = dataset.map(lambda line: tf.cast(vocab_table.lookup(line), tf.int32))
    if src_reverse:
        dataset = dataset.map(lambda line: tf.reverse(line, axis=[0]))
    # Add in the word counts for each line.
    dataset = dataset.map(lambda line: (line, tf.size(line)))

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The entry is the source line rows;
            # this has unknown-length vectors.  The last entry is
            # the source row size; this is a scalar.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([])),  # src_len
            # Pad the source sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(eos_id,  # src
                            0))  # src_len -- unused

    batched_dataset = batching_func(dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    (ids, seq_length) = batched_iter.get_next()
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=ids,
        target_input=None,
        target_output=None,
        source_sequence_length=seq_length,
        target_sequence_length=None
    )


def get_iterator(src_dataset,
                 tgt_dataset,
                 vocab_table,
                 batch_size,
                 sos,
                 eos,
                 src_reverse,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_threads=4,
                 output_buffer_size=None,
                 skip_count=None):
    """
    Create iterator for the training or evaluation graph.
    :param sos: The 'start of string' string.
    :param eos: The 'end of string' string
    :param src_reverse: Whether to reverse the input.
    :param random_seed: Seed used to fuel a pseudo-random number generator.
    :param num_threads: The number of threads to use for processing elements in parallel.
    :param output_buffer_size: The number of elements from this dataset from which the new dataset will sample
    :param skip_count: The number of elements of this dataset that should be skipped to form the new dataset.
    :return:
    """
    if not output_buffer_size:
        output_buffer_size = batch_size * 1000
    sos_id = tf.cast(vocab_table.lookup(tf.constant(sos)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(eos)), tf.int32)

    dataset = tf.contrib.data.Dataset.zip((src_dataset, tgt_dataset))
    # Skip the first skip_count elements.
    if skip_count is not None:
        dataset = dataset.skip(count=skip_count)
    # Shuffle the dataset.
    dataset = dataset.shuffle(output_buffer_size, random_seed)
    # Split the lines into tokens.
    dataset = dataset.map(lambda src, tgt: (tf.string_split([src]).values, tf.string_split([tgt]).values),
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)

    # Filter zero length input sequences
    dataset = dataset.filter(lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

    if src_max_len:
        dataset = dataset.map(lambda src, tgt: (src[:src_max_len], tgt),
                              num_threads=num_threads,
                              output_buffer_size=output_buffer_size)
    if tgt_max_len:
        dataset = dataset.map(lambda src, tgt: (src, tgt[:tgt_max_len]),
                              num_threads=num_threads,
                              output_buffer_size=output_buffer_size)
    if src_reverse:
        dataset = dataset.map(lambda src, tgt: (tf.reverse(src, axis=0), tgt),
                              num_threads=num_threads,
                              output_buffer_size=output_buffer_size)

    # Convert the word strings to ids.  Word strings that are not in the
    # vocab get the lookup table's default_value integer.
    dataset = dataset.map(lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                                            tf.cast(vocab_table.lookup(tgt), tf.int32)),
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)
    # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
    dataset = dataset.map(lambda src, tgt: (src,
                                            tf.concat(([sos_id], tgt), axis=0),   # target input
                                            tf.concat((tgt, [eos_id]), axis=0)),  # target output, the input shifted
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)

    # Add in the word counts.  Subtract one from the target to avoid counting
    # the target_input <eos> tag (resp. target_output <sos> tag) (has not been done) .
    dataset = dataset.map(lambda src, tgt_in, tgt_out: (src, tgt_in, tgt_out,
                                                        tf.size(src), tf.size(tgt_in)),
                          num_threads=num_threads,
                          output_buffer_size=output_buffer_size)

    # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([None]),  # tgt_input
                           tf.TensorShape([None]),  # tgt_output
                           tf.TensorShape([]),  # src_len
                           tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(eos_id,  # src
                            eos_id,  # tgt_input
                            eos_id,  # tgt_output
                            0,  # src_len -- unused
                            0))  # tgt_len -- unused

    if num_buckets > 1:
        def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
            # Pairs with length [0, bucket_width) go to bucket 0, length
            # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
            # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
            # If there is a max length find the width so that we equally split data in buckets.
            # Calculate bucket_width by maximum source sequence length.
            if src_max_len:
                bucket_width = (src_max_len + num_buckets - 1) // num_buckets
            else:
                bucket_width = 10

            # Bucket sentence pairs by the length of their source sentence and target
            # sentence.
            bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(unused_key, windowed_data):
            return batching_func(windowed_data)
        # Maps each consecutive elements in this dataset to a key using key_func to at
        # most window_size elements matching the same key.
        batched_dataset = dataset.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size
        )
    else:
        batched_dataset = batching_func(dataset)
    batched_iter = batched_dataset.make_initializable_iterator()
    # Get a sample of what the data looks like.
    (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (
        batched_iter.get_next())
    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_input_ids,
        target_output=tgt_output_ids,
        source_sequence_length=src_seq_len,  # Vector containing the sizes of the sequences without padding.Test - 173
        target_sequence_length=tgt_seq_len)


