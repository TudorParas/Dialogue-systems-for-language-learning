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

Tests for end2end_iterator_utils.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from utils import end2end_iterator_utils


class IteratorUtilsTest(tf.test.TestCase):
    def testGetIterator(self):
        src_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            tf.constant(["a b b a eou a c c c",
                         "a c eou c a",
                         "a b c eou b",
                         "a"])
        )
        tgt_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            tf.constant(["c a b",
                         "b f a f",
                         "f a",
                         "b "])
        )
        vocab_table = lookup_ops.index_table_from_tensor(
            tf.constant(["a", "b", "c", "sos", "eos", "eou"])
        )

        batch_size = 2
        src_reverse = False
        sos = 'sos'
        eos = "eos"
        eou = "eou"
        random_seed = 53
        num_buckets = 2
        src_max_len = 3
        tgt_max_len = 2
        skip_count = None

        iterator = end2end_iterator_utils.get_iterator(src_dataset,tgt_dataset, vocab_table, batch_size, sos, eos, eou, src_reverse,
                                                       random_seed, num_buckets, src_max_len,
                                                       tgt_max_len, skip_count=skip_count)
        source = iterator.source
        target_in = iterator.target_input
        target_out = iterator.target_output
        source_len = iterator.source_sequence_length
        target_len = iterator.target_sequence_length
        dialogue_len = iterator.dialogue_length

        self.assertEqual([None, None, None], source.shape.as_list())
        self.assertEqual([None, None], target_in.shape.as_list())
        self.assertEqual([None, None], target_out.shape.as_list())
        self.assertEqual([None, None], source_len.shape.as_list())
        self.assertEqual([None], target_len.shape.as_list())
        self.assertEqual([None], dialogue_len.shape.as_list())

        with self.test_session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)

            (src_eval, tgt_in, tgt_out, src_seq_len, tgt_seq_len, diag_len) = sess.run((
                source, target_in, target_out, source_len, target_len, dialogue_len))
            print(src_eval)
            self.assertAllEqual(
                [[[0, 1, 1],  # a b b, cut off because of src_max_len
                  [0, 2, 2]],  # a c c
                 # These two are batched together because of bucketing
                 [[0, 1, 2],  # a b c
                  [1, 4, 4]]],  # c pad=eos pad because we use eos for padding. I will differentiate them in comments
                src_eval
            )
            self.assertAllEqual(
                [[3, 2, 0],  # sos c a. Truncated because of tgt_max_len

                 [3, -1, 0]],  # sos f='unknown' a
                tgt_in
            )
            self.assertAllEqual(
                [[2, 0, 4],  # c a eos

                 [-1, 0, 4]],  # f='unknown' a eos
                tgt_out
            )

            self.assertAllEqual(
                [[3, 3],  # length of first utterance, length of second utterance in first dialogue
                 [3, 1]],  # first and second utterance in second dialogue
                src_seq_len
            )
            self.assertAllEqual(
                [3, 3],  # we include the sos and eos symbols

                tgt_seq_len
            )
            self.assertAllEqual(
                [2, 2],  # Dialogue lengths
                diag_len
            )

            # Get next batch
            (src_eval, tgt_in, tgt_out, src_seq_len, tgt_seq_len, diag_len) = sess.run((
                source, target_in, target_out,  source_len, target_len, dialogue_len))

            self.assertAllEqual(
                [[[0, 4],  # a pad
                  [4, 4]],  # pad pad, because the next in the batch has this src
                 # In this order because when carrying on we first look at the next elem, 4, and then batch em.
                 [[0, 2],  # a c
                  [2, 0]]],  # c a
                # Note that it has been cut of short because of dialogue_max_len
                src_eval
            )
            self.assertAllEqual(
                [[3, 1, 4],  # sos b pad

                 [3, 1, -1]],  # sos b f='unknown'
                tgt_in
            )
            self.assertAllEqual(
                [[1, 4, 4],  # b pad eos

                 [1, -1, 4]],  # b f='unknown' eos
                tgt_out
            )
            self.assertAllEqual(
                [[1, 0],  # Second utterance in first dialogue is only padding, so len of 0 for it
                 [2, 2]],
                src_seq_len
            )
            self.assertAllEqual(
                [2, 3],   # we count padding as well
                tgt_seq_len
            )
            self.assertAllEqual(
                [1, 2],  # Only one exchange for the first one
                diag_len
            )

    def testGetInferIterator(self):
        dataset = tf.contrib.data.Dataset.from_tensor_slices(
            tf.constant(["a b b a eou c ", "a c eou f", "a eou b eou c", "d"])
        )
        vocab_table = lookup_ops.index_table_from_tensor(
            tf.constant(["a", "b", "c", "eos", "eou"])
        )

        batch_size = 2
        src_reverse = False
        eos = "eos"
        eou = "eou"
        src_max_len = 3
        dialogue_max_len = 2

        iterator = end2end_iterator_utils.get_infer_iterator(dataset, vocab_table,
                                                             batch_size, src_reverse, eos, eou,
                                                             src_max_len, dialogue_max_len)

        source = iterator.source
        seq_len = iterator.source_sequence_length
        diag_len = iterator.dialogue_length
        self.assertEqual([None, None, None], source.shape.as_list())
        self.assertEqual([None, None], seq_len.shape.as_list())
        self.assertEqual([None], diag_len.shape.as_list())
        with self.test_session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)

            (src_eval, seq_len_eval, diag_len_eval) = sess.run((source, seq_len, diag_len))

            self.assertAllEqual(
                [[[0, 1, 1],  # a b b, cut off because of utt_max_len
                  [2, 3, 3]],  # c pad pad, where pad is the eos in this case

                 [[0, 2, 3],  # a c pad, because it pads it to the previous one's length
                  [-1, 3, 3]]],  # f='unknown', pad pad
                src_eval
            )
            self.assertAllEqual(
                [[3, 1],
                [2, 1]],
                seq_len_eval
            )
            self.assertAllEqual(
                [2, 2],
                diag_len_eval
            )

            (src_eval, seq_len_eval, diag_len_eval) = sess.run((source, seq_len, diag_len))

            self.assertAllEqual(
                [[[0],  # a
                  [1]],  # b

                 [[-1],  # d=unknown
                  [3]]],  # pad
                src_eval
            )
            self.assertAllEqual([[1, 1],
                                 [1, 0]],
                                seq_len_eval)
            self.assertAllEqual([2, 1], diag_len_eval)


if __name__ == '__main__':
    tf.test.main()
