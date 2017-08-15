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
        utt_max_len = 3
        dialogue_max_len = 2

        iterator = end2end_iterator_utils.get_infer_iterator(dataset, vocab_table,
                                                             batch_size, src_reverse, eos, eou,
                                                             utt_max_len, dialogue_max_len)

        source = iterator.source
        seq_len = iterator.source_sequence_length
        diag_len = iterator.dialogue_length
        self.assertEqual([None, None, None], source.shape.as_list())
        self.assertEqual([None], seq_len.shape.as_list())
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
                [3, 2],  # a b b (because of utt_max_len) and a c
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
            self.assertAllEqual([1, 1], seq_len_eval)
            self.assertAllEqual([2, 1], diag_len_eval)

if __name__ == '__main__':
    tf.test.main()