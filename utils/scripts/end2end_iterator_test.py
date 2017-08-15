import os

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from utils import end2end_iterator_utils
from utils import vocab_utils


def iterator_split():
    text_file_path = os.path.abspath("test_file.txt")

    dataset = tf.contrib.data.TextLineDataset(text_file_path)

    dataset = dataset.map(lambda line: tf.cast(tf.string_split([line], "</s>").values, tf.string))
    # Split into words
    dataset = dataset.map(lambda line: tf.sparse_tensor_to_dense(tf.string_split(line),
                                                                 default_value='<pad>'))

    dataset = dataset.map(lambda dialogue: (dialogue, tf.equal(dialogue, '<pad>')))
    # dataset = dataset.map(lambda indices, shape, values: (tf.sparse_to_dense(sparse_indices=indices,
    #                                                        output_shape=shape,
    #                                                        sparse_values=values,
    #                                                        default_value='</pad>'),
    #                                             shape))
    # dataset = dataset.map(lambda dialogue: (dialogue, tf.cast(tf.constant(not dialogue == '<pad>'), tf.int32)))

    print("mapped")
    print_dataset(dataset)


def print_dataset(dataset):
    tf.InteractiveSession()
    iterator = dataset.make_one_shot_iterator()

    (tens, weights) = iterator.get_next()
    # tens = iterator.get_next()
    tens = tens.eval()
    weights = weights.eval()
    print(tens, tens.shape)
    print(weights, weights.shape)


def infer_iter():
    file_path = os.path.abspath("test_files/en2end_iterator.txt")
    dataset = tf.contrib.data.TextLineDataset(file_path)

    eou = '</u>'
    eos = '</s>'
    src_reverse = False
    batch_size = 1
    utt_max_len = 20
    dialogue_max_len = 20

    vocab_table = lookup_ops.index_table_from_tensor(
        tf.constant([""])
    )
    dataset = tf.contrib.data.Dataset.from_tensor_slices(
        tf.constant(["a b c </u> a a b </u>", "c a b c a </u> c b c a a </u>"])
    )

    iterator = end2end_iterator_utils.get_infer_iterator(dataset, vocab_table, batch_size, src_reverse,
                                                         eos, eou, utt_max_len, dialogue_max_len)

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        for i in range(2):
            source, lengths = sess.run([iterator.source, iterator.source_sequence_length])
            print(source)
            print(lengths)


infer_iter()
