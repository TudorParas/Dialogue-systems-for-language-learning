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

To perform inference on given a trained simple model."""
from __future__ import print_function

import codecs
import collections
import time
import random

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from chatbot.models.simple_model import SimpleModel
from chatbot.models.hier_model import HierarchicalModel
from chatbot.models import model_helper
from utils import iterator_utils
from utils import end2end_iterator_utils
from utils import misc_utils as utils
from utils import chatbot_utils
from utils import vocab_utils
from utils import preprocessing_utils
from assessment.input_assessment import get_user_input

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model", "src_placeholder",
                            "batch_size_placeholder", "iterator"))):
    pass


def create_infer_model(model_creator, get_infer_iterator, hparams, verbose=True, scope=None):
    """Create the inference model"""
    graph = tf.Graph()
    vocab_file = hparams.vocab_file

    with graph.as_default():
        # Create the lookup tables
        vocab_table = vocab_utils.create_vocab_tables(vocab_file)
        ids_to_words = lookup_ops.index_to_string_table_from_file(
            vocabulary_file=vocab_file,
            default_value=vocab_utils.UNK
        )
        # Define data placeholders
        src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        # Create the dataset and iterator
        src_dataset = tf.contrib.data.Dataset.from_tensor_slices(
            src_placeholder)
        iterator = get_infer_iterator(
            dataset=src_dataset,
            vocab_table=vocab_table,
            batch_size=batch_size_placeholder,
            src_reverse=hparams.src_reverse,
            eos=hparams.eos,
            src_max_len=hparams.src_max_len_infer
        )
        # Create the model
        model = model_creator(
            hparams=hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.INFER,
            vocab_table=vocab_table,
            verbose=verbose,
            ids_to_words=ids_to_words,
            scope=scope)

    return InferModel(
        graph=graph,
        model=model,
        src_placeholder=src_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)


def _decode_inference_indices(model, sess,
                              output_infer_file,
                              output_infer_summary_prefix,
                              inference_indices,
                              eos,
                              bpe_delimiter,
                              number_token=None,
                              name_token=None):
    """
    Decoding only a specific set of sentences indicated by inference_indices
    :param output_infer:
    :param output_infer_summary_prefix:
    :param inference_indices: A list of sentence indices
    :param eos: the eos token
    :param bpe_delimiter: delimiter used for byte-pair entries
    :return:
    """
    utils.print_out("  decoding to output %s , num sents %d." %
                    (output_infer_file, len(inference_indices)))
    start_time = time.time()
    with codecs.getwriter("utf-8")(tf.gfile.GFile(output_infer_file, 'wb')) as f:
        f.write("")  # Write empty string to ensure that the file is created
        # Get the outputs
        outputs, infer_summary = model.decode(sess)

        # Iterate over the sentences we want to process. Use the index to process sentences and the
        # decode_id to create logs
        for sentence_id, decode_id in enumerate(inference_indices):
            # Get the response
            response = chatbot_utils.postprocess_output(outputs, sentence_id=sentence_id, eos=eos,
                                                        bpe_delimiter=bpe_delimiter, number_token=number_token,
                                                        name_token=name_token)
            # TODO: add inference_summary if deciding to use attention

            # Write the response to file
            f.write("%s\n" % response)
            utils.print_out("%s\n" % response)
    utils.print_time("  done", start_time)

def load_data(inference_input_file, lines_read=2000, hparams=None):
    """
    Load inference data from file. The lines read argument makes it so that we don't test on everything.
    """
    with codecs.getreader('utf-8')(tf.gfile.GFile(inference_input_file, 'rb')) as f:
        inference_data = f.read().splitlines()
        # Trim the data. inference_indices is a list of indices of the sentences we want to inference
        if hparams and hparams.inference_indices:
            inference_data = [index for index in hparams.inference_indices]
        else:
            inference_data = random.sample(inference_data, lines_read)

        return inference_data


def inference(checkpoint, inference_input_file, inference_output_file,
              hparams, scope=None):
    """Create the responses."""
    # TODO: can add multiple number of workers if so desired. This function is the helper _single_worker_inference
    # in the original model

    # Read the data
    infer_data = load_data(inference_input_file, hparams)


    # Containing the graph, model, source placeholder, batch_size placeholder and iterator
    infer_model = create_infer_model(hparams, scope)
    # ToDo: adapt for architectures
    with tf.Session(graph=infer_model.graph, config=utils.get_config_proto()) as sess:
        # Load the model from the checkpoint
        loaded_infer_model = model_helper.load_model(model=infer_model.model, ckpt=checkpoint,
                                                     session=sess, name="infer")
        # Initialize the iterator
        sess.run(infer_model.iterator.initializer,
                 feed_dict={
                     infer_model.src_placeholder: infer_data,
                     infer_model.batch_size_placeholder: hparams.infer_batch_size
                 })
        # Decode
        utils.print_out("# Starting Decoding")
        # Decode only a specific set of indices
        if hparams.inference_indices:
            _decode_inference_indices(model=loaded_infer_model,
                                      sess=sess,
                                      output_infer_file=inference_output_file,
                                      output_infer_summary_prefix=inference_output_file,
                                      inference_indices=hparams.inference_indices,
                                      eos=hparams.eos,
                                      bpe_delimiter=hparams.bpe_delimiter,
                                      number_token=hparams.number_token,
                                      name_token=hparams.name_token)
        else:
            chatbot_utils.decode_and_evaluate(name="infer",
                                              model=loaded_infer_model,
                                              sess=sess,
                                              output_file=inference_output_file,
                                              reference_file=None,
                                              metrics=hparams.metrics,
                                              bpe_delimiter=hparams.bpe_delimiter,
                                              beam_width=hparams.beam_width,
                                              eos=hparams.eos,
                                              number_token=hparams.number_token,
                                              name_token=hparams.name_token)


def chat(checkpoint, chat_logs_output_file, hparams, scope=None):
    # Containing the graph, model, source placeholder, batch_size placeholder and iterator

    if hparams.architecture == "simple":
        model_creator = SimpleModel
        get_infer_iterator = iterator_utils.get_infer_iterator

        def update_dialogue(so_far, utterance, response):
            # Dialogue so far does not matter
            return utterance

    elif hparams.architecture == "hier":
        model_creator = HierarchicalModel
        # Parse some of the arguments now
        get_infer_iterator = lambda dataset, vocab_table, batch_size, src_reverse, eos, src_max_len: \
            end2end_iterator_utils.get_infer_iterator(dataset, vocab_table, batch_size, src_reverse, eos,
                                                      src_max_len=src_max_len, eou=hparams.eou,
                                                      dialogue_max_len=hparams.dialogue_max_len)
        def update_dialogue(so_far, utterance, response):
            # Dialogue so far is considered
            if response == "":
                return utterance
            return so_far + " " + hparams.eou + " " + response + " " + hparams.eou + " " +  utterance
    else:
        raise ValueError("Unkown architecture", hparams.architecture)

    infer_model = create_infer_model(model_creator, get_infer_iterator, hparams=hparams, verbose=False, scope=scope)

    with tf.Session(graph=infer_model.graph, config=utils.get_config_proto()) as sess:
        # Load the model from the checkpoint
        # ToDo: adapt for architectures
        loaded_infer_model = model_helper.load_model(model=infer_model.model, ckpt=checkpoint,
                                                     session=sess, name="infer", verbose=False)
        utils.print_out("Welcome to ChatBro! If you have any better names please let me know.")
        dialogue_so_far = ""
        response = ""
        # Leave it in chat mode until interrupted
        while True:
            # Read utterance from user.
            utterance = get_user_input()
            # Preprocess it into the familiar format for the machine
            utterance = preprocessing_utils.tokenize_line(utterance, number_token=hparams.number_token,
                                                          name_token=hparams.name_token, gpe_token=hparams.gpe_token)
            dialogue_so_far = update_dialogue(dialogue_so_far, utterance, response)
            # Transform it into a batch of size 1
            batched_dialogue = [dialogue_so_far]
            # Initialize the iterator
            sess.run(infer_model.iterator.initializer,
                     feed_dict={
                         infer_model.src_placeholder: batched_dialogue,
                         infer_model.batch_size_placeholder: 1
                     })

            response = chatbot_utils.decode_utterance(model=loaded_infer_model, sess=sess, output_file=chat_logs_output_file,
                                           bpe_delimiter=hparams.bpe_delimiter, beam_width=hparams.beam_width,
                                           utterance=utterance, top_responses=hparams.top_responses, eos=hparams.eos,
                                           number_token=hparams.number_token, name_token=hparams.name_token)