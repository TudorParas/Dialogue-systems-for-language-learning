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

"Utility function specifically for running the chatbot
"""

from __future__ import print_function

import time
import random

import tensorflow as tf

from utils import evaluation_utils
from utils import misc_utils as utils


def decode_and_evaluate(name,
                        model,
                        sess,
                        output_file,
                        reference_file,
                        metrics,
                        bpe_delimiter,
                        beam_width,
                        eos,
                        number_token=None,
                        name_token=None,
                        decode=True):
    """Decode a test set and compute a score according to the evaluation task."""
    # Decode
    if decode:
        utils.print_out("  decoding to output %s." % output_file)
        start_time = time.time()
        num_sentences = 0
        with tf.gfile.GFile(output_file, mode="w+") as out_f:
            out_f.write("")  # Write empty string to ensure file is created.

            while True:
                try:
                    # Get the response(s) for each input in the batch (whole file in this case)
                    # ToDo: adapt for architectures
                    outputs, infer_summary = model.decode(sess)

                    if beam_width > 0:
                        # Get the top response if we used beam_search
                        outputs = outputs[0]

                    num_sentences += len(outputs)
                    # Iterate over the outputs an write them to file
                    for sent_id in range(len(outputs)):
                        response = postprocess_output(outputs,sent_id, eos,
                                                      bpe_delimiter, number_token, name_token)
                        out_f.write("%s\n" % response)
                except tf.errors.OutOfRangeError:
                    utils.print_time("  done, num sentences %d" % num_sentences,
                                     start_time)
                    break

    # Evaluation
    evaluation_scores = {}
    if reference_file and tf.gfile.Exists(output_file):
        for metric in metrics:
            score = evaluation_utils.evaluate(
                ref_file=reference_file,
                trans_file=output_file,
                metric=metric,
                bpe_delimiter=bpe_delimiter
            )
            evaluation_scores[metric] = score
            utils.print_out("  %s %s: %.1f" % (metric, name, score))

    return evaluation_scores


def decode_utterance(model, sess, output_file, bpe_delimiter, beam_width, utterance,
                     top_responses, eos, number_token=None, name_token=None):
    """Creates a response to the user's utterance and writes it to file"""
    # Get the response for the utterance
    outputs, infer_summary = model.decode(sess)

    if beam_width > 0:
        # Do random sampling over top k responses
        response_id = random.randint(0, min(top_responses, len(outputs)) - 1)
        outputs = outputs[response_id]

    # We use postprocess_output as it is a batch of size 1
    response = postprocess_output(outputs, 0, eos, bpe_delimiter,
                                  number_token, name_token)
    if output_file:
        with open(output_file, mode='a+') as out_f:

            # Write the response and the utterance to file
            out_f.write("Human: %s\nChatbot: %s\n" % (utterance, response))
    print('ChatBro: ', response)


def postprocess_output(outputs, sentence_id, eos, bpe_delimiter,
                       number_token=None, name_token=None):
    """Given batch decoding outputs, select a sentence and postprocess it."""
    # Select the sentence
    output = outputs[sentence_id, :].tolist()
    # It doesn't cut off at </s> because of mismatch between </s> and b'</s>', the output of the lookup table

    # The lookup-table outputs are in bytes. We need this for the equality check
    eos = bytes(eos, encoding='utf-8')
    if eos and eos in output:
        output = output[:output.index(eos)]
    if number_token:
        number_token_bytes = bytes(number_token, encoding='utf8')
        output = [b"53" if word == number_token_bytes else word for word in output]
    if name_token:
        name_token_bytes = bytes(name_token, encoding='utf8')
        output = [b"Batman" if word == name_token_bytes else word for word in output]

    if bpe_delimiter:
        response = utils.format_bpe_text(output, bpe_delimiter)
    else:
        response = utils.format_text(output)

    return response


def get_user_input():
    """Get the user's input which we will produce an answer to"""
    return input()