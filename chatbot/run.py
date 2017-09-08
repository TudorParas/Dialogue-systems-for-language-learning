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

In this file we have the 'control panel' of the chatbot, where
we parse the arguments and run it accordingly.
"""
from __future__ import print_function

import argparse
import os
import random

import numpy as np
import tensorflow as tf

# Make the tf library stop printing warnings about cpu modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from chatbot import inference
from chatbot import train
from utils import evaluation_utils
from utils import misc_utils as utils
from chatbot import argument_parser

utils.check_tensorflow_version()


def main(flags, unused_argv):
    # Get the random seed and seed numpy and random with it
    random_seed = flags.random_seed
    if random_seed is not None and random_seed > 0:
        if flags.verbose:
            utils.print_out("# Set random seed to %d" % random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Create the output directory
    out_dir = flags.out_dir
    if not tf.gfile.Exists(out_dir):
        tf.gfile.MakeDirs(out_dir)

    # Load the hyperparameters
    default_hparams = argument_parser.create_hparams(flags)
    hparams = argument_parser.create_or_load_hparams(out_dir, default_hparams, flags)


    # The place where we decide if we train or if we do inference
    # ToDo: Add ability to chat based on the chat argument
    if flags.chat:
        chat_logs_output_file = flags.chat_logs_output_file
        ckpt = flags.ckpt
        if not ckpt:
            # If a checkpoint has not been provided then load the latest one
            ckpt = tf.train.latest_checkpoint(out_dir)
        # Initiate chat mode
        inference.chat(checkpoint=ckpt, chat_logs_output_file=chat_logs_output_file, hparams=hparams)

    elif flags.inference_input_file:
        # Inference indices
        hparams.inference_indices = None
        if flags.inference_list:
            (hparams.inference_indices) = (
                [int(token) for token in flags.inference_list.split(",")])

        # Inference
        inference_output_file = flags.inference_output_file
        ckpt = flags.ckpt
        if not ckpt:
            # If a checkpoint has not been provided then load the latest one
            ckpt = tf.train.latest_checkpoint(out_dir)
        # Get responses to the utterances and write them to file
        inference.inference(ckpt, flags.inference_input_file, inference_output_file, hparams)

        # Compute scores for the reference file
        ref_file = flags.inference_ref_file
        if ref_file and tf.gfile.Exists(inference_output_file):
            for metric in hparams.metrics:
                score = evaluation_utils.evaluate(
                    ref_file,
                    inference_output_file,
                    metric,
                    hparams.bpe_delimiter)
                if flags.verbose:
                    utils.print_out("  %s: %.1f" % (metric, score))

    else:
        # Start training
        train.train(hparams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    argument_parser.add_arguments(parser)
    flags, unparsed = parser.parse_known_args()
    # Run the app
    main(flags, unparsed)
