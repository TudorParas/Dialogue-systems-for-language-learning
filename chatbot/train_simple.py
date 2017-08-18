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
"""For training Chatbot models."""

# tp423- Added comments, minor changes such as removing the model_creator variable
# so that it fits more with the current architecture

from __future__ import print_function

import collections
import math
import os
import random
import time

import tensorflow as tf

from chatbot.models.simple_model import SimpleModel
from chatbot.models.hier_model import HierarchicalModel
from chatbot import inference
from chatbot.models import model_helper
from utils import iterator_utils
from utils import misc_utils as utils
from utils import chatbot_utils
from utils import vocab_utils

utils.check_tensorflow_version()


def train(hparams, scope=None, target_session=''):
    """Train the chatbot"""
    # Initialize some local hyperparameters
    log_device_placement = hparams.log_device_placement
    out_dir = hparams.out_dir
    num_train_steps = hparams.num_train_steps
    steps_per_stats = hparams.steps_per_stats
    steps_per_external_eval = hparams.steps_per_external_eval
    steps_per_eval = 10 * steps_per_stats
    if not steps_per_external_eval:
        steps_per_external_eval = 5 * steps_per_eval

    # Create three models which share parameters through the use of checkpoints
    train_model = create_train_model(hparams, scope)
    eval_model = create_eval_model(hparams, scope)
    infer_model = inference.create_infer_model(hparams, scope)

    # Preload the data to use for sample decoding

    dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
    dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
    sample_src_data = inference.load_data(dev_src_file)
    sample_tgt_data = inference.load_data(dev_tgt_file)

    summary_name = "train_log"
    model_dir = hparams.out_dir

    # Log and output files
    log_file = os.path.join(out_dir, "log_%d" % time.time())
    log_f = tf.gfile.GFile(log_file, mode="a")
    utils.print_out("# log_file=%s" % log_file, log_f)

    avg_step_time = 0.0

    # Create the configurations for the sessions
    config_proto = utils.get_config_proto(log_device_placement=log_device_placement)
    # Create three sessions, one for each model
    train_sess = tf.Session(target=target_session, config=config_proto, graph=train_model.graph)
    eval_sess = tf.Session(target=target_session, config=config_proto, graph=eval_model.graph)
    infer_sess = tf.Session(target=target_session, config=config_proto, graph=infer_model.graph)

    # Load the train model from checkpoint or create a new one
    with train_model.graph.as_default():
        loaded_train_model, global_step = model_helper.create_or_load_model(train_model.model, model_dir,
                                                                            train_sess, name="train")

    # Summary writer
    summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, summary_name), train_model.graph)
    # First evaluation
    run_full_eval(
        model_dir, infer_model, infer_sess,
        eval_model, eval_sess, hparams,
        summary_writer, sample_src_data,
        sample_tgt_data)

    last_stats_step = global_step
    last_eval_step = global_step
    last_external_eval_step = global_step

    # This is the training loop.
    # Initialize the hyperparameters for the loop.
    step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
    checkpoint_total_count = 0.0
    speed, train_ppl = 0.0, 0.0
    start_train_time = time.time()

    utils.print_out(
        "# Start step %d, lr %g, %s" %
        (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
         time.ctime()),
        log_f)

    # epoch_step records where we were within an epoch. Used to skip trained on examples
    skip_count = hparams.batch_size * hparams.epoch_step
    utils.print_out("# Init train iterator, skipping %d elements" % skip_count)
    # Initialize the training iterator
    train_sess.run(
        train_model.iterator.initializer,
        feed_dict={train_model.skip_count_placeholder: skip_count})

    # Train until we reach num_steps.
    while global_step < num_train_steps:
        # Run a step
        start_step_time = time.time()
        try:
            step_result = loaded_train_model.train(train_sess)
            (_, step_loss, step_predict_count, step_summary, global_step,  # The _ is the output of the update op
             step_word_count, batch_size) = step_result
            hparams.epoch_step += 1
        except tf.errors.OutOfRangeError:
            # Finished going through the training dataset.  Go to next epoch.
            hparams.epoch_step = 0
            utils.print_out(
                "# Finished an epoch, step %d. Perform external evaluation" %
                global_step)
            # Decode and print a random sentence
            run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                              sample_src_data, sample_tgt_data)
            # Perform external evaluation to save checkpoints if this is the best for some metric
            dev_scores, test_scores, _ = run_external_evaluation(infer_model, infer_sess, model_dir, hparams,
                                                                 summary_writer, save_on_best_dev=True)
            # Reinitialize the iterator from the beginning
            train_sess.run(train_model.iterator.initializer,
                           feed_dict={train_model.skip_count_placeholder: 0})
            continue

        # Write step summary.
        summary_writer.add_summary(step_summary, global_step)

        # update statistics
        step_time += (time.time() - start_step_time)

        checkpoint_loss += (step_loss * batch_size)
        checkpoint_predict_count += step_predict_count
        checkpoint_total_count += float(step_word_count)

        # Once in a while, we print statistics.
        if global_step - last_stats_step >= steps_per_stats:
            last_stats_step = global_step

            # Print statistics for the previous epoch.
            avg_step_time = step_time / steps_per_stats
            train_ppl = utils.safe_exp(checkpoint_loss / checkpoint_predict_count)
            speed = checkpoint_total_count / (1000 * step_time)
            utils.print_out(
                "  global step %d lr %g "
                "step-time %.2fs wps %.2fK ppl %.2f %s" %
                (global_step,
                 loaded_train_model.learning_rate.eval(session=train_sess),
                 avg_step_time, speed, train_ppl, _get_best_results(hparams)),
                log_f)
            if math.isnan(train_ppl):
                # The model has screwed up
                break

            # Reset timer and loss.
            step_time, checkpoint_loss, checkpoint_predict_count = 0.0, 0.0, 0.0
            checkpoint_total_count = 0.0

        if global_step - last_eval_step >= steps_per_eval:
            # Perform evaluation. Start by reassigning the last_eval_step variable to the current step
            last_eval_step = global_step
            # Print the progress and add summary
            utils.print_out("# Save eval, global step %d" % global_step)
            utils.add_summary(summary_writer, global_step, "train_ppl", train_ppl)

            # Save checkpoint
            loaded_train_model.saver.save(train_sess, os.path.join(out_dir, "chatbot.ckpt"), global_step=global_step)
            # Decode and print a random sample
            run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                              sample_src_data, sample_tgt_data)
            # Run internal evaluation, and update the ppl variables. The data iterator is instantieted in the method.
            dev_ppl, test_ppl = run_internal_eval(eval_model, eval_sess, model_dir, hparams, summary_writer)

        if global_step - last_external_eval_step >= steps_per_external_eval:
            # Run the external evaluation
            last_external_eval_step = global_step
            # Save checkpoint
            loaded_train_model.saver.save(train_sess, os.path.join(out_dir, "chatbot.ckpt"), global_step=global_step)
            # Decode and print a random sample
            run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                              sample_src_data, sample_tgt_data)
            # Run external evaluation, updating metric scores in the meanwhile. The unneeded output is the global step.
            dev_scores, test_scores, _ = run_external_evaluation(infer_model, infer_sess, model_dir, hparams,
                                                                 summary_writer, save_on_best_dev=True)

    # Done training. Save the model
    loaded_train_model.saver.save(
        train_sess,
        os.path.join(out_dir, "chatbot.ckpt"),
        global_step=global_step)

    result_summary, _, dev_scores, test_scores, dev_ppl, test_ppl = run_full_eval(
        model_dir, infer_model, infer_sess,
        eval_model, eval_sess, hparams,
        summary_writer, sample_src_data,
        sample_tgt_data)
    utils.print_out(
        "# Final, step %d lr %g "
        "step-time %.2f wps %.2fK ppl %.2f, %s, %s" %
        (global_step, loaded_train_model.learning_rate.eval(session=train_sess),
         avg_step_time, speed, train_ppl, result_summary, time.ctime()),
        log_f)
    utils.print_time("# Done training!", start_train_time)

    utils.print_out("# Start evaluating saved best models.")
    for metric in hparams.metrics:
        best_model_dir = getattr(hparams, "best_" + metric + "_dir")
        result_summary, best_global_step, _, _, _, _ = run_full_eval(
            best_model_dir, infer_model, infer_sess, eval_model, eval_sess, hparams,
            summary_writer, sample_src_data, sample_tgt_data)
        utils.print_out("# Best %s, step %d "
                        "step-time %.2f wps %.2fK, %s, %s" %
                        (metric, best_global_step, avg_step_time, speed,
                         result_summary, time.ctime()), log_f)

    summary_writer.close()
    return (dev_scores, test_scores, dev_ppl, test_ppl, global_step)



class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "iterator", "skip_count_placeholder"))):
    """Interface for the model used for training."""
    pass


def create_train_model(hparams, scope=None):
    """Create the training graph, model and iterator"""

    # Get the files by concatting prefixes and outputs.
    src_file = "%s.%s" % (hparams.train_prefix, hparams.src)
    tgt_file = "%s.%s" % (hparams.train_prefix, hparams.tgt)
    vocab_file = hparams.vocab_file
    # Define the graph
    graph = tf.Graph()

    with graph.as_default():
        vocab_table = vocab_utils.create_vocab_tables(vocab_file)
        # Create datasets from file
        src_dataset = tf.contrib.data.TextLineDataset(src_file)
        tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file)
        # The number of elements of this dataset that should be skipped to form the new dataset.
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)
        # Iterator
        iterator = iterator_utils.get_iterator(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            vocab_table=vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            src_reverse=hparams.src_reverse,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len,
            tgt_max_len=hparams.tgt_max_len,
            skip_count=skip_count_placeholder
        )
        # Model. We don't give ids_to_words arg because we don't need it for training
        model = SimpleModel(
            hparams=hparams,
            mode=tf.contrib.learn.ModeKeys.TRAIN,
            iterator=iterator,
            vocab_table=vocab_table,
            scope=scope
        )

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator,
        skip_count_placeholder=skip_count_placeholder
    )


class EvalModel(collections.namedtuple("EvalMode", ("graph", "model", "src_file_placeholder",
                                                    "tgt_file_placeholder", "iterator"))):
    """Interface for an evaluation model"""
    pass


def create_eval_model(hparams, scope=None):
    """Create train graph, model, src/tgt file holders, and iterator."""
    vocab_file = hparams.vocab_file
    # Define the graph
    graph = tf.Graph()

    with graph.as_default():
        vocab_table = vocab_utils.create_vocab_tables(vocab_file)
        # Create placeholders for the file location
        src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        # Create the datasets from file
        src_dataset = tf.contrib.data.TextLineDataset(src_file_placeholder)
        tgt_dataset = tf.contrib.data.TextLineDataset(tgt_file_placeholder)
        # Create the iterator for the dataset. We do not use skip_count here as we evaluate on the full file
        iterator = iterator_utils.get_iterator(
            src_dataset=src_dataset,
            tgt_dataset=tgt_dataset,
            vocab_table=vocab_table,
            batch_size=hparams.batch_size,
            sos=hparams.sos,
            eos=hparams.eos,
            src_reverse=hparams.src_reverse,
            random_seed=hparams.random_seed,
            num_buckets=hparams.num_buckets,
            src_max_len=hparams.src_max_len_infer,
            tgt_max_len=hparams.tgt_max_len_infer
        )
        # Create a simple model
        model = SimpleModel(
            hparams=hparams,
            iterator=iterator,
            mode=tf.contrib.learn.ModeKeys.EVAL,
            vocab_table=vocab_table,
            scope=scope
        )

    return EvalModel(
        graph=graph,
        model=model,
        src_file_placeholder=src_file_placeholder,
        tgt_file_placeholder=tgt_file_placeholder,
        iterator=iterator
    )


def run_sample_decode(infer_model, infer_sess, model_dir, hparams, summary_writer,
                      src_data, tgt_data):
    """
    Sample decode a random sentence from the source data. Used to print the tangible progress of the model.
    :param infer_model: The model used to produce the response.
    :param model_dir: directory which contains the trained model
    :param summary_writer: An instance of a tensorflow Summary writer
    :return:
    """
    with infer_model.graph.as_default():
        # Load the model from checkpoint. It automatically loads the latest checkpoint
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            model=infer_model.model,
            model_dir=model_dir,
            session=infer_sess,
            name="infer"
        )
        _sample_decode(model=loaded_infer_model, global_step=global_step, sess=infer_sess, hparams=hparams,
                       iterator=infer_model.iterator, src_data=src_data, tgt_data=tgt_data,
                       iterator_src_placeholder=infer_model.src_placeholder,
                       iterator_batch_size_placeholder=infer_model.batch_size_placeholder,
                       summary_writer=summary_writer)


def run_internal_eval(eval_model, eval_sess, model_dir, hparams, summary_writer):
    """Compute internal evaluation (perplexity) for both dev / test."""
    with eval_model.graph.as_default():
        # Load the latest checkpoint from file
        loaded_eval_model, global_step = model_helper.create_or_load_model(
            model=eval_model.model,
            model_dir=model_dir,
            session=eval_sess,
            name="eval"
        )
        # Fill the feed_dict for the evaluation
        dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
        dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)
        dev_eval_iterator_feed_dict = {
            eval_model.src_file_placeholder: dev_src_file,
            eval_model.tgt_file_placeholder: dev_tgt_file
        }
        # Run evaluation on the development (validation) dataset
        dev_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                                 iterator=eval_model.iterator,
                                 iterator_feed_dict=dev_eval_iterator_feed_dict,
                                 summary_writer=summary_writer,
                                 label='dev')

        test_ppl = None
        if hparams.test_prefix:
            # Create the test data
            test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
            test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
            test_eval_iterator_feed_dict = {
                eval_model.src_file_placeholder: test_src_file,
                eval_model.tgt_file_placeholder: test_tgt_file
            }
            # Run evaluation on the test dataset
            test_ppl = _internal_eval(loaded_eval_model, global_step, eval_sess,
                                      iterator=eval_model.iterator,
                                      iterator_feed_dict=test_eval_iterator_feed_dict,
                                      summary_writer=summary_writer,
                                      label='test')

    return dev_ppl, test_ppl


def run_external_evaluation(infer_model, infer_sess, model_dir, hparams,
                            summary_writer, save_on_best_dev):
    with infer_model.graph.as_default():
        # Load the model from checkpoint. It automatically loads the latest checkpoint
        loaded_infer_model, global_step = model_helper.create_or_load_model(
            model=infer_model.model,
            model_dir=model_dir,
            session=infer_sess,
            name="infer"
        )
        # Fill the feed_dict for the evaluation
        dev_src_file = "%s.%s" % (hparams.dev_prefix, hparams.src)
        dev_tgt_file = "%s.%s" % (hparams.dev_prefix, hparams.tgt)

        inference_dev_data = inference.load_data(dev_src_file)
        dev_infer_iterator_feed_dict = {
            infer_model.src_placeholder: inference_dev_data,
            infer_model.batch_size_placeholder: hparams.infer_batch_size
        }

        dev_scores = _external_eval(
            model=loaded_infer_model,
            global_step=global_step,
            sess=infer_sess,
            hparams=hparams,
            iterator=infer_model.iterator,
            iterator_feed_dict=dev_infer_iterator_feed_dict,
            tgt_file=dev_tgt_file,
            label="dev",
            summary_writer=summary_writer,
            save_on_best_dev=save_on_best_dev
        )

        test_scores = None
        if hparams.test_prefix:
            # Create the test data
            test_src_file = "%s.%s" % (hparams.test_prefix, hparams.src)
            test_tgt_file = "%s.%s" % (hparams.test_prefix, hparams.tgt)
            inference_test_data = inference.load_data(test_src_file)
            test_infer_iterator_feed_dict = {
                infer_model.src_placeholder: inference_test_data,
                infer_model.batch_size_placeholder: hparams.infer_batch_size
            }
            # Run evaluation on the test dataset
            test_scores = _external_eval(
                model=loaded_infer_model,
                global_step=global_step,
                sess=infer_sess,
                hparams=hparams,
                iterator=infer_model.iterator,
                iterator_feed_dict=test_infer_iterator_feed_dict,
                tgt_file=test_tgt_file,
                label="test",
                summary_writer=summary_writer,
                save_on_best_dev=False  # We do not use the test set at all in training as that means overfitting
            )

    return dev_scores, test_scores, global_step


def run_full_eval(model_dir, infer_model, infer_sess, eval_model, eval_sess,
                  hparams, summary_writer, sample_src_data, sample_tgt_data):
    """
    Wrapper for running sample_decode, internal_eval and external_eval. We have different models and different
    sessions for each.
    """
    # Run decoding on a random sentence from the source data to show the progress in the stdout
    run_sample_decode(infer_model=infer_model,
                      infer_sess=infer_sess,
                      model_dir=model_dir,
                      hparams=hparams,
                      summary_writer=summary_writer,
                      src_data=sample_src_data,
                      tgt_data=sample_tgt_data)
    # Get the development and test perplexity
    dev_ppl, test_ppl = run_internal_eval(eval_model=eval_model,
                                          eval_sess=eval_sess,
                                          model_dir=model_dir,
                                          hparams=hparams,
                                          summary_writer=summary_writer)
    # Get the scores for the metrics
    dev_scores, test_scores, global_step = run_external_evaluation(infer_model=infer_model,
                                                                   infer_sess=infer_sess,
                                                                   model_dir=model_dir,
                                                                   hparams=hparams,
                                                                   summary_writer=summary_writer,
                                                                   save_on_best_dev=True)  # Save the model which gets the best metric
    # Create a results string
    result_summary = _format_results("dev", dev_ppl, dev_scores, hparams.metrics)
    if hparams.test_prefix:
        result_summary += ", " + _format_results("test", test_ppl, test_scores,
                                                 hparams.metrics)

    return result_summary, global_step, dev_scores, test_scores, dev_ppl, test_ppl


def _sample_decode(model, global_step, sess, hparams, iterator, src_data,
                   tgt_data, iterator_src_placeholder, iterator_batch_size_placeholder,
                   summary_writer):
    """
    Pick a random sentence and decode it.
    Args:
            iterator_src_placeholder, iterator_batch_size_placeholder: used to initialize the model
    """
    decode_id = random.randint(0, len(src_data) - 1)
    utils.print_out("  Decoding sentence %d" % decode_id)
    # Format the random sentence into a batch_size of 1 format.
    sentence = [src_data[decode_id]]
    # Create the feed-dict for the iterator
    iterator_feed_dict = {
        iterator_src_placeholder: sentence,
        iterator_batch_size_placeholder: 1
    }
    # Initialize the iterator
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    # Get the response. The summary is only used in attention models, which we do not use atm
    response, attention_summary = model.decode(sess)

    if hparams.beam_width > 0:
        response = response[0]
    # Postprocess the response
    response = chatbot_utils.postprocess_output(response, sentence_id=0, eos=hparams.eos,
                                                bpe_delimiter=hparams.bpe_delimiter,
                                                number_token=hparams.number_token, name_token=hparams.name_token)

    # ToDo: Add attention summary here if deciding to use attention
    # Add the print to check the model's progress
    utils.print_out("    src: %s" % src_data[decode_id])
    utils.print_out("    ref: %s" % tgt_data[decode_id])
    utils.print_out("    Chatbot: %s" % response)


def _internal_eval(model, global_step, sess, iterator, iterator_feed_dict, summary_writer, label):
    """Used to complute perplexity on the dataset provided through the iterator"""
    # Initialize the iterator using the feed dict
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    # Compute the perplexity
    ppl = model_helper.compute_perplexity(model, sess, name=label)
    # Add summary for the ppl to the summary writer
    utils.add_summary(summary_writer, global_step, tag="%s_ppl" % label, value=ppl)

    return ppl


def _external_eval(model, global_step, sess, hparams, iterator,
                   iterator_feed_dict, tgt_file, label, summary_writer,
                   save_on_best_dev):
    """External evaluation such as BLEU and ROUGE scores. If save on best then keep the best scores in the hparams"""
    out_dir = hparams.out_dir
    # Avoids running eval when global step is 0
    decode = global_step > 0
    if decode:
        utils.print_out("# External evaluation, global step %d" % global_step)
    # Initialize the iterator
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    # Create the output file for the logs
    output_file = os.path.join(out_dir, "output_%s" % label)
    # Get the scores for the metrics
    scores = chatbot_utils.decode_and_evaluate(
        name=label,
        model=model,
        sess=sess,
        output_file=output_file,
        reference_file=tgt_file,
        metrics=hparams.metrics,
        bpe_delimiter=hparams.bpe_delimiter,
        beam_width=hparams.beam_width,
        eos=hparams.eos,
        number_token=hparams.number_token,
        name_token=hparams.name_token,
        decode=decode
    )
    # Create the summaries and also save the best
    if decode:
        for metric in hparams.metrics:
            # Create the summary
            utils.add_summary(summary_writer, global_step, "%s_%s" % (label, metric),
                              scores[metric])
            # Is the current metric score better than the last
            if save_on_best_dev and scores[metric] > getattr(hparams, "best_" + metric):
                # Update the hparams score
                setattr(hparams, "best_" + metric, scores[metric])
                # Save the model which got the best for this metric to file
                model.saver.save(sess,
                                 os.path.join(getattr(hparams, "best_" + metric + "_dir"), "dialogue.ckpt"),
                                 global_step=model.global_step)  # For safety
    # Save the hparams to file
    utils.save_hparams(out_dir, hparams, verbose=True)

    return scores


def _format_results(name, ppl, scores, metrics):
    """Format results. Round ppl to 2 sf and metrics to 1sf"""
    result_str = "%s ppl %.2f" % (name, ppl)
    if scores:
        for metric in metrics:
            result_str += ", %s %s %.1f" % (name, metric, scores[metric])
    return result_str


def _get_best_results(hparams):
    """Summary of the current best results."""
    tokens = []
    for metric in hparams.metrics:
        tokens.append("%s %.2f" % (metric, getattr(hparams, "best_" + metric)))
    return ", ".join(tokens)
