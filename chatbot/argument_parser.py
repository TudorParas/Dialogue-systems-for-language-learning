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

Script which we use to create the hparams
"""


from __future__ import print_function

import os

import tensorflow as tf

from utils import misc_utils as utils
from utils import vocab_utils

utils.check_tensorflow_version()


def add_arguments(parser):
    """Build the arguements parser"""
    # Register the boolean type. This allows us to use booleans as arguments
    parser.register('type', 'bool', lambda v: v.lower == "true")

    # Hyperparameters regarding the neural network
    parser.add_argument("--num_units", type=int, default=32, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Network depth.")
    parser.add_argument("--encoder_type", type=str, default="uni", help="""\
        uni | bi. For bi, we build num_layers/2 bi-directional layers.""")
    parser.add_argument("--residual", type="bool", nargs="?", const=True,
                        default=False,
                        help="Whether to add residual connections.")
    parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                        default=True,
                        help="Whether to use time-major mode for dynamic RNN. This would change the shape of the \
                             data. Useful because of performance reason, how C++ keeps data in memory")
    # Hyperparameters regarding the optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam should use: 0.001 | 0.0001")
    # Learning rate decay. Used with the sgd optimizer
    parser.add_argument("--start_decay_step", type=int, default=0,
                        help="When we start to decay")
    parser.add_argument("--decay_steps", type=int, default=10000,
                        help="How frequent we decay")
    parser.add_argument("--decay_factor", type=float, default=0.98,
                        help="How much we decay.")
    # How many times we want the model to be trained for.
    parser.add_argument(
        "--num_train_steps", type=int, default=12000, help="Num steps to train.")
    parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                        const=True,
                        default=True,
                        help=("Whether try colocating gradients with "
                              "corresponding op"))

    # Data paths
    parser.add_argument("--src", type=str, default=None,
                        help="Source suffix, e.g., en.")
    parser.add_argument("--tgt", type=str, default=None,
                        help="Target suffix, e.g., de.")
    parser.add_argument("--train_prefix", type=str, default=None,
                        help="Train prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--dev_prefix", type=str, default=None,
                        help="Dev prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--test_prefix", type=str, default=None,
                        help="Test prefix, expect files with src/tgt suffixes.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")
    # Vocab path
    parser.add_argument("--vocab_file", type=str, default=None, help="""Path to the file containing the vocab""")
    # Tokens
    parser.add_argument("--sos", type=str, default="<s>",
                        help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>",
                        help="End-of-sentence symbol.")
    parser.add_argument("--number_token", type=str, default=None,
                        help="Token used to replace seen numbers")
    parser.add_argument("--name_token", type=str, default=None,
                        help="Token used to replace seen numbers")
    parser.add_argument("--gpe_token", type=str, default=None,
                        help="Token used to replace seen geopolitical entities")

    # Sequence lengths
    parser.add_argument("--src_max_len", type=int, default=50,
                        help="Max length of src sequences during training.")
    parser.add_argument("--tgt_max_len", type=int, default=50,
                        help="Max length of tgt sequences during training.")
    parser.add_argument("--src_max_len_infer", type=int, default=None,
                        help="Max length of src sequences during inference.")
    parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                        help="""\
        Max length of tgt sequences during inference.  Also use to restrict the
        maximum decoding length.\
        """)

    # Default settings works well (rarely need to change)
    parser.add_argument("--unit_type", type=str, default="lstm",
                        help="lstm | gru | layer_norm_lstm")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="Clip gradients to this norm in order to avoid exploding gradients.")
    parser.add_argument("--initial_weight", type=float, default=0.1,
                        help="Initial weights from [-this, this].")
    parser.add_argument("--src_reverse", type="bool", nargs="?", const=True,
                        default=False, help="Reverse source sequence.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")

    parser.add_argument("--steps_per_stats", type=int, default=100,
                        help=("How many training steps to do per stats logging."
                              "Save checkpoint every 10x steps_per_stats"))
    parser.add_argument("--max_train", type=int, default=0,
                        help="Limit on the size of training data (0: no limit).")
    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")
    parser.add_argument("--decoding_length_factor", type=float, default=2.0,
                        help="""\
                        If tgt_max_len_infer has not been provided then it assigns it the value of the max encoding
                         times this length factor \
                        """)

    # BPE
    parser.add_argument("--bpe_delimiter", type=str, default=None,
                        help="Set to @@ to activate BPE")

    # Misc
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of gpus in each worker.")
    parser.add_argument("--log_device_placement", type="bool", nargs="?",
                        const=True, default=False, help="Debug GPU allocation.")
    parser.add_argument("--metrics", type=str, default="bleu",
                        help=("Comma-separated list of evaluations "
                              "metrics (bleu,rouge,accuracy)"))
    parser.add_argument("--steps_per_external_eval", type=int, default=None,
                        help="""\
        How many training steps to do per external evaluation.  Automatically set
        based on data if None.\
        """)
    parser.add_argument("--scope", type=str, default=None,
                        help="scope to put variables under")
    parser.add_argument("--hparams_path", type=str, default=None,
                        help=("Path to standard hparams json file that overrides"
                              "hparams values from flags."))
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")

    # Inference
    parser.add_argument("--ckpt", type=str, default="",
                        help="Checkpoint file to load a model for inference.")
    # Chat
    parser.add_argument("--chat", type=bool, default=False,
                        help="Whether you would like to converse with the bot")
    parser.add_argument("--chat_logs_output_file", type=str, default=None,
                        help="File to write the conversations out to")
    parser.add_argument("--top_responses", type=int, default=1,
                        help="Sample over the top responses when using beam search in chat mode.")
    # Infer from file
    parser.add_argument("--inference_input_file", type=str, default=None,
                        help="Set to the text to decode.")

    parser.add_argument("--inference_list", type=str, default=None,
                        help=("A comma-separated list of sentence indices "
                              "(0-based) to decode."))
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Batch size for inference mode.")
    parser.add_argument("--inference_output_file", type=str, default=None,
                        help="Output file to store decoding results.")
    parser.add_argument("--inference_ref_file", type=str, default=None,
                        help=("""\
          Reference file to compute evaluation scores (if provided).\
          """))
    parser.add_argument("--beam_width", type=int, default=0,
                        help=("""\
          beam width when using beam search decoder. If 0 (default), use standard
          decoder with greedy helper.\
          """))
    parser.add_argument("--length_penalty_weight", type=float, default=0.0,
                        help="Length penalty for beam search.")


def create_hparams(flags):
    """Create training hparams."""
    return tf.contrib.training.HParams(
        # Data
        src=flags.src,
        tgt=flags.tgt,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_file=flags.vocab_file,
        out_dir=flags.out_dir,

        # Networks
        num_units=flags.num_units,
        num_layers=flags.num_layers,
        dropout=flags.dropout,
        unit_type=flags.unit_type,
        encoder_type=flags.encoder_type,
        residual=flags.residual,
        time_major=flags.time_major,

        # Train
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        initial_weight=flags.initial_weight,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        start_decay_step=flags.start_decay_step,
        decay_factor=flags.decay_factor,
        decay_steps=flags.decay_steps,
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,

        # Data constraints
        num_buckets=flags.num_buckets,
        max_train=flags.max_train,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,
        src_reverse=flags.src_reverse,

        # Inference
        top_responses=flags.top_responses,
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,
        decoding_length_factor=flags.decoding_length_factor,

        # Vocab
        sos=flags.sos if flags.sos else vocab_utils.SOS,
        eos=flags.eos if flags.eos else vocab_utils.EOS,
        number_token=flags.number_token,
        name_token=flags.name_token,
        gpe_token=flags.gpe_token,
        bpe_delimiter=flags.bpe_delimiter,

        # Misc
        forget_bias=flags.forget_bias,
        num_gpus=flags.num_gpus,
        epoch_step=0,  # record where we were within an epoch.
        steps_per_stats=flags.steps_per_stats,
        steps_per_external_eval=flags.steps_per_external_eval,
        metrics=flags.metrics.split(","),
        log_device_placement=flags.log_device_placement,
        random_seed=flags.random_seed,
    )


def extend_hparams(hparams):
    """Extend training hparams."""
    # Sanity checks
    if hparams.encoder_type == "bi" and hparams.num_layers % 2 != 0:
        raise ValueError("For bi, num_layers %d should be even" %
                         hparams.num_layers)
    if hparams.top_responses < 1:
        raise ValueError("We need to choose from the top responses. %s is not \
                         a valid value" % hparams.top_responses)

    # flags
    utils.print_out("# hparams:")
    utils.print_out("  src=%s" % hparams.src)
    utils.print_out("  tgt=%s" % hparams.tgt)
    utils.print_out("  train_prefix=%s" % hparams.train_prefix)
    utils.print_out("  dev_prefix=%s" % hparams.dev_prefix)
    utils.print_out("  test_prefix=%s" % hparams.test_prefix)
    utils.print_out("  out_dir=%s" % hparams.out_dir)

    # Set num_residual_layers
    if hparams.residual and hparams.num_layers > 1:
        num_residual_layers = hparams.num_layers - 1
    else:
        num_residual_layers = 0
    hparams.add_hparam("num_residual_layers", num_residual_layers)

    # Vocab
    if hparams.vocab_file:
        vocab_size, vocab_file = vocab_utils.check_vocab(hparams.vocab_file, out_dir=hparams.out_dir,
                                                         sos=hparams.sos, eos=hparams.eos, unk=vocab_utils.UNK)
    else:
        raise ValueError("A vocab_file must be provided by using --vocab_file=<vocab path>")
    # Add the vocab size and override the vocab_file
    hparams.add_hparam("vocab_size", vocab_size)
    hparams.parse("vocab_file=%s" % vocab_file)

    # Check out_dir
    if not tf.gfile.Exists(hparams.out_dir):
        utils.print_out("# Creating output directory %s ..." % hparams.out_dir)
        tf.gfile.MakeDirs(hparams.out_dir)

    # Evaluation
    for metric in hparams.metrics:
        hparams.add_hparam("best_" + metric, 0)  # larger is better
        best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
        hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
        tf.gfile.MakeDirs(best_metric_dir)

    return hparams


def ensure_compatible_hparams(hparams, default_hparams, flags):
    """Make sure the loaded hparams is compatible with new changes."""
    default_hparams = utils.maybe_parse_standard_hparams(
        default_hparams, flags.hparams_path, verbose=not flags.chat)

    # For compatible reason, if there are new fields in default_hparams,
    #   we add them to the current hparams
    default_config = default_hparams.values()
    config = hparams.values()
    for key in default_config:
        if key not in config:
            hparams.add_hparam(key, default_config[key])

    # Make sure that the loaded model has latest values for the below keys
    updated_keys = [
        "out_dir", "num_gpus", "test_prefix", "beam_width",
        "length_penalty_weight", "num_train_steps", "number_token",
        "name_token", "gpe_token"
    ]
    for key in updated_keys:
        if key in default_config and getattr(hparams, key) != default_config[key]:
            if not flags.chat:
                utils.print_out("# Updating hparams.%s: %s -> %s" %
                            (key, str(getattr(hparams, key)), str(default_config[key])))
            setattr(hparams, key, default_config[key])
    return hparams


def create_or_load_hparams(out_dir, default_hparams, flags):
    """Create hparams or load hparams from out_dir."""
    hparams = utils.load_hparams(out_dir, verbose=not flags.chat)
    if not hparams:
        # Parse the ones from the command line
        hparams = default_hparams
        hparams = utils.maybe_parse_standard_hparams(
            hparams, flags.hparams_path, verbose=not flags.chat)
        hparams = extend_hparams(hparams)
    else:
        hparams = ensure_compatible_hparams(hparams, default_hparams, flags)

    # Save HParams
    utils.save_hparams(out_dir, hparams, verbose=not flags.chat)

    for metric in hparams.metrics:
        utils.save_hparams(getattr(hparams, "best_" + metric + "_dir"), hparams, verbose=not flags.chat)

    # Print HParams
    if not flags.chat:
        utils.print_hparams(hparams)
    return hparams

