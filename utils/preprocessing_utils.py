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

Useful functions for postprocessing data
"""

import codecs
import nltk
import itertools
import os

from tqdm import tqdm
import tensorflow as tf


def create_vocab(src_file, out_dir, vocab_size,
                 eos, sos, unk):
    with codecs.getreader("utf-8")(tf.gfile.GFile(src_file, "rb")) as f:
        print("Creating vocabulary")
        lines = f.readlines()
        # We use split instead of the nltk tokenizer because this is what we'll use later.
        lines = [line.split() for line in lines]
        vocab = nltk.FreqDist(itertools.chain(*lines))
        vocab = vocab.most_common(vocab_size - 3)
        # Extract the words, disregard their frequencies.
        vocab = [tup[0] for tup in vocab]
        # Replace apostrophes with &apos;
        # Put in the tokens
        vocab.insert(0, eos)
        vocab.insert(0, sos)
        vocab.insert(0, unk)

    # Write it to file
    with codecs.getwriter("utf-8")(tf.gfile.GFile(os.path.join(out_dir, 'vocab'), 'w+')) as f:
        for word in vocab:
            f.write("%s\n" % word)


def tokenize_conversations(conversations, max_line_length=None, number_token=None,
                           name_token=None, gpe_token=None):
    """
    The data is fed in a string tokenized using split(), so we have to
        split the data with the same tokenizer which we use to build the vocab.
        We also do Name Entity replacement
        """
    tokenized_convos = []
    for convo in tqdm(conversations, "Tokenizing conversations"):
        token_conv = []
        for line in convo:
            tokenized_line = tokenize_line(line,
                                           max_line_length=max_line_length,
                                           number_token=number_token,
                                           name_token=name_token,
                                           gpe_token=gpe_token)
            token_conv.append(tokenized_line)
        tokenized_convos.append(token_conv)
    return tokenized_convos


def tokenize_line(line, max_line_length=None, number_token=None, name_token=None, gpe_token=None):
    """Preprocesses a sentence so that it can later be tokenized using the split() command"""
    new_line = word_tokenize(line,
                             number_token=number_token,
                             name_token=name_token,
                             gpe_token=gpe_token)
    if max_line_length:
        new_line = new_line[:max_line_length]
    new_line = vector_to_string(new_line)
    return new_line


def word_tokenize(sentence, number_token=None, name_token=None, gpe_token=None):
    """Tokenized the sentence and does name-entity recognition, replacing them with tokens"""
    tokenized_sentence = nltk.word_tokenize(sentence)
    # Verify that the tokens are lowercase or none and then do the replacement
    _check_tokens(number_token, name_token, gpe_token)

    tokenized_sentence = number2token(tokenized_sentence, number_token)
    tagged_sentence = entities2token(tokenized_sentence, name_token, gpe_token)

    return tagged_sentence


def number2token(tokenized_sentence, number_token=None):
    # Check whether we do number replacement
    if number_token is not None:
        # Tag the tokens
        tagged = nltk.pos_tag(tokenized_sentence)
        # Tokenize the numbers
        tagged = [number_token if tag == "CD" else token for (token, tag) in tagged]
    else:
        tagged = tokenized_sentence
    return tagged


def entities2token(tokenized_sentence, name_token=None, gpe_token=None):
    # Check whether we do any NE replacement. Avoids building the tree in some cases
    if name_token is not None or gpe_token is not None:

        tagged = nltk.pos_tag(tokenized_sentence)
        # Tag the named entities
        ne_tagged = nltk.tree2conlltags(nltk.ne_chunk(tagged))

        # Replace names
        if name_token is not None:
            ne_tagged = [(name_token, tag, ne_tag) if ne_tag.endswith('PERSON')
                         else (token, tag, ne_tag)
                         for (token, tag, ne_tag) in ne_tagged]
        # Replace geopolitical entities
        if gpe_token is not None:
            ne_tagged = [(gpe_token, tag, ne_tag) if ne_tag.endswith('GPE')
                         else (token, tag, ne_tag)
                         for (token, tag, ne_tag) in ne_tagged]

        # Discard the NE tokens
        tagged = [(token, tag) for (token, tag, ne_tag) in ne_tagged]

        # Recollect the tokens
        tokens = [token for (token, tag) in tagged]
    else:
        tokens = tokenized_sentence
    # Convert them to lowercase
    tokens = [token.lower() for token in tokens]
    return tokens


def _check_tokens(number_token=None, name_token=None, gpe_token=None):
    """Check that the tokens are lowercase"""
    assert number_token is None or number_token == number_token.lower(), \
        "Tokens need to be lowercase: %s" % number_token
    assert name_token is None or name_token == name_token.lower(), \
        "Tokens need to be lowercase: %s" % name_token
    assert gpe_token is None or gpe_token == gpe_token.lower(), \
        "Tokens need to be lowercase: %s" % gpe_token


def vector_to_string(token_vector):
    string = ''
    for token in token_vector:
        string += token + ' '
    return string


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
