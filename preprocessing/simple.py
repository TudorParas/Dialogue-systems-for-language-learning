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

Base code for preprocessing data into a question-answer format to feed into a basic seq2seq model
"""
import codecs
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from utils import preprocessing_utils as base
def question_answers(conversations):
    """ Divide the dataset into two sets: questions and answers. """
    questions, answers = [], []
    for convo in conversations:
        for index in range(len(convo) - 1):
            questions.append(convo[index])
            answers.append(convo[index + 1])
    assert len(questions) == len(answers)
    return questions, answers



def prepare_dataset(questions, answers, processed_dir_path, val_test_fraction=0.02):
    # create path to store all the train & validation & test encoder & decoder
    base.make_dir(processed_dir_path)

    X_train, X_test, y_train, y_test = train_test_split(questions, answers,
                                                        test_size=val_test_fraction, random_state=53)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=53)

    filenames = ['train.enc', 'val.enc', 'test.enc', 'train.dec', 'val.dec', 'test.dec']
    # filenames = ['train.vi', 'tst2012.vi', 'tst2013.vi', 'train.en', 'tst2012.en', 'tst2013.en']
    paths = [os.path.join(processed_dir_path, file) for file in filenames]

    for index, data in enumerate([X_train, X_val, X_test, y_train, y_val, y_test]):
        print('Writing item  %s' % filenames[index])
        with codecs.getwriter("utf-8")(tf.gfile.GFile(paths[index], "wb+")) as f:
            for line in data:
                f.write("%s\n" % line)
    # File from which we'll compute the vocab.
    return paths[0]