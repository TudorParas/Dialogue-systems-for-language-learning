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
Base methods for preprocessing the Cornell Movie-Dialogs dataset into conversations.

Code adapted from
https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/assignments/chatbot/data.py.
"""

import os


DATA_PATH = os.path.abspath('../../data/cornell')
LINE_FILE = "movie_lines.txt"
PROCESSED_DIR_PATH = os.path.join(DATA_PATH, 'processed')
CONVO_FILE = 'movie_conversations.txt'


def get_lines():
    id2line = {}
    file_path = os.path.join(DATA_PATH, LINE_FILE)
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                if parts[4][-1] == '\n':
                    parts[4] = parts[4][:-1]
                id2line[parts[0]] = parts[4]
    return id2line


def get_convos():
    """ Get conversations from the raw data """
    file_path = os.path.join(DATA_PATH, CONVO_FILE)
    convos = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 4:
                convo = []
                for line in parts[3][1:-2].split(', '):
                    convo.append(line[1:-1])
                convos.append(convo)

    return convos


def build_conv(id2line, convos):
    """Create a matrix of conversations, each row representing a vector of lines"""
    conversations = []
    for convo in convos:
        current_conv = []
        for index, line in enumerate(convo):
            current_conv.append(id2line[convo[index]])
        conversations.append(current_conv)
    return conversations

def load_conversations():
    return build_conv(get_lines(), get_convos())

