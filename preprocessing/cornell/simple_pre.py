"""
A dialogue system meant to be used for language learning.

This is based on Google Neural Machine Tranlation model
https://github.com/tensorflow/simple
which is based on Thang Luong's thesis on
Neural Machine Translation: https://github.com/lmthang/thesis

And on the paper Building End-To-End Dialogue Systems
Using Generative Hierarchical Neural Network Models:
https://arxiv.org/pdf/1507.04808.pdf

Created by Tudor Paraschivescu for the Cambridge UROP project
"Dialogue systems for language learning"

For processing data from the Cornell Movie-Dialogs Corpus
for use in the simple model
"""

import os

from preprocessing import simple
import preprocessing.cornell.base as base
from utils import preprocessing_utils

PROCESSED_DIR_PATH = os.path.join(base.PROCESSED_DIR_PATH, 'simple')

UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'

# For name-entity extraction
NUMBER_TOKEN = '<number>'
NAME_TOKEN = '<person>'
# Not using the gpe
GPE_TOKEN = None

VOCAB_SIZE = 10000

def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')
    id2line = base.get_lines()
    convos = base.get_convos()
    conversations = base.build_conv(id2line, convos)
    tokenized_conv = preprocessing_utils.tokenize_conversations(conversations, number_token=NUMBER_TOKEN,
                                                 name_token=NAME_TOKEN, gpe_token=GPE_TOKEN)
    questions, answers = simple.question_answers(tokenized_conv)
    src_file = simple.prepare_dataset(questions, answers, PROCESSED_DIR_PATH)
    preprocessing_utils.create_vocab(src_file,out_dir=PROCESSED_DIR_PATH,
                                     vocab_size=VOCAB_SIZE, eos=EOS,
                                     sos=SOS, unk=UNK)


if __name__ == '__main__':
    prepare_raw_data()
