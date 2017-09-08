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
for use in the hierarchical model
"""

import os
import sys

sys.path.append(os.getcwd())

from preprocessing import simple
import preprocessing.cornell.base as base
from utils import preprocessing_utils

PROCESSED_DIR_PATH = os.path.join(base.PROCESSED_DIR_PATH, 'hier')

UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
EOU =  '-EOU- '

# For name-entity extraction
NUMBER_TOKEN = '<number>'
NAME_TOKEN = '<person>'
# Not using the gpe
GPE_TOKEN = None

VOCAB_SIZE = 10000

def dialogue_response(conversations, max_conv_length=4):
    """ Divide the dataset into two sets: dialogue so far and answers. """
    previous_dialogues, answers = [], []
    for convo in conversations:
        dialogue_so_far = []
        for index in range(len(convo) - 1):
            dialogue_so_far .append("%s%s" % (convo[index], EOU))
            # Trim it to max_conv_length
            dialogue_so_far = dialogue_so_far[-max_conv_length:]
            # Remove the EOU from the last utteranceg
            previous_dialogues.append("".join(dialogue_so_far[:-1] + [dialogue_so_far[-1].split(EOU)[0]]))
            answers.append(convo[index + 1])


    assert len(previous_dialogues) == len(answers)
    return previous_dialogues, answers

def prepare_raw_data():
    print('Preparing raw data into train set and test set ...')

    conversations = base.load_conversations()
    tokenized_conv = preprocessing_utils.tokenize_conversations(conversations, number_token=NUMBER_TOKEN,
                                                 name_token=NAME_TOKEN, gpe_token=GPE_TOKEN)
    previous_dialogues, answers = dialogue_response(tokenized_conv)
    src_file = simple.prepare_dataset(previous_dialogues, answers, PROCESSED_DIR_PATH)
    preprocessing_utils.create_vocab(src_file,out_dir=PROCESSED_DIR_PATH,
                                     vocab_size=VOCAB_SIZE, eos=EOS,
                                     sos=SOS, unk=UNK)

if __name__ == '__main__':
    prepare_raw_data()