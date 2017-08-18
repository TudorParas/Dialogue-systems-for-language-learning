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

For processing data from the Cornell Movie-Dialogs Corpus
for use in the end2end model
"""

import os
import codecs

from tqdm import tqdm
import tensorflow as tf

from sklearn.model_selection import train_test_split
from preprocessing.cornell import base
from utils import preprocessing_utils

PROCESSED_DIR_PATH = os.path.join(base.PROCESSED_DIR_PATH, 'end2end')

UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'
EOU = '-eou-'

# For name-entity extraction
NUMBER_TOKEN = '<number>'
NAME_TOKEN = None
# Not using the gpe
GPE_TOKEN = None

VOCAB_SIZE = 10000


def write_dataset(tokenized_conversations):
    """Do a train, test, val split and write the dialogues to file"""
    preprocessing_utils.make_dir(PROCESSED_DIR_PATH)

    train, test = train_test_split(tokenized_conversations, test_size=0.02, random_state=53)
    test, val = train_test_split(test, test_size=0.5, random_state=53)


    filenames = ['train', 'val', 'test']
    paths = [os.path.join(PROCESSED_DIR_PATH, file) for file in filenames]
    # Write the train and the test to file
    for index, data in enumerate([train, val, test]):
        with codecs.getwriter("utf-8")(tf.gfile.GFile(paths[index], "wb+")) as f:
            for conv in tqdm(data, "Processing %s conversations" % filenames[index]):
                for line in conv:
                    # Split the lines using the EOS string
                    f.write("%s %s " % (line
                                        , EOU))
                # Each conversation is written on a separate line
                f.write('\n')


def preprocesss():
    id2line, convos = base.get_lines(), base.get_convos()
    conversations = base.load_conversations()
    tokenized_conversations = preprocessing_utils.tokenize_conversations(conversations,
                                                                         number_token=NUMBER_TOKEN,
                                                                         name_token=NAME_TOKEN,
                                                                         gpe_token=GPE_TOKEN)
    write_dataset(tokenized_conversations)
    preprocessing_utils.create_vocab(src_file=os.path.join(PROCESSED_DIR_PATH, 'train'),
                                     out_dir=PROCESSED_DIR_PATH,
                                     vocab_size=VOCAB_SIZE,
                                     eos=EOS,
                                     sos=SOS,
                                     unk=UNK)


if __name__ == '__main__':
    preprocesss()
