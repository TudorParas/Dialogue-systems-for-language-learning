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

Base code for preprocessing data into a question-answer format to feed into a hierarchical seq2seq model
"""

def dialogue_response(conversations, EOU, max_conv_length=4):
    """ Divide the dataset into two sets: dialogue so far and answers. """
    previous_dialogues, answers = [], []
    for convo in conversations:
        dialogue_so_far = []
        if not valid_conversation(convo):
            continue
        # Check if the conversation has blank utterances
        for index in range(len(convo) - 1):
            dialogue_so_far .append("%s%s" % (convo[index], EOU))
            # Trim it to max_conv_length
            dialogue_so_far = dialogue_so_far[-max_conv_length:]
            # Remove the EOU from the last utterance
            dialogue_so_far_trimmed = "".join(dialogue_so_far[:-1] + [dialogue_so_far[-1].split(EOU)[0]])
            previous_dialogues.append(dialogue_so_far_trimmed)
            answers.append(convo[index + 1])

    return previous_dialogues, answers

def valid_conversation(conv):
    for line in conv:
        if line == '':
            return False
    return True
