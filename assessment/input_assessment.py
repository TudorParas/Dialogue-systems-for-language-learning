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

To assess the user's utterance and pass it to the chatbot."""

import requests
import random
import time
from json import encoder
def get_user_input(hparams=None):
    """Get the user's input, assess it and return it to the bot."""
    utterance = input()
    # End the chat
    if utterance == 'end()':
        return utterance, utterance
    # Authorization not present
    if not hparams or not hparams.UNAME or not hparams.TOKEN:
        return utterance, utterance

    results = get_results(hparams, utterance)
    textual_errors = results['textual_errors']

    correct_sent = correct_utterance(textual_errors, utterance)
    if not correct_sent == utterance:
        print_result(results)
        print("The correct form is: ", correct_sent)
        print()
    # Return the correct sentence and the original utterance
    return correct_sent, utterance

def overall_score(user_utterances, hparams):
    if not hparams or not hparams.UNAME or not hparams.TOKEN:
        return
    aggregated_utterances = " ".join(user_utterances)
    print(aggregated_utterances)
    # Returns an overall score for the chat
    results = get_results(hparams, aggregated_utterances)
    print('\n\nOverall Score: ', results['overall_score'])


def correct_utterance(textual_errors, utterance):
    correct_sent = utterance
    # While we correct the characters get shifter as we add new characters.
    #  We account for that with this by keeping track of the change
    shift = 0
    for error in textual_errors:
        error_begin_index = error[0] + shift
        error_end_index = error[1] + shift
        correct_word = error[2]

        incorrect_word_len = error_end_index - error_begin_index
        shift = shift + (len(correct_word) - incorrect_word_len)

        correct_sent = correct_sent[:error_begin_index] + correct_word + correct_sent[error_end_index:]
    return correct_sent

def print_result(results):
    for key in results:
        print('\t\t\t\t\t', end='')
        print(key, ': ', end='')
        print(results[key])

def get_results(hparams, text):
    # Generate an id for the utterance
    UID = random.randint(0, 10000000)

    headers = {
        'Authorization': 'Token token=%s' % hparams.TOKEN,
        'Content-Type': 'application/json',
    }

    data = {"author_id": "tp423", "task_id": "UROPintern", "question_text": "chatbot", "text": text, "test": 1}

    link = 'https://api-staging.englishlanguageitutoring.com/v2.0.0/account/%s/text/%s' % (
        hparams.UNAME, UID)
    put_req = requests.put(link, headers=headers, json=data)

    # Get response
    headers = {
        'Authorization': 'Token token=%s' % hparams.TOKEN,
    }
    results_link = link + '/results'
    # Loop until we get an answer
    while True:
        # Wait 5 seconds
        time.sleep(1)

        request = requests.get(results_link, headers=headers)
        results = request.json()
        if not results['type'] == 'results_not_ready':
            break
    return results

