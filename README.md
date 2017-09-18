#                          Dialogue systems for language learning
            
What is it?
-----------
  
A dialogue system meant to be used for language learning.

Based on:

1) [Google Neural Machine Tranlation model](https://github.com/tensorflow/nmt) which is based on [Thang Luong's thesis on Neural Machine Translation](https://github.com/lmthang/thesis)

2)  [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models](https://arxiv.org/pdf/1507.04808.pdf)

Created by Tudor Paraschivescu for the Cambridge UROP project "Dialogue systems for language learning".


Dependencies
-------
Overall: tensorflow >= 1.2.1, numpy, nltk
Preprocessing: scikit-learn (for train-test split), tqdm (for checking progress)

Tensorflow can be installed by following the [tensorflow installation instructions](https://www.tensorflow.org/install/). Note that a virtualenv installation is recommended, with pip install, and that you need pip version >=8.1.


Data
-------

For training a model download the [Cornell Movie-Dialogs Corpus](http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip)

Change to the `Dialogue-systems-for-language-learning` (root) directory, create a 'data/cornell' directory path and unzip the 'cornell movie-dialogs corpus' folder from the zip file into it.

Make sure you're in the root directory again and run the script 'simple_pre.py' located in 'preprocessing/cornell'. This will take care of the preprocessing. 


Training
--------
  
To begin training a simple nmt model run 'chatbot/run.py' using the arguments:
  
        --src=enc --tgt=dec \
        --vocab_file="<repo-path>\data\cornell\processed\nmt\vocab"  \
        --train_prefix="<repo-path>\data\cornell\processed\nmt\train" \
        --dev_prefix="<repo-path>\data\cornell\processed\nmt\val"  \
        --test_prefix="<repo-path>\data\cornell\processed\nmt\test" \
        --out_dir="<repo-path>\output\cornell" \
        --num_train_steps=12000 \
        --steps_per_stats=100 \
        --num_layers=2 \
        --num_units=128 \
        --dropout=0.2 \
        --metrics=bleu
        
To begin training a hierarchical model run 'chatbot/run.py' using the arguments:

    --src=enc --tgt=dec \
    --vocab_file="<repo-path>\data\cornell\processed\nmt\vocab"  \
    --train_prefix="<repo-path>\data\cornell\processed\nmt\train" \
    --dev_prefix="<repo-path>\data\cornell\processed\nmt\val"  \
    --test_prefix="<repo-path>\data\cornell\processed\nmt\test" \
    --out_dir="<repo-path>\output\cornell" \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu \
    --architecture=hier \
    --context_num_layers=2
        

        
This will run the training for 12000 iterations. The hyperparameters used are the standard ones from the NMT guide. The following hyperparameters can be tweaked to change the model:
    
    num_train_steps: Overall training steps executed before stopping.
    num_units: Number of units of the hidden layer of the encoder and decoder RNNs
    num_layers: Number of layers used by the RNNs
    encoder_type: uni | bi. Default is uni. Chooses whether the encoder is unidirectional or bidirectional
    residual: Whether to add residual connections
    optimizer: sgd | adam. Choose the optimizer used for training.
    learning_rate: Default is 1.0. Should change to between 0.001 to 0.0001 if using adam.
    start_decay_step, decay_steps, decay_factor: hyperparameters which affect the learning rate decay.
    unit_type: lstm | gru | layer_norm_lstm. Type of the RNN cell used.
    forget_bias: Forget bias for BasicLSTMCell.
    src_reverse: Whether to reverse the source utterance.
    num_buckets: Number of bucket in which we put data of similar length.
    num_gpus: Number of GPUs of the machine. Default is 1.
    metrics: Comma-separated list of evaluations. Can be bleu,rouge,accuracy.
    context_num_layer: The number of layers of the context encoder.
    
For more information all the arguments are parsed in the 'chatbot/argument_parser.py' file.

Chatting
-------
To chat with your model run it with the arguments:

    --chat=True \
    --chat_logs_output_file="<repo-path>\output\cornell\chat_logs.txt" \
    --out_dir="<repo-path>\Chatbot\output\cornell" \
    --architecture=hier \
    --beam_width=5 \
    --top_responses=3 \
    --number_token=<number> \
    --name_token=<person> \

The hyperparameters which can be tweaked for a different experience are:

    chat_logs: The output file where your chat will be recorded. If none is provided then there will be no record of the chat.
    out_dir: should always point to the output directory from the training stage.
    beam_width: number of nodes expanded in the beam-search.
    top_responses: sample over this number of responses. Useful when also using beam_width.
    number_token: the token used for replacing numbers. Used for posprocessing, which can be changed by tweaking the          'postprocess_output' method in 'utils/chatbot_utils'.
    name_token: similar to number_token, but for names.


Issues
-------

1) InvalidArgumentError: Multiple OpKernel registrations match NodeDef: tensorflow bug, see [this issue](https://github.com/tensorflow/tensorflow/issues/11277). It is a bug in tensorflow, if it persists deactivate beam-search by setting beam_width=0
2) Import errors: all commands should be run from the home directory.