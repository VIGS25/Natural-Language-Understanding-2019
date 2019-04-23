from model import LanguageModel, Dataset
import numpy as np
import tensorflow as tf
import pickle
import os
import sys
import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

def run():
    """Runs the specified experiment."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir", dest="input_dir", default=DATA_DIR, help="input directory")
    parser.add_argument("--save_dir", dest="save_dir", default=DATA_DIR, help="Save results to")
    parser.add_argument("--log_dir", dest="logdir", default=LOG_DIR, help="Directory to save checkpoints to.")
    parser.add_argument("--exp", default="a", help="Which experiment to run")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size used.")
    parser.add_argument("-l", "--max_length", type=int, default=30, help="Maximum length of sentence to use.")
    parser.add_argument("--lstm_size", type=int, default=512, "Number of hidden units in LSTM.")
    parser.add_argument("--projection_size", type=int, default=512, help="Projection size")
    parser.add_argument("--learning_rate", dest="lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs used.")
    parser.add_argument("--max_train_time", type=int, default=4 * (3600), help="Max training time in seconds.")
    parser.add_argument("--embedding_size", type=int, default=100, help="Size of embeddings used.")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise DirectoryNotFoundError("Input directory is missing. Please set it up and download the data.")

    if args.exp_name == "a":
        experiment_A(args)
    elif args.exp_name == "b":
        experiment_B(args)
    elif args.exp_name == "c":
        experiment_C(args)

def experiment_A(args):
    """Runs experiment A."""
    dataset = Dataset(input_dir=args.input_dir)
    dataset.generate_vocab(max_sen_len=args.max_length, topk=20000, save=True)
    dataset.parse_sentences(mode="train", save=True, verbose=True)
    dataset.parse_sentences(mode="eval", save=True)
    dataset.parse_sentences(mode="test")

    # Create session object inside the Language Model tomorrow.
    model = LanguageModel(dataset=dataset, lstm_hidden_size=args.lstm_size,
                          embedding_size=args.embedding_size, project=False,
                          pretrained=True, lr=args.lr)
    model.fit() # Add option for summary
    model.predict()

    # Do the perplexity calculations per sentence
    perpl_file = open("group04.perplexityA", "w")
    for i in range(len(test_batches)):
        perplexities = sess.run(model.perplexity,
                                feed_dict={model.sentence_placeholder: test_batches[i]})
        if i + 1 == len(test_batches):  # last batch case, reduce result
            to_take = test_data.shape[0] % batch_size
            perplexities = perplexities[:to_take]
        # write to file
        for p in perplexities:
            perpl_file.write(str(p) + "\n")
    perpl_file.close()

def experiment_B(args):
    """Runs experiment B."""
    dataset = Dataset(input_dir=args.input_dir)

    with tf.Session() as sess:
        model = LanguageModel(num_hidden_units=args.lstm_size, time_steps=29,
                              len_corpus=len(words), batch_size=args.batch_size,
                              vocab=words, session=sess, lr=args.lr,
                              embed_path="..", #TODO:
                              embedding_length=args.embedding_length)
        model.train() # Add option for summary
        model.predict()

def experiment_C(args):
    """Runs experiment C."""
    dataset = Dataset(input_dir=args.input_dir)
    with tf.Session() as sess:
        model = LanguageModel(num_hidden_units=args.lstm_size, time_steps=29,
                              len_corpus=len(words), batch_size=args.batch_size,
                              add_projection=True, projection_size=args.projection_size,
                              vocab=words, session=sess, lr=args.lr,
                              embedding_length=args.embedding_length)
        model.train() # Add option for summary
        model.predict()

if __name__ == "__main__":
    run()
