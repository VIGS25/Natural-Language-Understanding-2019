"""
Train script for RNN model.
"""

import argparse
from datetime import datetime as dt
import os
import numpy as np
import logging

from story_cloze import Dataset
from story_cloze.embeddings import SkipThoughts, UniversalEncoder
from story_cloze.models import RNN

DEFAULT_INPUT_DIR = os.path.join(os.environ["SCRATCH"], "data")
DEFAULT_LOG_DIR = "./logs"
DEFAULT_MODEL_DIR = "./checkpoints"

def main():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Directory where data is present.")
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR, help="Where to save Tensorboard-Logs to")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="Where to save models to.")
    parser.add_argument("--restore_from", default=None, help="Where to restore pretrained model from.")
    parser.add_argument("--max_checkpoints_to_keep", default=5, type=int, help="How many checkpoints to keep.")

    # Setup specific
    parser.add_argument("--batch_size", default=64, type=int, help="Batch Size used.")
    parser.add_argument("--rnn_type", default="gru", help="Type of RNN used.")
    parser.add_argument("--num_hidden_units", default=1000, type=int, help="Number of hidden units in RNN Cell")
    parser.add_argument("--encoder_type", default="skipthoughts", choices=["skipthoughts", "universal"], help="Encoder type")
    parser.add_argument("--embed_mode", default="bi", choices=["bi", "uni", "both"], help="Embeddings to use for SkipThoughts")
    parser.add_argument("--clip_norm", type=float, default=10.0, help="Gradient clipping norm")
    parser.add_argument("--trainable_zero_state", action="store_true", help="Whether to train zero state.")

    # Dataset specific
    parser.add_argument("--story_length", type=int, default=4, help="Size of story used.")
    parser.add_argument("--n_random", type=int, default=1, help="Number of random endings generated")
    parser.add_argument("--n_backward", type=int, default=1, help="Number of backward_endings.")

    # Training specific
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate used.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--log_every", type=int, default=100, help="Log stats every.")
    parser.add_argument("--print_every", type=int, default=100, help="Print stats every.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    if args.encoder_type == "skipthoughts":
        embedding_dir = os.path.join(args.input_dir, "embeddings", "skip_thoughts")
        if args.embed_mode == "uni" or args.embed_mode == "bi":
            embedding_dim = 2400
        else:
            embedding_dim = 4800
        encoder = SkipThoughts(embed_dir=embedding_dir, mode=args.embed_mode)
    elif args.encoder_type == "universal":
        encoder = UniversalEncoder()
        embedding_dim = 512
    else:
        raise ValueError("Encoder of type {} is not supported.".format(args.encoder_type))

    exp_name = "Roemelle_RNN_" + dt.now().strftime("%d-%m-%Y--%H-%M-%S")
    model_dir = os.path.join(args.model_dir, exp_name)
    log_dir = os.path.join(args.log_dir, exp_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger.info("Starting to run the experiment {}".format(exp_name))
    logger.info("Parameters used: ")
    dataset = Dataset(story_length=args.story_length,
                      input_dir=args.input_dir,
                      n_random=args.n_random,
                      n_backward=args.n_backward)

    logger.info("Building the model...")
    model = RNN(encoder=encoder,
                embedding_dim=embedding_dim,
                rnn_type=args.rnn_type,
                learning_rate=args.learning_rate,
                num_hidden_units=args.num_hidden_units,
                n_story_sentences=args.story_length,
                clip_norm=args.clip_norm,
                model_dir=model_dir, log_dir=log_dir,
                max_checkpoints_to_keep=args.max_checkpoints_to_keep,
                trainable_zero_state=args.trainable_zero_state,
                restore_from=args.restore_from)

    logger.info("Training the model...")
    model.fit(dataset, nb_epochs=args.num_epochs, display_eval=True)

if __name__ == "__main__":
    main()
