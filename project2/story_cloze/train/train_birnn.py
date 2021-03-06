"""
Train script for RNN model.
"""

import argparse
from datetime import datetime as dt
import os
import numpy as np
import logging

from story_cloze import Dataset, UniversalEncoderDataset
from story_cloze.embeddings import SkipThoughts, UniversalEncoder
from story_cloze.models import BiRNN

SCRATCH_DIR = os.environ["SCRATCH"]
INPUT_DIR = os.path.join(SCRATCH_DIR, "data")
OUT_DIR = os.path.join(SCRATCH_DIR, "outputs")

def main():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--input_dir", default=INPUT_DIR, help="Directory where data is present.")
    parser.add_argument("--log_dir", default=OUT_DIR, help="Where to save Tensorboard-Logs to")
    parser.add_argument("--model_dir", default=OUT_DIR, help="Where to save models to.")
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
    parser.add_argument("--use_attn", action="store_true", help='Whether to use Attention')
    parser.add_argument("--attn_type", default="multiplicative", choices=["multiplicative", "additive"], help="Type of attention used.")

    # Dataset specific
    parser.add_argument("--story_length", type=int, default=4, help="Size of story used.")
    parser.add_argument("--n_random", type=int, default=6, help="Number of random endings generated")

    # Training specific
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate used.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--log_every", type=int, default=100, help="Log stats every.")
    parser.add_argument("--print_every", type=int, default=100, help="Print stats every.")
    parser.add_argument("--eval_every", type=int, default=100, help="Eval every.")
    parser.add_argument("--flush_log", action="store_true", help="Flag to flush logger to file.")

    args = parser.parse_args()
    att = "None" if not args.use_attn else args.attn_type
    exp_name = f'Roemelle_BiRNN_{att}_{args.rnn_type}_{dt.now().strftime("%d-%m-%Y--%H-%M-%S")}'

    log_params = {'level': logging.DEBUG, 'format': "%(asctime)s - [%(levelname)s] %(message)s"}
    if args.flush_log:
        log_params.update({'filename': os.path.join("logs", exp_name), 'filemode': 'a'})

    logging.basicConfig(**log_params)
    logger = logging.getLogger(__name__)


    if args.encoder_type == "skipthoughts":
        embedding_dir = os.path.join(args.input_dir, "embeddings", "skip_thoughts")
        if args.embed_mode == "uni" or args.embed_mode == "bi":
            embedding_dim = 2400
        else:
            embedding_dim = 4800
        encoder = SkipThoughts(embed_dir=embedding_dir, mode=args.embed_mode)
    elif args.encoder_type == "universal":
        embedding_dim = 512
    else:
        raise ValueError("Encoder of type {} is not supported.".format(args.encoder_type))

    model_dir = os.path.join(args.model_dir, exp_name, "checkpoints")
    log_dir = os.path.join(args.log_dir, exp_name, "logs")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger.info("Starting to run the experiment {}".format(exp_name))
    logger.info("Parameters used: {}".format(args))

    if args.encoder_type != "universal":
        dataset = Dataset(encoder=encoder,
                          story_length=args.story_length,
                          input_dir=args.input_dir,
                          n_random=args.n_random)
    else:
        dataset = UniversalEncoderDataset(story_length=args.story_length,
                                          input_dir=args.input_dir,
                                          n_random=args.n_random)

    logger.info("Building the model...")
    model = BiRNN(embedding_dim=embedding_dim,
                rnn_type=args.rnn_type,
                learning_rate=args.learning_rate,
                num_hidden_units=args.num_hidden_units,
                n_story_sentences=args.story_length,
                clip_norm=args.clip_norm,
                model_dir=model_dir, log_dir=log_dir,
                max_checkpoints_to_keep=args.max_checkpoints_to_keep,
                use_attn=args.use_attn,
                attn_type=args.attn_type,
                trainable_zero_state=args.trainable_zero_state,
                restore_from=args.restore_from)

    logger.info("Training the model...")
    model.fit(dataset,
              batch_size=args.batch_size,
              nb_epochs=args.num_epochs,
              log_every=args.log_every,
              print_every=args.print_every,
              eval_every=args.eval_every)

if __name__ == "__main__":
    main()
