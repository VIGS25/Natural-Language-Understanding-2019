import numpy as np
import tensorflow as tf
import pickle
import os
import time
import argparse
from datetime import datetime

from language_model import LanguageModel
from dataset import Dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "results")
SAVE_BASENAME = "group22."
PERP_FILE_BASENAME = SAVE_BASENAME + "perplexity"
CONTINUATION_FILE = "sentences.continuation"

def run():
    """Runs the specified experiment."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir", dest="input_dir", default=DATA_DIR, help="input directory")
    parser.add_argument("--save_dir", dest="save_dir", default=SAVE_DIR, help="Save results to")
    parser.add_argument("--log_dir", dest="log_dir", default=LOG_DIR, help="Directory to save checkpoints to.")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum length of sentence for parsing.")
    parser.add_argument("--exp_type", default="a", help="Which experiment to run")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs used.")
    parser.add_argument("--restore_epoch", type=int, default=10, help="Epoch to restore model from.")
    parser.add_argument("--model_dir", default=None, help="Saved model path.")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise DirectoryNotFoundError("Input directory is missing. Please set it up and download the data.")

    global dataset
    print("Preparing dataset...")
    dataset = Dataset(input_dir=args.input_dir)
    dataset.generate_vocab(max_sen_len=args.max_length, topk=20000, save=True)
    dataset.parse_train(max_sen_length=args.max_length, save=True, reload=True)
    dataset.parse_eval(max_sen_length=args.max_length, save=True, reload=True)
    dataset.parse_test(max_sen_length=args.max_length, save=True, reload=True)

    perp_savefile = PERP_FILE_BASENAME + args.exp_type.upper()

    if args.exp_type == "a":
        model = experiment_A(args)
    elif args.exp_type == "b":
        model = experiment_B(args)
    elif args.exp_type == "c":
        model = experiment_C(args)
    elif args.exp_type == "d":
        experiment_D(args)
        return
    else:
        raise ValueError("Experiment of type {} not found.".format(args.exp_type))

    if args.model_dir is None:
        print("Training model...")
        model.fit(num_epochs=args.num_epochs, eval_every=50, batch_size=64, verbose=True)
    else:
        print("Computing test perplexities...")
        model.save_perplexity_to_file(filename=perp_savefile)

def experiment_A(args):
    exp_name = "Experiment_A_" + datetime.now().strftime("%H-%M-%S")
    exp_logdir = os.path.join(args.log_dir, exp_name)
    exp_savedir = os.path.join(args.save_dir, exp_name)

    restore_from = None
    if args.model_dir is not None:
        restore_from = os.path.join(args.model_dir, str(args.restore_epoch), "model.ckpt")

    # Setup experiment log and save directories
    if not os.path.exists(exp_logdir):
        os.makedirs(exp_logdir)

    if not os.path.exists(exp_savedir):
        os.makedirs(exp_savedir)

    model = LanguageModel(dataset=dataset,
                          lstm_hidden_size=512,
                          embedding_size=100,
                          project=False,
                          pretrained=False,
                          save_dir=exp_savedir,
                          log_dir=exp_logdir,
                          restore_from=restore_from)
    return model

def experiment_B(args):
    exp_name = "Experiment_B_" + datetime.now().strftime("%H-%M-%S")
    exp_logdir = os.path.join(args.log_dir, exp_name)
    exp_savedir = os.path.join(args.save_dir, exp_name)

    restore_from = None
    if args.model_dir is not None:
        restore_from = os.path.join(args.model_dir, str(args.restore_epoch), "model.ckpt")

    # Setup experiment log and save directories
    if not os.path.exists(exp_logdir):
        os.makedirs(exp_logdir)

    if not os.path.exists(exp_savedir):
        os.makedirs(exp_savedir)

    model = LanguageModel(dataset=dataset,
                          lstm_hidden_size=512,
                          embedding_size=100,
                          project=False,
                          pretrained=True,
                          save_dir=exp_savedir,
                          log_dir=exp_logdir,
                          restore_from=restore_from)
    return model

def experiment_C(args):
    exp_name = "Experiment_C_" + datetime.now().strftime("%H-%M-%S")
    exp_logdir = os.path.join(args.log_dir, exp_name)
    exp_savedir = os.path.join(args.save_dir, exp_name)

    restore_from = None
    if args.model_dir is not None:
        restore_from = os.path.join(args.model_dir, str(args.restore_epoch), "model.ckpt")

    # Setup experiment log and save directories
    if not os.path.exists(exp_logdir):
        os.makedirs(exp_logdir)

    if not os.path.exists(exp_savedir):
        os.makedirs(exp_savedir)

    model = LanguageModel(dataset=dataset,
                          lstm_hidden_size=1024,
                          embedding_size=100,
                          project=True,
                          project_size=512,
                          pretrained=True,
                          log_dir=exp_logdir,
                          save_dir=exp_savedir,
                          restore_from=restore_from)
    return model

def experiment_D(args):
    exp_name = "Experiment_D_" + datetime.now().strftime("%H-%M-%S")
    exp_logdir = os.path.join(args.log_dir, exp_name)
    exp_savedir = os.path.join(args.save_dir, exp_name)

    if args.model_dir is None:
        raise ValueError("Experiment D requires a trained model. Please supply the directory.")
    restore_from = os.path.join(args.model_dir, str(args.restore_epoch), "model.ckpt")

    data_filename = os.path.join(args.input_dir, CONTINUATION_FILE)
    sol_filename = SAVE_BASENAME + "continuation"

    model = LanguageModel(dataset=dataset,
                          lstm_hidden_size=1024,
                          embedding_size=100,
                          project=True,
                          project_size=512,
                          pretrained=True,
                          log_dir=exp_logdir,
                          save_dir=None,
                          restore_from=restore_from)

    model.complete_sentences(data_filename, sol_filename, max_len=20, log_every=1000)

if __name__ == "__main__":
    run()
