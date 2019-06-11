import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime as dt
import pandas as pd
import os
import logging

from story_cloze import Dataset
from story_cloze.embeddings import SkipThoughts, UniversalEncoder
from story_cloze.models import BiRNN, RNN, FFN

DEFAULT_INPUT_DIR = os.path.join(os.environ.get("SCRATCH", "./"), "data")
DEFAULT_LOG_DIR = "./local/logs"
DEFAULT_MODEL_DIR = "./local/checkpoints"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_DICT = {"RNN": RNN, "BiRNN": BiRNN, "FFN": FFN}
ENCODERS = {"SkipThoughts": SkipThoughts, "UniversalEncoder": UniversalEncoder}

def transform(value):
    if value.isnumeric():
        value = int(value)
    elif "." in value:
        value = float(value)
    elif value in ["False", "True", "None"]:
        if value == "False":
            value = False
        elif value == "True":
            value = True
        else:
            return None
    elif "'" in value:
        value = value[1:-1]
    return value

def parse_args_from_lsf(lsf_file):
    with open(lsf_file, "r") as f:
        lines = f.readlines()

    args_line = None
    for line in lines:
        if "Namespace" in line:
            args_line = line
            break

    args_line = args_line.split(":")[-1].strip()
    args_line = args_line.split("(")[-1]
    args_line = args_line.split(")")[0]

    split_args = args_line.split(",")
    cleaned_args = list()

    for arg in split_args:
        cleaned_args.append(arg.strip())

    parsed = dict()
    for arg in cleaned_args:
        key, value = arg.split("=")
        key = transform(key)
        if "_dir" in key:
            continue
        value = transform(value)

        parsed[key] = value

    return parsed


def main():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Directory where data is present.")
    parser.add_argument("--test_file", default="stories.spring2016.csv", help="File to test on")
    parser.add_argument("--model_name", help="Name of the model to restore.", choices=["BiRNN", "RNN", "FFN"])
    parser.add_argument("--restore_from", help="Where to restore pretrained model from.")
    parser.add_argument("--mode", help="Whether to evaluate or predict")

    args = parser.parse_args()

    files_in_dir = os.listdir(args.restore_from)
    lsf_file = None
    for file in files_in_dir:
        if "lsf.o" in file:
            lsf_file = file
            break

    lsf_file = os.path.join(args.restore_from, lsf_file)

    parsed = parse_args_from_lsf(lsf_file)
    model_type = MODEL_DICT[args.model_name]

    if parsed["encoder_type"] == "skipthoughts" and (parsed["embed_mode"] == "bi" or parsed["embed_mode"] == "uni"):
        parsed["embedding_dim"] = 2400
    elif parsed["encoder_type"] == "skipthoughts" and parsed["embed_mode"] == "both":
        parsed["embedding_dim"] = 4800
    elif parsed["encoder_type"] == "universal":
        parsed["embedding_dim"] = 512

    ENCODER_NAMES = {"skipthoughts": "SkipThoughts", "universal": "UniversalEncoder"}
    encoder_name = ENCODER_NAMES[parsed["encoder_type"]]

    restore_from = os.path.join(args.restore_from, "checkpoints", "model.ckpt")

    if args.model_name in ["BiRNN", "RNN"]:
        model = model_type(embedding_dim=parsed["embedding_dim"],
                    rnn_type=parsed["rnn_type"],
                    learning_rate=parsed["learning_rate"],
                    num_hidden_units=parsed["num_hidden_units"],
                    n_story_sentences=parsed["story_length"],
                    clip_norm=parsed["clip_norm"],
                    model_dir=DEFAULT_MODEL_DIR, log_dir=DEFAULT_LOG_DIR,
                    max_checkpoints_to_keep=parsed["max_checkpoints_to_keep"],
                    use_attn=parsed["use_attn"],
                    attn_type=parsed["attn_type"],
                    trainable_zero_state=parsed["trainable_zero_state"],
                    restore_from=restore_from)
    else:
        model = FFN(embedding_dim=parsed["embedding_dim"],
                    rnn_type=parsed["rnn_type"],
                    learning_rate=parsed["learning_rate"],
                    num_hidden_units=parsed["embedding_dim"],
                    n_story_sentences=parsed["story_length"],
                    clip_norm=parsed["clip_norm"],
                    model_dir=DEFAULT_MODEL_DIR, log_dir=DEFAULT_LOG_DIR,
                    max_checkpoints_to_keep=parsed["max_checkpoints_to_keep"],
                    trainable_zero_state=parsed["trainable_zero_state"],
                    restore_from=restore_from)

    test_file = os.path.join(args.input_dir, args.test_file)
    test_df = pd.read_csv(test_file)

    if args.mode == "evaluate":
        correct_ending_idxs = test_df["AnswerRightEnding"] - 1
        test_correct_endings = correct_ending_idxs.values

        if encoder_name == "SkipThoughts":
            test_path = os.path.join(args.input_dir, "eval_embeddings_" + encoder_name + "_" + parsed["embed_mode"] + ".npy")
        else:
            test_path = os.path.join(args.input_dir, "eval_embeddings_" + encoder_name + ".npy")

        logger.info("Loading embeddings from: {}".format(test_path))
        test_data = np.load(test_path)
        logger.info("Test data loaded. Shape: {}".format(test_data.shape))

        labels = list()
        preds = list()

        for idx in range(test_data.shape[0]):
            test_sentences = test_data[idx]
            test_sentences = np.expand_dims(test_sentences, axis=0)
            test_labels = test_correct_endings[idx]
            labels.append(test_labels)

            results = model._evaluate_batch(test_sentences)
            predictions, _, _ = results
            preds.append(predictions)

        labels = np.squeeze(labels)
        preds = np.squeeze(preds)

        eval_accuracy = np.mean(labels == preds)
        print("Accuracy on given test set: {}".format(eval_accuracy))

    elif args.mode == "predict":
        preds = list()
        if encoder_name == "SkipThoughts":
            test_path = os.path.join(args.input_dir, "test_embeddings_" + encoder_name + "_" + parsed["embed_mode"] + ".npy")
        else:
            test_path = os.path.join(args.input_dir, "test_embeddings_" + encoder_name + ".npy")

        logger.info("Loading embeddings from: {}".format(test_path))
        test_data = np.load(test_path)
        logger.info("Test data loaded. Shape: {}".format(test_data.shape))

        for idx in range(test_data.shape[0]):
            test_sentences = test_data[idx]
            test_sentences = np.expand_dims(test_sentences, axis=0)
            results = model._evaluate_batch(test_sentences)
            predictions, _, _ = results
            preds.append(predictions + 1)

        preds = np.squeeze(preds)
        save_file = os.path.join(args.restore_from, "predictions.csv")
        pd.DataFrame(preds).to_csv(save_file, index=False, header=False)
        logger.info("Saved predictions to {}".format(save_file))

if __name__ == "__main__":
    main()
