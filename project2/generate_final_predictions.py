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

model1 = BiRNN(embedding_dim=4800,
            rnn_type="gru",
            learning_rate=0.001,
            num_hidden_units=1000,
            n_story_sentences=4,
            clip_norm=10.0,
            model_dir=DEFAULT_MODEL_DIR,
            log_dir=DEFAULT_LOG_DIR,
            max_checkpoints_to_keep=10,
            use_attn=True,
            attn_type="additive",
            trainable_zero_state=False,
            restore_from="./Best-Models/Roemmele-BiGRU-Add-Attn/model.ckpt")

model2 = BiRNN(embedding_dim=4800,
            rnn_type="gru",
            learning_rate=0.001,
            num_hidden_units=1000,
            n_story_sentences=4,
            clip_norm=10.0,
            model_dir=DEFAULT_MODEL_DIR,
            log_dir=DEFAULT_LOG_DIR,
            max_checkpoints_to_keep=10,
            use_attn=True,
            attn_type="additive",
            trainable_zero_state=False,
            restore_from="./Best-Models/Roemmele-BiGRU/model.ckpt")

model3 = FFN(embedding_dim=4800,
            rnn_type="gru",
            learning_rate=0.001,
            num_hidden_units=4800,
            n_story_sentences=4,
            clip_norm=5.0,
            model_dir=DEFAULT_MODEL_DIR,
            log_dir=DEFAULT_LOG_DIR,
            max_checkpoints_to_keep=10,
            trainable_zero_state=False,
            restore_from="./Best-Models/FFN-Last-Sentence/model.ckpt")

test_file = os.path.join(args.input_dir, args.test_file)
test_df = pd.read_csv(test_file)

preds1 = list()
preds2 = list()
preds3 = list()
test_path = os.path.join(DEFAULT_INPUT_DIR, "test_embeddings_SkipThoughts_both.npy")

logger.info("Loading embeddings from: {}".format(test_path))
test_data = np.load(test_path)
logger.info("Test data loaded. Shape: {}".format(test_data.shape))

for idx in range(test_data.shape[0]):
    test_sentences = test_data[idx]
    test_sentences = np.expand_dims(test_sentences, axis=0)
    results1 = model1._evaluate_batch(test_sentences)
    results2 = model2._evaluate_batch(test_sentences)
    results3 = model3._evaluate_batch(test_sentences)

    predictions1, _, _ = results1
    predictions2, _, _ = results2
    predictions3, _, _ = results3

    preds1.append(predictions1 + 1)
    preds2.append(predictions2 + 1)
    preds3.append(predictions3 + 1)

preds1 = pd.DataFrame(np.squeeze(preds1))
preds2 = pd.DataFrame(np.squeeze(preds2))
preds3 = pd.DataFrame(np.squeeze(preds3))

preds = pd.concat([preds1, preds2, preds3], axis=1).values

final_preds = list()
for predictions in preds:
    unique_vals, counts = np.unique(predictions, return_counts=True)
    final_preds.append(unique_vals[np.argmax(counts)])

final_preds = np.squeeze(final_preds)
pd.DataFrame(final_preds).to_csv("final_predictions.csv", index=False, header=False)
