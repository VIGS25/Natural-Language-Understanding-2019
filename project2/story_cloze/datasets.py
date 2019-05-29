"""
Dataset class for handling data loading and manipulations
"""

import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple
import logging

from .embeddings.sentence_encoders import SentenceEncoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/"))


class Dataset:
    train_file = "stories.train.csv"
    train_small_file = "stories.train.small.csv"
    eval_small_file = "stories.eval.small.csv"
    eval_file = "stories.eval.csv"
    test_file = "stories.test.csv"

    def __init__(self,
                 encoder,
                 input_dir: str = DATA_DIR,
                 story_length: int = 4,
                 preprocessors: List = None,
                 add_neg: bool = True,
                 n_random: int = 4,
                 n_backward: int = 2,
                 use_small: bool = False) -> None:

        self.input_dir = input_dir
        self.encoder = encoder
        self.story_length = story_length
        self.add_neg = add_neg
        self.use_small = use_small
        self.n_random = n_random
        self.n_backward = n_backward

        if use_small:
            train_file = self.train_small_file
        else:
            train_file = self.train_file

        self._process_train(train_file)
        self._process_eval()
        self._encode_train()
        self._encode_eval()

    def _process_train(self, train_file):
        """Processes training set and augments it with negative endings."""
        self.train_df = pd.read_csv(os.path.join(self.input_dir, train_file))
        self.train_df.drop(["storyid", "storytitle"], axis=1, inplace=True)
        self.train_cols = ["sentence_{}".format(i) for i in range(1, 5)] + ["ending"]
        self.train_df.columns = self.train_cols

        self.train_sentences = self.train_df[self.train_cols].values
        self.train_labels = np.ones((len(self.train_sentences), 1))
        self.n_train_stories = len(self.train_labels)

        logger.info("Train sentences Shape: ", self.train_sentences.shape)
        logger.info("Train labels shape: ", self.train_labels.shape)

        if self.add_neg:
            logger.info("Adding negative endings.")
            self._add_negative_endings(n_random=self.n_random, n_backward=self.n_backward)

        self.train_data = self.train_df.apply(lambda x: list([x[col] for col in self.train_cols]),axis=1)
        del self.train_df

    def _add_negative_endings(self, n_random: int =0, n_backward: int =0):
        """Adds specified number of backward and random negative endings for each story."""
        if n_random:
            logger.info("Sampling {} random endings per story.".format(n_random))
            random_endings = self.sample_random_endings(n_samples=n_random).reshape(-1, 1)
            train_story_augment = np.array([self.train_sentences[:, :self.story_length]]*n_random)
            train_story_augment = train_story_augment.reshape(-1, 4)

            train_augment = np.hstack([train_story_augment, random_endings])
            assert train_augment.shape[-1] == self.story_length + 1

            train_labels = np.zeros(shape=(train_augment.shape[0], 1))
            self.train_sentences = np.vstack([self.train_sentences, train_augment])
            self.train_labels = np.vstack([self.train_labels, train_labels])
            assert len(self.train_sentences) == len(self.train_labels)

            logger.info("After adding random endings..")
            logger.info("Train sentences shape: {}".format(self.train_sentences.shape))
            logger.info("Train labels shape: {}".format(self.train_labels.shape))

        train_sent_df = pd.DataFrame(self.train_sentences, columns=self.train_cols)
        self.train_df = train_sent_df
        del self.train_sentences

    def _process_eval(self):
        self.eval_df = pd.read_csv(os.path.join(self.input_dir, self.eval_file))
        correct_ending_idxs = self.eval_df["AnswerRightEnding"] - 1
        self.eval_df.drop(["InputStoryid", "AnswerRightEnding"], axis=1, inplace=True)
        eval_cols = ["sentence_{}".format(i) for i in range(1, 5)] + ["ending1", "ending2"]
        self.eval_df.columns = eval_cols

        self.eval_data = self.eval_df.apply(lambda x: list([x[col] for col in eval_cols]),axis=1)
        self.eval_correct_endings = correct_ending_idxs.values

        logger.info("Eval sentences shape: {}".format(self.eval_data.shape))
        logger.info("Eval endings shape: {}".format(self.eval_correct_endings.shape))
        del self.eval_df

        assert len(self.eval_data) == len(self.eval_correct_endings), "All sentences should have endings."

    def sample_random_endings(self, n_samples=1):
        ending_idxs = list()
        for _ in range(n_samples):
            ending_idxs.append(np.random.permutation(self.n_train_stories))
        ending_idxs = np.asarray(ending_idxs)
        return self.train_sentences[ending_idxs, -1]

    def _encode_train(self):
        logger.info("Encoding train sentences...")
        self.train_data = np.array([self.encoder.encode(x) for x in self.train_data])

    def _encode_eval(self):
        logger.info("Encoding eval sentences...")
        self.eval_data = np.array([self.encoder.encode(x) for x in self.eval_data])

    def batch_generator(self, mode="train", batch_size=64, shuffle=True):
        """Generates batches of data for training.

        Parameters
        ----------
        mode: str, default train
            Whether we want to generate batches for train, eval or test dataset
        batch_size: int, default 64
            Batch size used
        shuffle: bool, default True
            Whether to shuffle before generating batches
        """
        if mode == "train":
            data = (self.train_data, self.train_labels)
        elif mode == "eval":
            data = (self.eval_data, self.eval_correct_endings)
        elif mode == "test":
            data = (self.test_data, self.test_correct_endings)

        n_samples = data[0].shape[0]
        if shuffle:
            shuffled = np.random.permutation(n_samples)
        else:
            shuffled = np.arange(n_samples)
        for idx in range(0, n_samples, batch_size):
            yield data[0][shuffled[idx: idx + batch_size]], data[1][shuffled[idx: idx + batch_size]]
