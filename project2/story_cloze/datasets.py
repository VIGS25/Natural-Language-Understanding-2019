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
                 encoder: SentenceEncoder,
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

        self.read_train_and_eval()
        self._process_train()
        self._process_eval()
        self._encode_train()
        self._encode_eval()

    def read_train_and_eval(self) -> None:
        """Reads the train and eval stories."""
        logger.info("Reading train and eval stories from {}".format(
            self.input_dir))
        if self.use_small:
            self.train_df = pd.read_csv(os.path.join(self.input_dir, self.train_small_file))
            self.eval_df = pd.read_csv(os.path.join(self.input_dir, self.eval_small_file))
        else:
            self.train_df = pd.read_csv(os.path.join(self.input_dir, self.train_file))
            self.eval_df = pd.read_csv(os.path.join(self.input_dir, self.eval_file))

    def _process_train(self):
        """Processes training set and augments it with negative endings."""
        self.train_df.drop(["storyid", "storytitle"], axis=1, inplace=True)
        train_cols = ["sentence_{}".format(i) for i in range(1, 5)] + ["ending"]
        self.train_df.columns = train_cols

        self.train_stories = self.train_df[train_cols[:-1]].values
        self.train_endings = self.train_df[train_cols[-1]].values

        self.train_sentences = self.train_df[train_cols].values
        self.train_labels = np.ones((len(self.train_sentences), 1))
        self.n_train_stories = len(self.train_labels)

        print("Train sentences Shape: ", self.train_sentences.shape)
        print("Train labels shape: ", self.train_labels.shape)

        del self.train_df

        if self.add_neg:
            print("Adding negative endings.")
            self._add_negative_endings(n_random=self.n_random, n_backward=self.n_backward)

    def _add_negative_endings(self, n_random: int =0, n_backward: int =0):
        """Adds specified number of backward and random negative endings for each story."""
        if n_random:
            print("Sampling {} random endings per story.".format(n_random))
            random_endings = self.sample_random_endings(n_samples=n_random).reshape(-1, 1)
            train_story_augment = np.array([self.train_stories]*n_random)
            train_story_augment = train_story_augment.reshape(-1, 4)

            train_augment = np.hstack([train_story_augment, random_endings])
            assert train_augment.shape[-1] == self.story_length + 1

            train_labels = np.zeros(shape=(train_augment.shape[0], 1))
            self.train_sentences = np.vstack([self.train_sentences, train_augment])
            self.train_labels = np.vstack([self.train_labels, train_labels])
            assert len(self.train_sentences) == len(self.train_labels)

            print("After adding random endings..")
            print("Train sentences shape: ", self.train_sentences.shape)
            print("Train labels shape: ", self.train_labels.shape)

        if n_backward:
            print("Sampling {} backward endings per story.".format(n_backward))
            backward_endings = self.sample_backward_endings(n_samples=n_backward)
            train_story_augment = np.array([self.train_stories]*n_backward)
            train_story_augment = train_story_augment.reshape(-1, 4)

            train_augment = np.hstack([train_story_augment, backward_endings])
            assert train_augment.shape[-1] == self.story_length + 1

            train_labels = np.zeros(shape=(train_augment.shape[0], 1))
            self.train_sentences = np.vstack([self.train_sentences, train_augment])
            self.train_labels = np.vstack([self.train_labels, train_labels])
            assert len(self.train_sentences) == len(self.train_labels)

            print("After adding backward endings..")
            print("Train sentences shape: ", self.train_sentences.shape)
            print("Train labels shape: ", self.train_labels.shape)

    def _process_eval(self):
        correct_ending_idxs = self.eval_df["AnswerRightEnding"] - 1
        self.eval_df.drop(["InputStoryid", "AnswerRightEnding"], axis=1, inplace=True)
        eval_cols = ["sentence_{}".format(i) for i in range(1, 5)] + ["ending1", "ending2"]
        self.eval_df.columns = eval_cols

        self.eval_sentences = self.eval_df[eval_cols].values
        self.eval_correct_endings = correct_ending_idxs.values

        print("Eval sentences shape: ", self.eval_sentences.shape)
        print("Eval endings shape: ", self.eval_correct_endings.shape)
        del self.eval_df

        assert len(self.eval_sentences) == len(self.eval_correct_endings), "All sentences should have endings."

    def sample_random_endings(self, n_samples=1):
        ending_idxs = [np.random.choice(self.n_train_stories, size=n_samples, replace=False)
                       for _ in range(self.n_train_stories)]
        ending_idxs = np.asarray(ending_idxs)
        return self.train_endings[ending_idxs]

    def sample_backward_endings(self, n_samples=1):
        ending_idxs = [np.random.choice(self.story_length, size=n_samples)
                       for _ in range(self.n_train_stories)]
        ending_idxs = np.asarray(ending_idxs)
        backward_endings = list()
        for story_num, idx in enumerate(ending_idxs):
            backward_endings.append(self.train_stories[story_num, idx])

        backward_endings = np.asarray(backward_endings)
        return backward_endings

    def _encode_train(self):
        """Encodes train sentences and endings."""
        train_sentences_list = self.train_sentences.tolist()
        encoded_train = list()

        print("Encoding train sentences")
        for story in train_sentences_list:
            encoded_train.append(self.encoder.encode_sentences(story))

        print("Encoded train Shape: ", np.array(encoded_train).shape)
        self.train_embeddings = (np.array(encoded_train), np.array(self.train_labels))
        self.embedding_dim = self.train_embeddings[0].shape[-1]

    def _encode_eval(self):
        """Encodes train sentences and endings."""
        eval_sentences_list = self.eval_sentences.tolist()
        encoded_eval = list()

        print("Encoding eval sentences")
        for story in eval_sentences_list:
            encoded_eval.append(self.encoder.encode_sentences(story))

        print("Encoded eval Shape: ", np.array(encoded_eval).shape)
        self.eval_embeddings = (np.array(encoded_eval), self.eval_correct_endings)

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
            data = self.train_embeddings
        elif mode == "eval":
            data = self.eval_embeddings
        elif mode == "test":
            data = self.test

        n_samples = data[0].shape[0]
        if shuffle:
            shuffled = np.random.permutation(n_samples)
        else:
            shuffled = np.arange(n_samples)
        for idx in range(0, n_samples, batch_size):
            yield data[0][shuffled[idx: idx + batch_size]], data[1][shuffled[idx: idx + batch_size]]
