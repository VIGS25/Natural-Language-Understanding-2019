"""
Dataset class for handling data loading and manipulations
"""

import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data/"))


class Dataset:
    train_file = "stories.train.csv"
    train_small_file = "stories.train.small.csv"
    eval_small_file = "stories.eval.small.csv"
    eval_file = "stories.eval.csv"
    test_file = "stories.spring2016.csv"

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
            eval_file = self.eval_small_file
        else:
            train_file = self.train_file
            eval_file = self.eval_file

        self.load(train_file)
        self._process_eval(eval_file)
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

        logger.info("Train sentences Shape: {}".format(self.train_sentences.shape))
        logger.info("Train labels shape: {}".format(self.train_labels.shape))

        self.train_data = self.train_df.apply(lambda x: list([x[col] for col in self.train_cols]),axis=1)
        del self.train_df

    def _add_negative_endings(self, n_random: int = 0, n_backward: int = 0):
        """Adds specified number of backward and random negative endings for each story."""
        if n_random:
            logger.info("Sampling {} random endings per story.".format(n_random))
            ending_idxs = self.sample_random_endings(n_samples=n_random)
            train_story_augment = np.array([self.train_data[:, :self.story_length]]*n_random)
            train_story_augment = train_story_augment.reshape(-1, 4, self.train_data.shape[-1])
            train_endings_sampled = np.expand_dims(self.train_data[ending_idxs, -1], axis=1)
            assert len(train_story_augment) == len(train_endings_sampled)

            train_augment = np.concatenate([train_story_augment, train_endings_sampled], axis=1)
            assert train_augment.shape[1] == self.story_length + 1
            assert train_augment.shape[-1] == self.train_data.shape[-1]

            train_labels = np.zeros(shape=(train_augment.shape[0], 1))
            self.train_data = np.vstack([self.train_data, train_augment])
            self.train_labels = np.vstack([self.train_labels, train_labels])
            assert len(self.train_data) == len(self.train_labels)

            logger.info("After adding random endings..")
            logger.info("Train data shape: {}".format(self.train_data.shape))
            logger.info("Train labels shape: {}".format(self.train_labels.shape))

    def _process_eval(self, eval_file: str):
        self.eval_df = pd.read_csv(os.path.join(self.input_dir, eval_file))
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

    def sample_random_endings(self, n_samples: int = 1):
        ending_idxs = list()
        for _ in range(n_samples):
            ending_idxs.append(np.random.permutation(self.n_train_stories))
        ending_idxs = np.asarray(ending_idxs).flatten()
        return ending_idxs

    def _encode_train(self):
        logger.info("Encoding train sentences...")
        self.train_data = np.array([self.encoder.encode(x).astype(np.float32) for x in self.train_data])
        encoder_name = self.encoder.__class__.__name__
        embed_name = "train_embeddings_" + encoder_name

        if self.encoder.__class__.__name__ == "SkipThoughts":
            embed_name += "_" + self.encoder.mode  + ".npy"
        else:
            embed_name += ".npy"

        embed_name = os.path.join(self.input_dir, embed_name)
        np.save(embed_name, self.train_data, allow_pickle=False)
        logger.info("Saved training embeddings.")

    def _encode_eval(self):
        logger.info("Encoding eval sentences...")
        self.eval_data = np.array([self.encoder.encode(x).astype(np.float32) for x in self.eval_data])
        logger.info("Embeddings shape: {}".format(self.eval_data.shape))

    def load(self, train_file: str):
        logger.info("Loading the embeddings and labels...")
        encoder_name = self.encoder.__class__.__name__
        embed_name = "train_embeddings_" + encoder_name

        if self.encoder.__class__.__name__ == "SkipThoughts":
            embed_name += "_" + self.encoder.mode  + ".npy"
        else:
            embed_name += ".npy"

        embed_name = os.path.join(self.input_dir, embed_name)

        if not os.path.exists(embed_name):
            logger.warning("{} does not exist. Encoding embeddings.".format(embed_name))
            self._process_train(train_file)
            self._encode_train()

        else:
            self.train_data = np.load(embed_name).astype(np.float32)
            self.train_labels = np.ones((len(self.train_data), 1))
            self.n_train_stories = self.train_data.shape[0]

            logger.info("Train data shape: {}".format(self.train_data.shape))
            logger.info("Train labels shape: {}".format(self.train_labels.shape))

        if self.add_neg:
            logger.info("Adding negative endings.")
            self._add_negative_endings(n_random=self.n_random, n_backward=self.n_backward)

        logger.info("After adding negative endings..")
        logger.info("Train dataset shape: {}".format(self.train_data.shape))
        logger.info("Train labels shape: {}".format(self.train_labels.shape))

    def batch_generator(self, mode: str = "train", batch_size: int = 64, shuffle:bool =True):
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


class UniversalEncoderDataset(object):

    train_file = "stories.train.csv"
    train_small_file = "stories.train.small.csv"
    eval_small_file = "stories.eval.small.csv"
    eval_file = "stories.eval.csv"
    test_file = "stories.spring2016.csv"

    def __init__(self,
                 input_dir: str = DATA_DIR,
                 story_length: int = 4,
                 preprocessors: List = None,
                 add_neg: bool = True,
                 n_random: int = 4,
                 n_backward: int = 2,
                 use_small: bool = False,
                 mode: str = "train",
                 encode_only = False,
                 encoder = None) -> None:

        self.input_dir = input_dir
        self.story_length = story_length
        self.add_neg = add_neg
        self.use_small = use_small
        self.n_random = n_random
        self.n_backward = n_backward

        if mode == "train":
            train_file = self.train_file
            eval_file = self.eval_file
        elif mode == "test":
            train_file = self.train_file
            eval_file = self.test_file

        if encode_only:
            self.encoder = encoder
            self._process_train(train_file)
            self._process_eval(eval_file)
            self._encode_train()
            self._encode_eval()
            self._process_eval(self.test_file)
            self._encode_eval(mode="test")
        else:
            self.load(train_file)
            self._process_eval(eval_file)
            self.load_eval(eval_file)

    def _process_train(self, train_file):
        """Processes training set and augments it with negative endings."""
        self.train_df = pd.read_csv(os.path.join(self.input_dir, train_file))
        self.train_df.drop(["storyid", "storytitle"], axis=1, inplace=True)
        self.train_cols = ["sentence_{}".format(i) for i in range(1, 5)] + ["ending"]
        self.train_df.columns = self.train_cols

        self.train_sentences = self.train_df[self.train_cols].values
        self.train_labels = np.ones((len(self.train_sentences), 1))
        self.n_train_stories = len(self.train_labels)

        logger.info("Train sentences Shape: ".format(self.train_sentences.shape))
        logger.info("Train labels shape: ".format(self.train_labels.shape))

        self.train_data = self.train_df.values #self.train_df.apply(lambda x: list([x[col] for col in self.train_cols]),axis=1)
        del self.train_df

    def _add_negative_endings(self, n_random: int = 0, n_backward: int = 0):
        """Adds specified number of backward and random negative endings for each story."""
        if n_random:
            logger.info("Sampling {} random endings per story.".format(n_random))
            ending_idxs = self.sample_random_endings(n_samples=n_random)
            train_story_augment = np.array([self.train_data[:, :self.story_length]]*n_random)
            train_story_augment = train_story_augment.reshape(-1, 4, self.train_data.shape[-1])
            train_endings_sampled = np.expand_dims(self.train_data[ending_idxs, -1], axis=1)
            assert len(train_story_augment) == len(train_endings_sampled)

            train_augment = np.concatenate([train_story_augment, train_endings_sampled], axis=1)
            assert train_augment.shape[1] == self.story_length + 1
            assert train_augment.shape[-1] == self.train_data.shape[-1]

            train_labels = np.zeros(shape=(train_augment.shape[0], 1))
            self.train_data = np.vstack([self.train_data, train_augment])
            self.train_labels = np.vstack([self.train_labels, train_labels])
            assert len(self.train_data) == len(self.train_labels)
            logger.info("After adding random endings..")
            logger.info("Train data shape: {}".format(self.train_data.shape))
            logger.info("Train labels shape: {}".format(self.train_labels.shape))

    def _process_eval(self, eval_file: str):
        self.eval_df = pd.read_csv(os.path.join(self.input_dir, eval_file))
        correct_ending_idxs = self.eval_df["AnswerRightEnding"] - 1
        self.eval_df.drop(["InputStoryid", "AnswerRightEnding"], axis=1, inplace=True)
        eval_cols = ["sentence_{}".format(i) for i in range(1, 5)] + ["ending1", "ending2"]
        self.eval_df.columns = eval_cols

        self.eval_data = self.eval_df.values #self.eval_df.apply(lambda x: list([x[col] for col in eval_cols]),axis=1)
        self.eval_correct_endings = correct_ending_idxs.values

        logger.info("Eval sentences shape: {}".format(self.eval_data.shape))
        logger.info("Eval endings shape: {}".format(self.eval_correct_endings.shape))
        del self.eval_df

        assert len(self.eval_data) == len(self.eval_correct_endings), "All sentences should have endings."

    def sample_random_endings(self, n_samples: int = 1):
        ending_idxs = list()
        for _ in range(n_samples):
            ending_idxs.append(np.random.permutation(self.n_train_stories))
        ending_idxs = np.asarray(ending_idxs).flatten()
        return ending_idxs

    def load_eval(self, eval_file, mode="train"):
        logger.info("Loading eval sentences.")
        encoder_name = "UniversalEncoder"
        embed_name = "eval_" if mode == "train" else "test_"
        embed_name += "embeddings_" + encoder_name + ".npy"

        embed_name = os.path.join(self.input_dir, embed_name)

        self.eval_df = pd.read_csv(os.path.join(self.input_dir, eval_file))
        correct_ending_idxs = self.eval_df["AnswerRightEnding"] - 1
        self.eval_correct_endings = correct_ending_idxs.values

        with open(embed_name, "rb") as f:
            eval_data = pickle.load(f)

        eval_story = eval_data["data"].astype(np.float32)
        eval_endings1 = np.expand_dims(eval_data["endings1"].astype(np.float32), axis=1)
        eval_endings2 = np.expand_dims(eval_data["endings2"].astype(np.float32), axis=1)
        self.eval_data = np.concatenate([eval_story, eval_endings1, eval_endings2], axis=1)

        logger.info("Eval sentences shape: {}".format(self.eval_data.shape))
        logger.info("Eval endings shape: {}".format(self.eval_correct_endings.shape))

    def load(self, train_file: str):
        logger.info("Loading the embeddings and labels...")
        encoder_name = "UniversalEncoder"
        embed_name = "train_embeddings_" + encoder_name + ".npy"
        embed_name = os.path.join(self.input_dir, embed_name)

        if not os.path.exists(embed_name):
            logger.warning("{} does not exist. Encoding embeddings.".format(embed_name))
            self._process_train(train_file)
            self._encode_train()

        else:
            with open(embed_name, "rb") as f:
                train_data = pickle.load(f)

            train_story = train_data["data"].astype(np.float32)
            train_ending = np.expand_dims(train_data["endings"].astype(np.float32), axis=1)
            self.train_data = np.concatenate([train_story, train_ending], axis=1)
            self.train_labels = np.ones((len(self.train_data), 1))
            self.n_train_stories = self.train_data.shape[0]

            logger.info("Train data shape: {}".format(self.train_data.shape))
            logger.info("Train labels shape: {}".format(self.train_labels.shape))

        if self.add_neg:
            logger.info("Adding negative endings.")
            self._add_negative_endings(n_random=self.n_random, n_backward=self.n_backward)

        logger.info("After adding negative endings..")
        logger.info("Train dataset shape: {}".format(self.train_data.shape))
        logger.info("Train labels shape: {}".format(self.train_labels.shape))

    def _encode_train(self):
        logger.info("Encoding train sentences...")
        encoder_name = self.encoder.__class__.__name__
        dirname = os.path.join(self.input_dir, encoder_name)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        train_data  = self.encoder.encode(self.train_data[:, :4].flatten()).reshape(-1, 4, 512)
        train_endings = self.encoder.encode(self.train_data[:, 4])
        filename = os.path.join(dirname, "train_embeddings_UniversalEncoder.npy")
        logger.info("Embeddings shape: {}".format(train_data.shape))
        filedict = {
            "data": train_data,
            "labels": self.train_labels,
            "endings": train_endings
        }
        logger.info("Successfully generated embeddings.")
        with open(filename, "wb") as f:
            pickle.dump(filedict, f)
        logger.info("Saved training embeddings at {}".format(filename))

    def _encode_eval(self, mode="eval"):
        encoder_name = self.encoder.__class__.__name__
        dirname = os.path.join(self.input_dir, encoder_name)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        logger.info("Encoding eval sentences...")
        eval_data = self.encoder.encode(self.eval_data[:, :4].flatten()).reshape(-1, 4, 512)
        eval_endings1 = self.encoder.encode(self.eval_data[:, 4])
        eval_endings2 = self.encoder.encode(self.eval_data[:, 5])
        filename = os.path.join(dirname, "%s_embeddings_UniversalEncoder.npy" % mode)
        logger.info("Embeddings shape: {}".format(eval_data.shape))
        filedict = {"data": eval_data}
        filedict["endings1"] = eval_endings1
        filedict["endings2"] = eval_endings2
        filedict["correct_end"] = self.eval_correct_endings
        with open(filename, "wb") as f:
            pickle.dump(filedict, f)
        logger.info("Saved eval embeddings at {}.".format(filename))

    def batch_generator(self, mode: str = "train", batch_size: int = 64, shuffle:bool =True):
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

class ValDataset(object):
    def __init__(self,
                 encoder,
                 input_dir: str = DATA_DIR,
                 story_length: int = 4,
                 preprocessors: List = None) -> None:

        self.input_dir = input_dir
        self.encoder = encoder
        self.story_length = story_length

        train_file = "stories.eval.csv"
        test_file = "stories.spring2016.csv"
        self._load_train(train_file)
        self._load_test(test_file)
        self._encode_test()

    def _load_train(self, train_file: str):
        logger.info("Loading train data and labels.")
        encoder_name = self.encoder.__class__.__name__
        embed_name = "train_embeddings_" + encoder_name

        embed_name = os.path.join(self.input_dir, embed_name)

        if encoder_name != "SkipThoughts":
            embed_name = "eval_embeddings_" + encoder_name + ".npy"
            embed_name = os.path.join(self.input_dir, embed_name)

            with open(embed_name, "rb") as f:
                train_data = pickle.load(f)

            train_story = train_data["data"].astype(np.float32)
            train_endings1 = np.expand_dims(train_data["endings1"].astype(np.float32), axis=1)
            train_endings2 = np.expand_dims(train_data["endings2"].astype(np.float32), axis=1)
            target1 = (train_data["correct_end"] == 0)
            target2 = train_data["correct_end"]

            train_data1 = np.concatenate([train_story, train_endings1], axis=1)
            train_data2 = np.concatenate([train_story, train_endings2], axis=1)

        else:
            embed_name = "eval_embeddings_SkipThoughts_both.npy"
            embed_name = os.path.join(self.input_dir, embed_name)
            train_data = np.load(embed_name)

            train_data1 = train_data[:, :5]
            train_data2 = np.concatenate([train_data[:, :4], np.expand_dims(train_data[:, 5], 1)], axis=1)
            train_df = pd.read_csv(os.path.join(self.input_dir, train_file))
            target = train_df["AnswerRightEnding"].values - 1
            del train_df
            target1 = (target == 0)
            target2 = target

        self.train_data = np.concatenate([train_data1, train_data2], axis=0)
        self.train_labels = np.concatenate([target1, target2], axis=0)

        logger.info("Train data shape: {}".format(self.train_data.shape))
        logger.info("Train labels shape: {}".format(self.train_labels.shape))

    def _load_test(self, test_file: str):
        logger.info("Loading test data and labels.")

        self.test_df = pd.read_csv(os.path.join(self.input_dir, test_file))
        self.test_correct_endings = self.test_df["AnswerRightEnding"].values - 1
        # self.test_df.drop(["InputStoryid", "AnswerRightEnding"], axis=1, inplace=True)
        # test_cols = ["sentence_{}".format(i) for i in range(1, 5)] + ["ending1", "ending2"]
        # self.test_data = self.test_df.apply(lambda x: list([x[col] for col in test_cols]),axis=1)
        # self.test_df.columns = test_cols


        # logger.info("Test sentences shape: {}".format(self.test_data.shape))
        logger.info("Test endings shape: {}".format(self.test_correct_endings.shape))
        del self.test_df

        # assert len(self.test_data) == len(self.test_correct_endings), "All sentences should have endings."

    def _encode_test(self):
        logger.info("Encoding test data and labels.")
        encoder_name = self.encoder.__class__.__name__
        if encoder_name == "SkipThoughts":
            self.eval_data = np.load(os.path.join(self.input_dir, "test_embeddings_SkipThoughts_both.npy"))
            self.eval_correct_endings = self.test_correct_endings
        else:
            embed_name = "test_embeddings_" + encoder_name + ".npy"
            embed_name = os.path.join(self.input_dir, embed_name)

            with open(embed_name, "rb") as f:
                test_data = pickle.load(f)

            test_story = test_data["data"].astype(np.float32)
            test_endings1 = np.expand_dims(test_data["endings1"].astype(np.float32), axis=1)
            test_endings2 = np.expand_dims(test_data["endings2"].astype(np.float32), axis=1)
            target1 = (test_data["correct_end"] == 0)
            target2 = test_data["correct_end"]

            self.eval_data = np.concatenate([test_story, test_endings1, test_endings2], axis=1)
            self.eval_correct_endings = test_data["correct_end"]

        logger.info("Test sentences shape: {}".format(self.eval_data.shape))
        logger.info("Test endings shape: {}".format(self.eval_correct_endings.shape))


    def batch_generator(self, mode: str = "train", batch_size: int = 64, shuffle:bool =True):
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
