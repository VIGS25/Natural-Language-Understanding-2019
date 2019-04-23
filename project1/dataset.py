# coding: utf-8
import numpy as np
import os
import pickle
import argparse

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_FILE = "sentences.train"
EVAL_FILE = "sentences.eval"
TEST_FILE = "sentences_test.txt"
EMBEDDING_FILE = ""
BASE_VOCAB  = ["<bos>", "<eos>", "<pad>", "<unk>"]

SAVE_TRAIN = "train.pkl"
SAVE_EVAL = "eval.pkl"
SAVE_TEST = "test.pkl"

class Dataset(object):
    """
    Abstract class for holding train, eval, test datasets and parsing them.
    """
    vocab = BASE_VOCAB
    word_to_idx = None
    idx_to_word = None
    train = None
    test = None
    eval = None
    continuation = None

    def __init__(self, input_dir, embed_file, save_dir=None, **kwargs):
        """
        Parameters
        ----------
        input_dir: str
            Location of datasets
        embed_file: str, default None
            Location of word2vec embeddings
        save_dir: str, default None
            Where to save parsed datasets
        """
        self.input_dir = input_dir
        if save_dir is None:
            self.save_dir = input_dir
        else:
            self.save_dir = save_dir
        self.embedding_file = os.path.join(input_dir, embed_file)

    def generate_vocab(self, max_sen_len=30, topk=20000, save=False):
        """Generates the vocabulary used.

        Parameters
        ----------
        max_sen_len: int, default 30
            Maximum length of sentence used
        topk: int, default 20000
            Top k words to use for building vocabulary
        save: bool, default = False
            Whether to save the vocabulary
        """
        words = dict()
        with open(os.path.join(self.input_dir, TRAIN_FILE), "r") as fh:
            for l in fh.readlines():
                line = l.split()
                if len(line) > max_sen_len:
                    continue # ignore words from sentences longer than 30 words
                for w in line: # words in current line
                    if not words.get(w):
                        words[w] = 0  # new word
                    words[w] += 1

        word_by_count = sorted(words, key=words.get, reverse=True)[:topk-4]
        self.vocab += word_by_count
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

        if save:
            with open(os.path.join(self.save_dir, "vocab.pkl"), "wb") as f:
                pickle.dump(self.vocab, f)

    def pad(self, words, padded_len=30):
        """Pads the sentences to specified pad length.

        Parameters
        ----------
        words: list,
            List of words to add the padding symbol to
        padded_len: int, default 30
            Length after padding
        """
        words.extend(["<pad>"] * (padded_len - len(words)))
        return words

    def parse_sentence(self, sentence, max_sen_length=30):
        """Parses a single sentence.

        Parameters
        ----------
        sentence: str
            Sentence to parse
        max_sen_length: int, default 30
            Maximum length of allowed sentences
        """
        line = []
        parsed = False
        words = sentence.split()
        words.insert(0, "<bos>")
        words.append("<eos>")

        if(len(words) <= max_sen_length - 2):
            words = self.pad(words, padded_len=max_sen_length)
            for w in words:
                if w in self.vocab:
                    line.append(self.word_to_idx[w])
                else:
                    line.append(self.word_to_idx["<unk>"])
            parsed = True
            return line, parsed

        return line, parsed

    def parse_train(self, max_sen_length=30, verbose=False, save=False, reload=True):
        """Parses training sentences.

        Parameters
        ----------
        mode: str, default train
            Which dataset to parse
        max_sen_length: int, default 30
            Maximum allowed sentence length
        verbose: bool, default False
            Whether to print progress during parsing
        save: bool, default False
            Whether to save result of parsing.
        """
        self.train = self.parse_sentences(mode="train", max_sen_length=max_sen_length,
                                          verbose=verbose, save=save, reload=reload)

    def parse_test(self, max_sen_length=30, verbose=False, save=False, reload=True):
        """Parses test sentences.

        Parameters
        ----------
        mode: str, default train
            Which dataset to parse
        max_sen_length: int, default 30
            Maximum allowed sentence length
        verbose: bool, default False
            Whether to print progress during parsing
        save: bool, default False
            Whether to save result of parsing.
        """
        self.test = self.parse_sentences(mode="test", max_sen_length=max_sen_length,
                                         verbose=verbose, save=save, reload=reload)

    def parse_eval(self, max_sen_length=30, verbose=False, save=False, reload=True):
        """Parses eval sentences.

        Parameters
        ----------
        mode: str, default train
            Which dataset to parse
        max_sen_length: int, default 30
            Maximum allowed sentence length
        verbose: bool, default False
            Whether to print progress during parsing
        save: bool, default False
            Whether to save result of parsing.
        """
        self.eval = self.parse_sentences(mode="eval", max_sen_length=max_sen_length,
                                         verbose=verbose, save=save, reload=reload)

    def parse_sentences(self, mode="train", max_sen_length=30, verbose=False, save=False, reload=True):
        """Parses sentences depending on mode specified.

        Parameters
        ----------
        mode: str, default train
            Which dataset to parse
        max_sen_length: int, default 30
            Maximum allowed sentence length
        verbose: bool, default False
            Whether to print progress during parsing
        save: bool, default False
            Whether to save result of parsing.
        """
        load_filenames = {"train": TRAIN_FILE, "eval": EVAL_FILE, "test": TEST_FILE}
        save_filenames = {"train": SAVE_TRAIN, "eval": SAVE_EVAL, "test": SAVE_TEST}

        load_file = load_filenames[mode]
        save_file = save_filenames[mode]

        if reload:
            with open(os.path.join(self.save_dir, save_file), "rb") as f:
                return pickle.load(f)

        with open(os.path.join(self.input_dir, load_file), "r") as f:
            f.seek(0)
            output = []
            for sentence in f.readlines():
                if verbose and (len(output) % 20000) == 0:
                    print("[INFO]: Reading line", len(output))
                line, parsed = self.parse_sentence(sentence, max_sen_length=max_sen_length)
                if parsed:
                    output.append(line)
        if save:
            with open(os.path.join(self.save_dir, save_file), "wb") as f:
                pickle.dump(np.asarray(output), f)

        return np.asarray(output)

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
            data = self.train
        elif mode == "eval":
            data = self.eval
        elif mode == "test":
            data = self.test

        n_samples = data.shape[0]
        if shuffle:
            shuffled = np.random.permutation(n_samples)
        else:
            shuffled = np.arange(n_samples)
        for idx in range(0, n_samples, batch_size):
            yield data[shuffled[idx: idx + batch_size]]

def run():
    """Runs the specified experiment."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir", dest="input_dir", default=DATA_DIR, help="input directory")
    parser.add_argument("--save_dir", dest="save_dir", default=DATA_DIR, help="Save results to")

    args = parser.parse_args()
    dataset = Dataset(input_dir=args.input_dir, save_dir=args.save_dir)
    dataset.generate_vocab()
    dataset.parse_eval()
    print(dataset.eval)

if __name__ == "__main__":
    run()
