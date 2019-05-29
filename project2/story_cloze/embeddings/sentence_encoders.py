"""
Contains implementations of SentenceEncoders
"""

import numpy as np
import tensorflow as tf
import os
from typing import List, Dict, Tuple

from .skip_thoughts.encoder_manager import EncoderManager
from .skip_thoughts.configuration import model_config

class SentenceEncoder:
    """Abstract implementation of sentence encoder."""

    def encode(self, sentences: List[str]) -> np.ndarray:
        return self.encoder.encode(sentences)

class SkipThoughts(SentenceEncoder):
    """Implementation of SkipThoughts sentence encoder."""
    embed_dir = os.path.abspath("data/embeddings/skip_thoughts")

    def __init__(self, mode: str = "bi", embed_dir: str = None) -> None:
        self.mode = mode
        self.encoder = EncoderManager()
        if embed_dir is not None:
            self.embed_dir = embed_dir
        self.vocab_file = os.path.join(self.embed_dir, "vocab.txt")
        self.load_model()
        super(SkipThoughts, self).__init__()

    def get_bi_metadata(self) -> Tuple[str, str, str]:
        """Returns files corresponding to bidirectional encodings."""
        config = model_config(bidirectional_encoder=True)
        ckpt_file = os.path.join(self.embed_dir, "bi", "model.ckpt-500008")
        embedding_file = os.path.join(self.embed_dir, "bi", "embeddings.npy")
        return config, ckpt_file, embedding_file

    def get_uni_metadata(self) -> Tuple[str, str, str]:
        """Returns files corresponding to unidirectional sentence encoding."""
        config = model_config()
        ckpt_file = os.path.join(self.embed_dir, "uni", "model.ckpt-501424")
        embedding_file = os.path.join(self.embed_dir, "uni", "embeddings.npy")
        return config, ckpt_file, embedding_file

    def get_both_metadata(self) -> Tuple[str, str, str]:
        """Returns files corresponding to both types of encoding."""
        config_uni, ckpt_uni, embedding_uni = self.get_uni_metadata()
        config_bi, ckpt_bi, embedding_bi = self.get_bi_metadata()

        config = [config_uni, config_bi]
        ckpt_file = [ckpt_uni, ckpt_bi]
        embedding_file = [embedding_uni, embedding_bi]
        return config, ckpt_file, embedding_file

    def load_model(self) -> None:
        """Load the encoder model."""
        load_dict = {"bi": self.get_bi_metadata, "uni": self.get_uni_metadata,
                     "both": self.get_both_metadata}
        metadata_fn = load_dict[self.mode]
        config, ckpt_file, embedding_file = metadata_fn()

        if not isinstance(embedding_file, list):
            embedding_file = [embedding_file]
            config = [config]
            ckpt_file = [ckpt_file]

        for idx, configuration in enumerate(config):
            self.encoder.load_model(configuration, self.vocab_file, embedding_file[idx], ckpt_file[idx])

class UniversalEncoder(SentenceEncoder):
    import tensorflow_hub as hub
    encoder = None
    built = False

    def encode_sentence(self, sentence: str) -> np.ndarray:
        """Returns encoding of the sentence."""
        if not self.built:
            self.encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
            self.built = True
        return self.encoder(sentence)

if __name__ == "__main__":
    encoder = SkipThoughts()
    sentences = ["The dog is a cat", "The dog is a cat"]
    encoded = encoder.encode_sentences(sentences)
    print(encoded.shape)
