import numpy as np
import tensorflow as tf
import pandas as pd
import os
import logging
import argparse

from story_cloze.embeddings import SkipThoughts, UniversalEncoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.environ["SCRATCH"], "data")

def main():

    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--input_dir", default=DATA_DIR, help="Directory where data is present.")
    parser.add_argument("--file", default="stories.spring2016.csv", help="File to load stories and compute embeddings from.")
    parser.add_argument("--encoder_type", default="skipthoughts", choices=["universal", "skipthoughts"],
                        help="Which encoder to use.")
    parser.add_argument("--embed_mode", default="both", choices=["uni", "bi", "both"], help="Embedding mode for SkipThoughts.")
    parser.add_argument("--mode", default="eval", choices=["eval", "test"], help="Whether to generate eval or test embeddings.")

    file = os.path.join(args.input_dir, args.file)

    if args.encoder_type == "skipthoughts":
        filename = args.mode + "_embeddings_SkipThoughts" + args.embed_mode + ".npy"
        embedding_dir = os.path.join(args.input_dir, "embeddings", "skip_thoughts")
        if args.embed_mode == "uni" or args.embed_mode == "bi":
            embedding_dim = 2400
        else:
            embedding_dim = 4800
        encoder = SkipThoughts(embed_dir=embedding_dir, mode=args.embed_mode)
    elif args.encoder_type == "universal":
        filename = args.mode + "_embeddings_UniveralEncoder.npy"
        embedding_dim = 512
        encoder = UniversalEncoder()
    else:
        raise ValueError("Encoder of type {} is not supported.".format(args.encoder_type))

    df = pd.read_csv(file)
    if mode == "eval":
        correct_ending_idxs = df["AnswerRightEnding"] - 1
        df.drop(["InputStoryid", "AnswerRightEnding"], axis=1, inplace=True)
    cols = ["sentence_{}".format(i) for i in range(1, 5)] + ["ending1", "ending2"]
    df.columns = cols

    data = df.apply(lambda x: list([x[col] for col in cols]),axis=1)
    encoded_data = np.array([encoder.encode(x).astype(np.float32) for x in data])

    logger.info("Saving embeddings for given file.")
    filename = os.path.join(args.input_dir, filename)
    np.save(filename, encoded_data, allow_pickle=False)

if __name__ == "__main__":
    main()
