import numpy as np
import tensorflow as tf
import pandas as pd
import os
import logging
import argparse

from story_cloze.embeddings import SkipThoughts, UniversalEncoder

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.environ.get("SCRATCH", "./"), "data")

def main():

    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--input_dir", default=DATA_DIR, help="Directory where data is present.")
    parser.add_argument("--file", default="stories.spring2016.csv", help="File to load stories and compute embeddings from.")
    parser.add_argument("--embed_mode", default="both", choices=["uni", "bi", "both"], help="Embedding mode for SkipThoughts.")
    parser.add_argument("--mode", default="eval", choices=["eval", "test"], help="Whether to generate eval or test embeddings.")

    args = parser.parse_args()

    file_to_read = os.path.join(args.input_dir, args.file)

    filename = args.mode + "_embeddings_SkipThoughts_" + args.embed_mode + ".npy"
    embedding_dir = os.path.join(args.input_dir, "embeddings", "skip_thoughts")
    if args.embed_mode == "uni" or args.embed_mode == "bi":
        embedding_dim = 2400
    else:
        embedding_dim = 4800
    encoder = SkipThoughts(embed_dir=embedding_dir, mode=args.embed_mode)

    df = pd.read_csv(file_to_read)
    if "InputStoryid" in df.columns:
        df.drop(["InputStoryid"], axis=1, inplace=True)

    if "AnswerRightEnding" in df.columns:
        df.drop(["AnswerRightEnding"], axis=1, inplace=True)

    cols = ["sentence_{}".format(i) for i in range(1, 5)] + ["ending1", "ending2"]
    df.columns = cols

    data = df.apply(lambda x: list([x[col] for col in cols]),axis=1)
    logger.info("Encoding the data.")
    encoded_data = np.array([encoder.encode(x).astype(np.float32) for x in data])
    logger.info("Data Encoded, Shape: {}".format(encoded_data.shape))

    logger.info("Saving embeddings for given file to {}".format(filename))
    filename = os.path.join(args.input_dir, filename)
    np.save(filename, encoded_data, allow_pickle=False)

if __name__ == "__main__":
    main()
