"""
Script to generate Universal Sentence Encoders for training and eval datasets. 
"""
import numpy as np
import logging
import datetime
import os

from .datasets import UniversalEncoderDataset
from .embeddings.sentence_encoders import UniversalEncoder

logFormatter = "%(asctime)s - [%(levelname)s] %(message)s"
logging.basicConfig(filename="gen-logs/%s.log" % datetime.datetime.now().strftime('%d-%m--%H-%M'), filemode='a', format=logFormatter, level=logging.INFO)
logger = logging.getLogger(__name__)

dataset = UniversalEncoderDataset(input_dir=os.path.join(os.environ['SCRATCH'], "data"), encode_only=True, encoder=UniversalEncoder())

