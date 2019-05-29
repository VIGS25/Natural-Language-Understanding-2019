"""
Implementation of Abstract Model class
"""

import numpy as np
import tensorflow as tf
import os
from typing import List, Tuple, Dict
import time
import logging

from story_cloze import Dataset

logger = logging.getLogger(__name__)

class Model(object):

    def __init__(self,
                 learning_rate: float = 0.001,
                 model_dir: str = None,
                 log_dir: str = None,
                 graph: tf.Graph() = None,
                 configproto: tf.ConfigProto = None,
                 max_checkpoints_to_keep: int = 5,
                 restore_from: str = None,
                 seed: int =42,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        model_dir: str, default None
            Directory in which models are saved
        batch_size: int, default 64
            Batch size used for training and eval
        learning_rate: float, default 0.001
            Learning rate used for training
        optimizer: tf.train.Optimizer, default None
            Optimizer used for training
        tensorboard: bool, default True
            Whether to save results to tensorboard

        """
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        self.seed = seed
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.tf_objects = {"Graph": graph}

        if model_dir is not None:
            self.save_file = os.path.join(self.model_dir,  "models")
        self.log_dir = log_dir
        self._built = False
        self._training_ops_built = False
        self.restore_from = restore_from

        self._train_summaries  = list()
        self._eval_summaries = list()

    def _build_model_graph(self):
        raise NotImplementedError("Subclasses must implement for themselves.")

    def _build_tf_objects(self):
        """Setups the associated TF primitives."""
        if self.tf_objects["Graph"] is None:
            self.tf_objects["Graph"] = tf.Graph()
            self.tf_objects["Graph"].seed = self.seed
            self.tf_objects["Session"] = tf.Session(graph=self.tf_objects["Graph"])

        with self.tf_objects["Graph"].as_default():
            self.tf_objects["GlobalStep"] = tf.Variable(0, trainable=False)
            self._build_placeholders()

            logger.info("Build model graph for training.")
            self._build_model_graph(mode="train")

            logger.info("Building model graph for evaluation.")
            self._build_model_graph(mode="eval")

            self.tf_objects["FileWriter"] = tf.summary.FileWriter(logdir=self.log_dir)
            self.tf_objects["FileWriter"].add_graph(self.tf_objects["Graph"])

            self._build_train_summaries()
            self._build_eval_summaries()

            self.tf_objects["Saver"] = tf.train.Saver(max_to_keep=self.max_checkpoints_to_keep)

            if self.restore_from is not None:
                logger.info("Restoring variable values from.")
                self._get_tf_object("Saver").restore(self._get_tf_object("Session"), self.restore_from)
            else:
                logger.info("Initializing variable values.")
                self._get_tf_object("Session").run(tf.global_variables_initializer())

    def save(self, path):
        self._get_tf_object("Saver").save(sess=self._get_tf_object("Session"), save_path=path)

    def _get_tf_object(self, obj):
        try:
            return self.tf_objects[obj]
        except KeyError:
            raise ValueError(obj + " missing in the tf_objects dictionary")

    def _log_tensorboard(self, summary):
        """Logs summary to TensorBoard."""
        global_step = self._get_tf_object("Session").run(self._get_tf_object("GlobalStep"))
        writer = self._get_tf_object("FileWriter")
        writer.reopen()
        writer.add_summary(summary, global_step=global_step)
        writer.close()

    def evaluate(self, dataset, epoch, verbose=False):
        """Computes accuracy on eval data."""
        labels = list()
        preds = list()

        probs1 = list()
        probs2 = list()

        for idx, eval_batch in enumerate(dataset.batch_generator(mode="eval", batch_size=1, shuffle=False)):
            eval_sentences, eval_labels = eval_batch
            labels.extend(eval_labels)

            results = self._evaluate_batch(eval_sentences)
            predictions, eval_probs1, eval_probs2 = results
            preds.append(predictions)
            probs1.append(eval_probs1)
            probs2.append(eval_probs2)

        labels = np.squeeze(labels)
        preds = np.squeeze(preds)

        probs1 = np.reshape(probs1, (-1, 1))
        probs2 = np.reshape(probs2, (-1, 1))

        eval_accuracy = np.mean(labels == preds)

        if verbose:
            logger.info("Epoch Num: {}, Eval Accuracy: {}".format(epoch, eval_accuracy))
            logger.info("\n")

        fetches = self.merged_eval_summaries
        feed_dict = {self.eval_accuracy_ph: eval_accuracy, self.eval_act1_ph: probs1, self.eval_act2_ph: probs2}
        results = self._get_tf_object("Session").run(fetches=fetches, feed_dict=feed_dict)
        timestep = self._get_tf_object("Session").run(self._get_tf_object("GlobalStep"))
        self._get_tf_object("FileWriter").add_summary(results, timestep)

    def fit(self, dataset, batch_size=64, nb_epochs=10, log_every=100, print_every=100, display_eval=True):
        """Fits the model, and computes train-eval statistics."""
        start_time = time.time()

        for epoch in range(nb_epochs):
            model_dir_epoch = os.path.join(self.model_dir, str(epoch+1))
            if not os.path.exists(model_dir_epoch):
                os.makedirs(model_dir_epoch)

            logger.info("Computing train statistics, Epoch {}".format(epoch+1))
            for n_batch, train_batch in enumerate(dataset.batch_generator(mode="train", batch_size=batch_size, shuffle=True)):
                self._train_batch(train_batch=train_batch, add_summary=n_batch % log_every, verbose=n_batch % print_every)

            logger.info("Computing eval statistics...".format(epoch+1))
            self.evaluate(dataset=dataset, epoch=epoch, verbose=display_eval)
            model_savepath = os.path.join(model_dir_epoch, "model.ckpt")
            self.save(model_savepath)

        total_time = time.time() - start_time
        logger.info("Finished training the model. Time taken: {} seconds".format(total_time))
