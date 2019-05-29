"""
Implementation of RNN class.
This model is derived from the one presented in An RNN Based Binary Classifier for the Story Cloze test.
(https://www.aclweb.org/anthology/W17-0911)
"""

import numpy as np
import tensorflow as tf
import os
from .base_model import Model
from typing import List, Dict, Tuple
import time
import logging

DEF_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))

def get_rnn_cell(rnn_type="gru", num_hidden_units=1000):
    """Returns initialized RNN cells based on type."""
    if rnn_type == "gru":
        return tf.nn.rnn_cell.GRUCell(num_units=num_hidden_units)
    elif rnn_type == "lstm":
        return tf.nn.rnn_cell.LSTMCell(num_units=num_hidden_units)
    elif rnn_type == "vanilla":
        return tf.nn.rnn_cell.RNNCell()
    else:
        raise ValueError("RNN type {} not supported.".format(RNN))

logger = logging.getLogger(__name__)

class RNN(Model):

    model_dir = DEF_MODEL_DIR

    train_states = list()
    eval1_states = list()
    eval2_states = list()

    train_outputs = list()
    eval1_outputs = list()
    eval2_outputs = list()

    def __init__(self,
                 encoder,
                 embedding_dim: int,
                 n_story_sentences: int = 4,
                 learning_rate: float = 0.001,
                 rnn_type: str = "gru",
                 num_hidden_units: int = 1000,
                 trainable_zero_state: bool = False,
                 max_checkpoints_to_keep: int = 5,
                 clip_norm: float = 10.0,
                 model_dir: str = None,
                 log_dir: str = None,
                 restore_from: str = None,
                 **kwargs):
        """
        Parameters
        ----------
        embedding_dim: int,
            Size of the sentence embeddings
        n_story_sentences: int, default 4
            Number of sentences per story
        rnn_type: str, default gru
            Type of RNN cell
        num_hidden_units: int, default 1000
            Number of hidden units in the RNN cell
        trainable_zero_state: bool, default False
            Whether to make zero state trainable
        tensorboard_log_frequency: int, default 10
            How often to log to tensorboard
        model_dir: str, default None
            Directory to save models to
        restore_from: str, default None
            Restore saved model from
        """
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.n_story_sentences = n_story_sentences
        self.rnn_type = rnn_type
        self.num_hidden_units = num_hidden_units
        self.trainable_zero_state = trainable_zero_state
        self.clip_norm = clip_norm

        super(RNN, self).__init__(learning_rate=learning_rate,
                                  model_dir=model_dir, log_dir=log_dir,
                                  max_checkpoints_to_keep=max_checkpoints_to_keep,
                                  restore_from=restore_from, **kwargs)

        self._build_tf_objects()

    def _build_placeholders(self):
        """Builds placeholders needed for training."""
        self.input_ph = tf.placeholder(dtype=tf.float32, name='Inputs',
                                      shape=[None, self.n_story_sentences + 1, self.embedding_dim])
        logger.info("Input Story: {}".format(self.input_ph.shape.as_list()))
        self.labels_ph = tf.placeholder(dtype=tf.float32, name='Labels', shape=[None, 1])
        self.batch_size = tf.shape(self.input_ph)[0]

        self.eval_story = tf.placeholder(dtype=tf.float32, name='Eval_Story', shape=[None, self.n_story_sentences, self.embedding_dim])
        self.eval_ending1 = tf.placeholder(dtype=tf.float32, name='Eval_Ending1', shape=[None, 1, self.embedding_dim])
        self.eval_ending2 = tf.placeholder(dtype=tf.float32, name='Eval_Ending2', shape=[None, 1, self.embedding_dim])

        self.eval_in1 = tf.concat([self.eval_story, self.eval_ending1], axis=1)
        self.eval_in2 = tf.concat([self.eval_story, self.eval_ending2], axis=1)

        self.eval_accuracy_ph = tf.placeholder(tf.float32)
        self.eval_act1_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Eval_act1')
        self.eval_act2_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Eval_act2")

        logger.info("Eval input1: {}".format(self.eval_in1.shape))
        logger.info("Eval input2: {}".format(self.eval_in2.shape))

    def _unroll_rnn_cell(self, state, mode="train"):
        """Unrolls the RNN cell."""

        if mode == "train":
            for time_step in range(self.n_story_sentences + 1):
                out, state = self.rnn_cell(self.input_ph[:, time_step, :], state)
                self.train_outputs.append(out)
                self.train_states.append(state)

        else:
            for time_step in range(self.n_story_sentences + 1):
                out, state = self.rnn_cell(self.eval_in1[:, time_step, :], state)
                self.eval1_outputs.append(out)
                self.eval1_states.append(state)

            for time_step in range(self.n_story_sentences + 1):
                out, state = self.rnn_cell(self.eval_in2[:, time_step, :], state)
                self.eval2_outputs.append(out)
                self.eval2_states.append(state)

    def _build_rnn(self, mode="train"):
        """Builds the RNN."""
        with tf.variable_scope(self.rnn_type.upper(), reuse=tf.AUTO_REUSE):
            self.rnn_cell = get_rnn_cell(rnn_type=self.rnn_type)
            if not self.trainable_zero_state:
                state = self.rnn_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            else:
                raise NotImplementedError("Trainable zero state not supported yet.")
            if mode == "train":
                self.train_states.append(state)
            else:
                self.eval1_states.append(state)
                self.eval2_states.append(state)
            self._unroll_rnn_cell(state=state, mode=mode)

    def _build_fc_layer(self, inputs, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("FC", reuse=reuse):
            logits = tf.layers.dense(inputs, units=1, activation=None, name="output")
            logger.info("FC output: {}".format(logits.shape.as_list()))

        return logits

    def _compute_loss(self, mode="train"):
        if mode == "train":
            rnn_final_state = self.train_states[-1]
            logger.info("Final RNN hidden state: {}".format(rnn_final_state.shape.as_list()))
            assert rnn_final_state.shape.as_list()[-1] == self.num_hidden_units

            self.train_logits = self._build_fc_layer(inputs=rnn_final_state, reuse=tf.AUTO_REUSE)
            self.train_probs = tf.sigmoid(self.train_logits)
            self.train_predictions = tf.round(self.train_probs)

            self.train_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.labels_ph, self.train_predictions)))

            with tf.name_scope("train_loss"):
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.train_logits, labels=self.labels_ph))

        else:
            rnn_final_state1 = self.eval1_states[-1]
            rnn_final_state2 = self.eval2_states[-1]

            logger.info("Final RNN hidden state1: {}".format(rnn_final_state1.shape.as_list()))
            assert rnn_final_state1.shape.as_list()[-1] == self.num_hidden_units
            logger.info("Final RNN hidden state2: {}".format(rnn_final_state2.shape.as_list()))
            assert rnn_final_state2.shape.as_list()[-1] == self.num_hidden_units

            self.eval_logits1 = self._build_fc_layer(inputs=rnn_final_state1, reuse=True)
            self.eval_logits2 = self._build_fc_layer(inputs=rnn_final_state2, reuse=True)

            self.eval_probs1 = tf.nn.sigmoid(self.eval_logits1)
            self.eval_probs2 = tf.nn.sigmoid(self.eval_logits2)

            self.eval_logits = tf.concat([self.eval_logits1, self.eval_logits2], axis=1)
            self.eval_predictions = tf.argmax(self.eval_logits, axis=1)

        with tf.name_scope("optimizer"):
            self._build_optimizer()
            gradients = self.optimizer.compute_gradients(self.loss)
            clipped_gradients = [(tf.clip_by_norm(gradient, self.clip_norm), var) for gradient, var in gradients]
            self.train_op = self.optimizer.apply_gradients(clipped_gradients, global_step=self._get_tf_object("GlobalStep"))

            variables = tf.trainable_variables()

    def _evaluate_batch(self, eval_sentences):
        """Computes metrics on eval batches."""

        eval_sentences_list = eval_sentences.tolist()
        encoded_eval = list()

        for story in eval_sentences_list:
            encoded_eval.append(self.encoder.encode_sentences(story))

        encoded_eval = np.array(encoded_eval)

        eval_story = encoded_eval[:, :self.n_story_sentences]
        eval_ending1 = encoded_eval[:, self.n_story_sentences]
        eval_ending2 = encoded_eval[:, self.n_story_sentences + 1]

        eval_ending1 = np.expand_dims(eval_ending1, axis=1)
        eval_ending2 = np.expand_dims(eval_ending2, axis=1)

        eval_in1 = np.concatenate([eval_story, eval_ending1], axis=1)
        eval_in2 = np.concatenate([eval_story, eval_ending2], axis=1)

        assert list(eval_in1.shape[1:]) == self.input_ph.get_shape().as_list()[1:]
        assert list(eval_in2.shape[1:]) == self.input_ph.get_shape().as_list()[1:]

        fetches = [self.eval_predictions, self.eval_probs1, self.eval_probs2]
        results = self._get_tf_object("Session").run(fetches=fetches, feed_dict={self.eval_in1: eval_in1, self.eval_in2: eval_in2,
                                                                                self.input_ph: eval_in1})

        return results

    def _train_batch(self, train_batch, add_summary=False, verbose=False):
        """Runs the training on every batch."""
        train_sentences, train_labels = train_batch

        train_sentences_list = train_sentences.tolist()
        encoded_train = list()

        for story in train_sentences_list:
            encoded_train.append(self.encoder.encode_sentences(story))

        encoded_train = np.array(encoded_train)

        fetches = [self.loss,  self.train_op, self.train_predictions, self.train_accuracy]
        if add_summary:
            fetches.append(self.merged_train_summaries)

        assert len(encoded_train.shape) == 3
        assert list(encoded_train.shape[1:]) == self.input_ph.get_shape().as_list()[1:]

        if len(train_labels.shape) == 1:
            train_labels = train_labels.reshape(-1, 1)

        feed_dict = {self.input_ph: encoded_train, self.labels_ph: train_labels}
        results = self._get_tf_object("Session").run(fetches=fetches, feed_dict=feed_dict)

        if add_summary:
            timestep = self._get_tf_object("Session").run(self._get_tf_object("GlobalStep"))
            self._get_tf_object("FileWriter").add_summary(results[-1], timestep)
        if verbose:
            logger.info("Loss: {0:.4f}, Train accuracy: {1:.3f}".format(results[0], results[3]))

    def _build_optimizer(self, optimizer=None):
        """Builds the optimizer."""
        if optimizer is None:
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.optimizer = optimizer

    def _build_train_summaries(self):
        self._train_summaries.extend([tf.summary.histogram("train/activations", self.train_probs),
                                      tf.summary.scalar("train/loss", self.loss),
                                      tf.summary.scalar("train/accuracy", self.train_accuracy)])

        self.merged_train_summaries = tf.summary.merge(self._train_summaries, name="train_summaries")

    def _build_eval_summaries(self):
        self._eval_summaries.extend([tf.summary.scalar('eval/accuracy', self.eval_accuracy_ph),
                                     tf.summary.histogram('eval/activations1', self.eval_act1_ph),
                                     tf.summary.histogram('eval/activations2', self.eval_act2_ph)])
        self.merged_eval_summaries = tf.summary.merge(self._eval_summaries, name="eval_summaries")

    def _build_model_graph(self, mode="train"):
        """Sets up the computational graph in the model."""
        with tf.variable_scope(self.__class__.__name__, reuse=tf.AUTO_REUSE):
            self._build_rnn(mode=mode)
            self._compute_loss(mode=mode)

if __name__ == "__main__":
    rnn_model = RNN(embedding_dim=2400)
