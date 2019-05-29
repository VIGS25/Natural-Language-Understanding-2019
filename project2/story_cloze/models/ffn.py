"""
Feed-forward Neural Network with customisable input modes
"""

import numpy as np
import tensorflow as tf
import os
from typing import List, Dict, Tuple

from .base_model import Model

DEF_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))


class FFN(Model):

    model_dir = DEF_MODEL_DIR

    def __init__(self,
                 embedding_dim: int = 4800,
                 input_mode: str = "full_context",
                 num_hidden_units: int  = 4800,
                 hidden_layer_sizes: List[int] = [256, 64],
                 n_story_sentences: int = 4,
                 trainable_zero_state: bool = False,
                 tensorboard_log_frequency: int = 10,
                 learning_rate: float = 0.01,
                 model_dir: str = None,
                 log_dir: str = None,
                 restore_from: str = None,
                 **kwargs):
        """
        Parameters
        ----------
        input_mode: str, default full_context:
            Determines what input is used
        hidden_layer_sizes: list
            Hidden layer sizes in the FFN used.
        trainable_zero_state: bool, default False
            Whether to have a trainable hidden state in the GRU.
            Only used in full context mode
        tensorboard_log_frequency: int, default 10
            How frequently to log to tensorboard.
        model_dir: str, default None
            Where to save the models
        log_dir: str, default None
            Where to write tensorboard logs to.
        restore_from: str, default None
            Where to restore saved models from.
        """
        super(FFN, self).__init__(**kwargs)
        if input_mode not in ["full_context", "last_sentence", "no_context"]:
            raise ValueError("Input mode {} not recognized.".format(input_mode))
        self.hidden_layer_sizes = hidden_layer_sizes
        self.input_mode = input_mode
        self.trainable_zero_state = trainable_zero_state
        self.embedding_dim = embedding_dim
        self.n_story_sentences = n_story_sentences
        self.num_hidden_units = num_hidden_units
        self.learning_rate = learning_rate
        if model_dir is not None:
            self.model_dir = model_dir
        self.tensorboard_log_frequency = tensorboard_log_frequency
        self.restore_from = restore_from

        self._build_tf_objects()

    def _build_placeholders(self):
        if self.input_mode == "full_context":
            self.input_ph = tf.placeholder(dtype=tf.float32, name='Full_context',
                                          shape=[None, self.n_story_sentences, self.embedding_dim])
        elif self.input_mode == "last_sentence":
            self.input_ph = tf.placeholder(dtype=tf.float32, name='Last_sentence', shape=[None, self.embedding_dim])
        else:
            self.input_ph = tf.zeros(shape=[None, self.embedding_dim])
        self.ending_ph = tf.placeholder(dtype=tf.float32, name='Ending', shape=[None, self.embedding_dim])
        self.label_ph = tf.placeholder(dtype=tf.float32, name='Labels', shape=[None, 1])
        self.batch_size = tf.shape(self.input_ph)[0]

    def _build_rnn(self):
        """Builds the GRU and performs the unrolling."""
        self.states = list()
        self.outputs = list()

        with tf.variable_scope("GRU", reuse=tf.AUTO_REUSE):
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_hidden_units)
            state = self.rnn_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            self.states.append(state)

            for timestep in range(self.n_story_sentences):
                out, state = self.rnn_cell(self.input_ph[:, timestep], state)
                self.states.append(state)
                self.outputs.append(out)

            self.context_tensor = out

    def _build_model_graph(self):
        if self.input_mode == "full_context":
            self._build_rnn()
        else:
            self.context_tensor = self.input_ph
        ffn_tensor = tf.add(self.context_tensor, self.ending_ph)

        with tf.variable_scope("FFN", reuse=tf.AUTO_REUSE):
            for idx, layer_size in enumerate(self.hidden_layer_sizes):
                ffn_tensor = tf.layers.dense(inputs=ffn_tensor, units=layer_size, activation=tf.nn.relu)
                print("Layer {} output shape: {} ".format(idx, ffn_tensor.shape))

            self.logits = tf.layers.dense(ffn_tensor, units=2, activation=tf.nn.relu)
            ending_probs = tf.nn.softmax(self.logits)
            self.correct_ending_prob = tf.reduce_max(ending_probs, axis=1, keepdims=True)

        labels = tf.one_hot(self.label_ph, depth=2)
        labels = tf.stop_gradient(labels)
        with tf.name_scope("Loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=labels)

        with tf.name_scope("optimizer"):
            self._build_optimizer()
            gradients = self.optimizer.compute_gradients(self.loss)
            clipped_gradients = [(tf.clip_by_norm(gradient, self.clip_norm), var) for gradient, var in gradients]
            self.train_op = self.optimizer.apply_gradients(clipped_gradients, global_step=self._get_tf_object("GlobalStep"))

            variables = tf.trainable_variables()

    def _build_optimizer(self, optimizer=None):
        if optimizer is None:
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            self.optimizer = optimizer

    def _evaluate_batch(self, eval_sentences):
        """Computes metrics on eval batches."""
        eval_story = eval_sentences[:, :self.n_story_sentences]
        eval_ending1 = eval_sentences[:, self.n_story_sentences]
        eval_ending2 = eval_sentences[:, self.n_story_sentences + 1]

        if self.input_mode == "full_context":
            assert list(eval_in1.shape[1:]) == tf.shape(self.input_ph).as_list()[1:]
            feed_dict_1 = {self.input_ph: eval_story, self.ending_ph: eval_ending1}
            feed_dict_2 = {self.input_ph: eval_story, self.ending_ph: eval_ending2}

        elif self.input_mode == "last_sentence":
            feed_dict_1 = {self.input_ph: eval_story[:, -1], self.ending_ph: eval_ending1}
            feed_dict_2 = {self.input_ph: eval_story[:, -1], self.ending_ph: eval_ending2}

        else:
            feed_dict_1 = {self.ending_ph: eval_ending1}
            feed_dict_2 = {self.ending_ph: eval_ending2}

        fetches = self.correct_ending_prob

        corr_ending_prob1 = self._get_tf_object("Session").run(feed_dict=feed_dict_1, fetches=fetches)
        corr_ending_prob2 = self._get_tf_object("Session").run(feed_dict=feed_dict_2, fetches=fetches)

        corr_ending_probs = np.concatenate([corr_ending_prob1, corr_ending_prob2], axis=1)
        endings_pred = np.argmax(corr_ending_probs)

        return endings_pred

    def _train_batch(self, train_batch, add_summary=False, verbose=False):
        """Runs the training on every batch."""
        train_sentences, train_labels = train_batch
        fetches = [self.loss,  self.train_op, self.logits]

        assert len(train_sentences.shape) == 3
        assert train_sentences.shape[1:] == tf.shape(self.input_ph).as_list()[1:]

        if len(train_labels.shape) == 1:
            train_labels = train_labels.reshape(-1, 1)

        feed_dict = {self.input_ph: train_sentences, self.label_ph: train_labels}
        loss, _, logits = self._get_tf_object("Session").run(fetches=fetches, feed_dict=feed_dict)

        predictions = np.argmax(logits, axis=1)
        train_accuracy_batch = np.mean(predictions == train_labels)

        if add_summary:
            pass
        if verbose:
            print("Loss: {0:.4f}, Train accuracy: {1:.3f}".format(loss, train_accuracy_batch))

if __name__ == "__main__":
    ffn = FFN()
