"""
Feed-forward Neural Network with customisable input modes
"""

import numpy as np
import tensorflow as tf
import os
from typing import List, Dict, Tuple
import logging

from .base_model import Model

DEF_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../outputs"))

def get_rnn_cell(rnn_type="gru", num_hidden_units=1000):
    """Returns initialized RNN cells based on type."""
    if rnn_type == "gru":
        return tf.nn.rnn_cell.GRUCell(num_units=num_hidden_units)
    elif rnn_type == "lstm":
        return tf.nn.rnn_cell.LSTMCell(num_units=num_hidden_units)
    elif rnn_type == "vanilla":
        return tf.nn.rnn_cell.BasicRNNCell(num_units=num_hidden_units)
    else:
        raise ValueError("RNN type {} not supported.".format(RNN))

logger = logging.getLogger(__name__)

class FFN(Model):

    model_dir = DEF_MODEL_DIR

    train_states = list()
    eval_states = list()

    train_outputs = list()
    eval_outputs = list()

    def __init__(self,
                 embedding_dim: int = 4800,
                 rnn_type: str = "gru",
                 input_mode: str = "full_context",
                 hidden_layer_sizes: List[int] = [256, 64],
                 n_story_sentences: int = 4,
                 trainable_zero_state: bool = False,
                 clip_norm: float = 10.0,
                 max_checkpoints_to_keep: int = 5,
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
        self.rnn_type = rnn_type
        self.input_mode = input_mode
        self.trainable_zero_state = trainable_zero_state
        self.embedding_dim = embedding_dim
        self.n_story_sentences = n_story_sentences
        self.clip_norm = clip_norm
        self.num_hidden_units = embedding_dim

        super(FFN, self).__init__(learning_rate=learning_rate,
                                  model_dir=model_dir, log_dir=log_dir,
                                  max_checkpoints_to_keep=max_checkpoints_to_keep,
                                  restore_from=restore_from, **kwargs)

        self._build_tf_objects()

    def _build_placeholders(self):
        if self.input_mode == "full_context":
            self.train_input = tf.placeholder(dtype=tf.float32, name='Full_context',
                                          shape=[None, self.n_story_sentences, self.embedding_dim])
            self.eval_input = tf.placeholder(dtype=tf.float32, name='Eval_context',
                                          shape=[None, self.n_story_sentences, self.embedding_dim])
        elif self.input_mode == "last_sentence":
            self.train_input = tf.placeholder(dtype=tf.float32, name='Last_sentence', shape=[None, 1, self.embedding_dim])
            self.eval_input = tf.placeholder(dtype=tf.float32, name='Last_eval_sentence', shape=[None, 1, self.embedding_dim])
        else:
            self.train_input = tf.zeros(shape=[None, self.embedding_dim])
            self.eval_input = tf.zeros(shape=[None, self.embedding_dim])

        self.ending_ph = tf.placeholder(dtype=tf.float32, name='Ending', shape=[None, 1, self.embedding_dim])
        self.labels_ph = tf.placeholder(dtype=tf.int32, name='Labels', shape=[None, 1])
        self.batch_size = tf.shape(self.train_input)[0]

        self.eval_ending1 = tf.placeholder(dtype=tf.float32, name='Eval_Ending1', shape=[None, 1, self.embedding_dim])
        self.eval_ending2 = tf.placeholder(dtype=tf.float32, name='Eval_Ending2', shape=[None, 1, self.embedding_dim])

        self.eval_accuracy_ph = tf.placeholder(tf.float32)
        self.eval_act1_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Eval_act1')
        self.eval_act2_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Eval_act2")

    def _unroll_rnn_cell(self, state, mode="train"):
        if mode == "train":
            logger.info("Unrolling RNN in train mode.")

            for timestep in range(self.n_story_sentences):
                out, state = self.rnn_cell(self.train_input[:, timestep], state)
                self.train_states.append(state)
                self.train_outputs.append(out)

        else:
            logger.info("Unrolling RNN in eval mode.")
            for time_step in range(self.n_story_sentences):
                out, state = self.rnn_cell(self.eval_input[:, time_step], state)
                self.eval_outputs.append(out)
                self.eval_states.append(state)

        return out

    def _build_rnn(self, mode="train"):
        """Builds the RNN and performs the unrolling."""
        with tf.variable_scope(self.rnn_type.upper(), reuse=tf.AUTO_REUSE):
            self.rnn_cell = get_rnn_cell(rnn_type=self.rnn_type, num_hidden_units=self.num_hidden_units)
            state = self.rnn_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            if mode == "train":
                self.train_states.append(state)
            else:
                self.eval_states.append(state)
            context_tensor = self._unroll_rnn_cell(state=state, mode=mode)

        return context_tensor

    def _build_fc_layers(self, inputs, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("Dense", reuse=tf.AUTO_REUSE):
            for idx, layer_size in enumerate(self.hidden_layer_sizes):
                inputs = tf.layers.dense(inputs=inputs, units=layer_size, activation=tf.nn.relu)
                logger.info("Layer {} output shape: {} ".format(idx, inputs.shape))

            logits = tf.layers.dense(inputs, units=2)
        return logits

    def _compute_loss(self, mode="train"):
        if self.input_mode == "full_context":
            context_tensor = self._build_rnn(mode=mode)
        else:
            if mode == "train":
                context_tensor = self.train_input
            else:
                context_tensor = self.eval_input

        if mode == "train":
            ending_ph = tf.squeeze(self.ending_ph, axis=1)
            inputs = tf.add(context_tensor, ending_ph)
            logger.info("Inputs to FFN: {}".format(inputs.shape))
            self.train_logits = self._build_fc_layers(inputs)
            self.train_probs = tf.reduce_max(tf.nn.softmax(self.train_logits), axis=1)
            self.train_predictions = tf.cast(tf.argmax(self.train_logits, axis=-1), tf.int32)

            logger.info("Train Logits: {}".format(self.train_logits.shape))
            logger.info("Train probs: {}".format(self.train_probs.shape))
            logger.info("Train Predictions: {}".format(self.train_predictions.shape))

            self.train_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.labels_ph, self.train_predictions)))

            labels = tf.to_float(tf.one_hot(self.labels_ph, depth=2))
            labels = tf.stop_gradient(labels)
            with tf.name_scope("Loss"):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.train_logits, labels=labels))

        else:
            eval_ending1 = tf.squeeze(self.eval_ending1, axis=1)
            eval_ending2 = tf.squeeze(self.eval_ending2, axis=1)

            inputs1 = tf.add(context_tensor, eval_ending1)
            inputs2 = tf.add(context_tensor, eval_ending2)

            logger.info("Eval input1: {}".format(inputs1.shape))
            logger.info("Eval input2: {}".format(inputs2.shape))

            self.eval_logits1 = self._build_fc_layers(inputs1)
            self.eval_logits2 = self._build_fc_layers(inputs2)

            logger.info("Eval logits1: {}".format(self.eval_logits1.shape))
            logger.info("Eval logits2: {}".format(self.eval_logits2.shape))

            eval_probs1 = tf.nn.softmax(self.eval_logits1)
            eval_probs2 = tf.nn.softmax(self.eval_logits2)

            self.eval_probs1 = tf.expand_dims(eval_probs1[:, 1], axis=1)
            self.eval_probs2 = tf.expand_dims(eval_probs2[:, 1], axis=1)

            correct_ending_probs = tf.concat([self.eval_probs1, self.eval_probs2], axis=1)
            self.eval_predictions = tf.argmax(correct_ending_probs, axis=1)

        with tf.name_scope("optimizer"):
            self._build_optimizer()
            gradients = self.optimizer.compute_gradients(self.loss)
            clipped_gradients = [(tf.clip_by_norm(gradient, self.clip_norm), var) for gradient, var in gradients]
            self.train_op = self.optimizer.apply_gradients(clipped_gradients, global_step=self._get_tf_object("GlobalStep"))

            variables = tf.trainable_variables()

    def _build_optimizer(self, optimizer=None):
        if optimizer is None:
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.optimizer = optimizer

    def _build_model_graph(self, mode="train"):
        """Sets up the computational graph in the model."""
        with tf.variable_scope(self.__class__.__name__, reuse=tf.AUTO_REUSE):
            self._compute_loss(mode=mode)

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

    def _evaluate_batch(self, eval_sentences):
        """Computes metrics on eval batches."""
        eval_story = eval_sentences[:, :self.n_story_sentences]
        eval_ending1 = eval_sentences[:, self.n_story_sentences]
        eval_ending2 = eval_sentences[:, self.n_story_sentences + 1]

        eval_ending1 = np.expand_dims(eval_ending1, axis=1)
        eval_ending2 = np.expand_dims(eval_ending2, axis=1)

        feed_dict = {}
        if self.input_mode == "full_context":
            feed_dict = {self.eval_input: eval_story, self.train_input: eval_story}

        elif self.input_mode == "last_sentence":
            feed_dict = {self.eval_input: eval_story[:, -1], self.train_input: eval_story[:, -1]}

        feed_dict[self.eval_ending1] = eval_ending1
        feed_dict[self.eval_ending2] = eval_ending2

        fetches = [self.eval_predictions, self.eval_probs1, self.eval_probs2]
        results = self._get_tf_object("Session").run(fetches=fetches, feed_dict=feed_dict)

        return results

    def _train_batch(self, train_batch, add_summary=False, verbose=False):
        """Runs the training on every batch."""
        encoded_train, train_labels = train_batch

        train_story = encoded_train[:, :self.n_story_sentences]
        train_ending = np.expand_dims(encoded_train[:, -1], axis=1)

        fetches = [self.loss,  self.train_op, self.train_predictions, self.train_accuracy]
        if add_summary:
            fetches.append(self.merged_train_summaries)

        assert len(encoded_train.shape) == 3

        if len(train_labels.shape) == 1:
            train_labels = train_labels.reshape(-1, 1)
        train_labels = train_labels.astype(np.int32)

        feed_dict = {}
        if self.input_mode == "full_context":
            feed_dict = {self.train_input: train_story}

        elif self.input_mode == "last_sentence":
            feed_dict = {self.train_input: train_story[:, -1]}

        feed_dict[self.labels_ph] = train_labels
        feed_dict[self.ending_ph] = train_ending
        results = self._get_tf_object("Session").run(fetches=fetches, feed_dict=feed_dict)

        if add_summary:
            timestep = self._get_tf_object("Session").run(self._get_tf_object("GlobalStep"))
            self._get_tf_object("FileWriter").add_summary(results[-1], timestep)
        if verbose:
            logger.info("Loss: {0:.4f}, Train accuracy: {1:.3f}".format(results[0], results[3]))

if __name__ == "__main__":
    ffn = FFN()
