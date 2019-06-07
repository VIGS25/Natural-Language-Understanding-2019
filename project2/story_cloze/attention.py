"""Implements Bahdanau's additive & Luong's Multiplicative Attention Mechanism.

	encoded_states (tf.Tensor): A tensor stack of unrolled (encoder) states.
								Also referred to as memory
	target_state (tf.Tensor): Tensor of the current input with which encoder states are compared.
							  Also referred to as query.
"""
import tensorflow as tf


class Attention(object):
    def __init__(self, att_type: str, num_units: int, memory_length: int = 4):
        super(Attention, self).__init__()
        self.score_fn = getattr(self, att_type)
        self.num_units = num_units

    def __call__(self, encoded_states: tf.Tensor, target_state: tf.Tensor):
        expanded_target_state = tf.expand_dims(target_state, 1)

        # Shape of encoded_states: (batch_size, memory_length, num_units)
        # Shape of target_state: (batch_size, 1, num_units)

        scores = self.score_fn(encoded_states, expanded_target_state)
        attention_weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(attention_weights * encoded_states, axis=1)
        final_embedding = tf.concat([context, target_state], axis=-1)
        return final_embedding

    def additive(self, encoded_states: tf.Tensor, target_state: tf.Tensor):
        memory_layer = tf.layers.dense(inputs=encoded_states,
                                       units=self.num_units,
                                       use_bias=False,
                                       activation=None)
        query_layer = tf.layers.dense(inputs=target_state,
                                      units=self.num_units,
                                      use_bias=False,
                                      activation=None)
        additive_input = tf.add(query_layer, memory_layer)
        scaled_ouput = tf.layers.dense(inputs=tf.nn.tanh(additive_input),
                                       units=1,
                                       use_bias=False,
                                       activation=None)
        return scaled_ouput

    def multiplicative(self, encoded_states: tf.Tensor,
                       target_state: tf.Tensor):
        memory_layer = tf.layers.dense(encoded_states,
                                       self.num_units,
                                       use_bias=False,
                                       activation=None)
        return tf.transpose(
            tf.matmul(target_state, tf.transpose(memory_layer, (0, 2, 1))),
            (0, 2, 1))
