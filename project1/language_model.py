import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import xavier_initializer as xav_init
from load_embeddings import load_embedding
from dataset import Dataset
import time

SEED = 42

class LanguageModel(object):
    built = False

    def __init__(self,
                 dataset,
                 lstm_hidden_size,
                 pretrained=False,
                 embedding_size=100,
                 project_size=512,
                 project=False,
                 restore_from=None,
                 save_dir=None,
                 log_dir=None):
        """
        Parameters
        ----------
        dataset: Dataset,
            Dataset instance holding train, test, eval datasets
        lstm_hidden_size: int,
            Number of hidden units in the LSTM
        pretrained: bool, default False
            Whether to use pretrained embeddings
        project: bool, False
            Whether to project after using larger LSTM
        project_size: int, default 512
            Final size to project to
        restore_from: str, default None
            Path to restore model from
        save_dir: str, default None
            Directory to save model to
        log_dir: str, default None
            Directory to write summaries to
        """
        graph = tf.Graph()
        graph.seed = SEED
        self.dataset = dataset
        self.lstm_hidden_size = lstm_hidden_size
        self.embedding_size = embedding_size
        self.project = False
        if project:
            self.project_size = project_size
        self.session = tf.Session(graph=graph)
        self.len_corpus = len(dataset.vocab)
        self.time_steps = dataset.train.shape[1] -1

        with self.session.graph.as_default():
            self._placeholders()
            self._embeddings()
            self._savers(log_dir=log_dir)
            self._summaries()
            self._compute_cross_entropy_loss()
            self._compute_perplexity()
            self._optimizer()

            if restore_from is not None:
                self.saver.restore(self.session, restore_from)
            else:
                self.session.run(tf.global_variables_initializer())

    def _placeholders(self):
        """Creates placeholders to be used for sentences and words."""
        self.sentence_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.time_steps + 1], name="Sentence_placeholder")
        self.bs_ph = tf.placeholder(dtype=tf.int32, shape=[], name="Batch_size_placeholder")
        self.train_loss_ph = tf.placeholder(tf.float32)
        self.eval_loss_ph = tf.placeholder(tf.float32)
        self.perplexity_ph = tf.placeholder(tf.float32)

    def _savers(self, log_dir=None):
        """Creates saver and summary writer.

        Parameters
        ----------
        log_dir: str, default None
            Directory to log results to
        """
        self.summary_writer = tf.summary.FileWriter(log_dir)
        self.summary_writer.add_graph(self.session.graph)
        self.saver = tf.train.Saver()

    def _embeddings(self, pretrained=False, scope_name=None):
        """Compute word embeddings for sentence.

        Parameters
        ----------
        pretrained: bool, default False
            Whether to use pretrained embeddings
        scope_name: str, default None
            Variable scope
        """
        if not scope_name:
            scope_name = "Embedding"

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.len_corpus, self.embedding_size],
                initializer=xav_init()
            )

            if pretrained:
                load_embedding(session=self.session,
                               vocab=self.dataset.vocab,
                               emb=self.embedding_matrix,
                               path=self.dataset.embedding_file,
                               vocab_size=self.len_corpus,
                               dim_embedding=self.embedding_size)

            self.word_embeddings = tf.nn.embedding_lookup(self.embedding_matrix,
                                                          self.sentence_ph)

    def _build_rnn(self, trainable_zero_state=False, scope_name=None):
        """Sets up the LSTM and its unrolling."""
        if not scope_name:
            scope_name = "LSTM"

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.lstm = BasicLSTMCell(num_units=self.lstm_hidden_size)
            if not trainable_zero_state:
                state = self.lstm.zero_state(batch_size=self.bs_ph, dtype=tf.float32)
            else:
                state = self._trainable_zero_state()
            if self.project:
                self._projection_layer()
            self._unroll_lstm(state=state)
            self._output_layer()
        self.built = True

    def _projection_layer(self, scope_name=None):
        """Creates the weight matrix for projection, when a larger LSTM is used."""
        if scope_name is None:
            scope_name = "Projection"

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.project_W = tf.get_variable(name="proj_weights",
                shape=[self.lstm_hidden_size, self.project_size],
                dtype=tf.float32,
                initializer=xav_init()
            )

    def _unroll_lstm(self, state):
        """Unrolls the LSTM."""
        outputs = list()
        for time_step in range(self.time_steps):
            out, state = self.lstm(self.word_embeddings[:, time_step, :], state)
            out = tf.reshape(out, [-1, 1, self.lstm_hidden_size])
            outputs.append(out)
        self.output = tf.concat(outputs, axis=1)

        if self.project:
            self.output = tf.tensordot(self.output, self.project_W, axes=1)

    def _output_layer(self, scope_name=None):
        """Self explanatory."""
        if scope_name is None:
            scope_name = "Output_layer"
        if self.project:
            shape = [self.project_size, self.len_corpus]
        else:
            shape = [self.lstm_hidden_size, self.len_corpus]

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.output_layer = dict()
            self.output_layer['weights'] = tf.get_variable(name="weights",
                shape=shape, dtype=tf.float32, initializer=xav_init())

            self.output_layer['bias'] = tf.get_variable(name='bias',
                shape=[self.len_corpus], dtype=tf.float32, initializer=xav_init()
            )

    # TODO(vsomnath): Loss needs to be fixed to include masking of <pad> tokens
    def _compute_cross_entropy_loss(self):
        """Computes the loss for the LSTM."""
        if not self.built:
            self._build_rnn()
        logits = tf.tensordot(self.output, self.output_layer["weights"], axes=1) # 64 x 29 x 20'000
        logits = tf.add(logits, self.output_layer["bias"]) # bias: 20'000

        # Add mask for pads
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=self.sentence_ph[:,1:]
        ) # 64 x 29
        self.avg_cross_entropy = tf.reduce_mean(self.cross_entropy, axis=1) # 64, cross entropy for every sentence
        self.loss = tf.reduce_mean(self.avg_cross_entropy) # 1, avg cross entropy. Trains also over <pad> tokens

    # TODO(vsomnath): Fix perplexity computations
    def _compute_perplexity(self):
        """Computes perplexity and average."""
        # Have to work with only non-<pad> tokens, zero out other values
        pads = tf.equal(self.sentence_ph[:,1:], 2)
        pad1s = tf.cast(pads, tf.int32)
        sentence_lengths = self.time_steps - tf.reduce_sum(pad1s, axis=1)
        sentence_lengths = tf.cast(sentence_lengths, dtype=tf.float32)
        # cross entropy, filled with 0s where sentence ends
        filtered_cross_entropy = tf.where(pads, tf.zeros_like(self.cross_entropy), self.cross_entropy) # 64 x 29
        self.perplexity = tf.reduce_sum(filtered_cross_entropy, axis=1)/sentence_lengths # 64
        self.perplexity = tf.exp(self.perplexity) # 64, perplexity for all sentences
        self.perplexity_avg = tf.reduce_mean(self.perplexity) # 1, avg perplexity

    def _summaries(self):
        """Creates summaries to log."""
        self.summary_loss = tf.summary.scalar('loss training', self.train_loss_ph)
        self.summary_loss_eval = tf.summary.scalar('loss evaluation', self.eval_loss_ph)
        self.summary_perplexity = tf.summary.scalar('perplexity evaluation', self.perplexity_ph)

    def _optimizer(self):
        """Defines the optimizer."""
        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.AdamOptimizer()
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
            self.optimize_op = self.optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, timestep=None, verbose=False):
        """Computes loss on the eval dataset."""
        eval_loss = 0
        fetches = self.loss
        n_samples = self.dataset.eval.shape[0]

        for batch in self.dataset.batch_generator(mode="eval", batch_size=n_samples):
            feed_dict = {self.sentence_ph: batch, self.bs_ph: n_samples}
            eval_loss += self.session.run(fetches=fetches, feed_dict=feed_dict)

        fetches = self.summary_loss_eval
        feed_dict = {self.eval_loss_ph: eval_loss / self.dataset.eval.shape[0]}
        logging_eval_loss = self.session.run(fetches=fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(logging_eval_loss, timestep)

        if verbose:
            print("The Evaluation Loss is {0:.3f}".format(eval_loss))

    def fit(self, num_epochs=10, batch_size=64, eval_every=10, verbose=False):
        """Trains the LSTM."""
        start_time = time.time()
        for epoch in range(num_epochs):
            batch_count = 0
            for train_batch in self.dataset.batch_generator(mode="train",
                                            batch_size=batch_size, shuffle=True):
                fetches = [self.loss, self.optimize_op]
                feed_dict = {self.sentence_ph: train_batch, self.bs_ph: batch_size}
                timestep = self.dataset.train.shape[0]/batch_size * epoch + batch_count

                loss, _ = self.session.run(fetches=fetches, feed_dict=feed_dict)
                batch_count += 1
                logging_loss = self.session.run(self.summary_loss, feed_dict={self.train_loss_ph: loss})
                self.summary_writer.add_summary(logging_loss, timestep)

                if (batch_count+1) % 10 == 0:
                    self.evaluate(timestep=timestep, verbose=verbose)

    # TODO(vsomnath): Get rid of this and complete "complete_sentences"
    def _predict_nodes(self):
        # User feeds these states every step. First step: all zeros
        self.state_c = tf.placeholder(tf.float32, [1, self.lstm_hidden_size])
        self.state_h = tf.placeholder(tf.float32, [1, self.lstm_hidden_size])
        state = tf.contrib.rnn.LSTMStateTuple(self.state_c, self.state_h)
        out, self.next_state = self.lstm(self.embedding_word, state)
        if(self.project):
            out = tf.matmul(out, self.projection['Weights'])
        logits = tf.matmul(out, self.output_layer['Weights']) + self.output_layer['Bias']
        self.logits = tf.reshape(logits, [20000])

    def complete_sentence(self, words, max_len):
        pass

    def complete_sentences(self):
        pass

    def compute_perplexity(self, batch):
        fetches = self.perplexity
        feed_dict = {self.sentence_ph: batch}
        return self.session.run(fetches, feed_dict)

    def save_perplexity_to_file(self, filename):
        """Saves perplexity computations to file."""
        with open(filename, "w") as f:
            for test_sentence in self.dataset.batch_generator(mode="test",
            batch_size=1, shuffle=False):
                perplexity = self.compute_perplexity(test_sentence)
               f.write(str(perplexity) + "\n")
