import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import xavier_initializer as xav_init
from load_embeddings import load_embedding
from dataset import Dataset
import os
import time

SEED = 42

class LanguageModel(object):
    built = False
    sen_comp_setup = False

    def __init__(self,
                 dataset,
                 lstm_hidden_size,
                 pretrained=False,
                 embedding_size=100,
                 project_size=512,
                 project=False,
                 restore_from=None,
                 model_dir=None,
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
        model_dir: str, default None
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
        self.model_dir = model_dir

        with self.session.graph.as_default():
            self._embeddings(pretrained=pretrained)
            self._compute_cross_entropy_loss()
            self._optimizer()
            self._sentence_completion_setup()
            self._savers(log_dir=log_dir)
            self._summaries()

            if restore_from is not None:
                self.saver.restore(self.session, restore_from)
            else:
                self.session.run(tf.global_variables_initializer())

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

        self.sentence_ph = tf.placeholder(dtype=tf.int32, shape=[None, self.time_steps + 1],
                                        name="Sentence_placeholder")

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.len_corpus, self.embedding_size],
                initializer=xav_init()
            )

            if pretrained:
                print("Loading pretrained embeddings...")
                load_embedding(session=self.session,
                               vocab=self.dataset.word_to_idx,
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
            batch_size = tf.shape(self.sentence_ph)[0]
            if not trainable_zero_state:
                state = self.lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
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

    def _compute_cross_entropy_loss(self):
        """Computes the loss for the LSTM. Masks out <pad> tokens from final loss."""
        if not self.built:
            print("Building the RNN Graph...")
            self._build_rnn()
        # Expected shape: 64 x 29 x 20000
        logits = tf.tensordot(self.output, self.output_layer["weights"], axes=1)
        logits = tf.add(logits, self.output_layer["bias"])

        #Expected shape: 64 x 29
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=self.sentence_ph[:,1:]
        )

        # Include a mask that filters out the pad tokens from the loss computation.
        pad_index = self.dataset.word_to_idx["<pad>"]
        # Mask Tensor, with 0s whereever <pad> token is present
        self.not_pads = tf.not_equal(self.sentence_ph[:, 1:], 2)
        self.not_pads = tf.cast(self.not_pads, cross_entropy.dtype)

        self.cross_entropy_masked = tf.multiply(cross_entropy, self.not_pads)
        self.sentence_lengths = tf.reduce_sum(self.not_pads, axis=1)
        # Expected shape: (64, )
        cross_entropy_batch = tf.reduce_sum(self.cross_entropy_masked, axis=1)
        self.batch_loss = cross_entropy_batch / self.sentence_lengths
        self.batch_perplexity = tf.exp(self.batch_loss)
        self.loss_avg = tf.reduce_mean(self.batch_loss)
        self.perplexity_avg = tf.reduce_mean(self.batch_perplexity) # Batch averaged perplexity

    def _summaries(self):
        """Creates summaries to log."""
        # Train summaries
        self.train_loss_summary = tf.summary.scalar('train/batch_averaged_loss', self.loss_avg)
        self.train_perplexity_summary = tf.summary.scalar('train/batch_averaged_perplexity', self.perplexity_avg)
        train_summaries = [self.train_loss_summary, self.train_perplexity_summary]
        self.train_summaries = tf.summary.merge(train_summaries, name="train_summaries")

        # Test summaries
        self.eval_loss_ph = tf.placeholder(tf.float32)
        self.eval_perplexity_ph = tf.placeholder(tf.float32)
        self.eval_loss_summary = tf.summary.scalar('eval/averaged_loss', self.eval_loss_ph)
        self.eval_perplexity_summary = tf.summary.scalar('eval/averaged_perplexity', self.eval_perplexity_ph)
        eval_summaries = [self.eval_loss_summary, self.eval_perplexity_summary]
        self.eval_summaries = tf.summary.merge(eval_summaries, name="eval_summaries")

    def _optimizer(self):
        """Defines the optimizer."""
        with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.AdamOptimizer()
            gradients, variables = zip(*self.optimizer.compute_gradients(self.loss_avg))
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
            self.optimize_op = self.optimizer.apply_gradients(zip(gradients, variables))

    def evaluate(self, batch_size=64, timestep=None, verbose=False):
        """Computes loss and perplexity on the eval dataset."""
        losses, perplexities = [], []
        fetches = [self.batch_loss, self.batch_perplexity]

        for batch in self.dataset.batch_generator(mode="eval", batch_size=batch_size):
            feed_dict = {self.sentence_ph: batch}
            batch_loss, batch_perplexity = self.session.run(fetches=fetches, feed_dict=feed_dict)
            losses.extend(batch_loss)
            perplexities.extend(batch_perplexity)

        mean_eval_loss = np.mean(losses)
        mean_eval_perplexity = np.mean(perplexities)

        fetches = self.eval_summaries
        feed_dict = {self.eval_loss_ph: mean_eval_loss,
                     self.eval_perplexity_ph: mean_eval_perplexity}
        eval_summaries = self.session.run(fetches=fetches, feed_dict=feed_dict)
        self.summary_writer.add_summary(eval_summaries, timestep)

        if verbose:
            print("Evaluation Loss: {0:.3f}".format(mean_eval_loss))
            print("Evaluation Perplexity: {0:.3f}".format(mean_eval_perplexity))

    def fit(self, num_epochs=10, batch_size=64, eval_every=10, verbose=False):
        """Trains the LSTM."""
        start_time = time.time()
        for epoch in range(num_epochs):
            model_dir_epoch = os.path.join(self.model_dir, str(epoch+1))
            if not os.path.exists(model_dir_epoch):
                os.makedirs(model_dir_epoch)

            for n_batch, train_batch in enumerate(self.dataset.batch_generator(mode="train", batch_size=batch_size, shuffle=True)):
                fetches = [self.loss_avg, self.perplexity_avg, self.optimize_op, self.train_summaries]
                feed_dict = {self.sentence_ph: train_batch}
                timestep = self.dataset.train.shape[0]/batch_size * epoch + n_batch

                loss, perplexity, _, train_summaries = self.session.run(fetches=fetches, feed_dict=feed_dict)

                if (n_batch + 1) % eval_every == 0:
                    self.summary_writer.add_summary(train_summaries, timestep)
                    if verbose:
                        print("Epoch {}, Batch: {}".format(epoch+1, n_batch+1))
                        print("Training loss: {0:.3f}".format(loss))
                        print("Training perplexity: {0:.3f}".format(perplexity))

            print("Computing loss and perplexity on eval data. Epoch {}, Timestep: {}".format(epoch+1, timestep))
            self.evaluate(timestep=timestep, verbose=verbose)
            print()

            model_savepath = os.path.join(model_dir_epoch, "model.ckpt")
            save_path = self.saver.save(sess=self.session, save_path=model_savepath)

    def _sentence_completion_setup(self):
        """Setup for the sentence completion task."""
        self.state_c = tf.placeholder(tf.float32, [1, self.lstm_hidden_size])
        self.state_h = tf.placeholder(tf.float32, [1, self.lstm_hidden_size])
        self.word_ph = tf.placeholder(dtype=tf.int32, shape=[1], name="Word_placeholder")
        self.word_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.word_ph)

        state = tf.contrib.rnn.LSTMStateTuple(self.state_c, self.state_h)
        out, self.next_state = self.lstm(self.word_embedding, state)

        if(self.project):
            out = tf.matmul(out, self.projection["weights"])
        logits_word = tf.matmul(out, self.output_layer["weights"])
        logits_word = tf.add(logits_word, self.output_layer["bias"])
        self.logits = tf.reshape(logits_word, [20000])
        self.sen_comp_setup = True

    # TODO (vsomnath): This could still be a little buggy.
    def complete_sentence(self, words, max_len=20):
        """Completes a sentence, given the initial words.

        Parameters
        ----------
        words: list,
            List of starting words
        max_len: int, default 20
            Maximum length of sentence if <eos> is not generated.
        """
        words_copied = words.copy()
        words_copied.insert(0, "<bos>")

        sentence = list()
        state_c = np.zeros((1, self.lstm_hidden_size))
        state_h = np.zeros((1, self.lstm_hidden_size))
        word_predicted = None

        step = 0
        sentence_length = 0
        unk_idx = self.dataset.word_to_idx["<unk>"]

        while (sentence_length < max_len and word_predicted != "<eos>"):
            if sentence_length < len(words_copied):
                word = words_copied[step]
            else:
                word = word_predicted
            word_idx = self.dataset.word_to_idx.get(word, unk_idx)

            fetches = [self.next_state, self.logits]
            word_idx_array = np.array([word_idx])
            feed_dict = {self.word_ph: word_idx_array,
                         self.state_c: state_c,
                         self.state_h: state_h}
            state, logits = self.session.run(fetches, feed_dict)
            state_c, state_h = (state.c, state.h)

            # Decide next word
            logits[0] = np.finfo(float).min
            logits[2:4] = np.finfo(float).min
            word_predicted = self.dataset.idx_to_word[np.argmax(logits)]

            if sentence_length < len(words_copied) - 1:
                sentence.append(words_copied[step + 1])
            else:
                sentence.append(word_predicted)

            step += 1
            sentence_length += 1
        sentence = " ".join([pred_word for pred_word in sentence])
        return sentence

    def complete_sentences(self, data_filename, sol_filename, max_len=20, log_every=100):
        """Completes the sentences in given file.

        Parameters
        ----------
        data_filename: str,
            Filename containing the sentences to complete
        sol_filename: str,
            Filename to write the completed sentence to
        max_len: int, default 20
            Maximum allowed length of sentence.
        """
        if not self.sen_comp_setup:
            self._sentence_completion_setup()

        print("Starting to write sentences...")
        f1 = open(sol_filename, "w")
        f2 = open(data_filename, "r")
        num_lines = 0
        for idx, sentence in enumerate(f2.readlines()):
            words = sentence.split(" ")
            completed_sentence = self.complete_sentence(words, max_len=max_len)
            f1.write(completed_sentence + "\n")
            num_lines += 1
            if num_lines % log_every == 0:
                print("Finished writing {} sentences.".format(num_lines))
        f1.close()
        f2.close()
        print("Finished writing sentences.")

    def compute_perplexity(self, batch):
        """Wrapper function to compute batch perplexity, one for each sentence."""
        fetches = self.perplexity_avg
        feed_dict = {self.sentence_ph: batch}
        return self.session.run(fetches, feed_dict)

    def save_perplexity_to_file(self, filename, log_every=100):
        """Saves perplexity computations to file.

        Parameters
        ----------
        Filename: str,
            File to write perplexity values to
        """
        print("Starting to save perplexity values...")
        with open(filename, "w") as f:
            num_lines = 0
            for idx, test_sentence in enumerate(self.dataset.batch_generator(mode="test", batch_size=1, shuffle=False)):
                perplexity = self.compute_perplexity(test_sentence)
                f.write(str(perplexity) + "\n")
                num_lines += 1
                if num_lines % log_every == 0:
                    print("Finished calculating perplexity for {} sentences.".format(num_lines))
        print("Finished writing perplexity values.")
