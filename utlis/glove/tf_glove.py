from __future__ import division
from collections import Counter, defaultdict
import os
from random import shuffle
import tensorflow as tf
import logging
import pickle
import os
import random
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
from data_preprocess.utils import update_cooccurrence

class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class GloVeModel():
    def __init__(self, embedding_size, context_size, max_vocab_size=100000, min_occurrences=1,
                 scaling_factor=3.0/4.0, cooccurrence_cap=100, batch_size=256,
                 learning_rate=0.05, regularization=0.005, sample=0.5, force_reload=False, config_file=None, test_mode=False):
        self.embedding_size = embedding_size
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.__words = None
        self.__word_to_id = None
        self.__cooccurrence_matrix = None
        self.__embeddings = None
        self.id_words = {}  ## {1: 'some'}
        self.occurance_file = 'occurance.pickle'
        self.id_to_words = {}
        self.regularization = regularization
        self.sample = sample
        self.force_reload = force_reload
        self.vis_labels = 'latest_vec_log/metadata.tsv'
        self.need_write = False
        self.test_mode=test_mode
        self.config_file = config_file

    def initial_words_frequency(self, coocurrance_records, words_count):
        self.__get_words_count(words_count, self.max_vocab_size)
        self.__get_coocurrance_matrix(coocurrance_records)
        self.__build_graph()

    def fit_to_corpus(self, corpus, words_count):
        if not self.force_reload and os.path.exists(self.occurance_file):
            occurence_data = pickle.load(open(self.occurance_file, 'rb'))
            self.__word_to_id = occurence_data['words_to_id']
            self.__words = occurence_data['words']
            self.__cooccurrence_matrix = occurence_data['cooccurrence_matrix']
            self.id_to_words = occurence_data['id_to_words']
            logging.info('load pickle finished!')

            # with open(self.vis_labels, 'w') as f:
            #     f.write('Index\tLabel\n')
            #     for index, word in self.id_to_words.items():
            #         f.write("{}\t{}\n".format(index, word))
        else:
            if os.path.exists(self.occurance_file):
                os.remove(self.occurance_file)
                logging.info('delete occurence file')

            self.__get_words_count(words_count, self.max_vocab_size)
            self.__fit_to_corpus(corpus, self.left_context, self.right_context)

        self.__build_graph()

    def __get_words_count(self, word_counts, vocab_size):
        self.__words = []
        self.__word_to_id = {}
        self.id_to_words = {}
        if self.test_mode: vocab_size /= 10
        for index, word_count_fre in enumerate(word_counts):
            if index >= vocab_size: break
            if index % 1000 == 0: print('words counts:{}'.format(index))
            word, count, _ = word_count_fre
            self.__words.append(word)
            self.__word_to_id[word] = index
            self.id_to_words[index] = word

    @staticmethod
    def write_to_occurence_file(occurences : dict, file):
        for key, count in occurences.items():
            word1, word2 = key
            file.write("\t".join([str(word1), str(word2), str(count)])+'\n')

    def __get_coocurrance_matrix(self, coocurrance_records):
        # index = 0
        # self.__cooccurrence_matrix = []
        # for word1, word2, count in coocurrance_records:
        #     if random.random() > self.sample: continue
        #     print(index)
        #     self.__cooccurrence_matrix.append((int(word1), int(word2), float(count)))
        #     index += 1
        #
        # self.__cooccurrence_matrix = np.array(self.__cooccurrence_matrix)
        self.__cooccurrence_matrix = coocurrance_records

    def __fit_to_corpus(self, corpus, left_size, right_size):
        cooccurrence_counts = defaultdict(float)
        calculate_num = 0
        occurence_file = open('occurence.txt', 'w')
        # occ = update_cooccurrence.Occurence(':memory:', 'cooccurrence')
        OCC_NUM = 5000
        flush_num = 0
        occurence_num = OCC_NUM
        for index, region in enumerate(corpus):
            logging.info("{}".format(index))

            if self.test_mode and index > 100000: break

            if random.random() > self.sample:
                continue

            if self.sample < 1:
                if calculate_num >= int(1570000 * self.sample): break
                else: calculate_num += 1

            for l_context, word, r_context in _context_windows(region, left_size, right_size):
                logging.debug(word)
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    logging.debug("p2:"+word)
                    if word in self.__word_to_id and context_word in self.__word_to_id:
                        # occ.accumulate(word, context_word, 1/(i+1))
                        cooccurrence_counts[(self.__word_to_id[word], self.__word_to_id[context_word])] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    if word in self.__word_to_id and context_word in self.__word_to_id:
                        cooccurrence_counts[(self.__word_to_id[word], self.__word_to_id[context_word])] += 1 / (i + 1)
                        # occ.accumulate(word, context_word, 1/(i+1))

            occurence_num -= 1

            if occurence_num <= 0:
                print('flush to file.. {}'.format(flush_num))
                GloVeModel.write_to_occurence_file(cooccurrence_counts, occurence_file)
                cooccurrence_counts = defaultdict(float)
                occurence_num = OCC_NUM
                flush_num += 1

        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")

        # self.__cooccurrence_matrix = np.array([
        #     (self.__word_to_id[words[0]], self.__word_to_id[words[1]], count)
        #     for words, count in cooccurrence_counts.items()
        # ])

        # logging.info("write coocurrence, length: {}".format(len(self.__cooccurrence_matrix)))

        # with open(self.occurance_file, 'wb') as f:
        #     dump_data = {
        #         'words': self.__words,
        #         'words_to_id': self.__word_to_id,
        #         'id_to_words': self.id_to_words,
        #         'cooccurrence_matrix': self.__cooccurrence_matrix
            # }
            # pickle.dump(dump_data, f, pickle.HIGHEST_PROTOCOL)

    def __build_graph(self):
        self.__graph = tf.Graph()
        with self.__graph.as_default(), self.__graph.device(_device_for_node):
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32,
                                    name='max_cooccurrence_count')
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32,
                                         name="scaling_factor")

            self.__focal_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                name="focal_words")
            self.__context_input = tf.placeholder(tf.int32, shape=[self.batch_size],
                                                  name="context_words")
            self.__cooccurrence_count = tf.placeholder(tf.float32, shape=[self.batch_size],
                                                       name="cooccurrence_count")

            focal_embeddings = tf.get_variable(name='focal_embeddings',
                                               shape=[self.vocab_size, self.embedding_size],
                                               # initializer=tf.random_uniform_initializer(minval=-0.2, maxval=0.2))
                                               initializer=tf.contrib.layers.xavier_initializer())
                                               # initializer=tf.truncated_normal_initializer(stddev=-0.5))

            context_embeddings = tf.get_variable(name='context_embeddings',
                                                 shape=[self.vocab_size, self.embedding_size],
                                                 # initializer=tf.random_uniform_initializer(minval=-0.2, maxval=0.2))
                                                 initializer=tf.contrib.layers.xavier_initializer())
                                                 # initializer=tf.truncated_normal_initializer(stddev=-0.5))

            focal_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                       name='focal_biases')
            context_biases = tf.Variable(tf.random_uniform([self.vocab_size], 1.0, -1.0),
                                         name="context_biases")

            focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.__focal_input)
            context_embedding = tf.nn.embedding_lookup([context_embeddings], self.__context_input)
            focal_bias = tf.nn.embedding_lookup([focal_biases], self.__focal_input)
            context_bias = tf.nn.embedding_lookup([context_biases], self.__context_input)

            weighting_factor = tf.minimum(
                1.0,
                tf.pow(
                    tf.div(self.__cooccurrence_count, count_max),
                    scaling_factor))

            embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

            log_cooccurrences = tf.log(tf.to_float(self.__cooccurrence_count))

            distance_expr = tf.square(tf.add_n([
                embedding_product,
                focal_bias,
                context_bias,
                tf.negative(log_cooccurrences)]))

            single_losses = tf.multiply(weighting_factor, distance_expr)

            if self.regularization > 0 and random.random() < .3:
                # regularizer = tf.contrib.layers.l1_regularizer(scale=self.regularization)
                # regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, focal_embedding)
                # reg = tf.reduce_sum(tf.losses.l1_loss(focal_embeddings))
                # self.need_write = False
                # regularization_loss = 1/2 * self.regularization * reg
                regularization_loss = tf.reduce_sum(tf.nn.l2_loss(focal_embeddings)) \
                                      # + tf.reduce_sum(tf.nn.l2_loss(context_embeddings)) \
                                      # + tf.reduce_sum(tf.nn.l2_loss(focal_biases)) \
                                      # + tf.reduce_sum(tf.nn.l2_loss(context_biases))
                regularization_loss = 1/2 * self.regularization * regularization_loss
                # regularization_loss = self.regularization * tf.reduce_sum(tf.abs(focal_embedding))
            else:
                regularization_loss = 0
            self.need_write = True

            # self.__total_loss = tf.reduce_sum(single_losses) + regularization_loss
            self.__total_loss = tf.reduce_mean(single_losses) + regularization_loss
            tf.summary.scalar("GloVe_loss", self.__total_loss)
            tf.summary.histogram('focal_embedding', focal_embeddings)

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()

            global_step = tf.Variable(0, trainable=False)
            #
            learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 2000, 0.95, staircase=True)

            self.__optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.__total_loss)
            # self.__optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.__total_loss)
            self.merged = tf.summary.merge_all()

            self.__combined_embeddings = tf.add(focal_embeddings, context_embeddings,
                                                name="combined_embeddings")

    def train(self, num_epochs, log_dir=None, summary_batch_interval=300,
              tsne_epoch_interval=None):

        # TODO: Implement Early Stop

        embedding_visualiza_interval = 500000
        should_write_summaries = log_dir is not None and summary_batch_interval
        should_generate_tsne = log_dir is not None and tsne_epoch_interval
        total_steps = 0
        #
        coccurrence_num = len(self.__cooccurrence_matrix)
        batches_num = coccurrence_num // self.batch_size

        previous_loss = 0

        with tf.Session(graph=self.__graph) as session:
            if should_write_summaries:
                summary_writer = tf.summary.FileWriter(log_dir, graph=session.graph)
            tf.global_variables_initializer().run()
            logging.info('initial variables')
            epoch = 0
            target_epoch_num = num_epochs
            while epoch < target_epoch_num:
            # for epoch in range(num_epochs):
                print('epoch: {}.'.format(epoch))
                # shuffle(batches)
                batches = self.__prepare_batches()
                for batch_index, batch in enumerate(batches):
                    # indices = np.random.choice(range(coccurrence_num), size=self.batch_size, replace=True)
                    # choosen_pairs = self.__cooccurrence_matrix[indices]
                    # i_s, j_s, counts = choosen_pairs[:, 0], choosen_pairs[:, 1], choosen_pairs[:, 2]
                    i_s, j_s, counts = batch
                    if len(counts) != self.batch_size:
                        continue
                    # logging.debug(self.__words[int(i_s[0])], ' ', self.__words[int(j_s[0])])
                    feed_dict = {
                        self.__focal_input: i_s,
                        self.__context_input: j_s,
                        self.__cooccurrence_count: counts}
                    _ = session.run([self.__optimizer], feed_dict=feed_dict)
                    if self.need_write and should_write_summaries and (total_steps + 1) % summary_batch_interval == 0:
                        L = session.run([self.__total_loss], feed_dict=feed_dict)
                        logging.info('{}/{} loss == {}'.format(batch_index, batches_num, L))
                        summary_str = session.run(self.merged, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, total_steps)

                    if total_steps % embedding_visualiza_interval == 0:
                        saver = tf.train.Saver()
                        saver.save(session, os.path.join(log_dir, 'model.ckpt'), total_steps)

                    total_steps += 1
                # if should_generate_tsne and (epoch + 1) % tsne_epoch_interval == 0:
                #     current_embeddings = self.__combined_embeddings.eval()
                #     output_path = os.path.join(log_dir, "epoch{:03d}.png".format(epoch + 1))
                #     self.generate_tsne(output_path, embeddings=current_embeddings)

                epoch += 1
                target_epoch_num = self.get_target_epoch_num() or num_epochs

            self.__embeddings = self.__combined_embeddings.eval()
            if should_write_summaries:
                summary_writer.close()

    def get_target_epoch_num(self):
        epoch_num_file = self.config_file
        epoch = None

        if epoch_num_file is not None:
            with open(epoch_num_file) as f:
                for line in f:
                    name, num = line.split(':')
                    if name.strip() == 'epoch':
                        epoch = int(num.strip())
                        break
        return epoch

    def embedding_for(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.embeddings[self.__word_to_id[word_str_or_id]]
        elif isinstance(word_str_or_id, int):
            return self.embeddings[word_str_or_id]

    def __prepare_batches(self):
        if self.__cooccurrence_matrix is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        # cooccurrences = [(word_ids[0], word_ids[1], count)
        #                  for word_ids, count in self.__cooccurrence_matrix.items()]
        # i_indices, j_indices, counts = zip(*cooccurrences)
        return _batchify(self.batch_size, self.__cooccurrence_matrix)

    @property
    def vocab_size(self):
        return len(self.__words)

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return self.__word_to_id[word]

    def generate_tsne(self, path=None, size=(100, 100), word_count=1000, embeddings=None):
        if embeddings is None:
            embeddings = self.embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embeddings[:word_count, :])
        labels = self.words[:word_count]
        return _plot_with_labels(low_dim_embs, labels, path, size)

    def get_trained_embedding(self):
        embedding_result = {'embedding': self.embeddings}

        return embedding_result


def _context_windows(region, left_size, right_size):
    logging.debug('region == {}'.format(region))
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def _device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


def _batchify(batch_size, matrix):
    # np.random.shuffle(matrix)
    indices = np.random.choice(range(len(matrix)), size=len(matrix), replace=True)
    for i in range(0, len(indices), batch_size):
        sub_indices = indices[i: i+batch_size]
        sub_matrix = matrix[sub_indices]
        yield tuple(sub_matrix[:, i] for i in range(3))


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)
