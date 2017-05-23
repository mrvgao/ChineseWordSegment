from utlis.glove import tf_glove as glove
import jieba
import pickle

import logging
logging.basicConfig(level=logging.INFO)

total_cut_words = []

with open('all_regions', 'rb') as f:
    total_cut_words = pickle.load(f)

logging.info('pickle loaded done!')

model = glove.GloVeModel(embedding_size=250, context_size=10)

logging.debug(total_cut_words)
model.fit_to_corpus(total_cut_words)
model.train(num_epochs=20, log_dir='train_log')

logging.info('Get {} words'.format(len(model.embeddings)))

model_pickle = 'total_word_vectors.pickle'

with open(model_pickle, 'wb') as f:
    pickle.dump(model.get_trained_embedding(), f, pickle.HIGHEST_PROTOCOL)

logging.info('pick done!')
