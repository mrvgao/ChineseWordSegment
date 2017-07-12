from utlis.redis_manager import get_word_vector
from utlis.redis_manager import save_word_vector
import word2vec
import logging


def save():
    model = word2vec.load('data/vec_google.bin')

    logging.info('loading word vectors to redis')

    for index, word in enumerate(model.vocab):
        save_word_vector(word, model[word])

        if index % 1000 == 0 : logging.info(index)

