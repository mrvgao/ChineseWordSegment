import pickle
from utlis.redis_manager import save_word_count_frequency
import logging

logging.basicConfig(level=logging.INFO)


def save():
    count_frequency_file = 'data/words_count_and_frequency.pickle'

    test_mode = False

    with open(count_frequency_file, 'rb') as f:
        word_frequency = pickle.load(f)

    logging.info('loading words frequencies')

    for index, word in enumerate(word_frequency):
        frequency = word_frequency[word][1]
        save_word_count_frequency(word, frequency)
        logging.debug(word)
        if index % 1000 == 0: logging.info(index)

        if test_mode and index > 1000: break

