
## Get get_updated_crops;
## Get Mini_cut_words;
## Get Updated_crops;

from data_preprocess import get_updated_crops
from data_preprocess import get_mini_cut_words
from data_preprocess import get_mini_vectors
from data_preprocess import get_cms_news
import logging
import os

import time


def clear_file(file_list, dir=None):
    for file in file_list:
        if dir:
            path = os.path.join(dir, file)
        else:
            path = file
        if os.path.exists(path):
            logging.info('clearing...{}'.format(path))
            os.remove(path)


def get_backup_filename(directory, begin, end):
    file_name = "{}-{}.pickle".format(begin, end) if begin != end else begin + '.pickle'
    return os.path.join(directory, file_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    ratio = 1
    train_vector_config = {
        'embedding_size': 150,  # the last is 100
        'context_size': 5,
        'learning_rate': 0.0015,
        'sample': 0.20 * 5 * ratio,
        'force_reload': True,
        'regularization': 0.001,
        'batch_size': int(10000 * 5 * ratio),
        'max_vocab_size': 20000,
        'epoch': 50,
        'test_mode': False,
        'words_frequency': 'word_frequence/occurence_maxtrix_20000.pickle',
        'config_file': 'config/config.conf',
    }

    get_mini_vectors.get_words_vector(
        vector_config=train_vector_config,
        cut_crops='cutted_words/clear_wiki_and_news.txt',
        words_count='word_frequence/sorted_wiki_and_news_wc.txt',
        backup_pickle='word_vectors/wiki_and_news_glove_0.2loss.pickle',
    )
