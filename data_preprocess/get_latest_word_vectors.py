
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
    file_name = "{}-{}.pickle".format(begin, end) if begin != end else begin + 'pickle'
    return os.path.join(directory, file_name)


def get_vectors_from_scratch(target_news, force_load_news=False, force_reload_occorrence=False, file_type='.csv'):
    if file_type == '.csv':
        content_dir = "updated_news"
    else:
        content_dir = ""

    word_vector_dir = 'word_vectors'
    cut_word_dir = 'cutted_words'
    clearify_content_dir = 'clearify_content'

    vectors = get_backup_filename(word_vector_dir, target_news[0], target_news[-1])
    cut_words = get_backup_filename(cut_word_dir, target_news[0], target_news[-1])
    clearify_content = get_backup_filename(clearify_content_dir, target_news[0], target_news[-1])

    news_file = [string+file_type for string in target_news]
    news_file = [os.path.join(content_dir, f) for f in news_file]

    if force_load_news:
        clear_file([vectors, cut_words, clearify_content])
        get_cms_news.get_multiply_files_content(news_file, clearify_content)
        get_mini_cut_words.cut_all_words(content_file=clearify_content, cut_crops_save_file=cut_words)

    # get_mini_vectors.get_words_vector(cut_crops=cut_words, backup_pickle=vectors, force_reload=force_reload_occorrence)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    config = {
        'force_load_news': True,
        'force_reload_occorrence': False,
        # 'target_news' : ['20170522', '20170523', '20170524']
        'target_news': ['train_content'],
        'file_type': ".txt"
    }

    begin = time.time()

    get_vectors_from_scratch(**config)

    end = time.time()
    print('used time: {} s'.format(end-begin))
