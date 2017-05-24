
## Get get_updated_crops;
## Get Mini_cut_words;
## Get Updated_crops;

from data_preprocess import get_updated_crops
from data_preprocess import get_mini_cut_words
from data_preprocess import get_mini_vectors
import logging
import os


def clear_file(dir, file_list):
    for file in file_list:
        path = os.path.join(dir, file)
        if os.path.exists(path):
            logging.info('clearing...{}'.format(path))
            os.remove(path)


def get_vectors_from_scratch(content_dir, force=False):
    content_dir = content_dir or 'test_w2v'
    updated_news = 'updated_news.data'
    cut_crops = 'cut_crops.pickle'
    backup_pickle = 'latest_vectors.pickle'

    if force:
        clear_file(content_dir, [updated_news, cut_crops, backup_pickle])

    get_updated_crops.save(crops_dir=content_dir,
                           updated_news_file_backup=os.path.join(content_dir, updated_news))
    get_mini_cut_words.cut_all_words(content_file=os.path.join(content_dir, updated_news),
                                     cut_crops_save_file=os.path.join(content_dir, cut_crops))
    get_mini_vectors.get_words_vector(cut_crops=os.path.join(content_dir, cut_crops),
                                      backup_pickle=os.path.join(content_dir, backup_pickle))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    get_vectors_from_scratch('updated_news', force=True)
