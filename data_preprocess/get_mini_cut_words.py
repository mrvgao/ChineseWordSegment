from utlis.glove import tf_glove as glove
import jieba
import pickle
import logging

logging.basicConfig(level=logging.INFO)


def cut_all_words(content_file, cut_crops_save_file):
    sentences = []
    total_cut_word = [list(jieba.cut(line)) for line in open(content_file).readlines()]
    with open(cut_crops_save_file, 'wb') as f:
        pickle.dump(total_cut_word, f, pickle.HIGHEST_PROTOCOL)

    print('cut done!')


