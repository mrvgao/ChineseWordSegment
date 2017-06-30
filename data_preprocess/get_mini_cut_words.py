from utlis.glove import tf_glove as glove
from utlis.cut import cut
import pickle
import logging

logging.basicConfig(level=logging.INFO)


def get_cut_lines(f):
    get_cut_lines.called = 0

    def get_clean_line(line):
        split = '\t'
        get_cut_lines.called += 1
        if get_cut_lines.called % 100 == 0:
            print(get_cut_lines.called)
        return split.join([w for w in cut(line) if w != split])

    return (get_clean_line(line) for line in f.readlines())


def cut_all_words(content_file, cut_crops_save_file):
    split = '\t'
    backup = open(cut_crops_save_file, 'w')
    with open(content_file) as f:
        backup.writelines(get_cut_lines(f))

    print('cut done!')


if __name__ == '__main__':
    cut_all_words('clearify_content/train_content.txt', 'cutted_words/wiki_and_news.txt')


