import argparse
from data_preprocess import get_mini_vectors

FLAGS = None


def init_path():
    cut_crops = FLAGS.buckets
    pickle_path = FLAGS.pickle_path

    get_mini_vectors.get_words_vector(cut_crops=cut_crops, backup_pickle=pickle_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='', help='input date path')
    parser.add_argument('--pickle_path', type=str, default='', help='input embedding pickle path')

    FLAGS, _ = parser.parse_known_args()
