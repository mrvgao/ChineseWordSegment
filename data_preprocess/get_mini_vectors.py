from utlis.glove import tf_glove as glove
import numpy as np
import time
import pickle
import logging


def get_lines(file_name, delimiter='\t'):
    return (line.split(delimiter) for line in open(file_name))


def get_occurence_matrix(matrix_file, sample, vocab_length):
    logging.info("loading {}".format(matrix_file))
    start = time.time()
    with open(matrix_file, 'rb') as f:
        data = pickle.load(f)
        data = data[data[:, 0] < vocab_length]
        data = data[data[:, 1] < vocab_length]
        indices = np.random.choice(range(len(data)), size=int(len(data)*sample), replace=True)
        choosen_data = data[indices]
        del data

    end = time.time()
    print('get data with length: {}'.format(len(choosen_data)))
    print("load occurence used time: {}".format(end-start))

    return choosen_data


def get_words_vector(vector_config, cut_crops, words_count, backup_pickle):
    epoch = vector_config['epoch']
    del vector_config['epoch']

    words_count = get_lines(words_count)

    if 'words_frequency' in vector_config:
        words_frequency_file = vector_config['words_frequency']
        del vector_config['words_frequency']
        model = glove.GloVeModel(**vector_config)
        sample = vector_config['sample']
        vocab_length = vector_config['max_vocab_size']
        model.initial_words_frequency(get_occurence_matrix(words_frequency_file, sample, vocab_length), words_count)
    else:
        model = glove.GloVeModel(**vector_config)
        total_cut_words = get_lines(cut_crops, delimiter='\t')
        model.fit_to_corpus(total_cut_words, words_count)

    ## noting embedding_size for cms_news 0.5,  10 is so small.

    model.train(num_epochs=epoch, log_dir='latest_vec_log')

    with open(backup_pickle, 'wb') as f:
        pickle.dump(model.get_trained_embedding(), f, pickle.HIGHEST_PROTOCOL)

    print('train done!')
