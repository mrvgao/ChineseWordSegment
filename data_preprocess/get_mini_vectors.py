from utlis.glove import tf_glove as glove

import pickle


def get_total_cut_words(cut_files):
    split = '\t'
    return (line.split(split) for line in open(cut_files).readlines())


def get_words_vector(cut_crops, backup_pickle, force_reload=False):
    total_cut_words = get_total_cut_words(cut_crops)
    print('load file data done!')

    config = {
        'embedding_size': 100,
        'context_size': 3,
        'learning_rate': 0.015,
        'sample': 0.2,
        'force_reload': force_reload,
        'regularization': 0.03,
        'batch_size': 256,
    }
    model = glove.GloVeModel(**config)
    ## noting embedding_size for cms_news 0.5,  10 is so small.

    model.fit_to_corpus(total_cut_words)

    model.train(num_epochs=60, log_dir='latest_vec_log')

    with open(backup_pickle, 'wb') as f:
        pickle.dump(model.get_trained_embedding(), f, pickle.HIGHEST_PROTOCOL)

    print('train done!')
