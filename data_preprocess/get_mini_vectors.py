from utlis.glove import tf_glove as glove

import pickle


def get_words_vector(cut_crops, backup_pickle, force_reload=False):
    with open(cut_crops, 'rb') as f:
        total_cut_words = pickle.load(f)

    config = {
        'embedding_size': 15,
        'context_size': 3,
        'learning_rate': 0.03,
        'sample': 1.0,
        'force_reload': force_reload,
        'regularization': 0.01,
        'batch_size': 256,
    }
    model = glove.GloVeModel(**config)
    ## noting embedding_size for cms_news 0.5,  10 is so small.

    model.fit_to_corpus(total_cut_words)

    model.train(num_epochs=60, log_dir='latest_vec_log')

    with open(backup_pickle, 'wb') as f:
        pickle.dump(model.get_trained_embedding(), f, pickle.HIGHEST_PROTOCOL)

    print('train done!')
