from utlis.glove import tf_glove as glove

import pickle


def get_words_vector(cut_crops, backup_pickle):
    with open(cut_crops, 'rb') as f:
        total_cut_words = pickle.load(f)

    model = glove.GloVeModel(embedding_size=10, context_size=3, learning_rate=0.05)

    model.fit_to_corpus(total_cut_words)

    model.train(num_epochs=1000, log_dir='latest_vec_log')

    with open(backup_pickle, 'wb') as f:
        pickle.dump(model.get_trained_embedding(), f, pickle.HIGHEST_PROTOCOL)

    print('train done!')
