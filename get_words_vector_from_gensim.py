from gensim.models.word2vec import Word2Vec
import time
import pickle


def train_word2vec(file):
    with open(file, 'rb') as f:
        sentences = pickle.load(f)

    model = Word2Vec(sentences, size=250, window=5, min_count=1, workers=4)

    persist = 'data_preprocess/word_vectors/gensim_hudong_wiki_250.pickle'
    # persist = 'data_preprocess/word_vectors/gensim_fm_0524.pickle'
    model.save(persist)

if __name__ == '__main__':
    begin = time.time()
    train_word2vec('data_preprocess/cutted_words/train_contentpickle')
    end = time.time()
    print('use time {} s'.format(end - begin))
    print('get vector done!')
