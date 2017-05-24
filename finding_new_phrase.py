import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
'''
Use Word Vector way to get new phrase. 

First: we get all the word vectors of all the words, use a big crops;

Second: we get a sub word vectors from a mini crops which contains the new information updated;

Third: For any words pair W1, W2 we get the distance of those two pair in big corps and updated crops;

Forth: If the distance if more closed to the distance in big crops, we could classify this two words as a new
phrase.

Input is words, and these words's pos.
X_i = [glovel_vector; latest_vector; pos]

[X_i, X_[i+1]] shape = (2, D)

(2, D) * (D, W) = (2, W)
(3, D) * (D, W) = (3, W)

(2, W) * (W_i, W_j) = (2, W_j)

RNN
'''

import pickle
import jieba
import operator
from scipy import spatial

global_vector = pickle.load(open('data_preprocess/total_word_vectors_2.pickle', 'rb'))
mini_vector = pickle.load(open('data_preprocess/updated_news/latest_vectors.pickle', 'rb'))
global_word_to_id = {word: ID for ID, word in global_vector['id_to_word'].items()}
mini_word_to_id = {word: ID for ID, word in mini_vector['id_to_word'].items()}


def get_w2v(word, vector, word2id):
    if word in word2id:
        return vector['embedding'][word2id[word]]
    else:
        print('no *{}* in embedding'.format(word))
        return None


def get_vector(word, vector='global'):
    if vector == 'global':
        return get_w2v(word, global_vector, global_word_to_id)
    elif vector == 'latest':
        return get_w2v(word, mini_vector, mini_word_to_id)
    else:
        raise NameError('wrong vector name')


def distance(v1, v2):
    if v1 is not None and v2 is not None:
        return spatial.distance.cosine(v1, v2)
    else:
        return float('inf')


def get_new_phrase_probability(previous, word):
    GLOBAL, LATEST = 'global', 'latest'
    distance1 = distance(get_vector(previous, GLOBAL), get_vector(word, GLOBAL))
    distance2 = distance(get_vector(previous, LATEST), get_vector(word, LATEST))
    return (distance1 - distance2) / distance1


def analyse_new_phrase(sentence):
    '''
    :param sentence:  A input sentence to test if there is a new phrase.
    :return:  The new phrases sorted by the probability.
    '''

    cut_words = jieba.cut(sentence)

    previous = None

    new_phrases = {}
    for index, word in enumerate(cut_words):
        if index == 0:
            previous = word
            continue

        new_phrase_prob = get_new_phrase_probability(previous, word)
        new_phrases[(previous+word)] = new_phrase_prob
        previous = word

    return sorted(new_phrases.items(), key=operator.itemgetter(1), reverse=True)


if __name__ == '__main__':
    test_sentences ='''2015年7月， 今年以来我省主要市场进出口稳定，对“一带一路”沿线国家进出口额达2380.7亿元，同比增长19%，其中，对俄罗斯、伊朗、波兰、伊拉克的出口增速同比分别达到30%、34.1%、23.6%和28.5%，成为一大亮点。'''
    print(list(jieba.cut(test_sentences)))
    new_phrase_probs = analyse_new_phrase(test_sentences)

    for phrase in new_phrase_probs:
        print(phrase)

