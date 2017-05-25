import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
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
import logging
from data_preprocess.get_cms_news import clearify

global_vector = pickle.load(open('data_preprocess/total_word_vectors_2.pickle', 'rb'))
mini_vector = pickle.load(open('data_preprocess/updated_news/latest_vectors_0523.pickle', 'rb'))
global_word_to_id = {word: ID for ID, word in global_vector['id_to_word'].items()}
mini_word_to_id = {word: ID for ID, word in mini_vector['id_to_word'].items()}

cut = jieba.cut


def get_w2v(word, vector, word2id):
    if word in word2id:
        return vector['embedding'][word2id[word]]
    else:
        logging.info('no *{}* in embedding'.format(word))
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
    if distance1 > distance2:
        return (distance1 - distance2) / distance1, 'new'
    else:
        return distance1, 'old'


def is_phrase(Word1, Word2):
    '''
    Define word1 and word2 word1word2 could be a phrase or not. 
    :param word1: 
    :param word2: 
    :return: True or False
    '''
    decrease_threshold = 0.4
    distance_threshold = 1

    new_phrase_prob = get_new_phrase_probability(Word1.word, Word2.word)

    # if new_phrase_prob[1] == 'new' and new_phrase_prob[0] > decrease_threshold:
    #     return True
    # elif new_phrase_prob[1] == 'old' and new_phrase_prob[0] < distance_threshold:
    #     return True
    # else:
    #     return False

    if new_phrase_prob[1] == 'old':
        if 0 < new_phrase_prob[0] < distance_threshold:
            return True, 'old', new_phrase_prob[0] + 1
        else:
            return False, 'old', new_phrase_prob[0]
    else:
        if new_phrase_prob[0] > decrease_threshold:
            return True, 'new', new_phrase_prob[0]
        else:
            return False, 'new', new_phrase_prob[0]


def create_new_phrase(words_segment):
    """
    :param words_segment: [习近平, 主席] or [习近平]
    :return:  [习近平主席] or None
    """
    words = [x.word for x in words_segment]
    probs = [x.probability for x in words_segment]

    if len(words_segment) <= 1:
        return None
    else:
        return "".join(words), sum(probs[1:])/(len(probs)-1)
        ## if use np.mean directly, when [a, b] is a phrase, if b is probability 0.9,
        ## the average value of this [a, b] we only be 0.45, it's so small.


def collect_new_phrase(detected_new_phrases, new_phrase):
    if new_phrase:
        if detected_new_phrases is None:
            detected_new_phrases = []
        detected_new_phrases.append(new_phrase)

    return detected_new_phrases


class Word:
    def __init__(self, word, probability=0):
        self.word = word
        self.probability = probability


def analysis_one_segment(segment):
    detected_phrases = []

    cut_words = cut(segment)
    new_phrase_segments = []

    for word in cut_words:
        if len(new_phrase_segments) == 0:
            new_phrase_segments.append(Word(word, 0))
            continue

        _phrase, is_phrase_type, new_phrase_prob = is_phrase(new_phrase_segments[-1], Word(word))

        if _phrase:
            new_phrase_segments.append(Word(word, new_phrase_prob))
        else:
            new_phrase = create_new_phrase(new_phrase_segments)
            detected_phrases = collect_new_phrase(detected_phrases, new_phrase)
            new_phrase_segments = []

    if len(new_phrase_segments) != 0:
        detected_phrases = collect_new_phrase(detected_phrases,
                                              create_new_phrase(new_phrase_segments))

    return detected_phrases


def analyse_new_phrase(sentence, stop_word='|'):
    '''
    :param sentence:  A input sentence to test if there is a new phrase.
    :return:  The new phrases sorted by the probability.
    '''

    sentences = sentence.split(stop_word)
    detected_new_phrases = []

    for string in sentences:
        detected_new_phrases += analysis_one_segment(string)

    return detected_new_phrases


def get_sentence_from_file(file_name):
    sentences = ""

    with open(file_name) as f:
        for index, line in enumerate(f.readlines()):
            if index % 100 == 0:
                print(index)
            sentences += line + '\n'

    return sentences

if __name__ == '__main__':
    test_sentences = get_sentence_from_file('test_phrase.txt')
    stop_word =  '|'
    test_sentences = clearify(test_sentences, http_input=False)
    test_sentences = test_sentences.replace(' ', stop_word)
    print(list(jieba.cut(test_sentences)))
    new_phrases = analyse_new_phrase(test_sentences, stop_word)

    new_discoveries = sorted([p for p in new_phrases if p[1] <= 1], key=lambda x: x[1], reverse=True)
    tradition_phrases = sorted([p for p in new_phrases if p[1] > 1], key=lambda x: x[1])

    for phrase in new_discoveries:
        print('new phrase: {}, probability: {}'.format(phrase[0], phrase[1]))

    for phrase in tradition_phrases:
        print('tradition phrase: {}, probability: {}'.format(phrase[0], 2-phrase[1]))

