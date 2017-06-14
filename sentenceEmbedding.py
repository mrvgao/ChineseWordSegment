"""
Change a string to vector.
"""

GENSIM, GLOVE = 'gensim', 'glove'

MODEL = 'glove'

import jieba
import logging

if MODEL == GENSIM:
    from finding_new_phrase import get_consistent
    from finding_new_phrase import get_similarity as get_word_vector
else:
    from utlis.get_word_vector_by_glove import get_consistent
    from utlis.get_word_vector_by_glove import get_local_vector as get_word_vector


def get_word(word, word_left=None, word_right=None):
    weights = 1 + max(get_consistent(word_left, word), get_consistent(word, word_right))
    return weights


def sentence2Vec(string):
    string_padding = [None] + string + [None]
    sentence_vector = None
    for index in range(1, len(string_padding)-1):

        word_vector = get_word_vector(string_padding[index])

        if word_vector is None:
            continue

        weight = get_word(string_padding[index], string_padding[index-1], string_padding[index+1])
        vector = word_vector * weight
        if sentence_vector is not None:
            sentence_vector += vector
        else:
            sentence_vector = vector

    return sentence_vector


def get_sentence_vector(sentence):
    segments = list(jieba.cut(sentence))
    sentence_vector = sentence2Vec(segments)
    return sentence_vector


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_sentene = """"听附近的居民说"""

    sentence_vector = get_sentence_vector(test_sentene)

    assert sentence_vector is not None
    print(sentence_vector)
