from summary.text_summary import  *
import numpy as np


def get_fit_length(original_length):
    length_map = {
        lambda x: x > 1000: 250,
        lambda x: 500 < x < 1000: 200,
        lambda x: 300 < x < 500: 150,
        lambda x: x < 300: 100
    }
    for cond in length_map:
        if cond(original_length):
            return length_map[cond]


def get_title_distance(title, sentences):
    return get_text_sentences_distances(title, sentences)


def softmax(array):
    array = np.array(array)
    array -= np.max(array, axis=0)
    return np.exp(array) / sum(np.exp(array))


def get_complex_corelation(title_corelations, content_corelations):
    def f(title_corelation, content_corelation):
        p = 0.5
        return p * title_corelation + (1 - p) * content_corelation
    corelations = []
    for t_c, c_c in zip(title_corelations, content_corelations):
        corelations.append(f(t_c, c_c))
    return corelations


def is_outliner(x, array):
    _1st_percentile = np.percentile(array, 25)
    _3th_percentile = np.percentile(array, 75)
    threshold = 1.5
    if (_1st_percentile / x) > threshold or (x / _3th_percentile) > threshold:
        return True
    else:
        return False


def in_same_sentence(subsentence1, subsentence2, text):
    begin_index = text.index(subsentence1)
    if find_complete_sentence(subsentence1, text[begin_index:]) == find_complete_sentence(subsentence2, text[begin_index:]):
        return True
    else:
        return False