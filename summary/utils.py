from summary.text_summary import get_text_sentences_distances
from summary.text_summary import find_complete_sentence
import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt


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


def get_complex_correlation(title_correlations, content_correlations):
    def f(title_correlation, content_correlation):
        p = 0.5
        return p * title_correlation + (1 - p) * content_correlation
    correlations = []
    for t_c, c_c in zip(title_correlations, content_correlations):
        correlations.append(f(t_c, c_c))
    return correlations


def is_outliner(x, array):
    _1st_percentile = np.percentile(array, 25)
    _3th_percentile = np.percentile(array, 75)
    threshold = 1.5
    if (_1st_percentile / x) > threshold or (x / _3th_percentile) > threshold:
        return True
    else:
        return False


def clean_outliner(Xs):
    Xs = np.array(Xs)
    Xs = list(filter(lambda x: not is_outliner(x, Xs), Xs))
    return Xs


def in_same_sentence(subsentence1, subsentence2, text):
    begin_index = text.index(subsentence1)
    if find_complete_sentence(subsentence1, text[begin_index:]) == find_complete_sentence(subsentence2, text[begin_index:]):
        return True
    else:
        return False


def plot_correlation(correlation, sub_plot=None):
    mean = np.mean(correlation)
    _1st_percentile = np.percentile(correlation, 25)
    _3st_percentile = np.percentile(correlation, 75)

    if sub_plot:
        plt.subplot(*sub_plot)

    print('correlation length is: {}'.format(len(correlation)))
    plt.plot(range(len(correlation)), correlation, c=(0, 0, 0.2))
    plt.fill_between(range(len(correlation)), correlation)
    plt.plot([mean] * len(correlation), 'r--')
    plt.plot([_1st_percentile] * len(correlation), 'g+')
    plt.plot([_3st_percentile] * len(correlation), 'b*')


def accumulate(x):
    x = np.array(x)
    acc = [np.sum(x[:(index+1)]) for index in range(len(x))]
    return acc


def get_neighbor(values, index, neighbor):
    values = [v if not np.isnan(v) else 2 for v in values]
    candidates = [values[index]]

    left = index - 1
    left_neighbor = 1
    right = index + 1
    right_neighbor = 1

    while left_neighbor <= neighbor and left >= 0:
        candidates.append(values[left])
        left -= 1
        left_neighbor += 1

    while right_neighbor <= neighbor and right <= len(values) - 1:
        candidates.append(values[right])
        right += 1
        right_neighbor += 1

    return np.mean(candidates)


def k_nn(values, neighbor=1):
    new_values = []
    for index, v in enumerate(values):
        knn_value = get_neighbor(values, index, neighbor)
        new_values.append(knn_value)

    return new_values



