from summary.text_summary import get_text_sentences_distances
from summary.text_summary import find_complete_sentence
from sentence_manager.sentence_embedding import get_sentence_embedding
import numpy as np
from utlis.metrics import cosine


def get_fit_length(original_length):
    length_map = {
        lambda x: x > 1000: 300,
        lambda x: 500 < x < 1000: 250,
        lambda x: 300 < x < 500: 200,
        lambda x: x < 300: 150
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
    p = 0.5
    title_correlations = [c if c != float('inf') else 2 for c in title_correlations]
    content_correlations = [c if c != float('inf') else 2 for c in content_correlations]
    correlations = list(map(lambda t_c: t_c[0] * p + t_c[1] * (1-p),
                            zip(title_correlations, content_correlations)))
    return [c/np.sum(correlations) for c in correlations]


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


# def plot_correlation(correlation, sub_plot=None):
#     mean = np.mean(correlation)
#     _1st_percentile = np.percentile(correlation, 25)
#     _3st_percentile = np.percentile(correlation, 75)
#
#     if sub_plot:
#         plt.subplot(*sub_plot)
#
#     print('correlation length is: {}'.format(len(correlation)))
#     plt.plot(range(len(correlation)), correlation, c=(0, 0, 0.2))
#     plt.fill_between(range(len(correlation)), correlation)
#     plt.plot([mean] * len(correlation), 'r--')
#     plt.plot([_1st_percentile] * len(correlation), 'g+')
#     plt.plot([_3st_percentile] * len(correlation), 'b*')


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


def is_same_sentence(s1, s2):
    if s1 in s2 or s2 in s1:
        return True
    else:
        v1 = get_sentence_embedding(s1)
        v2 = get_sentence_embedding(s2)
        distance = cosine(v1, v2)
        threshold = 0.25
        if distance < threshold:
            return True
        else:
            return False


def contained_in_previous(new_sentence, previous_sentences):
    for s in previous_sentences:
        if is_same_sentence(new_sentence, s):
            return True
    return False


def is_empty_character_string(string):
    is_empty = True
    for s in string:
        if str(s).isalnum():
            is_empty = False
            break
    return is_empty


def top_n(correlations, sentences, fit_length, title=None):
    previous_choosen = []
    if title: previous_choosen.append(title)

    outliner_threshold = 1.3
    outliner_ceil = np.percentile(correlations, 75) * outliner_threshold

    sorted_array_with_sentences = sorted(zip(correlations, sentences), key=lambda x: x[0])
    max_correlations = []
    length = 0
    for c, s in sorted_array_with_sentences:
        if c >= outliner_ceil: break
        if is_empty_character_string(s): continue
        if not contained_in_previous(s, previous_choosen):
            length += len(s)
            max_correlations.append(c)
            previous_choosen.append(s)
            if length >= fit_length: break
    return max_correlations


def get_sentences_by_distances(distances, sentences, top_n_distances):
    total_sentence = [s for s, d in zip(sentences, distances) if d in top_n_distances]
    return total_sentence

