from summary.text_summary import *
from summary.utils import *


def get_complete_sentences_with_corelations(sentences, single_subsentence_corelations, text):
    complete_sentences = []
    complete_sentences_corelations = []

    corelations = [None if is_outliner(x, single_subsentence_corelations) else x for x in
                   single_subsentence_corelations]
    single_complete_corelations = [corelations[0]]
    single_complete_sentence = [sentences[0]]

    for index in range(1, len(sentences)):
        sub = sentences[index]
        last_word = single_complete_sentence[-1]
        if in_same_sentence(last_word, sub, text):
            single_complete_sentence.append(sub)
            single_complete_corelations.append(corelations[index])
        else:
            complete_sentences.append(single_complete_sentence)
            complete_sentences_corelations.append(single_complete_corelations)

            single_complete_sentence = [sub]
            single_complete_corelations = [corelations[index]]

        if index == len(sentences) - 1:
            complete_sentences.append(single_complete_sentence)
            complete_sentences_corelations.append(single_complete_corelations)

    return zip(complete_sentences, complete_sentences_corelations)


def get_text_corelation(text):
    text_sentences = get_text_sentence(text)
    distance_map = get_all_sentences_distance(text_sentences)
    distance_sentence_pair = [(string, distance_map[string]) for string in text_sentences]
    corelation = [1 - d for _, d in distance_sentence_pair]
    segments_with_index = [index_word for index_word in enumerate(text_sentences)]
    return segments_with_index, corelation, distance_sentence_pair


def get_one_file_complex_corelation(text, title):
    #    print("{} {}".format(text[:50], title))

    corelations = get_text_corelation(text)
    sentences = [s for s, d in corelations[2]]
    title_distance = get_title_distance(title, sentences)
    title_corelations = softmax([1 - d for _, d in title_distance])
    content_corelations = softmax(corelations[1])
    complex_corelation = get_complex_corelation(title_corelations, content_corelations)
    # plot_corelation(complex_corelation[0])
    return complex_corelation


def get_summary_with_nolinear(text, title, fit_length):
    complex_corelation = get_one_file_complex_corelation(text, title)
    sentences = get_text_sentence(text)
    complete_no_linear = get_complete_sentences_with_corelations(
        sentences, complex_corelation, get_text_content(text, escape_english=False))

    complete_no_linear = list(complete_no_linear)

    def f(array, sentences):
        total_words_length = len("".join(sentences))
        array = list(filter(lambda x: x is not None, array))
        content_ratio = len(array) / len(sentences)
        result = np.mean(array) * (1.05 ** (np.log(total_words_length * content_ratio + 1)))
        return result if not np.isnan(result) else -1

    def get_merged_corelation(single_nolinear_corelations, sentences):
        merged_corelation = f(single_nolinear_corelations, sentences)
        return merged_corelation

    def get_sentence_and_merged_corelation(subsentences, corelation):
        return (" ".join(subsentences), corelation)

    completed_sentences_with_corelations = []
    all_corelations = [c for s, c in complete_no_linear]
    #    _25th_all_corelations = np.

    for s, c in complete_no_linear:
        merged_corelation = get_merged_corelation(c, s)
        single_completed_sentence_with_corelation = get_sentence_and_merged_corelation(s, merged_corelation)
        completed_sentences_with_corelations.append(single_completed_sentence_with_corelation)

    # _25_percentile = np.percentile([c for s, c in completed_sentences_with_corelations], 25)
    #    _60_percentile = np.percentile([c for s, c in completed_sentences_with_corelations], 60)
    corelations = [c for s, c in completed_sentences_with_corelations]
    corelations = [x if not np.isnan(x) else -1 for x in corelations]
    sentences = [s for s, c in completed_sentences_with_corelations]

    top_corelations = top_n(corelations, sentences, fit_length)

    total_sentence = []
    total_length = 0
    min_single_length = 3
    for string, corelations in zip(sentences, corelations):
        if corelations in top_corelations:
            if len(string) >= min_single_length:
                total_length += len(string)
                total_sentence.append(string)

    return "。".join(total_sentence)


def top_n(corelations, sentences, fit_length):
    sorted_array_with_sentences = sorted(zip(corelations, sentences), key=lambda x: x[0], reverse=True)
    max_corelations = []
    length = 0
    for c, s in sorted_array_with_sentences:
        length += len(s)
        max_corelations.append(c)
        if length >= fit_length: break
    return max_corelations


def get_suitable_length_summary(text, title, fit_length):
    summary = get_summary_with_nolinear(text, title, fit_length)
    return summary


def readable_summary(text, title):
    fit_length = get_fit_length(len(get_text_content(text)))
    return title + ": " + get_suitable_length_summary(text, title, fit_length)


if __name__ == '__main__':
    target_file_path = '../experiment/error_analysis.txt'
    title = '“汉语桥”德国大区预选赛落幕'
    summary = readable_summary(target_file_path, title)
    print(summary)

