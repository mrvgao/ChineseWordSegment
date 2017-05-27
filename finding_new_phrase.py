import numpy as np
from word import Word, Segment
import pickle
import jieba
from scipy import spatial
import logging
from data_preprocess.get_cms_news import clearify
import jieba.posseg as pseg


global_vectors = pickle.load(open('data_preprocess/total_word_vectors_2.pickle', 'rb'))
local_vectors= pickle.load(open('data_preprocess/updated_news/latest_vectors_22_24.pickle', 'rb'))
global_word_to_id = {word: ID for ID, word in global_vectors['id_to_word'].items()}
local_word_to_id = {word: ID for ID, word in local_vectors['id_to_word'].items()}
global_vectors = global_vectors['embedding']
local_vectors = local_vectors['embedding']

cut = pseg.cut


def get_vector_by_word(word, word_id_map, vectors):
    return vectors[word_id_map[word]] if word in word_id_map else None


def get_global_vector(word): return get_vector_by_word(word, global_word_to_id, global_vectors)


def get_local_vector(word): return get_vector_by_word(word, local_word_to_id, local_vectors)


def get_distance(v1, v2, method='cosine'):
    method_f_map = {
        'cosine': spatial.distance.cosine,
        'correlation': spatial.distance.correlation,
        'euclidean': spatial.distance.euclidean,
    }
    return float('inf') if v1 is None or v2 is None else method_f_map[method](v1, v2)


def get_words_global_distance(word1, word2):
    return get_distance(get_global_vector(word1), get_global_vector(word2))


def get_words_local_distance(word1, word2):
    return get_distance(get_local_vector(word1), get_local_vector(word2))


def distance_decrease(u, v): return u - v / u


def get_consistent(word1, word2):
    """
    if word1 and word2 is consistent 
    :param word1: 
    :param word2: 
    :return: [0 - 1], 1 stands for very consistent.
    """
    global_distance = get_words_global_distance(word1, word2)
    local_distance = get_words_local_distance(word1, word2)
    pair_distance_decrease_ratio = distance_decrease(global_distance, local_distance)

    head_tail_distance = get_words_global_distance(word1[-1], word2[0])
    head_tail_local_distance = get_words_local_distance(word1[-1], word2[0])
    tail_head_decrease_ratio = distance_decrease(head_tail_distance, head_tail_local_distance)

    sig = lambda x: 1 / (1 + np.exp(-(x - 0.2)))
    already_consistent = sig(1 - min(global_distance, head_tail_distance))
    new_consistent = max(pair_distance_decrease_ratio, tail_head_decrease_ratio)
    return max(already_consistent, new_consistent)


def get_new_phrase_probability(words_pair_consistent): return words_pair_consistent


def could_concatenate(words_candidates, Word2):
    consistent_threshold = 0.4
    if words_candidates[-1].is_verb():
        consistent_threshold = 0.5

    enhance_ratio = 1.3
    consistent_threshold *= (enhance_ratio * len(words_candidates)-1)

    if not Word2.need_connect(phrase_strip=False):
        return False, None
    else:
        prob = get_new_phrase_probability(get_consistent(words_candidates[-1].word, Word2.word))
        return prob > consistent_threshold, prob


def collect_new_phrase(detected_new_phrases, new_phrase):
    if new_phrase:
        if detected_new_phrases is None: detected_new_phrases = []
        detected_new_phrases.append(new_phrase)

    return detected_new_phrases


def add_detected_new_phrase(new_phrase_segment, detected_phrases):
    new_phrase, new_phrase_prob = new_phrase_segment.get_new_phrase()
    detected_phrases = collect_new_phrase(detected_phrases, (new_phrase, new_phrase_prob, new_phrase_segment.word_segment))
    return detected_phrases


def analysis_one_segment(segment):
    detected_phrases = []

    cut_words = cut(segment)
    first_seg_word, first_seg_pos = next(cut_words)

    logging.debug("{} {}".format(first_seg_word, first_seg_pos))

    new_phrase_segments = Segment(init=Word(first_seg_word, first_seg_pos))

    new_cut_words = []

    for word, pos in cut_words:
        logging.debug("{} {}".format(word, pos))

        _phrase, consistent = could_concatenate(new_phrase_segments, Word(word, pos))

        if _phrase:
            new_phrase_segments.append(Word(word, pos, consistent))
        else:
            new_cut_words.append(new_phrase_segments.merge())
            detected_phrases = add_detected_new_phrase(new_phrase_segments, detected_phrases)
            new_phrase_segments = Segment(init=Word(word, pos))

    new_cut_words.append(new_phrase_segments.merge())

    if len(new_phrase_segments) > 1:
        detected_phrases = add_detected_new_phrase(new_phrase_segments, detected_phrases)

    return detected_phrases, new_cut_words


def analyse_new_phrase(sentence, stop_word='|'):
    '''
    :param sentence:  A input sentence to test if there is a new phrase.
    :return:  The new phrases sorted by the probability.
    '''

    sentences = sentence.split(stop_word)
    detected_new_phrases = []
    new_cut_words = []

    for index, string in enumerate(sentences):
        detected_phrases, cut_words = analysis_one_segment(string)
        detected_new_phrases += detected_phrases
        new_cut_words += cut_words

    return detected_new_phrases, new_cut_words


def get_sentence_from_file(file_name):
    sentences = ""

    with open(file_name) as f:
        for index, line in enumerate(f.readlines()):
            if index % 100 == 0:
                print(index)
            sentences += line + '\n'

    return sentences


def distinct(words):
    single_words = {}

    increase_ratio = 1.4

    for w, p, segments in words:
        if w in single_words:
            if p <=1:
                single_words[w][0] *= increase_ratio
            else:
                single_words[w][0] /= increase_ratio
        else:
            single_words[w] = [p, segments]

    return single_words


def test():
    vector = get_vector_by_word('一带', global_word_to_id, global_vectors)
    assert vector is not None
    vector = get_vector_by_word('3一带', global_word_to_id, global_vectors)
    assert vector is None

    print('test done')

if __name__ == '__main__':
    test()

    # logging.basicConfig(level=logging.DEBUG)
    test_sentences = get_sentence_from_file('test_phrase.txt')
    stop_word = '|'
    test_sentences = clearify(test_sentences, http_input=False)
    test_sentences = test_sentences.replace(' ', stop_word)
    print(list(jieba.cut(test_sentences)))
    new_phrases, new_cut = analyse_new_phrase(test_sentences, stop_word)

    data_collected = 'phrase_detected.csv'
    new_phrases = distinct(new_phrases)
    # new_discoveries = distinct([p for p in new_phrases if p[1] <= 1])
    # tradition_phrases = distinct([p for p in new_phrases if p[1] > 1])
    #
    # new_discoveries = sorted(new_discoveries.items(), key=lambda x: x[1], reverse=True)
    # tradition_phrases = sorted(tradition_phrases.items(), key=lambda x: x[1])
    #
    print(new_cut)
    #
    for word, p_segs in sorted(new_phrases.items(), key=lambda x: x[1][0], reverse=True):
        print('new phrase: {}, consistent: {}'.format(word, p_segs[0]))
    #
    # for phrase in tradition_phrases:
    #     print('tradition phrase: {}, probability: {}'.format(phrase[0], 2-phrase[1]))

# IDEA feature: User RNN?
