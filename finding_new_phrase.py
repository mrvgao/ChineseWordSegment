from word import Word, Segment
import jieba
import logging
from data_preprocess.get_cms_news import clarify
from utlis.get_word_vector_by_gensim import get_consistent
import jieba.posseg as pseg

cut = pseg.cut


def get_new_phrase_probability(words_pair_consistent): return words_pair_consistent


def could_concatenate(words_candidates, Word2):
    consistent_threshold = 0.5
    if words_candidates[-1].is_verb():
        consistent_threshold = 0.5

    enhance_ratio = 1.3
    consistent_threshold *= (enhance_ratio ** len(words_candidates))
    #
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


if __name__ == '__main__':

    # logging.basicConfig(level=logging.DEBUG)
    test_sentences = get_sentence_from_file('test_phrase.txt')
    stop_word = '|'
    test_sentences = clarify(test_sentences, http_input=False)
    test_sentences = test_sentences.replace(' ', stop_word)
    print(list(jieba.cut(test_sentences)))
    new_phrases, new_cut = analyse_new_phrase(test_sentences, stop_word)

    data_collected = 'new_phrase_detected.csv'
    new_phrases = distinct(new_phrases)
    # new_discoveries = distinct([p for p in new_phrases if p[1] <= 1])
    # tradition_phrases = distinct([p for p in new_phrases if p[1] > 1])
    #
    # new_discoveries = sorted(new_discoveries.items(), key=lambda x: x[1], reverse=True)
    # tradition_phrases = sorted(tradition_phrases.items(), key=lambda x: x[1])
    #
    print(new_cut)
    #
    with open(data_collected, 'w') as f:
        for word, p_segs in sorted(new_phrases.items(), key=lambda x: x[1][0], reverse=True):
            if word is not None:
                print('new phrase: {}, consistent: {}'.format(word, p_segs[0]))
                word_segments = [x.word for x in p_segs[1]]
                word_poses = [x.pos for x in p_segs[1]]
                words = "_".join(word_segments)
                words_poses = "_".join(word_poses)
                joined_words = word
                csv_format_string = ",".join([words, words_poses, joined_words])
                f.write(csv_format_string + '\n')
    #
    # for phrase in tradition_phrases:
    #     print('tradition phrase: {}, probability: {}'.format(phrase[0], 2-phrase[1]))

# IDEA feature: User RNN?
