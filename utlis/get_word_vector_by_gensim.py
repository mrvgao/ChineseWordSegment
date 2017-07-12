import jieba.posseg as pseg
from gensim.models.word2vec import Word2Vec

target = 'phrase'

if target == 'phrase':
    three_day_model = Word2Vec.load('/Users/kouminquan/AI-LAB/Chinese-New-Phrase-Detection/data_preprocess/word_vectors/gensim_20170522-0524.pickle')
elif target == 'fm':
    three_day_model = Word2Vec.load('/Users/kouminquan/AI-LAB/Chinese-New-Phrase-Detection/data_preprocess/word_vectors/gensim_fm_0524.pickle')

wiki_model = Word2Vec.load('/Users/kouminquan/AI-LAB/Chinese-New-Phrase-Detection/data_preprocess/word_vectors/gensim_hudong_wiki.pickle')

cut = pseg.cut

wiki_path = ''


def get_similarity_increase_ratio(word1, word2):
    wiki_similarity = get_similarity(word1, word2, model='wiki')
    three_day_similarity = get_similarity(word1, word2, model='three_day')

    increase_ratio = (three_day_similarity - wiki_similarity) / wiki_similarity

    return increase_ratio


def get_pair_similarity(word1, word2, chosen_model):
    if word1 in chosen_model and word2 in chosen_model:
        return chosen_model.wv.similarity(word1, word2)
    else:
        return None


def get_word_vector(word, chosen_model):
    if word in chosen_model:
        return chosen_model.wv[word]
    else:
        return None


def get_similarity(word1, word2=None, model='wiki'):
    eps = 1e-6

    model_map = {
        'wiki': wiki_model,
        'three_day': three_day_model
    }

    choosen_model = model_map[model]

    if word2 is not None:
        answer = get_pair_similarity(word1, word2, choosen_model)
    else:
        answer = get_word_vector(word1, choosen_model)

    return answer if answer is not None else eps


def get_consistent(word1, word2):
    """
    if word1 and word2 is consistent 
    :param word1: 
    :param word2: 
    :return: [0 - 1], 1 stands for very consistent.
    """
    # global_distance = get_words_global_distance(word1, word2)
    # local_distance = get_words_local_distance(word1, word2)
    # pair_distance_decrease_ratio = distance_decrease(global_distance, local_distance)

    if word1 is None or word2 is None:
        return 0

    pair_similarity_increase_ratio = get_similarity_increase_ratio(word1, word2)

    # head_tail_distance = get_words_global_distance(word1[-1], word2[0])
    # head_tail_local_distance = get_words_local_distance(word1[-1], word2[0])
    # tail_head_decrease_ratio = distance_decrease(head_tail_distance, head_tail_local_distance)

    # head_tail_similarity_increase_ratio = get_similarity_increase_ratio(word1[-1], word2[0])
    #
    # wiki_similarity = get_similarity(word1, word2, wiki_model)
    # wiki_head_tail_similarity = get_similarity(word1[-1], word2[0], wiki_model)
    #
    # sig = lambda x: 1 / (1 + np.exp(-(x - 0.2)))
    # wiki_consistent = sig(1 - max(wiki_similarity, wiki_head_tail_similarity))
    # new_consistent = max(pair_similarity_increase_ratio, head_tail_similarity_increase_ratio)
    # return max(wiki_consistent, new_consistent)

    return pair_similarity_increase_ratio
