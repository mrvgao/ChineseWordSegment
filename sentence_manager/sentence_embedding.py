'''
Embedding a sentence into a vector.

By the approach of described in 'A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SEN- TENCE EMBEDDINGS', Sanjeev Arora, et al. 

But with some modification / addition. 

If there are some words not exists in the vocabulary, give it a random value with uniform distribution in space

D^w

Author: Minquan 
Date: 2017-06-20
'''
import jieba
import numpy as np
import pickle
import os
from sentence_manager.pca import get_pca
import logging

from utlis.redis_manager import get_word_vector
from utlis.redis_manager import get_word_count_frequency

cache = {}


def get_sentence_words_vector(sentence):
    """
    :param A sentence. word1word2word3..word_n
    :return:  vectors of those words. [vector1, vector2, ... vector_n]
    """

    words = list(jieba.cut(sentence))

    test_word = '测试'
    test_vector = get_word_vector(test_word)
    assert test_vector is not None

    vectors = map(get_word_vector, words)

    try:
        vectors, words = zip(*([(v, w) for v, w in zip(vectors, words) if v is not None]))
    except ValueError:
        return [], []

    return vectors, words


def get_vector_dimension(vectors):
    """
    Get the right dimension of one vector array. 
    :param vectors: [V1, V2, .. None, VN], we need to know what's the dimension of 'None'
    :return: the dimension of not None vector. 
    """

    for vec in vectors:
        if vec is not None:
            vec = np.array(vec)
            dimension = vec.shape[0]
            break
    else:
        dimension = None

    return dimension


def change_none_vector_to_real_value(vectors):
    """
    There are several words not exist in vocabulary, and those word's vector will become None, 
    In this process, we change those None Value to Real Value. Based on the Word2Vec methods, 
    we could change those to a uniform distribution or Gaussian distribution based on the train methods.
    :param vectors:  [V1, V2, V3, ..None, . VN]
    :return: [V1, V2, V3, .. V_i, .. VN]
    """
    vector_length = get_vector_dimension(vectors)
    if vector_length is not None:
        new_vectors = [vec if vec is not None else get_random_vector(vector_length) for vec in vectors]
    else:
        new_vectors = None

    return new_vectors


def get_random_vector(vector_size):
    vocabulary_size = 1000

    # min_interval = -0.5
    # max_interval = 0.5

    random_vec = np.random.normal(0, 0.5, size=vector_size)

    return random_vec


def get_sentence_embedding(sentence):
    vectors, words = get_sentence_words_vector(sentence)

    a = 1.e-3
    three_times_frequency = 3.8e-08  ## if one word not occured in hudong wiki, we assume it have

    weighted_vectors = []

    for word, v in zip(words, vectors):
        frequency = get_word_count_frequency(word) or three_times_frequency

        if v is None or frequency is None:
            continue
        else:
            weight = a / (a + frequency)
            weighted_vector = weight * v
            weighted_vectors.append(weighted_vector)

    length = len(words)
    final_vector = np.sum(weighted_vectors, axis=0) / length

    # if length > 1:
    #     principle_component = get_pca(vectors)[0]
    #     dot = np.dot(principle_component, np.transpose(principle_component))
    #     final_vector = final_vector - dot * final_vector

    return final_vector


def test_long_sentence():
    sentence = """【环球网报道 记者 朱佩】英国首相特蕾莎∙梅日前称，由于曼彻斯特恐袭案，该国恐怖威胁级别从“严重”提高至“危急”。这意味着可能派遣军队保障安全。据俄新社5月24日报道，伦敦警察厅反恐部门负责人马克•罗利表示，希望恐怖威胁级别不会太长时间维持在最高级别。
    罗利在回答恐怖威胁“危急”水平制度要维持多久的问题时说道：“我不想预测未来，但如果你看看我们的历史，这样一个威胁级别是非常不寻常和罕见的措施。它从未维持很久，我们也希望这样。但在这样一个高风险期我们将竭尽所能，军队将帮助我们。”
    当地时间5月22日晚，自杀式恐怖分子在曼彻斯特竞技场音乐厅内实施了爆炸。爆炸造成22人死亡，59人受伤。伤亡者中有许多儿童。至少有8人失踪，恐怖组织“伊斯兰国”声称对爆炸负责。"""
    #
    # logging.basicConfig(level=logging.DEBUG)
    sentence_vec = get_sentence_embedding(sentence)
    assert sentence_vec is not None
    print(sentence_vec.shape)
    print(sentence_vec)
    print('test done!')

def test_special_case():
    sentence = '昨天'
    sentence_vec = get_sentence_embedding(sentence)
    return sentence_vec

if __name__ == '__main__':
    test_special_case()
