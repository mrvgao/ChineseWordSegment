import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
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
mini_vector = pickle.load(open('data_preprocess/updated_news/latest_vectors.pickle', 'rb'))
global_word_to_id = {word: ID for ID, word in global_vector['id_to_word'].items()}
mini_word_to_id = {word: ID for ID, word in mini_vector['id_to_word'].items()}


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
    threshold = 0.5
    # if distance1 < threshold:
    #     return 1 - distance1
    # elif distance1 < distance2:
    #     return distance1
    # else:
    if distance1 > distance2:
        return (distance1 - distance2) / distance1
    else:
        return distance1 + 10


def analyse_new_phrase(sentence):
    '''
    :param sentence:  A input sentence to test if there is a new phrase.
    :return:  The new phrases sorted by the probability.
    '''

    cut_words = jieba.cut(sentence)

    previous = None

    new_phrases = {}
    for index, word in enumerate(cut_words):
        if index == 0:
            previous = word
            continue

        new_phrase_prob = get_new_phrase_probability(previous, word)
        new_phrases[(previous+word)] = new_phrase_prob
        previous = word

    return new_phrases


def get_sentence_from_file(file_name):
    sentences = ""

    with open(file_name) as f:
        for index, line in enumerate(f.readlines()):
            if index % 100 == 0:
                print(index)
            sentences += line + '\n'

    return sentences

if __name__ == '__main__':
    test_sentences = '''5月15日，在“一带一路”国际合作高峰论坛的领导人圆桌峰会上，
    由龙泉青瓷制成的“丝路金桥”，连接起参加工作午宴的各国政要，以中国独特的文化符号，见证着古老的丝路精神。衢窑研究院·龙泉半闲堂所设计制作的青瓷成为2014APEC会议、2016杭州G20峰会和2017“一带一路”峰会唯一指定专用青瓷。2009年，龙泉青瓷传统烧制技艺被联合国教科文组织列入《人类非物质文化遗产代表作名录》。
    龙泉青瓷独立项目入选世界级文化遗产，标志着世界陶瓷领域里零的突破，成为目前全球唯一的陶瓷类“人类非遗”项目。
    新华社北京5月23日电  国家主席习近平23日就英国曼彻斯特市发生爆炸事件向英国女王伊丽莎白二世致慰问电，
    对无辜遇难者表示深切的哀悼，对伤者和遇难者家属表示诚挚的慰问。习近平指出，在这一艰难时刻，
    中国人民同英国人民坚定站在一起。\''''
    # test_sentences = get_sentence_from_file('test_phrase.txt')
    test_sentences = clearify(test_sentences, http_input=False)
    test_sentences = test_sentences.replace(' ', '|')
    print(list(jieba.cut(test_sentences)))
    new_phrase_probs = analyse_new_phrase(test_sentences)

    new_phrase_probs = sorted(new_phrase_probs.items(), key=operator.itemgetter(1), reverse=True)

    tradition_phrase = [p for p in new_phrase_probs if p[1] > 10]
    tradition_phrase = sorted(tradition_phrase, key=lambda x: x[1])

    new_phrase = [p for p in new_phrase_probs if p[1] > 0 and p[1] <= 1]
    new_phrase = sorted(new_phrase, key=lambda x: x[1], reverse=True)

    for phrase in new_phrase:
        print('new phrase: {}, probability:{}'.format(phrase[0], phrase[1]))

    for phrase in tradition_phrase:
        print('traditional phrase: {}', format(phrase[0]))

