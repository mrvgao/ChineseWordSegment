import pickle
from scipy import spatial

global_vectors = pickle.load(open('/Users/kouminquan/AI-LAB/Chinese-New-Phrase-Detection/data_preprocess/total_word_vectors_2.pickle', 'rb'))
local_vectors= pickle.load(open('/Users/kouminquan/AI-LAB/Chinese-New-Phrase-Detection/data_preprocess/word_vectors/sqlResult_1262716_0524pickle', 'rb'))
global_word_to_id = {word: ID for ID, word in global_vectors['id_to_word'].items()}
local_word_to_id = {word: ID for ID, word in local_vectors['id_to_word'].items()}
global_vectors = global_vectors['embedding']
local_vectors = local_vectors['embedding']


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
    if word1 is None or word2 is None:
        return 0

    global_distance = get_words_global_distance(word1, word2)
    local_distance = get_words_local_distance(word1, word2)

    distance_decrease_ratio = (global_distance - local_distance) / global_distance

    return distance_decrease_ratio



if __name__ == '__main__':
    word1 = '一带'
    word2 = '一路'

    assert get_words_local_distance(word1, word2) is not None
    assert get_words_global_distance(word1, word2) is not None


