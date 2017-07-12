import redis
import numpy as np
import ast

r = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

cache = {}


def save_word_vector(word, vector):
    array = list(vector)
    global r
    r.set(word, array)
    return array


def get_redis_after_cache(key, callback):
    global cache

    if key in cache:
        value = cache[key]
    else:
        value = r.get(key)
        if value:
            value = callback(value)
            cache[key] = value

    return value


def get_word_vector(word):
    return get_redis_after_cache(key=word, callback=lambda x: np.array(ast.literal_eval(x)))


def save_word_count_frequency(w, frequency):
    r.set("fre:{}".format(w), frequency)


def get_word_count_frequency(word):
    key = "fre:{}".format(word)
    return get_redis_after_cache(key=key, callback=float)


def test():
    r.set('test', '001')
    v = r.get('test')
    print(v)
    no = r.get('no')
    assert no is None


if __name__ == '__main__':
    word = 'test'
    vector = np.array([1., 2., 3.])

    saved_result = save_word_vector(word, vector)

    get_result = get_word_vector(word)

    np.testing.assert_equal(get_result, vector)

    save_word_count_frequency('高民权', 0.123)
    fre = get_word_count_frequency('高民权')
    assert fre == .123

    print('test done!')


