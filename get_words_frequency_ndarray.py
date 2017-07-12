import numpy as np
import random
import pickle

data = []

sample = 1.0
vab_size = 20000
test_mode = False

with open('data_preprocess/word_frequence/occurence_merged.txt') as f:
    index = 0
    for line in f:
        if random.random() > sample: continue

        if test_mode and index > 100: break

        print(index)

        index += 1

        word1, word2, count = line.split(',')
        word1 = int(word1)
        word2 = int(word2)
        if word1 < vab_size and word2 < vab_size:
            count = float(count)
            data.append((word1, word2, count))

data = np.array(data)

with open('occurence_maxtrix_{}.pickle'.format(vab_size), 'w+b') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
