import jieba
import pickle

import logging
logging.basicConfig(level=logging.INFO)

sentences = []

content_file = 'train_content.txt'

total_cut_words = []

mini_test_batch = None

with open(content_file) as f:
    for index, line in enumerate(f.readlines()):
        total_cut_words.append(list(jieba.cut(line)))
        if index % 100 == 0:
            print(index)

        if mini_test_batch is not None and index > mini_test_batch:
            break

all_regions = 'all_regions'

with open(all_regions, 'wb') as f:
    pickle.dump(total_cut_words, f, pickle.HIGHEST_PROTOCOL)

logging.info('pickle done!')

