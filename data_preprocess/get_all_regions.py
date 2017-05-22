import jieba
import pickle

import logging
logging.basicConfig(level=logging.INFO)

sentences = []

content_file = 'train_content.txt'

total_cut_words = []

with open(content_file) as f:
    for index, line in enumerate(f.readlines()):
        total_cut_words.append(list(jieba.cut(line)))
        if index % 100 == 0:
            print(index)

all_regions = 'all_regions'

with open(all_regions, 'wb') as f:
    pickle.dump(all_regions, f, pickle.HIGHEST_PROTOCOL)

logging.info('pickle done!')


