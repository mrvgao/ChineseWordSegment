from data_preprocess.word_frequence import create_word_distribute_file
from data_preprocess.word_frequence import get_words_frequency_pickle
import os

file_name = create_word_distribute_file([['a', 'b'],['1', 'a']], file_name='test_file')

assert os.path.isfile(file_name)

print('test done!')
