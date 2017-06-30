'''
Get word frequence of all the words in Hudong Wiki.
'''

import pickle
from collections import Counter
import csv
import pandas as pd


def get_cut_words(cut_words_file):
    with open(cut_words_file, 'rb') as f:
        cut_words = pickle.load(f)

    return cut_words


def get_word_count(cut_words):
    counter = Counter()
    length = len(cut_words)
    for index, words in enumerate(cut_words):
        print("{}%".format(index * 100/length))
        counter += Counter(words)

    backup = 'word_frequence/word_frequence.pickle'

    counter = dict(counter)

    with open(backup, 'wb') as f:
        pickle.dump(counter, f, pickle.HIGHEST_PROTOCOL)

    return counter


def create_word_distribute_file(all_words_crops, file_name='all_words'):
    '''
    Convert all words crop, which is an array [[w1, w2, w3, .. ], [wi, w_j, ..], []]
    into a file key, value
    :param all_words_crops: 
    :return: file_name
    '''
    backup_file = file_name

    f = open(backup_file, 'w')

    length = len(all_words_crops)

    for index, crops in enumerate(all_words_crops):
        for word in crops:
            f.write('{},{}\n'.format("".join(map(str, map(ord, word))), word))

        if index % 100 == 0:
            print("{}%".format(index*100/length))

    return backup_file


def change_file_to_csv(file):
    csv_file = file + '.csv'

    original_file = open(file).readlines()

    print('load completed')
    with open(csv_file, 'w', newline='') as f:
        spamwriter = csv.writer(f)
        for i, line in enumerate(original_file):
            try:
                index, word = line.strip().split(',')
                spamwriter.writerow([index, word])
            except ValueError:
                continue

            if i % 1000 == 0:
                print("{}%".format(i*100/132706151))


def get_word_frequency(file):
    f = open(file)

    word_counts = {}

    result = 'words_count.csv'

    all_numbers = 0

    for index, line in enumerate(f.readlines()):
        _ , word = line.strip().split(',')
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

        all_numbers += 1

        if index % 1000 == 0: print("{}%".format(index*100/130173393))

    with open(result, 'w') as f:
        spamwriter = csv.writer(f)
        length = len(word_counts)
        for index, word in enumerate(word_counts):
            spamwriter.writerow([word, word_counts[word], word_counts[word]/all_numbers])
            if index % 100 == 0: print('{}%'.format(index*100/length))


def get_words_frequency_pickle(words_counts, test_mode=False):
    '''
    Read from words count file, change this into a dictionary, which could get 
    the information of a word in random time.
    :param words_counts:  the file of words count. <words, count, frequency>
    :return: pickle file.
    '''
    words_frequency_dictionary = dict()
    file_content = pd.read_csv(words_counts)
    rows = file_content.iterrows()

    count = 10

    for index, r in enumerate(rows):
        word = r[1][0]
        count = r[1][1]
        frequency = r[1][2]
        words_frequency_dictionary[word] = (count, frequency)
        if test_mode and index >= count: break

        if index % 100 == 0: print(index)

    pickle_file = 'words_count_and_frequency.pickle'

    with open(pickle_file, 'wb') as f:
        pickle.dump(words_frequency_dictionary, f, pickle.HIGHEST_PROTOCOL)

    return pickle_file


if __name__ == '__main__':
    # cut_word = get_cut_words('cutted_words/train_content.pickle')
    # print('load data done!')
    # create_word_distribute_file(cut_word)
    # print('write all words done!')
    # change_file_to_csv('all_words')
    # get_word_frequency('all_words.csv')
    get_words_frequency_pickle('words_count.csv', test_mode=False)
    print('read done!')