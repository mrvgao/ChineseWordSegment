import jieba
import pickle

sentences = []

updated_news = 'updated_news.data'

total_cut_word = [list(jieba.cut(line)) for line in open(updated_news).readlines()]

with open('latest_crops.pickle', 'wb') as f:
    pickle.dump(total_cut_word, f, pickle.HIGHEST_PROTOCOL)
