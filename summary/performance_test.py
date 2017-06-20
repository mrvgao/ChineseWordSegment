import pandas as pd
from summary.nolinear_summary import  readable_summary
from summary.suitful import test_if_one_file_fit_summary
import time

import csv
import random
import os

test_file = 'test_summary_{}.csv'.format(time.strftime("%d%m%Y%H:%M:%S"))
csv_file = '../data_preprocess/updated_news/sqlResult_1262716_0524.csv'
contents = pd.read_csv(csv_file)
contents = contents.iterrows()
differents = []
test = []
length = 50

with open(test_file, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['id', 'title', 'content', 'description', 'score'])
    for C in contents:
        if length <= 0: break
        ID, content, title = C[1][0], C[1][4], C[1][2]
        if random.random() < 0.7: continue
        fit, var = test_if_one_file_fit_summary(content, title)
        if fit:
            try:
                summary = readable_summary(content, title)
            except IndexError:
                print('content: {}'.format(content))
                print('title: {}'.format(title))
            spamwriter.writerow([ID, title, content, summary, ''])
            length -= 1
            print(length)