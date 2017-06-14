from bs4 import BeautifulSoup
import pandas as pd
import re
import random
import logging


def clarify(content, http_input=True):
    if http_input:
        soup = BeautifulSoup(content, 'html.parser')
        content = soup.get_text()
    content = content.strip()
    white_space_regex = re.compile(r'[\n\r\t\xa0@]')
    content = white_space_regex.sub("。", content)
    quote_regx = re.compile(r'''[?？|'"、。，；;,!！“’”<> 《》() （）\---—]''')
    content = quote_regx.sub(" ", content)
    content = re.sub( '\s+', ' ', content).strip()
    return content


def get_multiply_files_content(new_data_files, pure_content_file, sample=1., original='cms'):
    func_map = {
        'cms': save_pure_content,
        'fm': get_fm_database
    }

    func = func_map[original]

    for file in new_data_files:
        func(file, pure_content_file, sample)
        logging.info('{}: read done!'.format(file))


def get_fm_database(new_file, pure_content_file, sample=1.):
    content = pd.read_csv(new_file)
    database = content.iterrows()

    with open(pure_content_file, 'a') as f:
        for index, r in enumerate(database):
            if index % 100 == 0:
                print(index)
            try:
                content = " ".join([r[1][2], r[1][3], r[1][4]])
                content = clarify(content)
                f.write(content)
            except TypeError:
                print(content)


def save_pure_content(new_file, pure_content_file, sample=1.):
    CSV, NORMAL = 'csv', 'normal'
    file_type = CSV if new_file.endswith('.csv') else NORMAL
    if file_type == CSV:
        cms = pd.read_csv(new_file)
        databases = cms.iterrows()
    else:
        databases = open(new_file).readlines()

    with open(pure_content_file, 'a') as f:
        for index, r in enumerate(databases):
            if index % 100 == 0:
                print(index)

            if random.random() > sample:
                continue

            try:
                if file_type == CSV:
                    content = r[1][2] + r[1][17]
                    content = clarify(content)
                else:
                    content = r
                f.write(content + '\n')
            except TypeError:
                print(r)

if __name__ == '__main__':
    yesterday = 'updated_news/20170523.csv'
    content_file = 'updated_news/20170523-pure.data'
    save_pure_content(yesterday, pure_content_file=content_file, sample=0.5)
    print('done!')
