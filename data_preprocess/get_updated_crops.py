'''
Get the news everyday.

Return: a file named updated.data
'''

import os
import logging


def write_file(filename, backup_file):
    with open(backup_file, 'a') as f:
        logging.debug(backup_file)
        with open(filename) as info:
            for line in info.readlines():
                f.write(line.strip() + '\n')


def read_dir(directory, backup_file):
    for filename in os.listdir(directory):
        logging.debug(filename)
        if filename.endswith('.txt'):
            write_file(os.path.join(directory, filename), backup_file)


def save(crops_dir, updated_news_file_backup=None):
    updated_news_file_backup = updated_news_file_backup or 'updated_news.data'
    read_dir(crops_dir, updated_news_file_backup)


if __name__ == '__main__':
    save()

