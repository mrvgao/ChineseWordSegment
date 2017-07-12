import re
import functools
import random
import pandas as pd
from data_preprocess.get_cms_news import clarify
import os
import logging
from sentence_manager.utils import line_to_sentences
from sentence_manager.utils import delete_bracket
from utlis.metrics import cosine


def delete_hidden_seperator(text):
    white_space_regex = re.compile(r"[\n\r\t\xa0@]")
    text = white_space_regex.sub('\n', text)
    text = text.replace("\u3000", '')
    return text


# def get_text_sentence(text, escape_english=True):
#     text = get_text_content(text, escape_english)
#     text_sentences = line_to_sentences(text)
#     if escape_english:
#         text_sentences = recovery_from_english(text_sentences)
#     return text_sentences
#
#
# def get_text_content(text, escape_english=True):
#     if os.path.isfile(text):
#         text = "".join(line for line in open(text).readlines())
#
#     text = delete_bracket(text)
#
#     if escape_english:
#         text = change_text_english(text)
#
#     text = delete_hidden_seperator(text)
#     return text

def forward_alpha(text : str, index, direction='right'):
    if direction == 'right': index += 1
    else: index -= 1

    if 0 <= index < len(text):
        return text[index].isalpha()

    return False


def get_sentence_begin_index(sub_str: str, text: str, end_marks: list, special_beginning):
    begin_index = text.index(sub_str)

    begin_length = 20
    while begin_index >= 0:
        if text[begin_index] in end_marks: break
        elif text[begin_index] == ' ' and not forward_alpha(text, begin_index, 'left'): break
        elif begin_index < begin_length and text[begin_index] in special_beginning: break
        begin_index -= 1

    return begin_index


def get_sentence_end_index(sub_str: str, text: str, end_marks: list, special_beginning):
    end_index = text.index(sub_str) + len(sub_str)
    begin_length = 20
    while end_index < len(text):
        if text[end_index] in end_marks:
            break
        elif text[end_index] == ' ' and not forward_alpha(text, end_index, 'right'): break
        elif end_index < begin_length and text[end_index] in special_beginning: break
        end_index += 1

    return end_index


def find_complete_sentence(sub_sentence: str, text: str) -> str:
    """
    find the complete sentence in the text. 
    What's the complete sentene? sentene between two "end mark", 
        such as .  。 ！ ! ？ ? \space  \n.
        :type sub_sentence: object
    """
    end_marks = ['。', '？', '！', '!', '?', '\n', '\xa0', '\r', '\t', '：']
    begining_end_marks = ['（', '）', '】', '【']
    begin = get_sentence_begin_index(sub_sentence, text, end_marks, begining_end_marks) + 1
    end = get_sentence_end_index(sub_sentence, text, end_marks, begining_end_marks) + 1
    return text[begin: end]


# def get_complete_summary(file=None, type="file"):
#     if type == 'file':
#         text_sentences = get_text_sentence(file)
#         text = "".join([line for line in open(file)])
#     elif type == 'text':
#         text_sentences = clarify(file).split()
#         text = file
#
#     summary_length = 60
#     summary = get_summary(text_sentences, summary_length=summary_length)
#     completed_summary = {}
#     for index, sentence in enumerate(text_sentences):
#         if sentence in summary:
#             complete_sentence = find_complete_sentence(sentence, text)
#             completed_summary[complete_sentence] = index
#
#     completed_summary = sorted(completed_summary.items(), key=lambda x: x[1])
#     completed_summary = "".join([s for s, _ in completed_summary])
#
#     return completed_summary
#

def show_summary_in_text(text_sentences, summary):
    for line in text_sentences:
        if len(line) < 1: continue
        if line in summary:
            print(line)
        else:
            print('--{}--'.format(line))


# def write_fm_news(f, contents, test_number):
#     number = 0
#     for row in contents:
#         if random.random() < 0.7: continue
#         if number > test_number: break
#
#         content = row[1][4]
#         if len(content) < 300: continue
#
#         summary = get_complete_summary(content, type='text')
#         f.write('--------------------------------------\n')
#         f.write("{}\n Content: \n {}\n Description: {}\n".format(number, content, summary))
#         print(number)
#         number += 1


# def get_test_result(content_csv):
#     test_file_name = 'summary_test_result_0605.txt'
#     contents = pd.read_csv(content_csv)
#     contents = contents.iterrows()
#     total_test_length = 25
#     with open(test_file_name, 'w') as f:
#         write_fm_news(f, contents, total_test_length)


def get_pure_content_from_csv(content_csv):
    test_file_name = 'pure_content.txt'
    contents = pd.read_csv(content_csv)
    contents = contents.iterrows()

    total_test_length = 50

    _info = 1
    _id, _title, _content = 0, 2, 4
    with open(test_file_name, 'w') as f:
        number = 0
        for row in contents:
            if random.random() < 0.7: continue
            if number >= total_test_length: break
            ID = row[_info][_id]
            title = row[_info][_title]
            content = row[_info][_content]
            f.write('**'*8)
            line = ['No: {}', 'ID: {}', 'title: {}', 'content: {}', 'fit: \n']
            line = "\n".join(line).format(number, ID, title, content)
            f.write(line)
            print(number)
            number += 1


if __name__ == '__main__':

    # summary = get_complete_summary('performace_test/test_summary.txt')
    # print(summary)
    # print('length: {}'.format(len(summary)))
    # get_test_result('data_preprocess/updated_news/sqlResult_1262716_0524.csv')

    get_pure_content_from_csv('data_preprocess/updated_news/sqlResult_1262716_0524.csv')