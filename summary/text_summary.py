from sentenceEmbedding import get_sentence_vector
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


def sentence_is_same(string1, string2):
    v1 = get_sentence_vector(string1)
    v2 = get_sentence_vector(string2)
    return cosine(v1, v2) < 1e-1


def is_space(char):
    if char == ' ':
        return True
    else:
        return False


def forward_is_english(index, text):
    while index < len(text):
        if is_space(text[index]):
            index += 1
            continue
        elif text[index].isalpha():
            return True
        else:
            return False


def change_text_english(text):
    if len(text) <= 1:
        return text

    new_text = text[0]
    placeholder = u'\U0001f604'
    for i in range(1, len(text)-1):
        current_char = text[i]
        if (is_space(current_char) and forward_is_english(i, text)) or (is_space(current_char) and text[i-1].isalpha()):
            new_text += placeholder
        else:
            new_text += current_char

    new_text += text[-1]

    return new_text


def recovery_from_english(sentences):
    new_sentences = []
    placeholder = u'\U0001f604'
    for s in sentences:
        s = s.replace(placeholder, ' ')
        new_sentences.append(s)

    return new_sentences


def delete_hidden_seperator(text):
    white_space_regex = re.compile(r"[\n\r\t\xa0@]")
    text = white_space_regex.sub('\n', text)
    text = text.replace("\u3000", '')
    return text


def get_text_sentence(text, escape_english=True):
    text = get_text_content(text, escape_english)
    text_sentences = line_to_sentences(text)
    if escape_english:
        text_sentences = recovery_from_english(text_sentences)
    return text_sentences


def get_text_content(text, escape_english=True):
    if os.path.isfile(text):
        text = "".join(line for line in open(text).readlines())

    text = delete_bracket(text)

    if escape_english:
        text = change_text_english(text)

    text = delete_hidden_seperator(text)
    return text


def get_two_sentence_distance(text1, text2):
    return cosine(get_text_vector(text1), get_text_vector(text2))


def get_text_vector(text):
    return get_sentence_vector(text)


def get_text_sentences_distances(text, sentences):
    text_vector = get_text_vector(text)
    sentences_vectors = [get_sentence_vector(string) for string in sentences]
    text_sentences_distances = [cosine(vec, text_vector) for vec in sentences_vectors]
    return list(zip(sentences, text_sentences_distances))


def get_all_sentences_distance(text_sentences):
    sentence_distances = get_text_sentences_distances(" ".join(text_sentences), text_sentences)
    sentence_distance_dic = {sentence: distance for sentence, distance in sentence_distances}
    return sentence_distance_dic


def clarify_duplicate_sentences(sentences_sorted_by_importance):
    deleted_index = []
    for index in range(len(sentences_sorted_by_importance) - 1):
        current_string = sentences_sorted_by_importance[index][0]
        next_string = sentences_sorted_by_importance[index + 1][0]
        if sentence_is_same(current_string, next_string):
            deleted_index.append(index + 1)

    for need_del in deleted_index:
        sentences_sorted_by_importance[need_del] = None

    return [s for s in sentences_sorted_by_importance if s is not None]


def get_important_sentences(sentences_distances: dict, keep_ratio=0.9):
    sorted_by_importance = sorted(sentences_distances.items(), key=lambda x: x[1])
    sorted_by_importance = clarify_duplicate_sentences(sorted_by_importance)
    important_sentences = [s for s, d in sorted_by_importance]
    return important_sentences[:int(len(sorted_by_importance)*keep_ratio)]


def get_summary(text_sentences,  summary_length, keep_ratio=0.9):
    sentences_importance = get_all_sentences_distance(text_sentences)
    keep_sentences = get_important_sentences(sentences_importance, keep_ratio=keep_ratio)
    total_sentence_number = 5
    if len(keep_sentences) < total_sentence_number: return keep_sentences
    return get_summary(keep_sentences, summary_length)


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


def get_complete_summary(file=None, type="file"):
    if type == 'file':
        text_sentences = get_text_sentence(file)
        text = "".join([line for line in open(file)])
    elif type == 'text':
        text_sentences = clarify(file).split()
        text = file

    summary_length = 60
    summary = get_summary(text_sentences, summary_length=summary_length)
    completed_summary = {}
    for index, sentence in enumerate(text_sentences):
        if sentence in summary:
            complete_sentence = find_complete_sentence(sentence, text)
            completed_summary[complete_sentence] = index

    completed_summary = sorted(completed_summary.items(), key=lambda x: x[1])
    completed_summary = "".join([s for s, _ in completed_summary])

    return completed_summary


def show_summary_in_text(text_sentences, summary):
    for line in text_sentences:
        if len(line) < 1: continue
        if line in summary:
            print(line)
        else:
            print('--{}--'.format(line))


def write_fm_news(f, contents, test_number):
    number = 0
    for row in contents:
        if random.random() < 0.7: continue
        if number > test_number: break

        content = row[1][4]
        if len(content) < 300: continue

        summary = get_complete_summary(content, type='text')
        f.write('--------------------------------------\n')
        f.write("{}\n Content: \n {}\n Description: {}\n".format(number, content, summary))
        print(number)
        number += 1


def get_test_result(content_csv):
    test_file_name = 'summary_test_result_0605.txt'
    contents = pd.read_csv(content_csv)
    contents = contents.iterrows()
    total_test_length = 25
    with open(test_file_name, 'w') as f:
        write_fm_news(f, contents, total_test_length)


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