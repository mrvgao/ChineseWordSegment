__author__ = 'minquan.gao'
__date__ = 'Mon 22 May 2017'

import json

json_file = './hudong_ret.json'

target_keys = ['detail', 'open_type', 'info_name', 'info_box', 'summary']

pure_content_file = './train_content.txt'

content_file = open(pure_content_file, 'a')


def special_char_convert(string):
    SPAN = '@#$#'
    return string.replace(SPAN, ';')

with open(json_file) as f:
    data = json.load(f)
    # for key in data:
    #     print(key['info_box'])

    for i, single_datum in enumerate(data):
        sentences = ""
        if i % 100 == 0:
            print(i)
        #     break
        for key in target_keys:
            content = single_datum[key]
            content = special_char_convert(content)
            sentences += content + '\n'

        if len(sentences.strip()) > 0:
            content_file.write(sentences)
