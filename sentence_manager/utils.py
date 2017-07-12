import re


def line_to_sentences(line, need_get_substring=True):
    line = change_text_english(line)
    split = '||'
    end_marks = r"""[\u3000\n\r\t\xa0@。；？！?？|;!！【】——]"""
    # line = replace_in_quote_end_mark(line, end_marks)
    white_space_regex = re.compile(end_marks)
    content = white_space_regex.sub(split, line)
    dont_need_mark = re.compile(r"[\"……]")
    content = dont_need_mark.sub(split, content)

    if need_get_substring:
        split_mark = re.compile(r"""[,，<> · () （）：)（）]""")
        content = split_mark.sub(split, content)

    # content = re.sub(split, content).strip()
    # content = re.sub("\s+", ' ', content).strip()
    sentences = content.split(split)
    sentences = filter(lambda x: len(x) >= 1, sentences)
    sentences = list(map(recovery_from_english, sentences))
    return sentences


def replace_in_quote_end_mark(string, end_mark):
    new_string = re.sub(r'(?!(([^“]*"){2})*[^”]*$)%s' % end_mark, ',', string)
    return new_string


def delete_bracket(string):
    '''
    delete the bracket content in a string. 
    e.g 北京办公楼外北侧的雨水收集池（位于建筑物20米开外）起火，原因是工人操作不当，引燃了塑料材料。目前火已扑灭，现场无人员伤亡，感谢大家的关心。”
        change to 北京办公楼外北侧的雨水收集池起火，原因是工人操作不当，引燃了塑料材料。目前火已扑灭，现场无人员伤亡，感谢大家的关心。”
    :param string: 
    :return: 
    '''
    brackets = "[\(\[\（\【].*?[\)\）\]\】]"
    string = re.sub(brackets, ",", string)
    return string


def change_text_english(text):
    if len(text) <= 1:
        return text

    new_text = text[0]
    placeholder = u'\U0001f604'
    for i in range(1, len(text)-1):
        current_char = text[i]
        if (is_space(current_char) and forward_is_english(i, text)) or (is_space(current_char) and 0 <= ord(text[i-1]) <= 127):
            new_text += placeholder
        else:
            new_text += current_char

    new_text += text[-1]

    return new_text


def is_between_english(previus, after):
    if ord(previus) >= 0 and ord(after) <= 127:
        return True
    else:
        return False


def recovery_from_english(sentence):
    assert len(sentence) >= 1, sentence

    placeholder = u'\U0001f604'

    new_sentence = sentence[0] if sentence[0] != placeholder else ""

    for index in range(1, len(sentence)-1):
        current_char = sentence[index]
        if current_char == placeholder:
            if is_between_english(sentence[index-1], sentence[index+1]):
                new_sentence += ' '
        else:
            new_sentence += current_char

    if sentence[-1] != placeholder:
        new_sentence += sentence[-1]

    return new_sentence


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
        elif 0 <= ord(text[index]) <= 127:
            return True
        else:
            return False


def recovery_punctuation(string: str, text: str):
    string = string.replace('。', ' ')
    sub_strings = string.split(' ')
    new_string = ""

    index = 0
    for sub_str in sub_strings:
        location = text[index:].index(sub_str)
        punctuation = text[index:][location + len(sub_str)]
        new_string += sub_str + punctuation
        index += len(sub_str)

    return new_string


def add_end_quote(string):
    new_string = string
    end_quote = '”'
    if string.find('“') >= 0 and string.find(end_quote) < 0:
        new_string += end_quote
    return new_string

