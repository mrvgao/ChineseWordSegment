import re


def line_to_sentences(line, need_get_substring=True):
    white_space_regex = re.compile(r"""[' '\n\r\t\xa0@。；？！?？|;!！【】]""")
    content = white_space_regex.sub("\n", line)
    dont_need_mark = re.compile(r"[\"…… /]")
    content = dont_need_mark.sub(" ", content)

    if need_get_substring:
        split_mark = re.compile(r"""[,，<> · () （）：)（）]""")
        content = split_mark.sub(" ", content)

    content = re.sub("\s+", ' ', content).strip()
    return content.split()


def delete_bracket(string, bracket='（', end_pair='）'):
    '''
    delete the bracket content in a string. 
    e.g 北京办公楼外北侧的雨水收集池（位于建筑物20米开外）起火，原因是工人操作不当，引燃了塑料材料。目前火已扑灭，现场无人员伤亡，感谢大家的关心。”
        change to 北京办公楼外北侧的雨水收集池起火，原因是工人操作不当，引燃了塑料材料。目前火已扑灭，现场无人员伤亡，感谢大家的关心。”
    :param string: 
    :return: 
    '''
    brackets = "[\(\[\（\【].*?[\)\）\]\】]"
    string = re.sub(brackets, "", string)
    return string


