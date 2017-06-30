import jieba


def cut(string):
    return list(jieba.cut(string))