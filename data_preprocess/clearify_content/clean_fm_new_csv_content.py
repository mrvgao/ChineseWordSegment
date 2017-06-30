import codecs


target_file_name = 'shuwen_fm_news_content.txt'

target_file = open(target_file_name, 'w')


with codecs.open('sqlResult_1307920.csv', 'r', encoding='gbk', errors='ignore') as f:
    for index, line in enumerate(f.readlines()):
        if index % 2 == 0: continue
        print(index)
        target_file.write(line)