## Get words counts.


def get_lines(file_name):
    return (line.split('\t') for line in open(file_name).readlines())


def all_is_chinese_word(words):
    first_chinese_word = '一'
    last_chinese_word = '鿏'

    is_chinese = True

    for w in words:
        if first_chinese_word <= w <= last_chinese_word:
            is_chinese = True
        else:
            is_chinese = False
            break

    return is_chinese


def all_is_english(word):
    return str.isalnum(word)


def all_is_character(word):
    return str.isalnum(word)


def count_words(lines, clear_cut_words_file, backup_file, test_mode=False):
    total_words = 0
    words_count_map = {}
    # each element in words_count_map is words_count_map[word] = [word, count, frequency]

    clear_cut_file = open(clear_cut_words_file, 'w')

    for index, line in enumerate(lines):
        if test_mode and index > 10: break

        # clear_line = [word for word in line if all_is_chinese_word(word) or all_is_english(word)]
        clear_line = [word for word in line if all_is_character(word)]

        if index % 100 == 0: print(index)

        for word in clear_line:
            if word in words_count_map:
                words_count_map[word] += 1
            else:
                words_count_map[word] = 1

        clear_cut_file.write("\t".join(clear_line)+'\n')

        total_words += 1

    back_up_file_name = backup_file
    back_up_file = open(back_up_file_name, 'w')

    for word, count in words_count_map.items():
        frequency = count/total_words
        back_up_file.write("\t".join([word, str(count), str(frequency)]) + "\n")

    back_up_file.close()


if __name__ == '__main__':
    assert all_is_chinese_word('以上')
    assert all_is_chinese_word('!!以上') == False
    assert all_is_chinese_word('dasja') == False
    assert all_is_english('test is english') == False
    assert all_is_english('test')
    assert all_is_english(';') == False
    assert all_is_english(':') == False
    assert all_is_character(':') == False
    assert all_is_english('AK47')
    assert all_is_character('AK47')
    assert all_is_english('G213')
    assert all_is_english('烫烫烫')
    assert all_is_character('烫烫烫')

    count_words(
        get_lines('cutted_words/wiki_and_news.txt'),
        'cutted_words/clear_wiki_and_news.txt',
        'word_frequence/wiki_and_news_wc.txt',
        test_mode=False
    )
