from jieba import posseg as pseg
from get_right_word import analyse_new_phrase


def get_elements(pos):
    return lambda words_seg: list(filter(lambda w_p: list(w_p)[1] == pos, words_seg))
get_location, get_number = get_elements('ns'), get_elements('m')


def is_m(pos): return pos == 'm'


def is_verb(pos): return pos.startswith('v') or pos.startswith('p')


def get_pos(words_pair):
    return list(words_pair)[1]


def is_pm25_number(words_seg, index):
    return is_m(get_pos(words_seg[index])) and not is_m(get_pos(words_seg[index+1])) and is_verb(get_pos(words_seg[index-1]))


def get_pm_number(words_seg):
    words_seg = list(words_seg)
    numbers = []
    for index in range(len(words_seg)-1):
        if is_pm25_number(words_seg, index):
            numbers.append(list(words_seg[index])[0])
    return numbers


def is_legal_format(words_seg):
    return len(get_location(words_seg)) > 0 and len(get_number(words_seg)) > 0


def main(string):
    words_seg = list(analyse_new_phrase(string))
    if len(words_seg) > 2 and is_legal_format(words_seg):
        return distinct_location(get_location(words_seg)), get_pm_number(words_seg)
    else:
        return None


def distinct_location(locations_pair):
    location = {}
    for w, p in locations_pair:
        if w in location:
            location[w] += 1
        else:
            location[w] = 0
    return [key for key, value in location.items()]


def change_to_segments(string):
    result = main(string)

    return result


def get_the_pair(result):
    length = min(len(result[0]), len(result[1]))
    result_pair = []

    for i in range(length):
        result_pair.append([result[0][i], result[1][i]])

    return result_pair

if __name__ == '__main__':
    test_string_1 = "北京在28日的雾霾高达187，创下了本月的最高"
    test_string = """
    北京局部PM2.5逼近1000，日均浓度值或超标10倍，发布首个霾橙色预警

    郑三波 徐勤

    全国多地严重雾霾天气仍在持续，截至13日零时，记者统计发现，在全国74个监测城市中，有33个城市的部分检测站点检测数据AQI（环境空气质量指数）数值超过300，即空气质量达到了严重污染。北京、河北昨日相继发布了霾橙色预警信号。
　　在这样的环境下，众多网友自嘲为“人肉吸尘器”，调侃称“空气如此糟糕，引无数美女戴口罩”。
　　北京局部PM2.5逼近1000

　　昨日，雾霾仍盘踞京城，北京已连续3天空气质量六级污染。上午9时监测数据显示，除定陵、八达岭、密云水库外，其余区域空气质量指数全部达到极值500，六级严重污染中的“最高级”。北京发布史上首个霾橙色预警。

　　而北京环保监测中心数据显示，12日23时，西直门北、南三环、奥体中心等监测点PM2.5实时浓度突破900微克，西直门北高达每立方米993微克。该中心预计，空气严重污染状况在未来三天仍将持续。

　　“新国标PM2.5日均浓度限值为每立方米75微克，这样看来，PM2.5日均浓度值将可能超标10倍。”北京大学教授朱彤说。
河北发布最高级雾霾预警

　　根据环保部公布的城市空气质量日报显示，12日空气污染指数最高的前10位城市中，河北有5个“上榜”，石家庄的可吸入颗粒物浓度为960微克/立方米。昨日下午4时，河北省气象台发布了最高级别的霾橙色预警信号。

　　从全国城市空气质量实时发布平台来看，华北的京津冀、东北三省、中部陕西、河南、湖北、湖南、安徽，以及东部沿海省市的部分城市，都出现了重度或严重污染，一条深褐色的“污染带”由东北往中部斜向穿越我国大部地区，小半个中国的空气质量都“脏”得要命。

    """
    result = get_the_pair(change_to_segments(test_string))

    print(result)



