"""
This file analysis the complete sentences correlation.

Define: Complete Sentence. e.g 5月24日，省交通运输厅发布消息：今年的端午小长假期间，高速仍正常收取通行费。受高速不免费影响，预计假日车流较“五一”假日有所下降，但高速上下站口仍会出现短时拥堵。

5月24日 and 省交通运输厅发布消息, etc called Sub Sentences, and the whole sentence which is a nature language 
sentence we called Complete Sentence.

"""

from sentence_manager.sentence_embedding import get_sentence_embedding
from sentence_manager.utils import line_to_sentences
from utlis.metrics import cosine
from summary.utils import k_nn
import logging
import numpy as np
from summary.utils import top_n, get_fit_length
from summary.utils import get_complex_correlation
from summary.utils import get_sentences_by_distances
from sentence_manager.utils import add_end_quote
from sentence_manager.utils import delete_bracket


def get_distances_between_segments_and_one_target(segments, target):
    e_f = get_sentence_embedding
    return [cosine(e_f(target), e_f(s)) for s in segments]


def get_distances_between_sentences_and_target_sentence(text, target_sentence, test_sub_string=False, sample=1.):
    sentences = line_to_sentences(text, need_get_substring=test_sub_string)

    if sample < 1:
        sentences = list(np.random.choice(sentences, int(len(sentences) * sample), replace=True))

    distances = get_distances_between_segments_and_one_target(sentences, target_sentence)
    return sentences, distances


def get_knn_distance_with_sentences_and_target_sentence(text, target_sentence, test_sub_string=False, neighbor=1, sample=1.):
    sentences, distances = get_distances_between_sentences_and_target_sentence(text, target_sentence, test_sub_string, sample=sample)
    sentences, distances = zip(*([(s, d)for s, d in zip(sentences, distances) if d != float('inf')]))
    knn_distances = k_nn(distances, neighbor=neighbor)
    return list(zip(sentences, knn_distances))


def is_verbose_begin(sentence, text):
    sub_strings, distances = zip(*get_knn_distance_with_sentences_and_target_sentence(sentence, text, test_sub_string=True, sample=1.))
    distance_mean = np.mean(distances)
    threshold = 1.1

    if distances[0] > distance_mean * threshold:
        return True, sub_strings
    else:
        return False, sub_strings


def clean_sentence_begin(sentence, text):
    verbose_begin, sub_strings = is_verbose_begin(sentence, text)
    if verbose_begin:
        return sentence[len(sub_strings[0])+1:].strip()
    else:
        return sentence


def recursive_clean_sentence_begin(sentence, text):
    new_sentence = clean_sentence_begin(sentence, text)
    if new_sentence != sentence:
        return recursive_clean_sentence_begin(new_sentence, text)
    else:
        return sentence


def get_clean_top_sentences(text, title=None):
    text = delete_bracket(text)
    sentences, text_self_dis = get_distances_between_sentences_and_target_sentence(text, text)

    if title:
        title = delete_bracket(title)
        sentences, text_title_dis = get_distances_between_sentences_and_target_sentence(text, title)
        complex_distances = get_complex_correlation(text_self_dis, text_title_dis)
    else:
        complex_distances = text_self_dis

    fit_length = get_fit_length(len(text))
    top_correlations = top_n(complex_distances, sentences, fit_length, title)
    top_sentences = get_sentences_by_distances(complex_distances, sentences, top_correlations)
    top_sentences = [add_end_quote(s) for s in top_sentences]
    clean_top_sentence = list(map(lambda s: recursive_clean_sentence_begin(s, text), top_sentences))
    summary = "。".join(clean_top_sentence) + '。'
    if title:
        summary = title + ": " + summary
    return summary


def test_text():
    text = """2017-06-28 07:17:24　来源: 封面新闻(成都)
（原标题：复兴号为何运营次日就晚点49分钟？纯属“躺枪”）
6月26日，在北京南站，“复兴号”G123次列车等候出发。当日，中国标准动车组“复兴号”在京沪高铁两端的北京南站和上海虹桥站双向首发，分别担当G123次和G124次高速列车。“复兴号”是由中国铁路总公司牵头组织研制、具有完全自主知识产权、达到世界先进水平的中国标准动车组。
新华社记者鞠焕宗摄
“晚点了49分钟！”6月27日下午，原定16点50分开抵上海虹桥的中国标准动车组G123次“复兴号”，直到17点39分才到达终点。接到读者“报料”，封面新闻第一时间联系了中国铁路客户服务中心，12306接线员向记者证实了G123次晚点一事，但对于原因，表示不知情。
与此同时，封面新闻发现，不少网友也注意到G123次晚点一事，甚至有网友担心“设备故障”，但也有不少网友力挺“复兴号”。一位博主更是坚信晚点跟“标动”本身没有关系，他还呼吁铁路部门发布晚点原因说明，“省的别有用心的人黑‘ 标动’”。
那么“复兴号”为何在开行次日晚点呢？是否真是设备故障呢？因为G123次始发站为北京南站，记者与北京铁路局取得了联系。因晚点发生地并不在北京，该局人员表示，具体原因建议问问上海铁路局。
上海铁路局会给出什么答案呢？该局证实，G123次确实发生了晚点，但晚点的远非“标动”。什么意思？看看补充说明——截至27日18时20分，该局数据显示仅京沪高铁，当日有14趟列车晚点。其中晚点时间最长的是北京南站11点10分发往上海虹桥的G125次，约晚了59分钟。此外，山东烟台发往上海的G459次和北京南站发往上海的G411次，均晚点54分钟。晚点时间最短的G137次，从北京去上海也“迟到”了23分钟。
所以，并非G123一趟车的事。
上海铁路局提供的数据还显示，当日京沪高铁计划开行373列动车组。虽然，晚点14趟并非“大范围”，但包括“复兴号”在内的晚点，究竟是何原因呢？遗憾的是，在这里也没能得到答案，但记者得到了一条重要线索——可能跟G329次有关。
封面新闻随后查询到，G329这趟由天津开往福州的列车确实发生了晚点，而且该线与京沪列车的多地存在“交集”，比如山东的济南、枣庄等。是不是这趟列车在山东出现了问题，导致路过济南、枣庄的G123等列车晚点呢？
封面新闻决定向济南铁路局求证。果不其然！该局证实，G123晚点原因确实是因为G329。原因是G329出现了“设备故障”，而引发晚点的时间，大约在中午12点47分。受影响的列车除了G123，还有G213、G323、G229、G121、G301、G15、G287、G163、G125等。
值得注意的是，济南铁路局提到的这个G125，就是上文上海铁路局提到的晚点时间最长的G125。
对于本次因设备故障引发的包括“复兴号”在内的多趟列车晚点，济南铁路方面直言“深表歉意”，并“敬请旅客谅解”。
多位铁路人士向记者表示，动车出现晚点其实正常，希望大家多理解，更希望不要“猜测”、“联想”。而对于G123次“复兴号”运营次日就遭遇晚点，一位资深铁路迷更是笑言，纯属“躺枪！”
免责声明：本文仅代表作者个人观点，与环球网无关。其原创性以及文中陈述文字和内容未经本站证实，对本文以及其中全部或者部分内容、文字的真实性、完整性、及时性本站不作任何保证或承诺，请读者仅作参考，并请自行核实相关内容。"""

    title = """复兴号为何运营次日就晚点49分钟？官方回应"""

    result = get_knn_distance_with_sentences_and_target_sentence(text, title, neighbor=1)

    for r in result:
        print(r)


def test_sentence():
    text = """腾讯体育5月23日讯 昨天，国际滑联（ISU）正式公布了2017/18赛季花样滑冰项目各大小赛程：于明年2月9日-25日在韩国平昌举行的第二十三届冬奥会，将成为检验索契之后各国花滑健儿们四年训练成果的大考！值得一提的是，根据赛程安排来看，本届冬奥会将在农历春节期间进行，相信会给广大中国观众不一样的观赛体验。
虽然每个赛季花样滑冰从8月份开始就有一些B级赛和挑战赛，但传统还是以每年10月份的大奖赛成年组分站赛作为赛季正式揭幕。今年首个成年组大奖赛分站赛是10月20-22日在莫斯科举行的俄罗斯站，然后按照每周一站的速度，分别在加拿大、中国、日本、法国和美国接力进行，并于12月7-10日在日本名古屋进行总决赛。
作为“未来的成年组如今的青年组”，每个赛季的青年组大奖赛则都会从8月份就开始，今年总共有7站，从8月21日的澳大利亚布里斯班开始，直到10月11日的意大利博尔扎诺结束。
值得一提的是，因为2022年北京冬奥会的成功申办，国家在加大花样滑冰项目投入的同时，经过几年的铺垫，国际滑联也开始有计划的在中国展开花滑推广计划。比如今年中国杯虽然是11月初在北京举行，但在11月24-26日，则将有“上海杯”这项B级赛事落户上海，以飨同好。
相比2017年还是有条不紊一张一弛的赛事节奏，从2018年1月15日的欧锦赛开始，则是一大波赛事接二连三袭来，足以称得上让广大选手只有招架之力没有喘息之机：欧锦赛结束后，无缝衔接进入于1月22日-27日在中国台北举行的四大洲比赛；四大洲赛后两个星期，平昌冬奥会也将上线！
而在平昌冬奥会结束10天后，世青赛将在保加利亚索菲亚上演，3月19日的意大利米兰世锦赛，同样会让经历了冬奥会大考的选手们疲于应付。
值得一提的是，4月13-15日，团体洲际挑战赛也将在美国上演。可以说，对于花滑选手们来说，他们将迎来一个无比忙碌的2017-18赛季。而他们能以怎么样的竞技状态来应对这样高密度赛事的挑战，给观众带来精彩的比赛，让我们拭目以待 。而作为作为中国重点夺牌项目的花样滑冰，尤其是隋文静/韩聪领衔的双人滑能有什么样的表现，同样牵动了全中国人的心。
免责声明：本文仅代表作者个人观点，与环球网无关。其原创性以及文中陈述文字和内容未经本站证实，对本文以及其中全部或者部分内容、文字的真实性、完整性、及时性本站不作任何保证或承诺，请读者仅作参考，并请自行核实相关内容。"""

    test_line = '腾讯体育5月23日讯 昨天，国际滑联,正式公布了2017/18赛季花样滑冰项目各大小赛程：于明年2月9日-25日在韩国平昌举行的第二十三届冬奥会，将成为检验索契之后各国花滑健儿们四年训练成果的大考'
    title = '2017/18花滑赛程 赛事紧凑冬奥会是大考'

    sub_distances = get_knn_distance_with_sentences_and_target_sentence(test_line, text, test_sub_string=True)

    for dis in sub_distances:
        print(dis)


def test_get_summary():
    text = """央广网大连5月24日消息（记者张四清 通讯员阎晓雨 刘芳婉）日前，大连旅顺开发区首批2家“双新”组织工会联合会。新成立的“双新”组织工会联合会分别是海花社区人来旺市场工会联合会和海霞社区小行业工会联合会，可切实维护17个工会小组的1800名自由职业者和小行业从业者的合法权益。
    旅顺开发区是国家级开发区，伴随着旅顺开发区经济社会的稳步发展，外来入驻企业、高校和购房落户人口逐年增多，超市、餐饮、大市场等一批新经济和社会组织随之发展，从业人员持续增加。从今年年初开始，旅顺开发区总工会率先着手筹备建立小行业工会联合会，在海花、海霞2家“双新”小行业比较集中的社区，探索建立适应“双新”组织从业者服务需求的新途径。
    新成立的小行业工会联合会将分散的职工凝聚到工会大家庭中来，依法维护职工政治、社会、经济、文化生活等合法权益，充分调动和发挥职工的积极性。以工会联合会建设为依托，对职工进行上岗培训、法律援助、劳动保障等服务，畅通职工的诉求渠道，为构建和谐稳定的劳动关系起到积极作用。
    免责声明：本文仅代表作者个人观点，与环球网无关。其原创性以及文中陈述文字和内容未经本站证实，对本文以及其中全部或者部分内容、文字的真实性、完整性、及时性本站不作任何保证或承诺，请读者仅作参考，并请自行核实相关内容。"""

    title = '旅顺开发区率先成立2家“双新”组织工会联合会'

    summary = get_clean_top_sentences(text, title)

    return summary


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # test_text()
    # test_sentence()

    summary = test_get_summary()
    print(summary)