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


def get_distances_between_sentences_and_target_sentence(text, target_sentence):
    target_vector = get_sentence_embedding(target_sentence)
    sentences = line_to_sentences(text, need_get_substring=False)
    distances = []
    for s in sentences:
        s_v = get_sentence_embedding(s)
        distance = cosine(target_vector, s_v)
        distances.append(distance)
    return sentences, distances


def get_knn_distance_with_sentences_and_target_sentence(text, target_sentence, neighbor=1):
    sentences, distances = get_distances_between_sentences_and_target_sentence(text, target_sentence)
    knn_distances = k_nn(distances, neighbor=neighbor)
    return list(zip(sentences, knn_distances))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
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

    result = get_knn_distance_with_sentences_and_target_sentence(text, title, neighbor=2)

    for r in result:
        print(r)

