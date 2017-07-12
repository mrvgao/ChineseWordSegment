import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance

random_choosen = "进一步提高  紫外线  201  Galaxy  免疫学  三角  海浪  出版业  西瓜  得不到  范围  天水  省级  可谓  IP  大海  当场  指定  应用软件  ①    菌  丛林  exe  有余  合肥  衣  外层  口语  CECT  晚清  精度  银奖  大赛  改革"
wiki_choosen = "大陆 政府 FULL 酷狗 葡萄牙文 22 对 对象 experience 最有 解释 2 舆论 崔胜铉 进行 性能 7 影响力 形成 总汇 2008 a 三个 首尔 ALIVE 提到 中文 推出 其中 度过 巡回演唱 地区 英文 Act"
web_news = "长足进展  主席  或  被  电视机  申请  只是  搬走  气愤  长期  方式  有  父子俩  项目  几十年  丰富  构成威胁  遭到  精神病  坚决  新闻  将  经济体  注入  多年  称  特点  人民法院  共识  访问学者  驳回  伤害  进展"

target_words = random_choosen.split() + wiki_choosen.split() + web_news.split()

assert len(target_words) == 100, len(target_words)
