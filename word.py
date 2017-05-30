CONNECT = 'c'
VERB = 'v'
VERB_SHI = 'vshi'
VERB_YOU = 'vyou'
ZHU_CI = 'u', # 助
FANG_WEI_CI = 'f', #方位词
FEI_YU_SU = 'x', #非语
BIAO_DIAN = 'w'
JIE_CI = 'p'
TAN_CI = 'e'
DAI_CI = 'r'
FU_CI = 'd'
ZHUANG_TAI_CI = 'z'


class Word:
    def __init__(self, word, pos, probability=0):
        self.word = word
        self.probability = probability
        self.pos = pos

    def need_connect(self, phrase_strip=True):
        skip_poses = [VERB_SHI,VERB_YOU, ZHU_CI, FANG_WEI_CI, FEI_YU_SU, BIAO_DIAN, JIE_CI, TAN_CI, DAI_CI, FU_CI]
        if phrase_strip:
            skip_poses += [VERB, CONNECT]
        if any([self.pos.startswith(x) for x in skip_poses]):
            return False
        else:
            return True

    def is_verb(self):
        return self.pos.startswith(VERB)

    def can_be_end(self):
        end_strip = [CONNECT, VERB, ZHUANG_TAI_CI]
        if any([self.pos.startswith(x) for x in end_strip]):
            return False
        else:
            return True

    def can_be_begin(self):
        begin_strip = [CONNECT, ZHUANG_TAI_CI]

        if any([self.pos.startswith(x) for x in begin_strip]):
            return False
        else:
            return True


class Segment:
    def __init__(self, init=None):
        self.word_segment = []

        if init:
            self.word_segment.append(init)

    def __getitem__(self, item):
        return self.word_segment[item]

    def __len__(self):
        return len(self.word_segment)

    def append(self, word):
        self.word_segment.append(word)

    def get_new_phrase(self):
        # self.strip()

        if len(self) > 1:
            probs = [x.probability for x in self.word_segment]
            return self.merge(), sum(probs[1:])/(len(probs)-1)
        else:
            return None, 0

    def merge(self):
        return "".join([x.word for x in self.word_segment])

    def strip_r(self):
        while len(self.word_segment) > 1 and not self.word_segment[-1].can_be_end():
            del self.word_segment[-1]

    def strip(self):
        self.strip_r()
        self.strip_l()

    def strip_l(self):
        while len(self.word_segment) > 1 and not self.word_segment[0].need_connect(phrase_strip=True):
            del self.word_segment[0]
