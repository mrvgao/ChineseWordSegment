from sentence_manager.utils import line_to_sentences
from sentence_manager.sentence_embedding import get_sentence_words_vector
from sentence_manager.sentence_embedding import get_vector_dimension
from sentence_manager.sentence_embedding import get_random_vector
from sentence_manager.sentence_embedding import change_none_vector_to_real_value
import numpy as np

text = """【环球网报道 记者 朱佩】英国首相特蕾莎∙梅日前称，由于曼彻斯特恐袭案，该国恐怖威胁级别从“严重”提高至“危急”。这意味着可能派遣军队保障安全。据俄新社5月24日报道，伦敦警察厅反恐部门负责人马克•罗利表示，希望恐怖威胁级别不会太长时间维持在最高级别。
罗利在回答恐怖威胁“危急”水平制度要维持多久的问题时说道：“我不想预测未来，但如果你看看我们的历史，这样一个威胁级别是非常不寻常和罕见的措施。它从未维持很久，我们也希望这样。但在这样一个高风险期我们将竭尽所能，军队将帮助我们。”
当地时间5月22日晚，自杀式恐怖分子在曼彻斯特竞技场音乐厅内实施了爆炸。爆炸造成22人死亡，59人受伤。伤亡者中有许多儿童。至少有8人失踪，恐怖组织“伊斯兰国”声称对爆炸负责。"""

sentences = line_to_sentences(text)

assert sentences[0] == '环球网报道'
assert sentences[1] == '记者'
assert sentences[3] == '英国首相特蕾莎∙梅日前称'
assert sentences[-1] == '恐怖组织“伊斯兰国”声称对爆炸负责'


sentences = line_to_sentences(text, need_get_substring=False)

assert sentences[0] == '环球网报道'
assert sentences[1] == '记者'
assert sentences[3] == '英国首相特蕾莎∙梅日前称，由于曼彻斯特恐袭案，该国恐怖威胁级别从“严重”提高至“危急”'
assert sentences[-1] == '至少有8人失踪，恐怖组织“伊斯兰国”声称对爆炸负责'

test_string = sentences[3]

vectors = get_sentence_words_vector(test_string)

assert isinstance(vectors[0], np.ndarray)

_array = [[0, 1], None, [2, 2], [3, 3], None]

dimension = get_vector_dimension(_array)
assert dimension == 2

_array = [None, [2, 2], [3, 3], None]
dimension = get_vector_dimension(_array)
assert dimension == 2

_array = [None, None]
dimension = get_vector_dimension(_array)
assert dimension is None

random_vector = get_random_vector(25)

assert random_vector.shape == (25,)

vectors = change_none_vector_to_real_value(vectors)

assert np.all(vectors is not None)

print('test done!')
