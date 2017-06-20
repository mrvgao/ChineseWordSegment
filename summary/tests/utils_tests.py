from summary.utils import delete_bracket

input = '北京办公楼外北侧的雨水收集池（位于建筑物20米开外）起火，原因是工人操作不当，' \
        '引燃了塑料材料。目前火已扑灭，现场无人员伤亡，感谢大家的关心。”'

output = '北京办公楼外北侧的雨水收集池起火，原因是工人操作不当，' \
         '引燃了塑料材料。目前火已扑灭，现场无人员伤亡，感谢大家的关心。”'

assert(delete_bracket(input)) == output

print('test done!')