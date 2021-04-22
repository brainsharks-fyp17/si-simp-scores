from cal_scores import create_orig_to_base_json

file_name = "../datasets/test.complexity.si"
file_cont = open(file_name,encoding='utf8')
lines = file_cont.readlines()
create_orig_to_base_json(lines)

