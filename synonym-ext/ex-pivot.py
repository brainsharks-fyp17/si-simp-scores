import json
import string
from collections import OrderedDict
inp = open("ppdb-si-pivot.txt").readlines()
sinhala_start = 3456
vowels_and_const_end = 3527
sinhala_end = 3573
def is_sinhala_letter(letter):
    unicode_val = ord(letter)
    # only Sinhala vowels and constants
    if sinhala_start <= unicode_val <= vowels_and_const_end:
        return True


def are_lexically_similar(word1: str, word2: str, similarity_percentage=0.3):
    word1_set = set([x for x in word1 if is_sinhala_letter(x)])  # get characters
    word2_set = set([x for x in word2 if is_sinhala_letter(x)])
    inter = len(word1_set.intersection(word2_set))  # intersection
    return inter > len(word1_set) * similarity_percentage  # intersection contains more than 30% of the first word


alnum = set(string.ascii_letters + string.digits)
lines = []
for line in inp:
    if len(set(line) & alnum) == 0:
        lines.append(line.strip()[:-1])
lines = [{line.split(":")[0]:line.strip().split(":")[1].strip().split(",")} for line in lines]

out = dict()
for i in lines:
    k = list(i.keys())[0]
    vals = list(i.values())[0]
    out[k] = []
    for j in vals:
        if not are_lexically_similar(k, j, 0.8):
            out[k].append(j)
    if len(out[k]) == 0:
        del out[k]

output_file = open("ppdb-si-p.txt", "w")
word_counts = json.load(open("/home/rumesh/Downloads/FYP/datasets/word_counts_beauti.json"))

for k in out:
    lst = [k]
    lst.extend(out[k])
    freq_dic = dict()
    for word in lst:
        if word in word_counts:
            freq_dic[word] = word_counts[word]
        else:
            freq_dic[word] = -1
    if len(freq_dic) < 2:
        continue
    freq_dict_sorted = sorted(freq_dic.items(), key=lambda item: item[1], reverse=False)
    for i in range(1, len(freq_dict_sorted)):
        # print(freq_dict_sorted[0],",",freq_dict_sorted[i],"\n")
        output_file.write(freq_dict_sorted[0][0]+","+freq_dict_sorted[i][0]+"\n")






