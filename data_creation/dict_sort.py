from collections import OrderedDict
import json

in_file = '/home/rumesh/Downloads/FYP/datasets/word_counts_beauti.json'
out_file = 'word_counts_beauti_sorted_by_freq.json'
d = json.load(open(in_file))
# od = OrderedDict(sorted(d.items()))
od = OrderedDict()
for k, v in sorted(d.items(), key=lambda item: item[1]):
    od.update({k: v})

json.dump(od, open(out_file, "w"), ensure_ascii=False, indent=4)
