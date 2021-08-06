from collections import OrderedDict
import json

in_file = 'comp_words.json'
out_file = 'comp_words_beauti_sorted_by_freq.json'
d = json.load(open(in_file))
# od = OrderedDict(sorted(d.items()))
od = OrderedDict()
count = 0
limit = 20000
for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True):
    count += 1
    if count > limit:
        break
    od.update({k: v})

json.dump(od, open(out_file, "w"), ensure_ascii=False, indent=4)
