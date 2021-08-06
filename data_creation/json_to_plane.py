import json

in_file = 'simp_words_beauti_sorted_by_freq.json'
out_file = open("frequent_simp.list", "w")
d = json.load(open(in_file))
for k, v in d.items():
    out_file.write(k + "\n")
out_file.close()
