import re
from mtranslate import translate

out_file = open("wordnet-si.txt", "w")
word_net_lines = open("core-wordnet.txt").readlines()
for line in word_net_lines:
    line = re.sub(" \d+", "", line)
    line = line.strip().replace("[", "").replace("]", "").replace(":", "")
    line = re.sub(r"(^|\W)\d+", " ", line)[2:]
    line = re.sub(r'[0-9]', " ", line)
    lst = list(set(line.split(" ")))
    print(lst)
    for word in lst:
        if not word:
            continue
        out_file.write(str(translate(word, 'si')) + "\t")
    out_file.write("\n")
    out_file.flush()
out_file.close()
