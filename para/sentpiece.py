import sentencepiece as spm
from string import digits, punctuation

from tqdm import tqdm

sp = spm.SentencePieceProcessor(model_file='sent-mbart.model')
file_name = "tokenized_shard_100000.txt"
original = [i.strip() for i in open(file_name).readlines()]
text = original


def clean_sent(sent):
    remove_digits = str.maketrans('', '', digits)
    sent = sent.translate(remove_digits)
    sent = sent.translate(str.maketrans('', '', punctuation))
    sent = " ".join(sent.strip().split())
    return sent


def set_similarity(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0
    # if (len(s2) - len(s1)) / len(s1) > 0.5:
    #     return 0
    return round(len(s1.intersection(s2)) / len(s1.union(s2)), 2)


text = [clean_sent(i) for i in text]
encoded = sp.encode(text, out_type=int)
sets = []
out_file_name = "para_"+file_name+".tsv"
outfile = open(out_file_name, "w")
for i in encoded:
    sets.append(set(i))
for i in tqdm(range(len(sets))):
    for j in range(i + 1, len(sets)):
        sim = set_similarity(sets[i], sets[j])
        if sim > 0.3:
            # print(original[i], '\n', original[j])
            outfile.write(original[i] + "\t" + original[j] + "\n")
            outfile.flush()
outfile.close()
