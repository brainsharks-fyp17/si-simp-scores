sinhala_start = 3456
vowels_and_const_end = 3527
sinhala_end = 3573


def is_sinhala_letter(letter):
    unicode_val = ord(letter)
    # only Sinhala vowels and constants
    if sinhala_start <= unicode_val <= vowels_and_const_end:
        return True


def is_similar_sent(word1: str, word2: str, length_per=0.8, char_per=0.8):
    """
    check whether two tokenized sentences are similar
    """
    lw1 = len(word1)
    lw2 = len(word2)
    dis_sim = abs(lw1 - lw2) / ((lw1 + lw2) / 2)  # calculate dissimilarity ratio of length
    # quickly remove sentences of very dissimilar lengths
    if 1 - dis_sim < length_per:
        return False
    word1_set = set([x for x in word1 if is_sinhala_letter(x)])  # get characters
    word2_set = set([x for x in word2 if is_sinhala_letter(x)])
    inter = len(word1_set.intersection(word2_set))  # intersection
    return inter > len(word1_set) * char_per  # intersection contains more than 80% of the first word


if __name__ == '__main__':
    file = '/home/rumesh/IdeaProjects/python/tokenize/all_data_dedupe_si.txt'
    similar_pairs = []
    with open(file) as file_data:
        lines = file_data.readlines()
    print("Number of sentences: " + str(len(lines)))
    for i in lines:
        i = i.strip()
        for j in lines:
            j = j.strip()
            if i == j:  # strictly equals are removed already
                continue
            if is_similar_sent(i, j):
                similar_pairs.append((i, j))
    print(len(similar_pairs))
    write_file = open("out.txt")
    for i in similar_pairs:
        i1, i2 = i[0], i[1]
        write_file.write(i1 + "\t" + i2 + "\n")

