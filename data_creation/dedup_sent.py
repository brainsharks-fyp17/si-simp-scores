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
    dis_sim = abs(lw1 - lw2) / ((lw1 + lw2) / 2)  # calculate dissimilarity ration of length
    # quickly remove sentences of very dissimilar lengths
    if 1 - dis_sim < length_per:
        return False
    word1_set = set([x for x in word1 if is_sinhala_letter(x)])  # get characters
    word2_set = set([x for x in word2 if is_sinhala_letter(x)])
    inter = len(word1_set.intersection(word2_set))  # intersection
    return inter > len(word1_set) * char_per  # intersection contains more than 80% of the first word
