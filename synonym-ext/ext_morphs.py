import json

fileName = './si_neighbors_unfiltered.json'
outFile = 'si_morphs.json'
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


with open(fileName) as f:
    all_dict = dict()
    all_dict = json.load(f)
    remove_keys = list()
    for word_base in all_dict:
        lst = all_dict[word_base]
        copy_list = list(lst)
        for word in lst:
            if not (word_base[:4] == word[:4]) or not (are_lexically_similar(word_base, word, 0.7)):
                copy_list.remove(word)
        all_dict[word_base] = copy_list
        if len(copy_list) < 2:
            remove_keys.append(word_base)
    for word_base in remove_keys:
        all_dict.pop(word_base, None)
    with open(outFile, "w") as fi:
        json.dump(all_dict, fi, ensure_ascii=False)
