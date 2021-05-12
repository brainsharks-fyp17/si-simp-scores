import json

import matplotlib.pyplot as plt
from sinling import word_splitter
import uuid

sinhala_start = 3456
vowels_and_const_end = 3527
sinhala_end = 3573
dataset_file_path = '../datasets/parallel-08.11.2020-Tr74K.si-en.si'


# dataset_file_path = '/home/rumesh/Downloads/FYP/datasets/common-crawl-si.txt'

def is_sinhala_letter(letter):
    unicode_val = ord(letter)
    # only Sinhala vowels and constants
    if sinhala_start <= unicode_val <= vowels_and_const_end:
        return True


def is_sinhala_word(word):
    # A word has more than one Sinhala letter
    # can have other letters/digits; Ex: 25වන
    letter_count = 0
    for letter in word:
        if is_sinhala_letter(letter):
            letter_count += 1
            if letter_count >= 2:
                return True
    return False


def is_strictly_sinhala_word(word):
    for ch in word:
        unicode_val = ord(ch)
        if not (sinhala_start <= unicode_val <= sinhala_end):
            return False
    return True


def word_length(word):
    length = 0
    for letter in word:
        if is_sinhala_letter(letter):
            length += 1
    return length


def words_in_sentence(sentence):
    count = 0
    words = sentence.split()  # splits on whitespaces
    for word in words:
        if is_sinhala_word(word):
            count += 1
    # if count == 0:
    #     print(sentence)
    return count


def init_complexity(corpus, words_dict):
    # corpus is a list of sentences
    # words_dict key->word | value->number of occurrences
    for sentence in corpus:
        words = sentence.split()
        for word in words:
            # word = word.replace('\u200d', '')
            if is_sinhala_word(word):
                # try:
                #     word = get_base(word)
                # except Exception as e:
                #     print("word:", word)
                #     print(e, "\n")
                if word in words_dict:
                    words_dict[word] += 1
                else:
                    words_dict[word] = 1


def get_base(word):
    return word_splitter.split(word)['base']


def create_orig_to_base_json(lines):
    # lines is a list of sentences
    from sinling import word_splitter as ws
    unique_file_name = str(uuid.uuid4()) + ".json"
    output_dict = dict()
    for sentence in lines:
        words = sentence.split()
        for word in words:
            # word = word.replace('\u200d', '')
            if is_sinhala_word(word):
                try:
                    base = ws.split(word)['base']
                    print(word, base)
                    if word not in output_dict:
                        output_dict[word] = base
                except Exception as e:
                    print("word:", word)
                    print(e, "\n")

    with open(unique_file_name, 'w', encoding='utf8') as outfile:
        json.dump(output_dict, outfile, ensure_ascii=False)


if __name__ == '__main__':
    all_words_dict = dict()
    reverse_dict = dict()
    len_dict = dict()  # key-> length; val-> number of words having that length
    sentence_len_dict = dict()  # key-> sentence; val->length of sentence
    sentence_len_dict_reverse = dict()  # key->legth of a sentence; val-> number of sentences with that length
    dataset = open(dataset_file_path)
    sentences = dataset.readlines()  # all sentences
    init_complexity(sentences, all_words_dict)

    # sentence length analysis
    for sen in sentences:
        sentence_len_dict[sen] = words_in_sentence(sen)
    for val in sentence_len_dict.values():
        if val in sentence_len_dict_reverse.keys():
            sentence_len_dict_reverse[val] += 1
        else:
            sentence_len_dict_reverse[val] = 1
    plt_sent_len = plt.figure("Words per sentence")
    wps_keys = sorted(sentence_len_dict_reverse.keys())
    wps_values = []
    for i in wps_keys:
        wps_values.append(sentence_len_dict_reverse[i])
    plt.bar(wps_keys, wps_values, align='center')
    plt.xticks(wps_keys)
    plt.xlabel("length")
    plt.ylabel("number of sentences with the length")

    # word length analysis
    words_list = all_words_dict.keys()
    for wrd in words_list:
        ln = word_length(wrd)
        if ln in len_dict:
            len_dict[ln] += 1
        else:
            len_dict[ln] = 1

    plot_len = plt.figure("Syllables per Word")
    len_dict_keys_sorted = sorted(len_dict.keys())
    len_dict_values_sorted = []
    for k in len_dict_keys_sorted:
        len_dict_values_sorted.append(len_dict[k])
    plt.bar(len_dict_keys_sorted, len_dict_values_sorted, align='center')
    plt.xticks(len_dict_keys_sorted)
    plt.xlabel("length")
    plt.ylabel("number of unique words")

    # frequency analysis
    for key in all_words_dict:
        val = all_words_dict[key]
        if val in reverse_dict:
            reverse_dict[val] += 1
        else:
            reverse_dict[val] = 1

    plot_freq = plt.figure("Frequency")
    k_rev = sorted(reverse_dict.keys())
    v_rev = []
    for k in k_rev:
        v_rev.append(reverse_dict[k])
    # print(k_rev, len(k_rev))
    # print(v_rev, len(v_rev))
    plt.plot(k_rev, v_rev, '.r')
    plt.xlabel("frequency")
    plt.ylabel("number of words with that frequency")

    unique_num_words = len(all_words_dict.keys())
    total_words = sum(all_words_dict.values())
    print("Total number of words: ", total_words)
    print("Number of unique words: ", unique_num_words)
    print("Number of sentences: ", len(sentences))
    print("Average number of words in a sentence: ", (sum(sentence_len_dict.values()) / len(sentences)))
    print("Max word frequency: ", max(all_words_dict.values()))
    print("Min word frequency: ", min(all_words_dict.values()))
    print("Average word frequency: ", (sum(all_words_dict.values()) / len(all_words_dict)))

    plt.show()
